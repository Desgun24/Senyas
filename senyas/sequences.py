"""Timed semantic feature sequences, motion descriptors, and portable caches."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import numpy as np

from .features import _prioritize_windows_system_dlls
from .semantic_features import FEATURE_NAMES, SPECS, FINGERS, SIDES, SCHEMA_VERSION, FeatureSpec, extract_result


def motion_specs():
    result = []
    for side in SIDES:
        for axis in "xy":
            result.append(FeatureSpec(f"{side}.motion.velocity_{axis}", "movement", "shoulder_units_per_second", "Timestamp-aware wrist derivative; camera-plane direction"))
        result.append(FeatureSpec(f"{side}.motion.speed", "movement", "shoulder_units_per_second", "Camera-plane wrist speed"))
        result.append(FeatureSpec(f"{side}.motion.signed_turn_rate", "movement", "radians_per_second", "Turning of wrist velocity: positive anticlockwise in unmirrored image; not proof of a complete circle"))
        for finger in FINGERS:
            result.append(FeatureSpec(f"{side}.{finger}.relative_tip_speed", "finger_motion", "palm_units_per_second", "Fingertip motion in rotating palm frame; distinguishes articulation from whole-hand translation"))
            result.append(FeatureSpec(f"{side}.{finger}.sway_extent", "finger_motion", "palm_units", "Fingertip extent within a trailing half-second in palm coordinates"))
            result.append(FeatureSpec(f"{side}.{finger}.sway_reversals", "finger_motion", "reversals_per_second", "Largest axis reversal rate in trailing half-second; a wiggle proxy, not a flailing label"))
            result.append(FeatureSpec(f"{side}.{finger}.shape_variation", "finger_motion", "fraction_pi", "Largest joint-flexion standard deviation in trailing half-second; separates shape change from stable-shape sway"))
    for axis in ("x", "y"):
        result.append(FeatureSpec(f"head.motion.{axis}_velocity", "head_motion_proxy", "ear_units_per_second", "Derivative of nose position relative to ears; NOT a verified nod/shake label", True))
    return tuple(result)


MOTION_SPECS = motion_specs()
SEQUENCE_SPECS = SPECS + MOTION_SPECS
SEQUENCE_NAMES = tuple(spec.name for spec in SEQUENCE_SPECS)


@dataclass
class FeatureSequence:
    values: np.ndarray
    timestamps: np.ndarray
    metadata: dict = field(default_factory=dict)
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)
    feature_names: tuple[str, ...] = SEQUENCE_NAMES

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if self.values.shape != (len(self.timestamps), len(self.feature_names)) or not len(self.timestamps):
            raise ValueError("Sequence features must have one row per timestamp and match the named schema")
        if not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if np.isinf(self.values).any():
            raise ValueError("Use NaN for missing measurements, never infinity")


def timed_derivative(values, timestamps, max_gap=.16):
    values = np.asarray(values, dtype=float)
    result = np.full_like(values, np.nan)
    if len(values) < 2:
        return result
    dt = np.diff(timestamps)
    valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    if values.ndim == 2:
        valid &= (dt <= max_gap)[:, None]
        result[1:] = np.where(valid, np.diff(values, axis=0) / dt[:, None], np.nan)
    else:
        valid &= dt <= max_gap
        result[1:] = np.where(valid, np.diff(values) / dt, np.nan)
    return result


def add_motion(base_values, timestamps):
    base_values = np.asarray(base_values, dtype=np.float32)
    if base_values.shape != (len(timestamps), len(FEATURE_NAMES)):
        raise ValueError("Base feature schema mismatch")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Motion features need finite increasing timestamps")
    lookup = {name: i for i, name in enumerate(FEATURE_NAMES)}
    data = {}
    for side in SIDES:
        positions = base_values[:, [lookup[f"{side}.wrist.body_{axis}"] for axis in "xy"]]
        velocity = timed_derivative(positions, timestamps)
        for i, axis in enumerate("xy"):
            data[f"{side}.motion.velocity_{axis}"] = velocity[:, i]
        speed = np.linalg.norm(velocity, axis=1)
        data[f"{side}.motion.speed"] = speed
        turning = np.full(len(timestamps), np.nan)
        if len(timestamps) > 2:
            first, second = velocity[1:-1], velocity[2:]
            cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
            dot = np.sum(first * second, axis=1)
            valid = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
            valid &= (np.linalg.norm(first, axis=1) > .02) & (np.linalg.norm(second, axis=1) > .02)
            valid &= np.diff(timestamps)[1:] <= .16
            turning[2:] = np.where(valid, np.arctan2(cross, dot) / np.diff(timestamps)[1:], np.nan)
        data[f"{side}.motion.signed_turn_rate"] = turning
        for finger in FINGERS:
            points = base_values[:, [lookup[f"{side}.{finger}.local_tip_{axis}"] for axis in "xyz"]]
            tip_velocity = timed_derivative(points,timestamps)
            data[f"{side}.{finger}.relative_tip_speed"] = np.linalg.norm(tip_velocity, axis=1)
            extent,reversals,variation = [np.full(len(timestamps),np.nan) for _ in range(3)]
            joints = ('base','tip') if finger=='thumb' else ('base','middle','tip')
            flexion = base_values[:,[lookup[f'{side}.{finger}.{joint}_flexion'] for joint in joints]]
            for i,current in enumerate(timestamps):
                start = int(np.searchsorted(timestamps,current-.5))
                window = points[start:i+1]
                if len(window)<5 or not np.isfinite(window).all() or np.any(np.diff(timestamps[start:i+1])>.16):
                    continue
                extent[i] = float(np.linalg.norm(np.ptp(window,axis=0)))
                rates=[]
                for axis in range(3):
                    speed=tip_velocity[start:i+1,axis]
                    significant=speed[np.isfinite(speed)&(np.abs(speed)>.1)]
                    flips=np.count_nonzero(np.diff(np.sign(significant))) if len(significant)>1 else 0
                    rates.append(flips/(current-timestamps[start]))
                reversals[i]=max(rates)
                shape=flexion[start:i+1]
                if np.isfinite(shape).all():
                    variation[i]=float(np.std(shape,axis=0).max())
            data[f'{side}.{finger}.sway_extent']=extent
            data[f'{side}.{finger}.sway_reversals']=reversals
            data[f'{side}.{finger}.shape_variation']=variation
    for axis in "xy":
        data[f"head.motion.{axis}_velocity"] = timed_derivative(base_values[:, lookup[f"head.nose_relative_{axis}"]], timestamps)
    return np.column_stack([base_values, *[data[spec.name] for spec in MOTION_SPECS]]).astype(np.float32)


def circle_measurements(sequence: FeatureSequence, side: str):
    """Descriptive trajectory diagnostics; never assert a circle from turning alone."""
    columns = [sequence.feature_names.index(f"{side}.wrist.body_{axis}") for axis in "xy"]
    points = sequence.values[:, columns]
    valid = np.isfinite(points).all(axis=1)
    if valid.mean() < .8 or valid.sum() < 12:
        return {"available": False, "reason": "insufficient tracked wrist positions"}
    # Missing internal runs must not be joined as if continuously observed.
    if np.any(np.diff(sequence.timestamps[np.where(valid)[0]]) > .16):
        return {"available": False, "reason": "tracking gap"}
    points = points[valid].astype(float)
    origin = points.mean(axis=0)
    centered = points - origin
    path_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if path_length < .1:
        return {"available": False, "reason": "too little wrist movement"}
    matrix = np.column_stack([2 * centered[:, 0], 2 * centered[:, 1], np.ones(len(points))])
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.sum(centered**2, axis=1), rcond=None)
    if rank < 3:
        return {"available": False, "reason": "trajectory is approximately a line"}
    radii = np.linalg.norm(centered - solution[:2], axis=1)
    radius = float(radii.mean())
    angles = np.unwrap(np.arctan2(centered[:, 1]-solution[1], centered[:, 0]-solution[0]))
    total = float(angles[-1] - angles[0])
    return {"available": True, "signed_turns": total / (2*np.pi),
            "viewpoint": "unmirrored camera image; positive is anticlockwise",
            "radius_shoulder_units": radius, "radius_relative_std": float(radii.std()/max(radius, 1e-8)),
            "closure_gap_over_path": float(np.linalg.norm(points[-1]-points[0])/path_length),
            "path_length_shoulder_units": path_length}


def save_sequence(path: str | Path, sequence: FeatureSequence):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, values=sequence.values, timestamps=sequence.timestamps,
                        feature_names=np.asarray(sequence.feature_names), schema=np.array(SCHEMA_VERSION),
                        metadata=np.array(json.dumps(sequence.metadata, allow_nan=False)),
                        **{f"landmarks_{key}": value for key, value in sequence.landmarks.items()})


def load_sequence(path: str | Path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != SCHEMA_VERSION or tuple(data["feature_names"].tolist()) != SEQUENCE_NAMES:
            raise ValueError("Unsupported semantic sequence schema")
        return FeatureSequence(data["values"], data["timestamps"], json.loads(str(data["metadata"])),
                               {name.removeprefix("landmarks_"): data[name] for name in data.files if name.startswith("landmarks_")})


class SequenceExtractor:
    """One tracking session per complete clip; no detector state crosses clips."""
    def __init__(self, target_fps=30., max_image_side=480, model_complexity=0):
        if not 1 <= target_fps <= 60 or max_image_side < 160:
            raise ValueError("Invalid sequence extraction settings")
        _prioritize_windows_system_dlls()
        import mediapipe as mp
        import cv2
        self.cv2 = cv2
        self.target_fps = float(target_fps)
        self.max_image_side = int(max_image_side)
        self.model_complexity = model_complexity
        self.holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=model_complexity,
            smooth_landmarks=True, enable_segmentation=False, refine_face_landmarks=False,
            min_detection_confidence=.5, min_tracking_confidence=.5)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.holistic.close()

    def _process(self, frame):
        height, width = frame.shape[:2]
        scale = min(1., self.max_image_side/max(height, width))
        if scale < 1:
            frame = self.cv2.resize(frame, None, fx=scale, fy=scale)
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        started = time.perf_counter()
        result = self.holistic.process(rgb)
        landmark_ms = (time.perf_counter()-started)*1000
        started = time.perf_counter()
        vector, arrays = extract_result(result, frame.shape[1]/frame.shape[0])
        return vector, arrays, landmark_ms, (time.perf_counter()-started)*1000

    def _assemble(self, observations, timestamps, started, source):
        if not observations:
            raise ValueError("No frames were decoded")
        base = np.vstack([item[0] for item in observations])
        values = add_motion(base, np.asarray(timestamps))
        hand_indices = [FEATURE_NAMES.index(f"{side}.present") for side in SIDES]
        hand_frames = np.any(base[:, hand_indices] > .5, axis=1)
        elapsed = time.perf_counter()-started
        landmark_ms = np.array([item[2] for item in observations])
        geometry_ms = np.array([item[3] for item in observations])
        metadata = {"schema": SCHEMA_VERSION, "source": source, "target_fps": self.target_fps,
            "frames": len(values), "hand_frames": int(hand_frames.sum()), "hand_coverage": float(hand_frames.mean()),
            "recording_duration_seconds": float(timestamps[-1]-timestamps[0]),
            "sampled_fps": float((len(values)-1)/(timestamps[-1]-timestamps[0])) if len(values)>1 else None,
            "wall_seconds": elapsed, "throughput_fps": len(values)/elapsed,
            "landmark_ms_mean": float(landmark_ms.mean()), "landmark_ms_p95": float(np.percentile(landmark_ms,95)),
            "geometry_ms_mean": float(geometry_ms.mean()), "geometry_ms_p95": float(np.percentile(geometry_ms,95)),
            "max_image_side": self.max_image_side, "model_complexity": self.model_complexity,
            "tracking_mode": True, "semantic_labels_trained": False,
            "meaning": "Numerical measurements only; no word recognizer or named hand-shape classifier is fitted"}
        landmarks = {key: np.stack([item[1][key] for item in observations]) for key in ("left", "right", "pose", "face")}
        return FeatureSequence(values, np.asarray(timestamps), metadata, landmarks)

    def extract_video(self, path: str | Path, max_seconds=10.):
        self.holistic.reset()
        started = time.perf_counter()
        capture = self.cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError("Video could not be opened")
        native_fps = float(capture.get(self.cv2.CAP_PROP_FPS))
        if not np.isfinite(native_fps) or native_fps <= 0:
            capture.release()
            raise ValueError("Video needs a valid frame rate for timed motion")
        observations, timestamps = [], []
        index, next_time = 0, 0.
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = index/native_fps
                index += 1
                if timestamp > max_seconds:
                    break
                if timestamp + 1e-7 < next_time:
                    continue
                observations.append(self._process(frame))
                timestamps.append(timestamp)
                next_time = len(observations)/min(native_fps, self.target_fps)
        finally:
            capture.release()
        return self._assemble(observations, timestamps, started, str(path))

    def extract_frames(self, frames, timestamps):
        times = np.asarray(timestamps, dtype=float)
        if len(frames) != len(times) or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError("One strictly increasing finite timestamp per camera frame is required")
        if len(times) > 600:
            raise ValueError("Attempt exceeds the sequence frame budget")
        self.holistic.reset()
        started = time.perf_counter()
        observations = [self._process(frame) for frame in frames]
        return self._assemble(observations, times-times[0], started, "live_camera")
