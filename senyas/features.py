from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

_WINDOWS_SYSTEM_DLL_HANDLE: object | None = None


def _prioritize_windows_system_dlls() -> None:
    """Prevent third-party runtime folders from shadowing Windows C++ DLLs."""
    global _WINDOWS_SYSTEM_DLL_HANDLE
    if os.name != "nt":
        return

    system32 = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
    if _WINDOWS_SYSTEM_DLL_HANDLE is None and hasattr(os, "add_dll_directory"):
        _WINDOWS_SYSTEM_DLL_HANDLE = os.add_dll_directory(system32)
    system32_key = os.path.normcase(os.path.normpath(system32))
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    path_entries = [
        entry
        for entry in path_entries
        if os.path.normcase(os.path.normpath(entry)) != system32_key
    ]
    os.environ["PATH"] = os.pathsep.join([system32, *path_entries])


_prioritize_windows_system_dlls()

import numpy as np
import pandas as pd


HAND_LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)


@dataclass(frozen=True)
class ExtractionInfo:
    frames_total: int
    frames_sampled: int
    hand_frames: int
    hand_detection_rate: float
    fps: float
    duration_seconds: float
    extraction_ms: float


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return 0.0
    cosine = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return float(np.arccos(cosine) / np.pi)


def frame_feature_names() -> list[str]:
    names = ["pose_present", "shoulder_width"]
    for side in ("left", "right"):
        names.extend(
            [
                f"{side}.present",
                f"{side}.wrist_abs_x",
                f"{side}.wrist_abs_y",
                f"{side}.wrist_body_x",
                f"{side}.wrist_body_y",
                f"{side}.palm_scale",
            ]
        )
        for landmark in HAND_LANDMARK_NAMES:
            names.extend(
                [
                    f"{side}.{landmark}_x",
                    f"{side}.{landmark}_y",
                    f"{side}.{landmark}_z",
                ]
            )
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            names.append(f"{side}.{finger}_bend")
            names.append(f"{side}.{finger}_tip_distance")
    names.extend(["both_hands_present", "hands_distance", "hands_vertical_offset"])
    return names


def temporal_feature_names(base_names: Sequence[str] | None = None) -> list[str]:
    base_names = list(base_names or frame_feature_names())
    statistics = (
        "mean",
        "std",
        "min",
        "max",
        "range",
        "start",
        "end",
        "delta",
        "velocity_mean",
        "velocity_max",
    )
    return [f"{name}.{statistic}" for name in base_names for statistic in statistics]


def _interpolate_missing(sequence: np.ndarray) -> np.ndarray:
    output = np.asarray(sequence, dtype=np.float32).copy()
    x = np.arange(len(output))
    for column in range(output.shape[1]):
        values = output[:, column]
        valid = np.isfinite(values)
        if not valid.any():
            output[:, column] = 0.0
        elif not valid.all():
            output[:, column] = np.interp(x, x[valid], values[valid])
    return output


def summarize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Convert variable-length frame descriptors into one fixed vector."""

    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError("Expected a non-empty [frames, features] sequence")
    values = _interpolate_missing(sequence)
    velocity = np.abs(np.diff(values, axis=0))
    if not len(velocity):
        velocity = np.zeros_like(values[:1])

    columns: list[np.ndarray] = []
    for column in range(values.shape[1]):
        series = values[:, column]
        speed = velocity[:, column]
        columns.append(
            np.asarray(
                [
                    series.mean(),
                    series.std(),
                    series.min(),
                    series.max(),
                    series.max() - series.min(),
                    series[0],
                    series[-1],
                    series[-1] - series[0],
                    speed.mean(),
                    speed.max(),
                ],
                dtype=np.float32,
            )
        )
    return np.concatenate(columns)


class LandmarkExtractor:
    """Extract interpretable hand-pose and motion features from a video."""

    def __init__(
        self,
        max_frames: int = 32,
        min_hand_frames: int = 3,
        max_image_side: int = 640,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        _prioritize_windows_system_dlls()
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Video feature extraction needs opencv-python and mediapipe. "
                "Run scripts/setup.ps1 first."
            ) from exc

        self.cv2 = cv2
        self.mp = mp
        self.max_frames = max_frames
        self.min_hand_frames = min_hand_frames
        self.max_image_side = max_image_side
        self.base_names = frame_feature_names()
        self.feature_names = temporal_feature_names(self.base_names)
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This feature extractor requires MediaPipe 0.10.21. "
                "Re-run scripts/setup.ps1 to install the pinned compatible version."
            )
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self.holistic.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _pose_reference(self, results: object) -> tuple[np.ndarray, float, float]:
        pose = getattr(results, "pose_landmarks", None)
        if pose is None:
            return np.asarray([0.5, 0.5], dtype=np.float32), 1.0, 0.0
        left = pose.landmark[11]
        right = pose.landmark[12]
        center = np.asarray([(left.x + right.x) / 2, (left.y + right.y) / 2], dtype=np.float32)
        width = float(np.linalg.norm(np.asarray([left.x - right.x, left.y - right.y])))
        return center, max(width, 1e-4), 1.0

    def _hand_vector(
        self,
        hand_landmarks: object | None,
        body_center: np.ndarray,
        shoulder_width: float,
    ) -> tuple[list[float], np.ndarray | None]:
        if hand_landmarks is None:
            return [0.0] + [np.nan] * (5 + 63 + 10), None

        points = np.asarray(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
            dtype=np.float32,
        )
        wrist = points[0]
        palm_scale = float(np.linalg.norm(points[9, :2] - wrist[:2]))
        palm_scale = max(palm_scale, 1e-4)
        normalized = (points - wrist) / palm_scale
        body_xy = (wrist[:2] - body_center) / shoulder_width
        values: list[float] = [
            1.0,
            float(wrist[0]),
            float(wrist[1]),
            float(body_xy[0]),
            float(body_xy[1]),
            palm_scale,
        ]
        values.extend(normalized.reshape(-1).astype(float).tolist())

        triples = ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
        tips = (4, 8, 12, 16, 20)
        for triple, tip in zip(triples, tips):
            values.append(_angle(normalized[triple[0]], normalized[triple[1]], normalized[triple[2]]))
            values.append(float(np.linalg.norm(normalized[tip, :2])))
        return values, wrist

    def _frame_vector(self, results: object) -> np.ndarray:
        body_center, shoulder_width, pose_present = self._pose_reference(results)
        left_values, left_wrist = self._hand_vector(
            getattr(results, "left_hand_landmarks", None), body_center, shoulder_width
        )
        right_values, right_wrist = self._hand_vector(
            getattr(results, "right_hand_landmarks", None), body_center, shoulder_width
        )
        both = left_wrist is not None and right_wrist is not None
        if both:
            hands_distance = float(np.linalg.norm(left_wrist[:2] - right_wrist[:2]) / shoulder_width)
            vertical_offset = float((left_wrist[1] - right_wrist[1]) / shoulder_width)
        else:
            hands_distance = np.nan
            vertical_offset = np.nan
        values = [pose_present, shoulder_width, *left_values, *right_values, float(both), hands_distance, vertical_offset]
        vector = np.asarray(values, dtype=np.float32)
        if len(vector) != len(self.base_names):
            raise AssertionError(f"Feature-name mismatch: {len(vector)} values vs {len(self.base_names)} names")
        return vector

    def extract_video(self, video_path: str | Path) -> tuple[np.ndarray, ExtractionInfo]:
        path = Path(video_path)
        capture = self.cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video: {path}")

        started = time.perf_counter()
        frame_total = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(self.cv2.CAP_PROP_FPS) or 0.0)
        if frame_total > 0:
            targets = set(np.linspace(0, frame_total - 1, min(frame_total, self.max_frames), dtype=int).tolist())
        else:
            targets = set(range(self.max_frames))

        frame_vectors: list[np.ndarray] = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in targets:
                    height, width = frame.shape[:2]
                    scale = min(1.0, self.max_image_side / max(height, width))
                    if scale < 1.0:
                        frame = self.cv2.resize(frame, None, fx=scale, fy=scale)
                    rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    frame_vectors.append(self._frame_vector(self.holistic.process(rgb)))
                    if len(frame_vectors) >= self.max_frames:
                        break
                frame_index += 1
        finally:
            capture.release()

        if not frame_vectors:
            raise ValueError(f"No frames could be decoded from {path}")
        sequence = np.vstack(frame_vectors)
        left_present = sequence[:, self.base_names.index("left.present")] > 0.5
        right_present = sequence[:, self.base_names.index("right.present")] > 0.5
        hand_frames = int(np.logical_or(left_present, right_present).sum())
        if hand_frames < self.min_hand_frames:
            raise ValueError(
                f"Only {hand_frames} sampled frames contained a detected hand; "
                f"at least {self.min_hand_frames} are required"
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        effective_total = frame_total if frame_total > 0 else frame_index
        info = ExtractionInfo(
            frames_total=effective_total,
            frames_sampled=len(frame_vectors),
            hand_frames=hand_frames,
            hand_detection_rate=hand_frames / len(frame_vectors),
            fps=fps,
            duration_seconds=(effective_total / fps) if fps > 0 else 0.0,
            extraction_ms=elapsed_ms,
        )
        return summarize_sequence(sequence), info


def extract_manifest(
    manifest: pd.DataFrame,
    extractor: LandmarkExtractor,
    limit: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    source = manifest.head(limit) if limit else manifest
    total = len(source)
    started_at = time.perf_counter()
    for position, (_, row) in enumerate(source.iterrows(), start=1):
        record = row.to_dict()
        try:
            vector, info = extractor.extract_video(row["video_path"])
            record.update(asdict(info))
            record.update(success=True, error="", array_index=len(vectors))
            vectors.append(vector)
        except Exception as exc:  # Each failed video is auditable without stopping the run.
            record.update(success=False, error=f"{type(exc).__name__}: {exc}", array_index=-1)
        rows.append(record)
        if position == 1 or position % 10 == 0 or position == total or not record["success"]:
            elapsed_seconds = time.perf_counter() - started_at
            eta_minutes = (elapsed_seconds / position) * (total - position) / 60
            status = "ok" if record["success"] else "failed"
            filename = Path(str(row["video_path"])).name
            print(
                f"[{position}/{total}] {status}: {filename} | "
                f"elapsed {elapsed_seconds / 60:.1f} min | ETA {eta_minutes:.1f} min",
                flush=True,
            )

    width = len(extractor.feature_names)
    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, width), dtype=np.float32)
    return matrix, pd.DataFrame(rows)


def save_feature_bundle(
    output_path: str | Path,
    matrix: np.ndarray,
    feature_names: Sequence[str],
    successful_index: pd.DataFrame,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    success = successful_index[successful_index["success"]].sort_values("array_index")
    np.savez_compressed(
        output,
        X=matrix,
        feature_names=np.asarray(feature_names, dtype=str),
        video_path=np.asarray(success["video_path"].astype(str).tolist(), dtype=str),
    )


def load_feature_bundle(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return (
            bundle["X"].astype(np.float32),
            bundle["feature_names"].astype(str).tolist(),
            bundle["video_path"].astype(str),
        )
