"""Versioned, human-readable geometry for ordered sign sequences.

These measurements are independent of the released 1,630-feature model.
Missing observations stay NaN; image proximity never asserts physical contact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np

SCHEMA_VERSION = "semantic-sequence-v1"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_POINTS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8),
                 "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
SIDES = ("left", "right")
BODY_REGIONS = ("forehead", "lips", "nose", "chin", "left_ear", "right_ear", "left_cheek",
                "right_cheek", "chest", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    unit: str
    meaning: str
    experimental: bool = False


def feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    def add(name, group, unit, meaning, experimental=False):
        specs.append(FeatureSpec(name, group, unit, meaning, experimental))
    for flag in ("pose.present", "face.present", "both_hands.present"):
        add(flag, "tracking", "flag", "Landmark sets available in this frame; not physical absence")
    for side in SIDES:
        add(f"{side}.present", "tracking", "flag", "Hand landmarks available")
        for finger in FINGERS:
            joints = ("base", "tip") if finger == "thumb" else ("base", "middle", "tip")
            for joint in joints:
                add(f"{side}.{finger}.{joint}_flexion", "finger_shape", "fraction_pi", "One minus joint angle divided by pi; greater means more flexed")
            for target in ("root", "palm"):
                add(f"{side}.{finger}.tip_to_{target}", "finger_shape", "palm_units", "Three-dimensional within-hand gap; estimated hand depth only")
            add(f"{side}.{finger}.tip_projection", "finger_shape", "palm_units", "Tip projection along hand longitudinal axis from palm centre")
            for axis in "xyz":
                add(f"{side}.{finger}.direction_{axis}", "finger_direction", "unit_vector", "Base-to-tip ray in unmirrored camera axes; unreliable for folded fingers")
                add(f"{side}.{finger}.local_tip_{axis}", "finger_motion", "palm_units", "Fingertip in rotating palm coordinates, for relative motion")
        for a, b in combinations(FINGERS, 2):
            add(f"{side}.tips.{a}_{b}_gap", "finger_proximity", "palm_units", "Within-hand fingertip separation; small gap is not proven contact")
        for a, b in zip(FINGERS[1:-1], FINGERS[2:]):
            add(f"{side}.spread.{a}_{b}", "finger_shape", "fraction_pi", "Angle between neighbouring finger rays")
        for family in ("hand_axis", "palm_normal_winding"):
            for axis in "xyz":
                add(f"{side}.{family}_{axis}", "orientation", "unit_vector", "Hand longitudinal axis" if family == "hand_axis" else "Ordered landmark cross-product; sign is NOT calibrated palm-facing direction", family != "hand_axis")
        add(f"{side}.tilt_from_vertical", "orientation", "degrees", "Undirected image-plane tilt: vertical zero, horizontal ninety")
        for axis in "xyz":
            add(f"{side}.thumb.palm_relative_{axis}", "finger_shape", "palm_units", "Thumb tip in palm coordinates; front/side semantic labels need examples")
        for axis in "xy":
            add(f"{side}.wrist.body_{axis}", "location", "shoulder_units", "Wrist relative to shoulder midpoint; x image-right, y image-up")
        add(f"{side}.wrist.body_depth_proxy", "depth_proxy", "shoulder_units", "Pose wrist depth relative to shoulders; NOT independent hand-landmark z", True)
        for region in BODY_REGIONS:
            for part in ("palm", "tips", "knuckles"):
                add(f"{side}.near.{region}.{part}", "body_proximity", "shoulder_units", "Minimum image-plane distance to visible body anchor; not physical touch")
        for axis in "xy":
            add(f"{side}.wrist.head_{axis}", "location", "head_units", "Wrist offset from head centre using visible ear gap")
    for key in ("wrist_gap", "wrist_horizontal_offset", "wrist_vertical_offset", "minimum_landmark_gap", "palm_gap"):
        add(f"hands.{key}", "two_hand_relation", "shoulder_units", "Image-plane relative geometry; wrist offsets are left minus right")
    add("hands.pose_depth_order_proxy", "depth_proxy", "shoulder_units", "Pose left wrist depth minus right wrist depth; not a calibrated front/back order", True)
    for side in SIDES:
        for finger in FINGERS:
            add(f"{side}.{finger}.near_other_palm", "two_hand_relation", "shoulder_units", "Image-plane gap to other palm centre; not wrapping or touch")
        add(f"{side}.palm.near_other_wrist", "two_hand_relation", "shoulder_units", "Image-plane palm-to-other-wrist gap")
    for key in ("nose_relative_x", "nose_relative_y", "ear_line_roll"):
        add(f"head.{key}", "head_motion_proxy", "degrees" if key.endswith("roll") else "ear_units",
            "Head-relative image geometry; a motion proxy, not verified yaw/pitch or a nod/shake label", True)
    return specs


SPECS = tuple(feature_specs())
FEATURE_NAMES = tuple(spec.name for spec in SPECS)
assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def spec_records():
    return [asdict(spec) for spec in SPECS]


def landmarks_array(result: Any, attribute: str, count: int, *, pose=False) -> np.ndarray:
    points = getattr(result, attribute, None)
    if points is None or len(points.landmark) < count:
        return np.full((count, 3), np.nan, dtype=np.float32)
    output = np.array([[p.x, p.y, p.z] for p in points.landmark[:count]], dtype=np.float32)
    if pose:
        for i, point in enumerate(points.landmark[:count]):
            if getattr(point, "visibility", 1.0) < .5:
                output[i] = np.nan
    return output


def _unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if np.isfinite(norm) and norm > 1e-7 else np.full_like(vector, np.nan)


def _flexion(a, b, c):
    first, second = _unit(a - b), _unit(c - b)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return np.nan
    return 1.0 - float(np.arccos(np.clip(first @ second, -1, 1)) / np.pi)


def _distance(a, b):
    return float(np.linalg.norm(a - b))


def _min_gap(points, target):
    distances = np.linalg.norm(points - target, axis=1)
    finite = distances[np.isfinite(distances)]
    return float(finite.min()) if len(finite) else np.nan


def frame_features(left: np.ndarray, right: np.ndarray, pose: np.ndarray, face: np.ndarray,
                   aspect_ratio: float) -> np.ndarray:
    """Convert one landmark observation into named continuous concepts.

    Image-normalized x and estimated hand z are scaled by width/height.
    The resulting camera convention is +x image-right, +y image-up, +z toward
    camera. Each hand's z has its own wrist origin and is never compared
    directly to the other hand or the body depth.
    """
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("A positive image aspect ratio is required")
    transform = np.array([aspect_ratio, -1., -aspect_ratio])
    hand_points = {"left": np.asarray(left, dtype=float) * transform,
                   "right": np.asarray(right, dtype=float) * transform}
    body = np.asarray(pose, dtype=float) * transform
    face_points = np.asarray(face, dtype=float) * transform
    values = dict.fromkeys(FEATURE_NAMES, np.nan)
    present = {side: bool(np.isfinite(points).all()) for side, points in hand_points.items()}
    pose_ok = bool(np.isfinite(body[[11, 12]]).all())
    face_ok = bool(np.isfinite(face_points[[1, 10, 13, 14, 152, 234, 454]]).all())
    values.update({"pose.present": float(pose_ok), "face.present": float(face_ok),
                   "both_hands.present": float(all(present.values()))})
    centre = (body[11] + body[12]) / 2
    width = _distance(body[11, :2], body[12, :2]) if pose_ok else np.nan
    if width < .02:  # Degenerate shoulder geometry cannot support normalization.
        width = np.nan
    regions = {"left_ear": body[7, :2], "right_ear": body[8, :2], "nose": body[0, :2],
               "left_shoulder": body[11, :2], "right_shoulder": body[12, :2],
               "left_elbow": body[13, :2], "right_elbow": body[14, :2]}
    if np.isfinite(body[[11, 12, 23, 24], :2]).all():
        # Upper-torso interior rather than shoulder midpoint.
        hips = (body[23, :2] + body[24, :2]) / 2
        regions["chest"] = .7 * centre[:2] + .3 * hips
    if face_ok:
        regions.update(forehead=face_points[10, :2], lips=(face_points[13, :2] + face_points[14, :2]) / 2,
                       nose=face_points[1, :2], chin=face_points[152, :2])
        # Resolve cheek sides using pose ears, never assume screen sides are anatomical sides.
        if np.isfinite(body[[7, 8], :2]).all():
            candidates = face_points[[234, 454], :2]
            first = int(np.argmin(np.linalg.norm(candidates - body[7, :2], axis=1)))
            regions.update(left_cheek=candidates[first], right_cheek=candidates[1-first])
    palms = {}
    ears_valid = np.isfinite(body[[7, 8], :2]).all()
    head_width = _distance(body[7, :2], body[8, :2]) if ears_valid else np.nan
    head_centre = (body[7, :2] + body[8, :2]) / 2
    for side, points in hand_points.items():
        values[f"{side}.present"] = float(present[side])
        if not present[side]:
            continue
        palm = points[[0, 5, 9, 13, 17]].mean(axis=0)
        palms[side] = palm
        scale = _distance(points[0], points[9])
        if scale < .005:
            continue
        longitudinal = _unit(points[9] - points[0])
        across = _unit(points[5] - points[17])
        normal = _unit(np.cross(across, longitudinal))
        across = _unit(np.cross(longitudinal, normal))
        basis = np.vstack([across, longitudinal, normal])
        for axis, component in zip("xyz", longitudinal):
            values[f"{side}.hand_axis_{axis}"] = float(component)
        for axis, component in zip("xyz", normal):
            values[f"{side}.palm_normal_winding_{axis}"] = float(component)
        values[f"{side}.tilt_from_vertical"] = float(np.degrees(np.arctan2(abs(longitudinal[0]), abs(longitudinal[1]))))
        for finger, (a, b, c, d) in FINGER_POINTS.items():
            triples = [("base", a, b, c), ("tip", b, c, d)] if finger == "thumb" else [
                ("base", 0, a, b), ("middle", a, b, c), ("tip", b, c, d)]
            for joint, first, second, third in triples:
                values[f"{side}.{finger}.{joint}_flexion"] = _flexion(points[first], points[second], points[third])
            values[f"{side}.{finger}.tip_to_root"] = _distance(points[d], points[a]) / scale
            values[f"{side}.{finger}.tip_to_palm"] = _distance(points[d], palm) / scale
            values[f"{side}.{finger}.tip_projection"] = float((points[d] - palm) @ longitudinal / scale)
            ray = points[d] - points[a]
            direction = _unit(ray) if np.linalg.norm(ray) / scale > .2 else np.full(3, np.nan)
            for axis, component in zip("xyz", direction):
                values[f"{side}.{finger}.direction_{axis}"] = float(component)
            for axis, component in zip("xyz", basis @ (points[d] - palm) / scale):
                values[f"{side}.{finger}.local_tip_{axis}"] = float(component)
        for first, second in combinations(FINGERS, 2):
            values[f"{side}.tips.{first}_{second}_gap"] = _distance(points[FINGER_POINTS[first][-1]], points[FINGER_POINTS[second][-1]]) / scale
        for first, second in zip(FINGERS[1:-1], FINGERS[2:]):
            a, b = FINGER_POINTS[first], FINGER_POINTS[second]
            r1, r2 = _unit(points[a[-1]] - points[a[0]]), _unit(points[b[-1]] - points[b[0]])
            values[f"{side}.spread.{first}_{second}"] = float(np.arccos(np.clip(r1 @ r2, -1, 1)) / np.pi)
        for axis, component in zip("xyz", basis @ (points[4] - palm) / scale):
            values[f"{side}.thumb.palm_relative_{axis}"] = float(component)
        if np.isfinite(width):
            for axis, component in zip("xy", (points[0, :2] - centre[:2]) / width):
                values[f"{side}.wrist.body_{axis}"] = float(component)
            pose_wrist = body[15 if side == "left" else 16]
            values[f"{side}.wrist.body_depth_proxy"] = float((pose_wrist[2] - centre[2]) / width)
            parts = {"palm": palm[None, :2], "tips": points[[4, 8, 12, 16, 20], :2], "knuckles": points[[2, 5, 9, 13, 17], :2]}
            for region, anchor in regions.items():
                for part, subset in parts.items():
                    values[f"{side}.near.{region}.{part}"] = _min_gap(subset, anchor) / width
        if np.isfinite(head_width) and head_width > .02:
            for axis, component in zip("xy", (points[0, :2] - head_centre) / head_width):
                values[f"{side}.wrist.head_{axis}"] = float(component)
    if all(present.values()) and np.isfinite(width) and len(palms) == 2:
        left_xy, right_xy = hand_points["left"][:, :2], hand_points["right"][:, :2]
        delta = (left_xy[0] - right_xy[0]) / width
        values.update({"hands.wrist_gap": float(np.linalg.norm(delta)),
                       "hands.wrist_horizontal_offset": float(delta[0]), "hands.wrist_vertical_offset": float(delta[1]),
                       "hands.minimum_landmark_gap": float(np.linalg.norm(left_xy[:, None] - right_xy[None], axis=2).min() / width),
                       "hands.palm_gap": _distance(palms["left"][:2], palms["right"][:2]) / width,
                       "hands.pose_depth_order_proxy": float((body[15, 2] - body[16, 2]) / width)})
        for side, other in [("left", "right"), ("right", "left")]:
            for finger in FINGERS:
                values[f"{side}.{finger}.near_other_palm"] = _distance(hand_points[side][FINGER_POINTS[finger][-1], :2], palms[other][:2]) / width
            values[f"{side}.palm.near_other_wrist"] = _distance(palms[side][:2], hand_points[other][0, :2]) / width
    if np.isfinite(head_width) and head_width > .02 and np.isfinite(body[0, :2]).all():
        delta = (body[0, :2] - head_centre) / head_width
        ordered = body[[7,8],:2][np.argsort(body[[7,8],0])]
        values.update({"head.nose_relative_x": float(delta[0]), "head.nose_relative_y": float(delta[1]),
                       "head.ear_line_roll": float(np.degrees(np.arctan2(ordered[1,1]-ordered[0,1],ordered[1,0]-ordered[0,0])))})
    vector = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    vector[~np.isfinite(vector)] = np.nan
    return vector


def extract_result(result, aspect_ratio):
    arrays = {"left": landmarks_array(result, "left_hand_landmarks", 21),
              "right": landmarks_array(result, "right_hand_landmarks", 21),
              "pose": landmarks_array(result, "pose_landmarks", 33, pose=True),
              "face": landmarks_array(result, "face_landmarks", 468)}
    return frame_features(**arrays, aspect_ratio=aspect_ratio), arrays
