"""Describe the actual tree inputs without numerical or anatomical overclaims.

All comparisons are relative to the learned branch. Coordinate descriptors keep
exactly the reference used by the feature extractor. Frame differences are not
velocities per second. Detection flags are not interpolated; other missing hand
measurements are interpolated or zero-filled by summarize_sequence.
"""
from __future__ import annotations

import math

DESCRIPTION_VERSION = 2
STATISTICS = {"mean", "std", "min", "max", "range", "start", "end", "delta", "velocity_mean", "velocity_max"}
TRACKING_FEATURES = {f"{name}.{stat}" for name in ("left.present", "right.present", "both_hands_present", "pose_present") for stat in ("mean", "start", "end")}


def _joint_name(descriptor: str) -> str:
    names = {"wrist": "wrist", "thumb_cmc": "thumb base", "thumb_mcp": "thumb knuckle", "thumb_ip": "thumb middle joint", "thumb_tip": "thumb tip"}
    if descriptor in names:
        return names[descriptor]
    for suffix, label in (("_mcp", "base knuckle"), ("_pip", "middle joint"), ("_dip", "end joint"), ("_tip", "fingertip")):
        if descriptor.endswith(suffix):
            return descriptor.removesuffix(suffix) + " " + label
    return descriptor.replace("_", " ")


def _feature(base: str) -> tuple[str, str]:
    side = "Left hand" if base.startswith("left.") else "Right hand" if base.startswith("right.") else "Body and hands"
    descriptor = base.split(".", 1)[-1]
    if descriptor.endswith("_bend"):
        return f"{side} · {descriptor.removesuffix('_bend')} finger bend", "finger angle"
    if descriptor.endswith("_tip_distance"):
        return f"{side} · {descriptor.removesuffix('_tip_distance')} fingertip reach", "fingertip distance from the wrist, relative to the palm reference"
    if descriptor == "palm_scale":
        return f"{side} · palm reference", "visible distance from the wrist to the middle finger's base"
    if descriptor == "shoulder_width":
        return "Visible shoulder gap", "shoulder gap in the recorded image"
    if descriptor == "hands_distance":
        return "Space between wrists", "wrist separation relative to the shoulder reference"
    if descriptor == "hands_vertical_offset":
        return "Relative wrist height", "left-to-right wrist height offset relative to the shoulder reference"
    if descriptor in {"present", "pose_present", "both_hands_present"}:
        title = side if descriptor == "present" else "Shoulder tracking" if descriptor == "pose_present" else "Both hands together"
        return title, "detection state"
    point = descriptor[:-2].replace("_abs", "").replace("_body", "")
    axis = descriptor[-1]
    direction = {"x": "sideways position", "y": "vertical position", "z": "estimated depth"}[axis]
    reference = "in the image" if "wrist_abs" in descriptor else "relative to the shoulder reference" if "wrist_body" in descriptor else "relative to the wrist and palm reference"
    return f"{side} · {_joint_name(point)}", f"{direction} {reference}"


def _tracking(base: str, stat: str, observations: dict) -> tuple[str | None, str]:
    """Return an unavailable-data explanation, plus any partial-tracking note."""
    descriptor = base.split(".", 1)[-1]
    if descriptor in {"present", "pose_present", "both_hands_present"}:
        return None, ""
    required = []
    if base.startswith(("left.", "right.")):
        required.append((base.split(".")[0] + ".present", "This hand", "interpolated or filled-in measurements"))
    if descriptor in {"hands_distance", "hands_vertical_offset"}:
        required.append(("both_hands_present", "Both hands together", "filled-in wrist comparisons"))
    if descriptor in {"shoulder_width", "hands_distance", "hands_vertical_offset", "wrist_body_x", "wrist_body_y"}:
        required.append(("pose_present", "Shoulders", "the fallback body reference"))
    notes = []
    for key, label, fallback in required:
        rate = observations.get(key + ".mean")
        if rate is None:
            return f"The tree used this feature, but {label.lower()} tracking coverage was not saved. Record again to verify the pose description.", ""
        if float(rate) == 0:
            return f"{label} were not detected in any sampled frame; the tree used {fallback}." if label != "This hand" else f"This hand was not detected in any sampled frame; the tree used {fallback}.", ""
        endpoint = observations.get(key + "." + stat) if stat in {"start", "end"} else None
        if endpoint is not None and float(endpoint) == 0:
            return f"{label} were not detected at the {'start' if stat == 'start' else 'end'}; this feature used {fallback}." if label != "This hand" else f"This hand was not detected at the {'start' if stat == 'start' else 'end'}; this feature used {fallback}.", ""
        if float(rate) < 1:
            if stat in {"start", "end"}:
                notes.append(f"Tracking for {label.lower()} was incomplete elsewhere in the sign; this endpoint was detected." if endpoint is not None else f"Tracking for {label.lower()} was incomplete, and endpoint coverage was not saved.")
            else:
                notes.append(f"Tracking for {label.lower()} was incomplete; this summary also includes {fallback}.")
    return None, " ".join(notes)


def _variation(subject: str, stat: str, higher: bool, value: float) -> str:
    if stat in {"velocity_mean", "velocity_max"}:
        which = "average" if stat == "velocity_mean" else "largest"
        if value == 0:
            return f"No change in {subject} was recorded between sampled frames."
        return f"The {which} change in {subject} between sampled frames was {'larger' if higher else 'smaller'} for this decision."
    if stat == "range":
        return f"The recorded {subject} covered a {'wider' if higher else 'narrower'} range for this decision."
    return f"The recorded {subject} was {'more' if higher else 'less'} spread out across the sign for this decision."


def _presence(descriptor: str, stat: str, value: float, higher: bool) -> str:
    subject = "This hand" if descriptor == "present" else "Shoulders" if descriptor == "pose_present" else "Both hands together"
    detected = value > .5
    if stat == "mean":
        state = "in every sampled frame" if value == 1 else "in none of the sampled frames" if value == 0 else "in only some sampled frames"
        return f"{subject}: detected {state}."
    if stat in {"start", "end"}:
        verb = "was" if subject == "This hand" else "were"
        return f"At the {'start' if stat == 'start' else 'end'}, {subject.lower()} {verb} {'' if detected else 'not '}detected."
    if stat == "min":
        return f"{subject}: detected in every sampled frame." if detected else f"{subject}: missed in at least one sampled frame."
    if stat == "max":
        return f"{subject}: detected in at least one sampled frame." if detected else f"{subject}: never detected in the sampled frames."
    if stat == "delta":
        return "Detection at the end matched the start; it may have changed in between." if value == 0 else "Detection was present at the end but absent at the start." if value > 0 else "Detection was absent at the end but present at the start."
    if value == 0:
        return "The detection state did not change between sampled frames."
    if stat == "velocity_mean":
        return f"Detection switched {'more' if higher else 'less'} often between sampled frames for this decision."
    return "The detection state changed within the sampled frames."


def describe_rule(rule: dict, observations: dict[str, float]) -> dict[str, str]:
    feature = rule["feature"]
    base, stat = feature.rsplit(".", 1)
    value = float(rule["value"])
    if stat not in STATISTICS or not math.isfinite(value) or rule["operator"] not in {">", "<="}:
        raise ValueError("Unsupported or invalid explanation feature")
    if "threshold" in rule:
        threshold = float(rule["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("Explanation branch has an invalid comparison")
        expected = ">" if value > threshold else "<="
        if rule["operator"] != expected:
            raise ValueError("Explanation branch does not match its recorded value")
    higher = rule["operator"] == ">"
    descriptor = base.split(".", 1)[-1]
    title, subject = _feature(base)
    result = {"feature": feature, "title": title, "description": ""}
    if descriptor in {"present", "pose_present", "both_hands_present"}:
        result["description"] = _presence(descriptor, stat, value, higher)
        return result
    unavailable, note = _tracking(base, stat, observations)
    if unavailable:
        result["description"] = unavailable
        return result
    if note:
        result["note"] = note
    if stat in {"std", "range", "velocity_mean", "velocity_max"}:
        result["description"] = _variation(subject, stat, higher, value)
        return result
    prefix = {"start": "At the start", "end": "At the end", "mean": "On average across the sign", "min": "At the smallest recorded value", "max": "At the largest recorded value"}.get(stat, "From the start to the end")
    if stat == "delta":
        if value == 0:
            text = f"The recorded {subject} finished where it started; it may have changed in between."
        elif descriptor.endswith("_bend"):
            text = "The finger angle ended straighter than it started." if value > 0 else "The finger angle ended more bent than it started."
        elif descriptor.endswith("_tip_distance"):
            text = "The fingertip reach ended farther from the wrist relative to the palm reference." if value > 0 else "The fingertip reach ended closer to the wrist relative to the palm reference."
        elif descriptor in {"palm_scale", "shoulder_width"}:
            text = f"The recorded {subject} increased." if value > 0 else f"The recorded {subject} decreased."
        elif descriptor in {"hands_distance", "hands_vertical_offset"}:
            text = f"The recorded {subject} increased." if value > 0 else f"The recorded {subject} decreased."
        else:
            axis = descriptor[-1]
            if axis == "z":
                text = "Estimated depth relative to the wrist and palm reference increased." if value > 0 else "Estimated depth relative to the wrist and palm reference decreased."
            else:
                reference = "in the image" if "wrist_abs" in descriptor else "relative to the shoulder reference" if "wrist_body" in descriptor else "relative to the wrist and palm reference"
                direction = ("farther right" if value > 0 else "farther left") if axis == "x" else ("lower" if value > 0 else "higher")
                text = f"The recorded position ended {direction} {reference}."
    elif descriptor.endswith("_bend"):
        text = f"{prefix}, the finger angle was {'relatively straighter' if higher else 'relatively more bent'} for this decision."
    elif descriptor.endswith("_tip_distance"):
        text = f"{prefix}, fingertip reach was {'farther from' if higher else 'closer to'} the wrist relative to the palm reference for this decision."
    elif descriptor in {"palm_scale", "shoulder_width", "hands_distance"}:
        text = f"{prefix}, the recorded {subject} was {'larger' if higher else 'smaller'} for this decision."
    elif descriptor == "hands_vertical_offset":
        position = "lower than" if value > 0 else "higher than" if value < 0 else "at the same height as"
        text = f"{prefix}, the normalized left wrist height was {position} the right wrist."
    else:
        axis = descriptor[-1]
        absolute = "wrist_abs" in descriptor
        body = "wrist_body" in descriptor
        midpoint = .5 if absolute else 0
        reference = "the image midpoint" if absolute else "the shoulder reference" if body else "the wrist"
        if axis == "z":
            position = "farther from the camera than" if value > 0 else "closer to the camera than" if value < 0 else "aligned with"
            text = f"{prefix}, estimated landmark depth was {position} the wrist, relative to the palm reference."
        elif value == midpoint:
            text = f"{prefix}, the recorded position aligned with {reference} along this direction."
        else:
            position = ("right of" if value > midpoint else "left of") if axis == "x" else ("below" if value > midpoint else "above")
            text = f"{prefix}, the recorded position was {position} {reference} in the unmirrored image."
    result["description"] = text
    return result


def describe_trace(trace: dict, observations: dict[str, float]) -> list[dict[str, str]]:
    rules = trace["rules"]
    if not all(rule.get("condition_holds", False) for rule in rules):
        raise ValueError("The explanation contains an unverified tree branch")
    return [describe_rule(rule, observations) for rule in rules]
