"""Verified qualitative descriptions of actual frozen Decision Tree branches."""
import math

from .sequences import SEQUENCE_SPECS

DESCRIPTION_VERSION = "semantic-branches-v1"
SPECS = {spec.name: spec for spec in SEQUENCE_SPECS if not spec.experimental}
SCOPES = {"global": "Across the active sign", "opening": "During the opening phase",
          "middle": "During the middle phase", "ending": "During the ending phase"}
STATS = {"median": "typical measurement", "p10": "lower-range measurement",
         "p90": "upper-range measurement", "end_minus_start": "change from opening to ending",
         "observed_fraction": "measurement availability"}


def describe_trace(trace, imputed_features=()):
    missing = set(imputed_features)
    output = []
    for rule in trace["rules"]:
        value, cutoff = float(rule["value"]), float(rule["threshold"])
        operator = rule["operator"]
        holds = (value <= cutoff) if operator == "<=" else (value > cutoff) if operator == ">" else False
        if not math.isfinite(value) or not math.isfinite(cutoff) or not holds or not rule["condition_holds"]:
            raise ValueError("Decision branch does not match its recorded measurement")
        name, scope, statistic = rule["feature"].rsplit(".", 2)
        if name not in SPECS or scope not in SCOPES or statistic not in STATS:
            raise ValueError("Unknown feature in the saved decision path")
        larger = operator == ">"
        relation = "above" if larger else "at or below"
        title = name.replace(".", " · ").replace("_", " ").capitalize()
        prefix = SCOPES[scope]
        note = "This describes a learned branch, not a universal hand-shape rule."
        if rule["feature"] in missing:
            description = f"{prefix}, this measurement was unavailable. Its training-set replacement followed the {('higher' if larger else 'lower')} branch."
            note = "The branch used an imputed value; it does not establish the observed hand state."
        elif statistic == "observed_fraction":
            description = f"{prefix}, this feature's tracking availability was {relation} the learned split."
            note = "Availability describes whether landmarks could be measured, not the physical position or presence of the hand."
        elif name.endswith(".present"):
            description = f"{prefix}, the {STATS[statistic]} of landmark availability was {relation} the learned split."
            note = "A missing landmark set does not prove that a hand or body part was physically absent."
        elif statistic == "end_minus_start":
            trend = "increased" if value > 0 else "decreased" if value < 0 else "stayed unchanged"
            description = f"The measurement {trend} from opening to ending; that change was {relation} the learned split."
            note = "The change compares opening and ending measurements; it does not describe every intervening frame."
        else:
            observation, caveat = _measurement(name, larger, value)
            description = f"{prefix}, the {STATS[statistic]} indicated {observation}, relative to the learned split."
            if caveat:
                note = caveat
        output.append({"title": title, "description": description, "note": note})
    return output


def _measurement(name, larger, value):
    if name.endswith("_flexion"):
        return ("a more bent finger joint" if larger else "a straighter finger joint"), "The joint angle supports relative bending; it does not alone establish a named hand shape."
    if ".spread." in name:
        return ("a wider angle between neighbouring fingers" if larger else "a narrower angle between neighbouring fingers"), ""
    if name.endswith("tilt_from_vertical"):
        return ("a hand axis closer to horizontal" if larger else "a hand axis closer to vertical"), "Tilt describes the image plane and does not establish palm-facing direction."
    if any(term in name for term in (".near.", "_gap", ".tip_to_", ".near_other_")):
        return ("a larger measured separation" if larger else "a smaller measured separation"), "Proximity is not proof of physical contact or fingers wrapping around another hand."
    if "direction_" in name or "hand_axis_" in name or ".velocity_" in name:
        axis = name[-1]
        directions = {"x": ("image-right", "image-left"), "y": ("upward", "downward"),
                      "z": ("toward the camera in estimated hand depth", "away from the camera in estimated hand depth")}
        direction = directions[axis][0 if value > 0 else 1] if value else "a neutral axis component"
        return f"{direction} with an axis component {'above' if larger else 'at or below'} the branch boundary", "Directions use the unmirrored image; folded-finger rays and estimated depth can be unreliable."
    if name.endswith("signed_turn_rate"):
        turn = "anticlockwise" if value > 0 else "clockwise" if value < 0 else "neutral"
        return f"{turn} turning with a signed rate {'above' if larger else 'at or below'} the branch boundary", "Turning of the wrist trajectory does not prove that a complete circle was performed."
    if name.endswith(("body_x", "head_x", "body_y", "head_y", "horizontal_offset", "vertical_offset")):
        vertical = name.endswith(("_y", "vertical_offset"))
        direction = ("above" if value > 0 else "below") if vertical else ("image-right of" if value > 0 else "image-left of")
        if not value:
            direction = "aligned with"
        reference = "the other wrist, using left minus right" if name.startswith("hands.") else "the shoulder midpoint" if ".body_" in name else "the head centre"
        return f"a wrist offset {direction} {reference}, {'above' if larger else 'at or below'} the branch boundary", "Positions are image-plane measurements from visible body anchors."
    if name.endswith(("speed", "sway_extent", "sway_reversals", "shape_variation")):
        terms = {"speed": "movement rate", "sway_extent": "fingertip movement extent",
                 "sway_reversals": "fingertip direction-reversal rate", "shape_variation": "finger-joint variation"}
        term = next(text for suffix, text in terms.items() if name.endswith(suffix))
        return f"{'greater' if larger else 'less'} {term}", "Motion rates use frame timestamps; finger motion is measured relative to the palm where specified."
    return ("a larger coordinate measurement" if larger else "a smaller coordinate measurement"), "This palm-relative measurement does not alone establish a front/back, thumb-placement, or named-shape label."
