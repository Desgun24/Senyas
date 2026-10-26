import copy

import numpy as np
import pytest

from senyas.explanations import describe_rule, describe_trace
from senyas.features import _angle, summarize_sequence

COMPLETE = {f"{name}.{stat}": 1 for name in ("left.present", "right.present", "both_hands_present", "pose_present") for stat in ("mean", "start", "end")}


def explanation(feature, value, operator=">", coverage=None):
    return describe_rule({"feature": feature, "value": value, "operator": operator}, COMPLETE if coverage is None else coverage)


def test_finger_bend_direction_matches_the_actual_angle_formula():
    straight = _angle(np.array([0., 0., 0.]), np.array([1., 0., 0.]), np.array([2., 0., 0.]))
    folded = _angle(np.array([0., 0., 0.]), np.array([1., 0., 0.]), np.array([.5, 0., 0.]))
    assert straight > folded
    assert "straighter" in explanation("left.index_bend.mean", straight)["description"]
    assert "more bent" in explanation("left.index_bend.mean", folded, "<=")["description"]


def test_zero_endpoint_change_does_not_claim_a_still_finger():
    # Finger bends and returns: no net change despite a large range.
    summary = summarize_sequence(np.array([[.9], [.2], [.9]], dtype=np.float32))
    assert summary[4] > .6 and summary[7] == 0
    text = explanation("right.index_bend.delta", float(summary[7]))["description"]
    assert "may have changed in between" in text
    assert "kept its bend" not in text


@pytest.mark.parametrize("stat,label", [("std", "spread out"), ("range", "range"), ("velocity_mean", "average change"), ("velocity_max", "largest change")])
def test_temporal_statistics_keep_their_distinct_meanings(stat, label):
    text = explanation(f"right.thumb_tip_y.{stat}", .25)["description"]
    assert label in text
    assert "steadier" not in text and "speed" not in text and "per second" not in text
    if stat.startswith("velocity"):
        assert "between sampled frames" in text


def test_relative_coordinate_delta_does_not_claim_whole_hand_motion():
    relative = explanation("right.index_tip_y.delta", -.2)["description"]
    absolute = explanation("right.wrist_abs_y.delta", -.2)["description"]
    body = explanation("right.wrist_body_y.delta", -.2)["description"]
    assert "higher relative to the wrist and palm reference" in relative
    assert "higher in the image" in absolute
    assert "higher relative to the shoulder reference" in body


@pytest.mark.parametrize("stat,phrase", [("start", "At the start"), ("end", "At the end"), ("min", "every sampled frame"), ("max", "at least one sampled frame"), ("mean", "only some sampled frames")])
def test_detection_statistics_are_not_all_reported_as_consistency(stat, phrase):
    value = .25 if stat == "mean" else 1
    text = explanation(f"right.present.{stat}", value)["description"]
    assert phrase in text


def test_simultaneous_hands_are_required_for_wrist_comparisons():
    coverage = {**COMPLETE, "left.present.mean": .5, "right.present.mean": .5, "both_hands_present.mean": 0}
    result = explanation("hands_distance.mean", 0, "<=", coverage)
    assert "Both hands together were not detected" in result["description"]
    assert "filled-in wrist comparisons" in result["description"]
    assert "closer" not in result["description"]


def test_missing_shoulders_are_handled_for_body_relative_changes():
    coverage = {**COMPLETE, "pose_present.mean": 0}
    text = explanation("right.wrist_body_y.delta", -.3, coverage=coverage)["description"]
    assert "fallback body reference" in text
    assert "higher" not in text


def test_missing_endpoint_does_not_assert_observed_pose():
    coverage = {**COMPLETE, "left.present.mean": .5, "left.present.start": 0}
    text = explanation("left.index_bend.start", .7, coverage=coverage)["description"]
    assert "not detected at the start" in text
    assert "straighter" not in text
    observed = explanation("left.index_bend.end", .7, coverage=coverage)
    assert "elsewhere in the sign" in observed["note"]
    assert "includes" not in observed["note"]


def test_partial_tracking_and_older_logs_are_explicit():
    result = explanation("right.index_bend.mean", .7, coverage={**COMPLETE, "right.present.mean": .75})
    assert "incomplete" in result["note"] and "interpolated" in result["note"]
    old = explanation("right.index_bend.mean", .7, coverage={})
    assert "tracking coverage was not saved" in old["description"]
    assert "straighter" not in old["description"]


def test_joint_titles_are_plain_language_and_zero_alignment_is_explicit():
    assert "thumb base" in explanation("right.thumb_cmc_x.mean", .2)["title"]
    assert "middle joint" in explanation("right.index_pip_y.mean", .2)["title"]
    assert "aligned" in explanation("right.wrist_body_y.mean", 0)["description"]


def test_unverified_or_inconsistent_branches_cannot_be_silently_omitted():
    rule = {"feature": "right.index_bend.mean", "value": .2, "threshold": .5, "operator": "<=", "condition_holds": True}
    trace = {"rules": [rule]}
    assert len(describe_trace(trace, COMPLETE)) == 1
    invalid = copy.deepcopy(trace)
    invalid["rules"][0]["condition_holds"] = False
    with pytest.raises(ValueError, match="unverified"):
        describe_trace(invalid, COMPLETE)
    invalid = copy.deepcopy(trace)
    invalid["rules"][0]["operator"] = ">"
    with pytest.raises(ValueError, match="branch"):
        describe_trace(invalid, COMPLETE)
