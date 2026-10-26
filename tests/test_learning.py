import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from senyas.assessments import create_assessment, submit_answer
from senyas.config import PROJECT_ROOT
from senyas.curriculum import bundled_video, categories, check_meaning, load_lessons
from senyas.explanations import describe_rule, describe_trace
from senyas.features import temporal_feature_names
from senyas.progress import category_status, load_progress, record_attempt, save_progress, set_lesson_stage, upgrade_progress


def test_portable_curriculum_matches_authoritative_labels():
    import pandas as pd
    source = pd.read_csv(PROJECT_ROOT / "config" / "labels.csv")
    lessons = load_lessons()
    assert len(lessons) == lessons["id"].nunique() == 105
    assert len(categories(lessons)) == 10
    assert lessons.set_index("id")["category"].to_dict() == source.set_index("id")["category"].to_dict()
    assert lessons.set_index("id")["label"].to_dict() == {int(row["id"]): row["label"].strip() for _, row in source.iterrows()}
    manifest = json.loads((PROJECT_ROOT / "assets/lessons/manifest.json").read_text())
    assert len(manifest["videos"]) == 105
    for video in manifest["videos"]:
        path = PROJECT_ROOT / video["video"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == video["sha256"]
        assert video["audio_tracks"] == 0
    assert all(bundled_video(row).is_file() for _, row in lessons.iterrows())
    assert all(check_meaning(row["filipino"], row) for _, row in lessons.iterrows())


@pytest.mark.parametrize("chosen,kind", [(["GREETING"], "category"), (["COLOR"], "custom"), (["GREETING", "COLOR"], "custom"), ([], "final")])
def test_assessments_cover_each_selected_word_once_and_balance(chosen, kind):
    lessons = load_lessons()
    expected = set(lessons["id"] if kind == "final" else lessons.loc[lessons["category"].isin(chosen), "id"])
    for seed in range(12):
        assessment = create_assessment(lessons, chosen, kind, "Decision Tree", seed=seed)
        ids = [row["class_id"] for row in assessment["questions"]]
        assert len(ids) == len(set(ids)) == len(expected)
        assert set(ids) == expected
        counts = Counter(row["mode"] for row in assessment["questions"])
        assert abs(counts["sign"] - counts["meaning"]) <= 1
        if kind == "final":
            assert sorted(counts.values()) == [52, 53]


def test_assessment_survives_refresh_without_reordering_and_submits_once(tmp_path):
    progress = upgrade_progress({})
    progress["active_assessment"] = create_assessment(load_lessons(), ["GREETING"], "category", "Decision Tree", seed=4)
    original = copy.deepcopy(progress["active_assessment"])
    path = tmp_path / "progress.json"
    save_progress(path, progress)
    progress = load_progress(path)
    assert progress["active_assessment"] == original
    for index in range(len(original["questions"])):
        assert submit_answer(progress, original["id"], index, {"correct": index % 2 == 0})
        assert not submit_answer(progress, original["id"], index, {"correct": True})
    assert progress["active_assessment"] is None
    assert progress["category_quizzes"]["GREETING"] == original["id"]
    assert progress["assessments"][0]["correct"] == 5


def test_migration_keeps_history_and_atomic_save_preserves_nested_state(tmp_path):
    legacy = {"completed_lessons": [0], "practice_attempts": [{"model": "Extra Trees", "capture_id": "old"}], "meaning_attempts": [{"answer": "salamat"}]}
    progress = upgrade_progress(legacy)
    assert progress["lesson_states"]["0"]["stage"] == "complete"
    assert progress["practice_attempts"] == legacy["practice_attempts"]
    updated = record_attempt(progress, "copy", {"capture_id": "new"})
    assert updated["settings"] == {"model": "Decision Tree", "developer_mode": False}
    assert len(progress["practice_attempts"]) == 1
    assert len(record_attempt(updated, "copy", {"capture_id": "new"})["practice_attempts"]) == 2
    path = tmp_path / "progress.json"
    save_progress(path, updated)
    assert load_progress(path) == updated
    assert not list(tmp_path.glob(".progress-*.json"))
    assert category_status(progress, [0, 1], "GREETING") == "started"
    set_lesson_stage(progress, 1, "complete")
    progress["category_quizzes"]["GREETING"] = "quiz"
    assert category_status(progress, [0, 1], "GREETING") == "complete"


def test_descriptions_cover_schema_without_numbers_or_invented_orientation():
    coverage = {"left.present.mean": 1, "right.present.mean": 1, "both_hands_present.mean": 1, "pose_present.mean": 1}
    for name in temporal_feature_names():
        rule = {"feature": name, "operator": ">", "value": .25, "threshold": .1, "condition_holds": True}
        result = describe_rule(rule, coverage)
        assert result["feature"] == name
        assert not re.search(r"\d|threshold|cutoff", result["description"])
        assert "facing outward" not in result["description"]
    missing = describe_rule({"feature": "left.index_bend.mean", "operator": "<=", "value": 0}, {"left.present.mean": 0})
    assert "not detected" in missing["description"]
    assert "bent" not in missing["description"]
    change = describe_rule({"feature": "hands_distance.delta", "operator": ">", "value": -.1}, coverage)
    assert "decreased" in change["description"] and "shoulder reference" in change["description"]
    depth = describe_rule({"feature": "left.index_tip_z.mean", "operator": ">", "value": -.2}, coverage)
    assert "closer to the camera" in depth["description"]
    trace = {"rules": [{"feature": "left.index_bend.mean", "operator": "<=", "value": .3, "condition_holds": True}]}
    assert "more bent" in describe_trace(trace, coverage)[0]["description"]
