from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import senyas.learning_app as ui
from senyas.assessments import create_assessment
from senyas.camera import CameraAttempt
from senyas.frozen_recognizer import PIPELINE_ID
from senyas.progress import load_progress, save_progress, set_lesson_stage, upgrade_progress

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def profile(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    monkeypatch.setenv("SENYAS_PROGRESS_PATH", str(path))
    save_progress(path, upgrade_progress({}))
    return path


def button(app, label):
    return next(item for item in app.button if item.label == label)


def mock_camera(monkeypatch):
    payload = {"capture_id": "test-capture", "context": None}
    enabled = [True]

    def capture(context, processed_capture_id=None, **kwargs):
        if payload["context"] is None:
            payload["context"] = context
        return payload if enabled[0] else None

    monkeypatch.setattr(ui, "live_camera", capture)
    monkeypatch.setattr(ui, "decode_camera_attempt", lambda value, context, **kwargs: CameraAttempt(
        value["capture_id"], context, [np.zeros((360, 640, 3), dtype=np.uint8)] * 32, 4
    ))
    extractor = MagicMock()
    artifact = ui.recognizer("Decision Tree")
    vector = artifact["feature_preparation"].medians
    from senyas.model import predict_vector, decision_trace
    extractor.extract_frames.return_value = {
        "prediction": predict_vector(artifact, vector), "trace": decision_trace(artifact, vector),
        "extraction": {"frames": 32}, "imputed_features": [], "pipeline_id": PIPELINE_ID,
        "model_sha256": artifact["runtime_sha256"],
    }
    monkeypatch.setattr(ui, "analyze_attempt", lambda artifact, attempt: extractor.extract_frames(attempt))
    return payload, enabled, extractor


def test_guided_camera_flow_deduplication_and_developer_descriptions(profile, monkeypatch):
    progress = load_progress(profile)
    set_lesson_stage(progress, 0, "copy")
    save_progress(profile, progress)
    payload, enabled, extractor = mock_camera(monkeypatch)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params.update(page="Lessons", word="0")
    app.run()
    assert not app.exception
    assert extractor.extract_frames.call_count == 1
    saved = load_progress(profile)
    assert len(saved["practice_attempts"]) == 1
    assert saved["last_diagnostic"]["trace"]["trace_source"] == "decision_tree_primary"
    assert not any("What the model noticed" in item.value for item in app.subheader)
    app.run()
    enabled[0] = False
    app.run()
    assert extractor.extract_frames.call_count == 1
    assert len(load_progress(profile)["practice_attempts"]) == 1
    saved["settings"]["developer_mode"] = True
    save_progress(profile, saved)
    app.run()
    assert any("Actual Decision Tree path" in item.value for item in app.markdown)
    observations = [item.value for item in app.markdown if 'class="observation"' in item.value]
    assert observations and all("cutoff" not in value and "Observed `" not in value for value in observations)
    button(app, "Next · Replay").click().run()
    assert load_progress(profile)["lesson_states"]["0"]["stage"] == "replay"
    button(app, "Next word").click().run()
    assert not app.exception
    assert 0 in load_progress(profile)["completed_lessons"]
    assert load_progress(profile)["current_lesson"] == 1
    assert not app.get("file_uploader") and not app.radio


def test_all_navigation_pages_and_developer_mode(profile):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert not app.selectbox and not app.radio
    # Native dialog fragment interactions are covered by the browser check;
    # AppTest in Streamlit 1.39 reruns the full script for those interactions.
    progress = load_progress(profile)
    progress["settings"] = {"model": "Extra Trees", "developer_mode": True}
    save_progress(profile, progress)
    app.run()
    assert not app.exception
    assert load_progress(profile)["settings"] == {"model": "Extra Trees", "developer_mode": True}
    for page in ["Lessons", "Quiz", "Final Exam", "Progress", "Home"]:
        button(app, page).click().run()
        assert not app.exception, page
        assert any(item.value == "What the model noticed" for item in app.subheader)
    assert not app.radio


def test_custom_quiz_selection_and_saved_final_exam(profile):
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params["page"] = "Quiz"
    app.run()
    next(item for item in app.checkbox if item.label == "Greetings").set_value(True).run()
    next(item for item in app.checkbox if item.label == "Colors").set_value(True).run()
    assert len(app.checkbox) == 10
    assert not app.multiselect
    button(app, "Start quiz").click().run()
    assert not app.exception
    active = load_progress(profile)["active_assessment"]
    assert len(active["questions"]) == 23
    questions = active["questions"]
    app.run()
    assert load_progress(profile)["active_assessment"]["questions"] == questions
    # End explicitly before beginning a different assessment.
    button(app, "End assessment").click().run()
    assert len(load_progress(profile)["abandoned_assessments"]) == 1
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    button(app, "Final Exam").click().run()
    button(app, "Start final exam").click().run()
    assert not app.exception
    saved = load_progress(profile)["active_assessment"]
    assert len(saved["questions"]) == 105
    assert saved["kind"] == "final"


def test_hidden_meaning_answer_and_completed_review(profile, monkeypatch):
    lessons = ui.curriculum()
    progress = load_progress(profile)
    active = create_assessment(lessons, ["GREETING"], "category", "Decision Tree", seed=2)
    # Keep a real balanced question list, with meaning questions first.
    active["questions"].sort(key=lambda item: item["mode"])
    progress["active_assessment"] = active
    progress["settings"]["developer_mode"] = True
    progress["last_diagnostic"] = {"model": "Decision Tree", "class_id": active["questions"][0]["class_id"], "assessment_id": None}
    save_progress(profile, progress)
    payload, enabled, extractor = mock_camera(monkeypatch)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params["page"] = "Quiz"
    app.run()
    assert not app.exception
    assert not app.header
    assert any("after you finish" in item.value for item in app.info)
    for index, question in enumerate(active["questions"]):
        if question["mode"] == "meaning":
            lesson = lessons.loc[lessons["id"].eq(question["class_id"])].iloc[0]
            app.text_input[0].set_value(lesson["filipino"]).run()
            next(item for item in app.button if item.label == "Submit answer").click().run()
        else:
            payload["capture_id"] = f"capture-{index}"
            payload["context"] = f"{PIPELINE_ID}:quiz:{active['id']}:{index}:Decision Tree"
            app.run()
            button(app, "Submit sign").click().run()
        assert not app.exception
    saved = load_progress(profile)
    assert saved["active_assessment"] is None
    assert saved["category_quizzes"]["GREETING"] == active["id"]
    assert len(saved["assessments"][0]["answers"]) == 10
    assert any(item.value == "Your results" for item in app.subheader)


def test_category_quiz_starts_only_after_all_words_are_finished(profile):
    progress = load_progress(profile)
    set_lesson_stage(progress, 9, "replay")
    save_progress(profile, progress)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params.update(page="Lessons", word="9")
    app.run()
    button(app, "Next word").click().run()
    saved = load_progress(profile)
    assert saved["active_assessment"] is None
    assert saved["current_lesson"] == 0
    for class_id in range(9):
        set_lesson_stage(saved, class_id, "complete")
    set_lesson_stage(saved, 9, "replay")
    save_progress(profile, saved)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params.update(page="Lessons", word="9")
    app.run()
    button(app, "Next · Category quiz").click().run()
    saved = load_progress(profile)
    assert saved["active_assessment"]["kind"] == "category"
    assert saved["active_assessment"]["categories"] == ["GREETING"]
    assert len(saved["active_assessment"]["questions"]) == 10


def test_model_change_ignores_stale_camera_capture(profile, monkeypatch):
    progress = load_progress(profile)
    set_lesson_stage(progress, 0, "copy")
    progress["settings"]["developer_mode"] = True
    save_progress(profile, progress)
    payload, enabled, extractor = mock_camera(monkeypatch)
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.query_params.update(page="Lessons", word="0")
    app.run()
    saved = load_progress(profile)
    saved["settings"]["model"] = "Extra Trees"
    save_progress(profile, saved)
    payload["capture_id"] = "stale-capture"
    app.run()
    assert not app.exception
    assert extractor.extract_frames.call_count == 1
    assert len(load_progress(profile)["practice_attempts"]) == 1
    assert any("selected model" in item.value for item in app.info)
