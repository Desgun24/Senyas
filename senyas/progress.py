from __future__ import annotations

import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROGRESS = {
    "schema_version": 2, "completed_lessons": [], "practice_attempts": [], "meaning_attempts": [],
    "lesson_states": {}, "current_lesson": 0, "assessments": [], "active_assessment": None,
    "category_quizzes": {}, "settings": {"model": "Decision Tree", "developer_mode": False},
    "last_diagnostic": None,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upgrade_progress(stored: dict) -> dict:
    """Add learning state while retaining historical attempts and extra fields."""
    progress = {**copy.deepcopy(DEFAULT_PROGRESS), **copy.deepcopy(stored)}
    progress["settings"] = {**DEFAULT_PROGRESS["settings"], **progress.get("settings", {})}
    progress["schema_version"] = 2
    for class_id in progress["completed_lessons"]:
        progress["lesson_states"].setdefault(str(class_id), {"stage": "complete"})
    return progress


def load_progress(path: str | Path) -> dict:
    progress_path = Path(path)
    if not progress_path.exists():
        return upgrade_progress({})
    with progress_path.open("r", encoding="utf-8") as stream:
        stored = json.load(stream)
    return upgrade_progress(stored)


def save_progress(path: str | Path, progress: dict) -> None:
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=progress_path.parent,
                                         prefix=".progress-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(progress, stream, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, progress_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def record_attempt(progress: dict, activity: str, payload: dict) -> dict:
    updated = copy.deepcopy(progress)
    key = "practice_attempts" if activity == "copy" else "meaning_attempts"
    capture_id = payload.get("capture_id")
    if capture_id and any(item.get("capture_id") == capture_id for item in updated.get(key, [])):
        return updated
    updated.setdefault(key, []).append(
        {**payload, "recorded_at": datetime.now(timezone.utc).isoformat()}
    )
    return updated


def lesson_stage(progress: dict, class_id: int) -> str:
    return progress["lesson_states"].get(str(class_id), {}).get(
        "stage", "complete" if class_id in progress["completed_lessons"] else "watch"
    )


def set_lesson_stage(progress: dict, class_id: int, stage: str) -> None:
    if stage not in {"watch", "copy", "replay", "complete"}:
        raise ValueError("Unknown lesson stage")
    progress["lesson_states"][str(class_id)] = {"stage": stage, "updated_at": now()}
    progress["current_lesson"] = class_id
    if stage == "complete" and class_id not in progress["completed_lessons"]:
        progress["completed_lessons"].append(class_id)


def category_status(progress: dict, class_ids: list[int], category: str) -> str:
    done = set(progress["completed_lessons"])
    if all(class_id in done for class_id in class_ids) and category in progress["category_quizzes"]:
        return "complete"
    if any(str(class_id) in progress["lesson_states"] or class_id in done for class_id in class_ids):
        return "started"
    return "untouched"

