from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROGRESS = {"completed_lessons": [], "practice_attempts": [], "meaning_attempts": []}


def load_progress(path: str | Path) -> dict:
    progress_path = Path(path)
    if not progress_path.exists():
        return {key: list(value) for key, value in DEFAULT_PROGRESS.items()}
    with progress_path.open("r", encoding="utf-8") as stream:
        stored = json.load(stream)
    return {**DEFAULT_PROGRESS, **stored}


def save_progress(path: str | Path, progress: dict) -> None:
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("w", encoding="utf-8") as stream:
        json.dump(progress, stream, indent=2, ensure_ascii=False)


def record_attempt(progress: dict, activity: str, payload: dict) -> dict:
    updated = {key: list(value) for key, value in progress.items()}
    key = "practice_attempts" if activity == "copy" else "meaning_attempts"
    updated.setdefault(key, []).append(
        {**payload, "recorded_at": datetime.now(timezone.utc).isoformat()}
    )
    return updated

