"""Persistent balanced questions shared by category, custom and final quizzes."""
from __future__ import annotations

import random
import uuid

from .progress import now


def create_assessment(lessons, categories: list[str], kind: str, model: str, seed=None) -> dict:
    if kind not in {"category", "custom", "final"}:
        raise ValueError("Unknown assessment type")
    chosen = lessons if kind == "final" else lessons.loc[lessons["category"].isin(categories)]
    ids = chosen["id"].drop_duplicates().astype(int).tolist()
    if not ids:
        raise ValueError("Choose at least one topic")
    rng = random.Random(seed)
    rng.shuffle(ids)
    sign_count = len(ids) // 2 + (rng.randrange(2) if len(ids) % 2 else 0)
    questions = [{"class_id": class_id, "mode": "sign" if index < sign_count else "meaning"}
                 for index, class_id in enumerate(ids)]
    rng.shuffle(questions)
    return {
        "id": str(uuid.uuid4()), "kind": kind, "categories": chosen["category"].drop_duplicates().tolist(),
        "model": model, "started_at": now(), "questions": questions, "answers": [],
    }


def submit_answer(progress: dict, assessment_id: str, index: int, answer: dict) -> bool:
    active = progress.get("active_assessment")
    if not active or active["id"] != assessment_id or len(active["answers"]) != index:
        return False
    question = active["questions"][index]
    active["answers"].append({**answer, **question, "submitted_at": now()})
    if len(active["answers"]) == len(active["questions"]):
        active["completed_at"] = now()
        active["correct"] = sum(bool(row["correct"]) for row in active["answers"])
        progress["assessments"].append(active)
        if active["kind"] == "category":
            progress["category_quizzes"][active["categories"][0]] = active["id"]
        progress["active_assessment"] = None
    return True
