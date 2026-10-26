from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).casefold()
    value = value.replace("’", "'")
    value = re.sub(r"[^a-z0-9\s]", "", value)
    return " ".join(value.split())


def load_lessons(path: str | Path | None = None) -> pd.DataFrame:
    lesson_path = Path(path) if path else PROJECT_ROOT / "config" / "lessons.csv"
    lessons = pd.read_csv(lesson_path).fillna("")
    if "display_label" not in lessons:
        lessons["display_label"] = lessons["label"]
    lessons["accepted_answers"] = lessons.apply(
        lambda row: sorted(
            {
                normalize_answer(answer)
                for answer in [row["label"], row["display_label"], row["filipino"], *str(row["synonyms"]).split("|")]
                if normalize_answer(answer)
            }
        ),
        axis=1,
    )
    return lessons.sort_values("beginner_order").reset_index(drop=True)


def bundled_video(lesson: pd.Series | dict) -> Path:
    """Resolve project-owned media without consulting the source dataset."""
    root = PROJECT_ROOT.resolve()
    path = (root / str(lesson["video"])).resolve()
    if not path.is_relative_to(root / "assets" / "lessons"):
        raise ValueError("Lesson video must belong to the bundled lesson assets")
    return path


def categories(lessons: pd.DataFrame) -> list[str]:
    return lessons["category"].drop_duplicates().tolist()


def category_title(category: str) -> str:
    return {"GREETING": "Greetings", "NUMBER": "Numbers", "COLOR": "Colors", "DRINK": "Drinks"}.get(category, category.title())


def check_meaning(answer: str, lesson: pd.Series | dict[str, object]) -> bool:
    accepted = lesson["accepted_answers"]
    return normalize_answer(answer) in set(accepted)


def find_demo_video(dataset_root: str | Path, class_id: int, preferred_signer: str = "0") -> Path | None:
    class_dir = Path(dataset_root) / "clips" / str(int(class_id))
    preferred = [class_dir / f"{preferred_signer}{suffix}" for suffix in (".MOV", ".mov", ".mp4")]
    for path in preferred:
        if path.exists():
            return path
    if not class_dir.exists():
        return None
    videos = sorted(path for path in class_dir.iterdir() if path.is_file())
    return videos[0] if videos else None

