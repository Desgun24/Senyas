from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv", ".webm"}


@dataclass(frozen=True)
class SplitSummary:
    split: str
    samples: int
    groups: int
    classes: int
    ratio: float


def _normalise_relative_path(value: str) -> Path:
    return Path(str(value).replace("\\", "/"))


def infer_signer_id(video_path: str | Path) -> str:
    """Infer the signer/session ID from the clip filename.

    FSL105 clips use the pattern ``clips/<class>/<signer>.MOV``. If a future
    dataset uses a different convention, add an explicit ``signer_id`` column
    to its manifest instead of relying on this fallback.
    """

    return Path(str(video_path).replace("\\", "/")).stem


def _load_source_tables(dataset_root: Path) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for name in ("train.csv", "test.csv"):
        path = dataset_root / name
        if path.exists():
            table = pd.read_csv(path)
            table["source_split"] = path.stem
            tables.append(table)

    if tables:
        frame = pd.concat(tables, ignore_index=True)
        required = {"vid_path", "id_label", "label"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Source CSV files are missing columns: {sorted(missing)}")
        return frame.drop_duplicates(subset=["vid_path"], keep="first")

    labels_path = dataset_root / "labels.csv"
    clips_dir = dataset_root / "clips"
    if not labels_path.exists() or not clips_dir.exists():
        raise FileNotFoundError(
            "Expected train.csv/test.csv or both labels.csv and clips/. "
            f"Dataset root: {dataset_root}"
        )

    labels = pd.read_csv(labels_path).set_index("id")
    rows: list[dict[str, object]] = []
    for video in sorted(clips_dir.rglob("*")):
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        class_id = int(video.parent.name)
        label_row = labels.loc[class_id]
        rows.append(
            {
                "vid_path": video.relative_to(dataset_root).as_posix(),
                "id_label": class_id,
                "label": label_row["label"],
                "category": label_row.get("category", "UNSPECIFIED"),
                "source_split": "scanned",
            }
        )
    return pd.DataFrame(rows)


def build_manifest(dataset_root: str | Path) -> pd.DataFrame:
    """Create a validated manifest for every source video."""

    root = Path(dataset_root).expanduser().resolve()
    frame = _load_source_tables(root).copy()
    frame["vid_path"] = frame["vid_path"].map(lambda p: _normalise_relative_path(p).as_posix())
    frame["video_path"] = frame["vid_path"].map(lambda p: str((root / p).resolve()))
    frame["signer_id"] = frame["vid_path"].map(infer_signer_id)
    frame["id_label"] = frame["id_label"].astype(int)
    frame["label"] = frame["label"].astype(str).str.strip()
    if "category" not in frame:
        frame["category"] = "UNSPECIFIED"
    frame["category"] = frame["category"].fillna("UNSPECIFIED").astype(str).str.strip()
    frame["exists"] = frame["video_path"].map(lambda p: Path(p).is_file())
    frame["extension"] = frame["video_path"].map(lambda p: Path(p).suffix.lower())

    duplicate_paths = frame["vid_path"].duplicated(keep=False)
    if duplicate_paths.any():
        duplicates = frame.loc[duplicate_paths, "vid_path"].tolist()[:5]
        raise ValueError(f"Duplicate video paths found, for example: {duplicates}")
    if not frame["exists"].all():
        missing = frame.loc[~frame["exists"], "video_path"].tolist()[:5]
        raise FileNotFoundError(f"Missing video files, for example: {missing}")
    invalid = ~frame["extension"].isin(VIDEO_EXTENSIONS)
    if invalid.any():
        raise ValueError(f"Unsupported video extensions: {sorted(frame.loc[invalid, 'extension'].unique())}")

    return frame.sort_values(["id_label", "signer_id", "vid_path"]).reset_index(drop=True)


def _score_partition(
    frame: pd.DataFrame,
    group_order: np.ndarray,
    train_end: int,
    val_end: int,
    group_sizes: dict[str, int],
    group_labels: dict[str, set[int]],
    ratios: tuple[float, float, float],
) -> float:
    partitions = (
        group_order[:train_end],
        group_order[train_end:val_end],
        group_order[val_end:],
    )
    total = len(frame)
    all_labels = set(frame["id_label"].unique())
    score = 0.0
    for groups, target in zip(partitions, ratios):
        count = sum(group_sizes[str(group)] for group in groups)
        actual = count / total
        labels: set[int] = set()
        for group in groups:
            labels.update(group_labels[str(group)])
        missing_fraction = len(all_labels.difference(labels)) / max(len(all_labels), 1)
        score += abs(actual - target) + (missing_fraction * 5.0)
    return score


def assign_group_splits(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    trials: int = 4000,
) -> pd.DataFrame:
    """Assign train/validation/test splits while keeping signers disjoint.

    Random candidate group orders are searched for the closest sample-level
    70/15/15 ratio. Class-coverage penalties prevent a deceptively close split
    that omits signs from validation or testing.
    """

    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    required = {"signer_id", "id_label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    result = frame.copy()
    groups = result["signer_id"].astype(str).unique()
    if len(groups) < 3:
        raise ValueError("At least three signer/session groups are required for group-aware splitting")

    sizes = result.groupby(result["signer_id"].astype(str)).size().to_dict()
    labels = {
        str(group): set(part["id_label"].astype(int).tolist())
        for group, part in result.groupby(result["signer_id"].astype(str))
    }
    rng = np.random.default_rng(seed)
    best: tuple[float, np.ndarray, int, int] | None = None
    target_train = ratios[0] * len(result)
    target_train_val = (ratios[0] + ratios[1]) * len(result)

    for _ in range(max(trials, 1)):
        order = rng.permutation(groups)
        cumulative = np.cumsum([sizes[str(group)] for group in order])
        train_candidates = np.argsort(np.abs(cumulative[:-2] - target_train))[:3]
        for train_index in train_candidates:
            train_end = int(train_index + 1)
            valid_positions = np.arange(train_end, len(order) - 1)
            if not len(valid_positions):
                continue
            val_index = valid_positions[
                np.argmin(np.abs(cumulative[valid_positions] - target_train_val))
            ]
            val_end = int(val_index + 1)
            score = _score_partition(result, order, train_end, val_end, sizes, labels, ratios)
            if best is None or score < best[0]:
                best = (score, order.copy(), train_end, val_end)

    if best is None:
        raise RuntimeError("Unable to construct a three-way group split")

    _, order, train_end, val_end = best
    mapping: dict[str, str] = {}
    for group in order[:train_end]:
        mapping[str(group)] = "train"
    for group in order[train_end:val_end]:
        mapping[str(group)] = "validation"
    for group in order[val_end:]:
        mapping[str(group)] = "test"
    result["split"] = result["signer_id"].astype(str).map(mapping)
    return result


def split_summary(frame: pd.DataFrame) -> list[SplitSummary]:
    total = len(frame)
    return [
        SplitSummary(
            split=name,
            samples=len(part),
            groups=part["signer_id"].nunique(),
            classes=part["id_label"].nunique(),
            ratio=len(part) / max(total, 1),
        )
        for name, part in frame.groupby("split", sort=False)
    ]


def assert_no_group_leakage(frame: pd.DataFrame) -> None:
    memberships = frame.groupby("signer_id")["split"].nunique()
    leaked = memberships[memberships > 1]
    if not leaked.empty:
        raise AssertionError(f"Signer/session leakage detected: {leaked.index.tolist()}")


def merge_condition_tags(frame: pd.DataFrame, condition_file: str | Path | None) -> pd.DataFrame:
    if not condition_file:
        return frame
    path = Path(condition_file)
    if not path.exists():
        raise FileNotFoundError(path)
    tags = pd.read_csv(path)
    if "vid_path" not in tags:
        raise ValueError("Condition file must contain a vid_path column")
    tags["vid_path"] = tags["vid_path"].map(lambda p: _normalise_relative_path(p).as_posix())
    columns = [column for column in tags.columns if column != "signer_id"]
    return frame.merge(tags[columns], on="vid_path", how="left", validate="one_to_one")


def available_condition_columns(frame: pd.DataFrame) -> Iterable[str]:
    for column in ("lighting", "background", "distance", "speed"):
        if column in frame and frame[column].notna().any():
            yield column

