from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dataset_dir() -> Path:
    """Return the configured dataset directory.

    A local ``dataset`` folder is preferred so the project remains portable.
    """

    configured = os.getenv("SENYAS_DATASET_DIR")
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT / "dataset"


def artifact_dir() -> Path:
    configured = os.getenv("SENYAS_ARTIFACT_DIR")
    path = Path(configured).expanduser() if configured else PROJECT_ROOT / "artifacts"
    return path.resolve()


def confidence_threshold() -> float:
    return float(os.getenv("SENYAS_CONFIDENCE_THRESHOLD", "0.60"))

