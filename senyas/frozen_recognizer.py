"""Runtime adapter for the immutable, held-out-tested semantic models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .model import decision_trace, load_artifact, predict_vector
from .semantic_tree import prepare_tree_input, summarize_semantic_sequence
from .sequences import SequenceExtractor

PIPELINE_ID = "frozen-three-phase-v1"
MODEL_FILES = {"Decision Tree": "decision_tree.joblib", "Extra Trees": "extra_trees.joblib"}


def load_recognizer(directory: Path, model_name: str) -> dict:
    if model_name not in MODEL_FILES:
        raise ValueError("Unknown recognition model")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    filename = MODEL_FILES[model_name]
    path = directory / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest["pipeline_id"] != PIPELINE_ID or digest != manifest["files"][filename]:
        raise ValueError("The frozen model bundle failed its integrity check")
    artifact = load_artifact(path)
    if (len(artifact["feature_names"]) != 3392 or artifact["model"].n_features_in_ != 3392
            or list(artifact["feature_preparation"].feature_names) != artifact["feature_names"]
            or artifact["representation"] != "phases"
            or artifact["metadata"]["extractor"] != manifest["extractor"]
            or artifact["feature_preparation"].training_rows != 1494
            or artifact["model"].classes_.tolist() != list(range(105))):
        raise ValueError("The frozen model schema is incompatible with this app")
    return {**artifact, "runtime_model_name": model_name, "runtime_sha256": digest,
            "pipeline_id": PIPELINE_ID}


def analyze_sequence(artifact: dict, sequence) -> dict:
    vector = prepare_tree_input(artifact, sequence)
    prediction = predict_vector(artifact, vector)
    trace = decision_trace(artifact, vector) if artifact["runtime_model_name"] == "Decision Tree" else None
    missing = []
    if trace:
        if not all(rule["condition_holds"] for rule in trace["rules"]):
            raise ValueError("The model's decision path could not be verified")
        raw = summarize_semantic_sequence(sequence, artifact["representation"])
        preparation = artifact["feature_preparation"]
        missing = [name for name, value in zip(artifact["feature_names"], raw[preparation.enabled])
                   if not np.isfinite(value)]
    return {"prediction": prediction, "trace": trace, "imputed_features": missing,
            "extraction": sequence.metadata, "pipeline_id": PIPELINE_ID,
            "model_sha256": artifact["runtime_sha256"]}


def analyze_attempt(artifact: dict, attempt) -> dict:
    if attempt.timestamps is None:
        raise ValueError("This recording has no frame timestamps. Please record again.")
    with SequenceExtractor(**artifact["metadata"]["extractor"]) as extractor:
        sequence = extractor.extract_frames(attempt.frames, attempt.timestamps)
    return analyze_sequence(artifact, sequence)
