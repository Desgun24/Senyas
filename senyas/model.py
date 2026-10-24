from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from .config import confidence_threshold


@dataclass(frozen=True)
class CandidateResult:
    criterion: str
    max_depth: int | None
    min_samples_leaf: int
    ccp_alpha: float
    validation_accuracy: float
    validation_macro_f1: float
    node_count: int


@dataclass(frozen=True)
class EnsembleCandidateResult:
    n_estimators: int
    max_features: str | float
    min_samples_leaf: int
    validation_accuracy: float
    validation_macro_f1: float


def _candidate_parameters() -> list[dict[str, Any]]:
    return [
        {
            "criterion": criterion,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "ccp_alpha": alpha,
        }
        for criterion in ("gini", "entropy")
        for depth in (6, 10, 14)
        for leaf in (2, 4, 8)
        for alpha in (0.0, 0.001)
    ]


def select_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int = 42,
) -> tuple[DecisionTreeClassifier, list[CandidateResult]]:
    """Tune only on the validation split and return the best transparent tree."""

    candidates: list[tuple[DecisionTreeClassifier, CandidateResult]] = []
    for parameters in _candidate_parameters():
        model = DecisionTreeClassifier(random_state=seed, class_weight="balanced", **parameters)
        model.fit(X_train, y_train)
        predicted = model.predict(X_validation)
        result = CandidateResult(
            **parameters,
            validation_accuracy=float(accuracy_score(y_validation, predicted)),
            validation_macro_f1=float(f1_score(y_validation, predicted, average="macro", zero_division=0)),
            node_count=int(model.tree_.node_count),
        )
        candidates.append((model, result))

    candidates.sort(
        key=lambda pair: (
            pair[1].validation_macro_f1,
            pair[1].validation_accuracy,
            -pair[1].node_count,
        ),
        reverse=True,
    )
    return candidates[0][0], [result for _, result in candidates]


def fit_final_tree(
    selected: DecisionTreeClassifier,
    X_train_validation: np.ndarray,
    y_train_validation: np.ndarray,
    seed: int = 42,
) -> DecisionTreeClassifier:
    params = {
        "criterion": selected.criterion,
        "max_depth": selected.max_depth,
        "min_samples_leaf": selected.min_samples_leaf,
        "ccp_alpha": selected.ccp_alpha,
        "class_weight": "balanced",
        "random_state": seed,
    }
    return DecisionTreeClassifier(**params).fit(X_train_validation, y_train_validation)


def select_extra_trees(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int = 42,
) -> tuple[ExtraTreesClassifier, list[EnsembleCandidateResult]]:
    """Select an accuracy-focused ensemble using validation data only."""

    candidates: list[tuple[ExtraTreesClassifier, EnsembleCandidateResult]] = []
    for max_features in ("sqrt", 0.1, 0.2, 0.35):
        for min_samples_leaf in (1, 2):
            model = ExtraTreesClassifier(
                n_estimators=300,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            predicted = model.predict(X_validation)
            result = EnsembleCandidateResult(
                n_estimators=300,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                validation_accuracy=float(accuracy_score(y_validation, predicted)),
                validation_macro_f1=float(
                    f1_score(y_validation, predicted, average="macro", zero_division=0)
                ),
            )
            candidates.append((model, result))

    candidates.sort(
        key=lambda pair: (pair[1].validation_macro_f1, pair[1].validation_accuracy),
        reverse=True,
    )
    return candidates[0][0], [result for _, result in candidates]


def fit_final_extra_trees(
    selected: ExtraTreesClassifier,
    X_train_validation: np.ndarray,
    y_train_validation: np.ndarray,
    seed: int = 42,
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=selected.n_estimators,
        max_features=selected.max_features,
        min_samples_leaf=selected.min_samples_leaf,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(X_train_validation, y_train_validation)


def save_artifact(
    path: str | Path,
    model: Any,
    feature_names: Sequence[str],
    label_table: pd.DataFrame,
    threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
    explainer_model: DecisionTreeClassifier | None = None,
) -> dict[str, Any]:
    label_lookup = {
        int(row["id_label"]): {"label": str(row["label"]), "category": str(row.get("category", ""))}
        for _, row in label_table.drop_duplicates("id_label").iterrows()
    }
    artifact = {
        "model": model,
        "feature_names": list(feature_names),
        "label_lookup": label_lookup,
        "confidence_threshold": threshold if threshold is not None else confidence_threshold(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }
    if explainer_model is not None:
        artifact["explainer_model"] = explainer_model
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    return artifact


def load_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def predict_vector(artifact: dict[str, Any], vector: np.ndarray) -> dict[str, Any]:
    model = artifact["model"]
    sample = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    started = time.perf_counter()
    probabilities = model.predict_proba(sample)[0]
    latency_ms = (time.perf_counter() - started) * 1000
    best_index = int(np.argmax(probabilities))
    class_id = int(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    threshold = float(artifact.get("confidence_threshold", 0.60))
    label = artifact["label_lookup"].get(class_id, {"label": str(class_id), "category": ""})
    return {
        "class_id": class_id,
        "label": label["label"],
        "category": label.get("category", ""),
        "confidence": confidence,
        "uncertain": confidence < threshold,
        "threshold": threshold,
        "prediction_ms": latency_ms,
        "probabilities": {
            int(class_value): float(probability)
            for class_value, probability in zip(model.classes_, probabilities)
        },
    }


def decision_trace(artifact: dict[str, Any], vector: np.ndarray) -> dict[str, Any]:
    primary_model = artifact["model"]
    model: DecisionTreeClassifier = artifact.get("explainer_model", primary_model)
    if not hasattr(model, "tree_"):
        raise ValueError("This artifact does not contain a Decision Tree explainer")
    feature_names: list[str] = artifact["feature_names"]
    sample = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    node_indicator = model.decision_path(sample)
    leaf_id = int(model.apply(sample)[0])
    nodes = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]
    rules: list[dict[str, Any]] = []
    for node_id in nodes:
        if node_id == leaf_id:
            continue
        feature_index = int(model.tree_.feature[node_id])
        threshold = float(model.tree_.threshold[node_id])
        value = float(sample[0, feature_index])
        went_left = value <= threshold
        rules.append(
            {
                "node": int(node_id),
                "feature": feature_names[feature_index],
                "value": value,
                "operator": "<=" if went_left else ">",
                "threshold": threshold,
                "condition_holds": bool(value <= threshold if went_left else value > threshold),
            }
        )
    prediction = predict_vector(artifact, vector)
    explainer_class_id = int(model.predict(sample)[0])
    explainer_label = artifact["label_lookup"].get(
        explainer_class_id, {"label": str(explainer_class_id), "category": ""}
    )
    return {
        "leaf_id": leaf_id,
        "rules": rules,
        "prediction": prediction,
        "explainer_prediction": {
            "class_id": explainer_class_id,
            "label": explainer_label["label"],
            "category": explainer_label.get("category", ""),
        },
        "agrees_with_primary": explainer_class_id == prediction["class_id"],
        "trace_source": "decision_tree_baseline",
    }


def friendly_feature_name(name: str) -> str:
    replacements = {
        "left.": "left hand ",
        "right.": "right hand ",
        "wrist_body": "wrist position relative to shoulders ",
        "velocity": "movement ",
        "tip_distance": "fingertip extension ",
        "bend": "finger bend ",
        "_": " ",
        ".": " - ",
    }
    friendly = name
    for source, destination in replacements.items():
        friendly = friendly.replace(source, destination)
    return " ".join(friendly.split())


def export_tree_metadata(artifact: dict[str, Any], path: str | Path) -> None:
    model = artifact["model"]
    explainer: DecisionTreeClassifier = artifact.get("explainer_model", model)
    summary = {
        "trained_at": artifact["trained_at"],
        "primary_model": type(model).__name__,
        "confidence_threshold": artifact["confidence_threshold"],
        "classes": model.classes_.astype(int).tolist(),
        "ensemble_trees": int(len(getattr(model, "estimators_", []))),
        "primary_parameters": artifact.get("metadata", {}).get("primary_parameters", {}),
        "explainer": {
            "model": type(explainer).__name__,
            "node_count": int(explainer.tree_.node_count),
            "max_depth": int(explainer.tree_.max_depth),
            "parameters": artifact.get("metadata", {}).get("explainer_parameters", {}),
        },
    }
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

