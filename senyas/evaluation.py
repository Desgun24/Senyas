from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def recognition_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = [
        int(value)
        for value in sorted(set(np.asarray(y_true).astype(int)).union(np.asarray(y_pred).astype(int)))
    ]
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def grouped_robustness(
    table: pd.DataFrame,
    y_true_column: str,
    y_pred_column: str,
    group_columns: Iterable[str],
) -> dict[str, list[dict[str, Any]]]:
    report: dict[str, list[dict[str, Any]]] = {}
    for column in group_columns:
        if column not in table or not table[column].notna().any():
            continue
        rows: list[dict[str, Any]] = []
        for value, part in table.dropna(subset=[column]).groupby(column):
            y_true = part[y_true_column].to_numpy()
            y_pred = part[y_pred_column].to_numpy()
            rows.append(
                {
                    "group": str(value),
                    "samples": int(len(part)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                }
            )
        report[column] = rows
    return report


def prediction_latency_ms(model: Any, X: np.ndarray) -> dict[str, float]:
    timings: list[float] = []
    for sample in X:
        started = time.perf_counter()
        model.predict(sample.reshape(1, -1))
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "mean": float(np.mean(timings)) if timings else 0.0,
        "p95": float(np.percentile(timings, 95)) if timings else 0.0,
        "max": float(np.max(timings)) if timings else 0.0,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
