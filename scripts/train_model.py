from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from senyas.config import artifact_dir  # noqa: E402
from senyas.dataset import available_condition_columns, split_summary  # noqa: E402
from senyas.evaluation import (  # noqa: E402
    grouped_robustness,
    prediction_latency_ms,
    recognition_metrics,
    save_json,
)
from senyas.features import load_feature_bundle  # noqa: E402
from senyas.model import (  # noqa: E402
    decision_trace,
    export_tree_metadata,
    fit_final_extra_trees,
    fit_final_tree,
    save_artifact,
    select_decision_tree,
    select_extra_trees,
)


def parse_args() -> argparse.Namespace:
    artifacts = artifact_dir()
    parser = argparse.ArgumentParser(description="Train an Extra Trees recognizer with a Decision Tree explainer.")
    parser.add_argument("--features", type=Path, default=PROJECT_ROOT / "data" / "processed" / "features.npz")
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "data" / "processed" / "feature_index.csv")
    parser.add_argument("--model-output", type=Path, default=artifacts / "hybrid_recognizer.joblib")
    parser.add_argument("--evaluation-output", type=Path, default=artifacts / "evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_confidence_threshold(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    confidence: np.ndarray,
    target_selective_accuracy: float = 0.95,
) -> tuple[float, float, float]:
    """Choose maximum validation coverage while retaining the target accepted-sample accuracy."""

    choices: list[tuple[float, float, float]] = []
    for threshold in np.arange(0.10, 0.81, 0.01):
        accepted = confidence >= threshold
        if not accepted.any():
            continue
        selective_accuracy = float((y_predicted[accepted] == y_true[accepted]).mean())
        coverage = float(accepted.mean())
        if selective_accuracy >= target_selective_accuracy:
            choices.append((coverage, float(threshold), selective_accuracy))
    if not choices:
        return 0.25, 0.0, 0.0
    coverage, threshold, selective_accuracy = max(choices, key=lambda item: item[0])
    return threshold, coverage, selective_accuracy


def main() -> None:
    args = parse_args()
    X, feature_names, video_paths = load_feature_bundle(args.features)
    index = pd.read_csv(args.index)
    success = index[index["success"].astype(bool)].sort_values("array_index").reset_index(drop=True)
    if len(success) != len(X):
        raise ValueError(f"Feature rows ({len(X)}) do not match successful index rows ({len(success)})")
    if not np.array_equal(success["video_path"].astype(str).to_numpy(), video_paths):
        raise ValueError("Feature bundle and feature index video order do not match")

    masks = {name: success["split"].eq(name).to_numpy() for name in ("train", "validation", "test")}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"No successful features are available for the {name} split")
    y = success["id_label"].astype(int).to_numpy()

    selected_tree, tree_candidates = select_decision_tree(
        X[masks["train"]],
        y[masks["train"]],
        X[masks["validation"]],
        y[masks["validation"]],
        seed=args.seed,
    )
    baseline_validation_predictions = selected_tree.predict(X[masks["validation"]])
    baseline_validation_metrics = recognition_metrics(
        y[masks["validation"]], baseline_validation_predictions
    )

    selected_ensemble, ensemble_candidates = select_extra_trees(
        X[masks["train"]],
        y[masks["train"]],
        X[masks["validation"]],
        y[masks["validation"]],
        seed=args.seed,
    )
    validation_predictions = selected_ensemble.predict(X[masks["validation"]]).astype(int)
    validation_probabilities = selected_ensemble.predict_proba(X[masks["validation"]])
    validation_confidence = validation_probabilities.max(axis=1)
    validation_metrics = recognition_metrics(y[masks["validation"]], validation_predictions)
    threshold, validation_coverage, validation_selective_accuracy = choose_confidence_threshold(
        y[masks["validation"]], validation_predictions, validation_confidence
    )

    train_validation = masks["train"] | masks["validation"]
    final_model = fit_final_extra_trees(
        selected_ensemble, X[train_validation], y[train_validation], seed=args.seed
    )
    final_explainer = fit_final_tree(
        selected_tree, X[train_validation], y[train_validation], seed=args.seed
    )
    test_predictions = final_model.predict(X[masks["test"]]).astype(int)
    test_probabilities = final_model.predict_proba(X[masks["test"]])
    test_confidence = test_probabilities.max(axis=1)
    test_metrics = recognition_metrics(y[masks["test"]], test_predictions)
    baseline_test_predictions = final_explainer.predict(X[masks["test"]]).astype(int)
    baseline_test_metrics = recognition_metrics(y[masks["test"]], baseline_test_predictions)

    best_ensemble = ensemble_candidates[0]
    best_tree = tree_candidates[0]
    artifact = save_artifact(
        args.model_output,
        final_model,
        feature_names,
        success,
        threshold=threshold,
        metadata={
            "primary_model": "ExtraTreesClassifier",
            "primary_parameters": {
                "n_estimators": best_ensemble.n_estimators,
                "max_features": best_ensemble.max_features,
                "min_samples_leaf": best_ensemble.min_samples_leaf,
            },
            "explainer_model": "DecisionTreeClassifier",
            "explainer_parameters": {
                "criterion": best_tree.criterion,
                "max_depth": best_tree.max_depth,
                "min_samples_leaf": best_tree.min_samples_leaf,
                "ccp_alpha": best_tree.ccp_alpha,
            },
            "validation_macro_f1": best_ensemble.validation_macro_f1,
        },
        explainer_model=final_explainer,
    )

    test_table = success.loc[masks["test"]].copy()
    test_table["predicted_id"] = test_predictions
    test_table["confidence"] = test_confidence
    test_table["uncertain"] = test_confidence < threshold
    test_table["baseline_predicted_id"] = baseline_test_predictions
    label_lookup = {int(key): value["label"] for key, value in artifact["label_lookup"].items()}
    test_table["predicted_label"] = test_table["predicted_id"].map(label_lookup)
    args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    test_table.to_csv(args.evaluation_output.parent / "test_predictions.csv", index=False)

    group_columns = ["signer_id", *list(available_condition_columns(test_table))]
    robustness = grouped_robustness(test_table, "id_label", "predicted_id", group_columns)
    extraction_times = test_table.get("extraction_ms", pd.Series(dtype=float)).dropna().astype(float)
    traces = [decision_trace(artifact, X[index_value]) for index_value in np.where(masks["test"])[0][:50]]
    trace_checks = [all(rule["condition_holds"] for rule in trace["rules"]) for trace in traces]
    trace_agreement = [bool(trace["agrees_with_primary"]) for trace in traces]

    report = {
        "split_summary": [asdict(item) for item in split_summary(success)],
        "model_selection": {
            "primary_model": "ExtraTreesClassifier",
            "primary_best": asdict(best_ensemble),
            "primary_candidate_count": len(ensemble_candidates),
            "primary_top_five": [asdict(candidate) for candidate in ensemble_candidates[:5]],
            "explainer_model": "DecisionTreeClassifier",
            "explainer_best": asdict(best_tree),
            "explainer_candidate_count": len(tree_candidates),
        },
        "validation": validation_metrics,
        "baseline_validation": baseline_validation_metrics,
        "test": test_metrics,
        "baseline_test": baseline_test_metrics,
        "comparison": {
            "validation_accuracy_gain_percentage_points": float(
                (validation_metrics["accuracy"] - baseline_validation_metrics["accuracy"]) * 100
            ),
            "test_accuracy_gain_percentage_points": float(
                (test_metrics["accuracy"] - baseline_test_metrics["accuracy"]) * 100
            ),
        },
        "uncertainty": {
            "threshold": threshold,
            "selection_target_accuracy": 0.95,
            "validation_coverage": validation_coverage,
            "validation_selective_accuracy": validation_selective_accuracy,
            "uncertain_samples": int(test_table["uncertain"].sum()),
            "coverage": float((~test_table["uncertain"]).mean()),
        },
        "robustness": robustness,
        "efficiency": {
            "feature_extraction_ms_mean": float(extraction_times.mean()) if len(extraction_times) else None,
            "feature_extraction_ms_p95": float(extraction_times.quantile(0.95)) if len(extraction_times) else None,
            "prediction_ms": prediction_latency_ms(final_model, X[masks["test"]]),
        },
        "explainability": {
            "traces_checked": len(trace_checks),
            "valid_traces": int(sum(trace_checks)),
            "all_displayed_conditions_correct": bool(all(trace_checks)),
            "primary_model": "Extra Trees ensemble",
            "trace_model": "Decision Tree baseline",
            "trace_primary_agreement": float(np.mean(trace_agreement)) if trace_agreement else 0.0,
            "tree_nodes": int(final_explainer.tree_.node_count),
            "tree_depth": int(final_explainer.tree_.max_depth),
        },
        "limitations": [
            "Lighting, background, distance, and signing-speed robustness require condition tags in the manifest.",
            "Filename-derived signer IDs should be confirmed against the FSL105 data documentation.",
            "Filipino translations and demonstration videos should be reviewed by a qualified FSL signer or instructor.",
            "Displayed rule traces come from the transparent Decision Tree baseline; primary recognition uses Extra Trees.",
        ],
    }
    save_json(args.evaluation_output, report)
    export_tree_metadata(artifact, args.evaluation_output.parent / "model_summary.json")
    concise_test = {
        key: test_metrics[key]
        for key in ("samples", "accuracy", "macro_precision", "macro_recall", "macro_f1")
    }
    print(
        json.dumps(
            {
                "model": str(args.model_output),
                "evaluation": str(args.evaluation_output),
                "primary_model": "ExtraTreesClassifier",
                "test": concise_test,
                "baseline_test_accuracy": baseline_test_metrics["accuracy"],
                "accuracy_gain_percentage_points": report["comparison"]["test_accuracy_gain_percentage_points"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
