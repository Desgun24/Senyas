import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from senyas.model import decision_trace, predict_vector, save_artifact, select_recognizer


def test_hybrid_artifact_predicts_and_traces(tmp_path) -> None:
    X = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0], [2.0, 2.0], [2.1, 2.0]],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 1, 1, 2, 2], dtype=int)
    primary = ExtraTreesClassifier(n_estimators=10, random_state=42).fit(X, y)
    explainer = DecisionTreeClassifier(random_state=42).fit(X, y)
    labels = pd.DataFrame(
        {
            "id_label": [0, 1, 2],
            "label": ["zero", "one", "two"],
            "category": ["test", "test", "test"],
        }
    )
    artifact = save_artifact(
        tmp_path / "hybrid.joblib",
        primary,
        ["feature.a", "feature.b"],
        labels,
        threshold=0.25,
        explainer_model=explainer,
    )

    prediction = predict_vector(artifact, X[0])
    trace = decision_trace(artifact, X[0])

    assert prediction["class_id"] == 0
    assert trace["trace_source"] == "decision_tree_baseline"
    assert trace["explainer_prediction"]["class_id"] == 0
    assert all(rule["condition_holds"] for rule in trace["rules"])


def test_selected_tree_predicts_and_explains_its_own_class() -> None:
    X = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1])
    ensemble = ExtraTreesClassifier(n_estimators=5, random_state=42).fit(X, y)
    tree = DecisionTreeClassifier(random_state=42).fit(X, 1 - y)
    bundle = {
        "model": ensemble,
        "explainer_model": tree,
        "feature_names": ["hand.position"],
        "label_lookup": {0: {"label": "zero"}, 1: {"label": "one"}},
        "confidence_threshold": 0.24,
    }
    selected = select_recognizer(bundle, "Decision Tree")
    prediction = predict_vector(selected, X[0])
    trace = decision_trace(selected, X[0])

    assert predict_vector(bundle, X[0])["class_id"] == 0
    assert prediction["class_id"] == 1
    assert trace["prediction"]["class_id"] == trace["explainer_prediction"]["class_id"] == 1
    assert trace["trace_source"] == "decision_tree_primary"
    assert trace["agrees_with_primary"]
    assert trace["leaf_id"] == int(tree.apply(X[:1])[0])
    assert all(rule["condition_holds"] for rule in trace["rules"])
    assert prediction["threshold"] is None
    assert not prediction["uncertain"]
    assert bundle["model"] is ensemble
    assert bundle["confidence_threshold"] == 0.24
    assert select_recognizer(bundle, "Extra Trees")["model"] is ensemble


def test_tree_uses_only_its_own_optional_confidence_cutoff() -> None:
    tree = DecisionTreeClassifier(max_depth=1, random_state=42).fit(
        [[0.0], [0.0], [1.0], [1.0]], [0, 1, 0, 1]
    )
    bundle = {
        "model": tree,
        "feature_names": ["hand.position"],
        "label_lookup": {0: {"label": "zero"}, 1: {"label": "one"}},
        "confidence_threshold": 0.24,
    }
    default = predict_vector(select_recognizer(bundle, "Decision Tree"), np.asarray([0.0]))
    assert default["confidence"] == 0.5
    assert not default["uncertain"]
    bundle["decision_tree_confidence_threshold"] = 0.75
    explicit = predict_vector(select_recognizer(bundle, "Decision Tree"), np.asarray([0.0]))
    assert explicit["threshold"] == 0.75
    assert explicit["uncertain"]
