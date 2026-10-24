import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

from senyas.model import decision_trace, predict_vector, save_artifact


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
