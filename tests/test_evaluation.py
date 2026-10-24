import json

import numpy as np

from senyas.evaluation import recognition_metrics


def test_recognition_metrics_are_json_serializable() -> None:
    metrics = recognition_metrics(
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([0, 2, 2], dtype=np.int32),
    )
    encoded = json.dumps(metrics)
    assert '"accuracy"' in encoded
    assert metrics["labels"] == [0, 1, 2]

