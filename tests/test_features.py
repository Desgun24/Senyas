import numpy as np
import pandas as pd

from senyas.features import (
    frame_feature_names,
    load_feature_bundle,
    save_feature_bundle,
    summarize_sequence,
    temporal_feature_names,
)


def test_temporal_summary_has_named_fixed_width() -> None:
    base = frame_feature_names()
    sequence = np.arange(4 * len(base), dtype=np.float32).reshape(4, len(base))
    vector = summarize_sequence(sequence)
    assert vector.shape == (len(temporal_feature_names(base)),)
    assert np.isfinite(vector).all()


def test_missing_values_are_interpolated_or_zero_filled() -> None:
    sequence = np.asarray([[1.0, np.nan], [np.nan, np.nan], [3.0, np.nan]], dtype=np.float32)
    vector = summarize_sequence(sequence)
    assert np.isfinite(vector).all()
    assert vector.shape == (20,)


def test_feature_bundle_uses_pickle_free_unicode_paths(tmp_path) -> None:
    matrix = np.asarray([[1.0, 2.0]], dtype=np.float32)
    index = pd.DataFrame(
        [{"success": True, "array_index": 0, "video_path": "clips/0/0.MOV"}]
    )
    output = tmp_path / "features.npz"
    save_feature_bundle(output, matrix, ["one", "two"], index)
    loaded, names, paths = load_feature_bundle(output)
    assert np.array_equal(loaded, matrix)
    assert names == ["one", "two"]
    assert paths.tolist() == ["clips/0/0.MOV"]

