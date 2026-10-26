from pathlib import Path
import json
import re

import numpy as np
import pytest

from senyas.frozen_recognizer import load_recognizer, analyze_sequence
from senyas.semantic_explanations import describe_trace, SPECS
from senyas.sequences import FeatureSequence, SEQUENCE_NAMES

ROOT = Path(__file__).resolve().parents[1]


def sequence():
    values = np.zeros((30, len(SEQUENCE_NAMES)), dtype=np.float32)
    values[:, SEQUENCE_NAMES.index('right.present')] = 1
    values[:, SEQUENCE_NAMES.index('right.index.middle_flexion')] = np.linspace(0, .8, 30)
    return FeatureSequence(values, np.arange(30) / 30)


@pytest.mark.parametrize('name', ['Decision Tree', 'Extra Trees'])
def test_frozen_runtime_shape_and_actual_explanation(name):
    artifact = load_recognizer(ROOT / 'artifacts/frozen', name)
    result = analyze_sequence(artifact, sequence())
    assert result['model_sha256'] == artifact['runtime_sha256']
    assert len(result['prediction']['probabilities']) == 105
    if name == 'Decision Tree':
        assert result['trace']['trace_source'] == 'decision_tree_primary'
        assert all(r['condition_holds'] for r in result['trace']['rules'])
        assert describe_trace(result['trace'], result['imputed_features'])
    else:
        assert result['trace'] is None


def test_no_hands_is_rejected_before_prediction():
    artifact = load_recognizer(ROOT / 'artifacts/frozen', 'Decision Tree')
    clip = sequence()
    clip.values[:, SEQUENCE_NAMES.index('right.present')] = 0
    with pytest.raises(ValueError, match='three frames'):
        analyze_sequence(artifact, clip)


def test_explanations_cover_every_used_split_and_verify_comparisons():
    artifact = load_recognizer(ROOT / 'artifacts/frozen', 'Decision Tree')
    used = {artifact['feature_names'][i] for i in artifact['model'].tree_.feature if i >= 0}
    assert len(used) == 135
    for feature in used:
        assert feature.rsplit('.', 2)[0] in SPECS
        for value, operator in [(0., '<='), (1., '>')]:
            rule = dict(feature=feature, value=value, operator=operator, threshold=.5, condition_holds=True)
            observation = describe_trace({'rules': [rule]})[0]
            assert not re.search(r'\d', ' '.join(observation.values()))
            replacement = describe_trace({'rules': [rule]}, [feature])[0]
            assert 'unavailable' in replacement['description']
            rule['condition_holds'] = False
            with pytest.raises(ValueError):
                describe_trace({'rules': [rule]})


def test_hash_failure_is_rejected(tmp_path):
    original = ROOT / 'artifacts/frozen'
    (tmp_path / 'manifest.json').write_bytes((original / 'manifest.json').read_bytes())
    (tmp_path / 'decision_tree.joblib').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        load_recognizer(tmp_path, 'Decision Tree')


def test_frozen_runtime_sources_are_byte_identical():
    import hashlib
    manifest = json.loads((ROOT / 'artifacts/frozen/manifest.json').read_text())
    for name, expected in manifest['runtime_source_lf_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
