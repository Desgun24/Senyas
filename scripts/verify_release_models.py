"""Replay saved held-out sequences through the app adapter without fitting."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from senyas.frozen_recognizer import load_recognizer, analyze_sequence
from senyas.sequences import load_sequence
from senyas.semantic_explanations import describe_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('test_output', type=Path)
    args = parser.parse_args()
    rows = pd.read_csv(args.test_output / 'feature_index.csv').sort_values('vid_path')
    records = {}
    for title, key in [('Decision Tree', 'decision_tree'), ('Extra Trees', 'extra_trees')]:
        artifact = load_recognizer(ROOT / 'artifacts/frozen', title)
        expected = pd.read_csv(args.test_output / key / 'test_predictions.csv').set_index('vid_path')
        probabilities = pd.read_csv(args.test_output / key / 'test_probabilities.csv', index_col='vid_path')
        correct = 0
        for row in rows.itertuples():
            result = analyze_sequence(artifact, load_sequence(row.cache_file))
            prediction = result['prediction']
            assert prediction['class_id'] == int(expected.loc[row.vid_path, 'predicted_id'])
            assert np.allclose(list(prediction['probabilities'].values()), probabilities.loc[row.vid_path].to_numpy(), atol=1e-12)
            if result['trace']:
                describe_trace(result['trace'], result['imputed_features'])
            correct += prediction['class_id'] == row.id_label
        records[title] = {'reproduced_predictions': len(rows), 'correct': correct, 'sha256': artifact['runtime_sha256']}
        print(title, records[title], flush=True)
    (args.test_output / 'release_adapter_verification.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
