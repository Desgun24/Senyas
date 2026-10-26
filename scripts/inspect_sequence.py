"""Extract, inspect, and benchmark semantic sequences without fitting a recognizer."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mediapipe  # noqa: E402,F401; initialize native libraries on the main thread.
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from senyas.sequences import SequenceExtractor, SEQUENCE_SPECS, circle_measurements, save_sequence  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--size', type=int, default=480)
    parser.add_argument('--complexity', type=int, choices=[0,1,2], default=0)
    parser.add_argument('--seconds', type=float, default=10.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(args.video):
        with SequenceExtractor(args.fps, args.size, args.complexity) as extractor:
            sequence = extractor.extract_video(path, max_seconds=args.seconds)
        stem = f'{index:02d}-{path.parent.name}-{path.stem}'
        save_sequence(args.output / (stem+'.npz'), sequence)
        pd.DataFrame(sequence.values, columns=sequence.feature_names).assign(timestamp_seconds=sequence.timestamps).to_csv(args.output/(stem+'.csv'),index=False)
        row = {**sequence.metadata, 'circle': {side: circle_measurements(sequence, side) for side in ['left','right']},
               'feature_coverage': {name: float(np.isfinite(sequence.values[:,i]).mean()) for i,name in enumerate(sequence.feature_names)}}
        (args.output/(stem+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False),encoding='utf-8')
        rows.append(sequence.metadata)
        print(json.dumps(sequence.metadata,indent=2),flush=True)
    pd.DataFrame([asdict(spec) for spec in SEQUENCE_SPECS]).to_csv(args.output/'feature_registry.csv',index=False)
    (args.output/'benchmark.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    main()
