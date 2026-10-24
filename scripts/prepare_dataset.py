from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from senyas.config import dataset_dir  # noqa: E402
from senyas.dataset import (  # noqa: E402
    assert_no_group_leakage,
    assign_group_splits,
    build_manifest,
    merge_condition_tags,
    split_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FSL videos and create a signer-disjoint split.")
    parser.add_argument("--dataset", type=Path, default=dataset_dir())
    parser.add_argument("--conditions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.dataset)
    manifest = merge_condition_tags(manifest, args.conditions)
    manifest = assign_group_splits(manifest, seed=args.seed)
    assert_no_group_leakage(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    summary = {
        "dataset": str(args.dataset.resolve()),
        "manifest": str(args.output.resolve()),
        "samples": len(manifest),
        "classes": int(manifest["id_label"].nunique()),
        "signers_or_sessions": int(manifest["signer_id"].nunique()),
        "splits": [item.__dict__ for item in split_summary(manifest)],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

