from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _restart_with_windows_system_dlls_first() -> None:
    """Restart once so MediaPipe cannot load conflicting third-party C++ DLLs."""
    if os.name != "nt" or os.environ.get("SENYAS_WINDOWS_DLL_PATH_READY") == "1":
        return

    system32 = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
    system32_key = os.path.normcase(os.path.normpath(system32))
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    if path_entries and os.path.normcase(os.path.normpath(path_entries[0])) == system32_key:
        return

    clean_environment = os.environ.copy()
    clean_environment["PATH"] = os.pathsep.join(
        [
            system32,
            *[
                entry
                for entry in path_entries
                if os.path.normcase(os.path.normpath(entry)) != system32_key
            ],
        ]
    )
    clean_environment["SENYAS_WINDOWS_DLL_PATH_READY"] = "1"
    completed = subprocess.run([sys.executable, *sys.argv], env=clean_environment, check=False)
    raise SystemExit(completed.returncode)


_restart_with_windows_system_dlls_first()

import mediapipe as _mediapipe  # noqa: E402,F401
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from senyas.features import LandmarkExtractor, extract_manifest, save_feature_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MediaPipe hand/pose and motion features.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "processed" / "manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "features.npz")
    parser.add_argument("--index-output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "feature_index.csv")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    with LandmarkExtractor(max_frames=args.max_frames) as extractor:
        matrix, index = extract_manifest(manifest, extractor, limit=args.limit)
        save_feature_bundle(args.output, matrix, extractor.feature_names, index)
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.index_output, index=False)
    failures = index.loc[~index["success"].astype(bool), ["video_path", "error"]]
    summary = {
        "attempted": len(index),
        "successful": int(index["success"].astype(bool).sum()),
        "failed": len(failures),
        "feature_count": int(matrix.shape[1]),
        "features": str(args.output.resolve()),
        "feature_index": str(args.index_output.resolve()),
        "failure_examples": failures.head(5).to_dict("records"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

