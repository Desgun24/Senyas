"""Package existing weights and portable results without fitting any model."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from senyas.frozen_recognizer import PIPELINE_ID
from senyas.sequences import SEQUENCE_SPECS
from dataclasses import asdict


def main():
    source = ROOT / "outputs/final-test-20261027"
    protocol = json.loads((source / "frozen_protocol.json").read_text())
    identity = protocol["identity"]
    for filename, expected in identity["protected_sha256"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source or weights changed: " + filename)
    destination = ROOT / "artifacts/frozen"
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, record in identity["models"].items():
        path = destination / (name + ".joblib")
        shutil.copyfile(record["path"], path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"pipeline_id": PIPELINE_ID, "frozen_at": protocol["frozen_at"],
                "files": hashes, "extractor": {key: identity["extraction_configuration"][key]
                    for key in ("target_fps", "max_image_side", "model_complexity")},
                "training_samples": 1494, "validation_samples_accepted": 320, "test_samples": 315,
                "raw_frame_measurements": 328, "supported_frame_measurements": 314,
                "summary_candidates": 3454, "model_inputs": 3392,
                "decision_tree_role": "interpretable primary model",
                "extra_trees_role": "data-driven baseline",
                "validation_accuracy": {"decision_tree": 0.846875, "extra_trees": 0.98125},
                "test_results": json.loads((source / "test_results.json").read_text()),
                "runtime_source_sha256": {str(Path(p).relative_to(ROOT)).replace("\\", "/"): digest
                    for p, digest in identity["protected_sha256"].items()
                    if Path(p).parent == ROOT / "senyas"},
                "runtime_source_lf_sha256": {str(Path(p).relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                    for p in identity["protected_sha256"] if Path(p).parent == ROOT / "senyas"}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "feature_schema.json").write_text(json.dumps([asdict(spec) for spec in SEQUENCE_SPECS], indent=2) + "\n", encoding="utf-8")
    results = destination / "results"
    results.mkdir(exist_ok=True)
    for filename in ("test_results.json", "verification.json"):
        shutil.copyfile(source / filename, results / filename)
    for name in identity["models"]:
        (results / name).mkdir(exist_ok=True)
        for filename in ("test_predictions.csv", "test_confusion_matrix.csv", "test_per_class.csv"):
            shutil.copyfile(source / name / filename, results / name / filename)
    print("Exported frozen models unchanged:", hashes)


if __name__ == "__main__":
    main()
