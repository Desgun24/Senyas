"""Reproduce a supplied split locally without changing the app model or test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source: Path, dataset: Path, output: Path):
    import pandas as pd
    frame = pd.read_csv(source, keep_default_na=False)
    required = {"vid_path", "id_label", "label", "category", "signer_id", "split"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    if set(frame.split) != {"train", "validation", "test"} or frame.vid_path.duplicated().any():
        raise ValueError("Expected three splits and unique clip paths")
    if (frame.groupby("signer_id").split.nunique() != 1).any():
        raise ValueError("Signer IDs overlap across splits")
    labels = pd.read_csv(dataset / "labels.csv").set_index("id")
    dataset = dataset.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for record in frame.to_dict("records"):
        relative = PurePosixPath(str(record["vid_path"]).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 3 or relative.parts[0] != "clips":
            raise ValueError(f"Unexpected relative clip path: {relative}")
        class_id = int(record["id_label"])
        if int(relative.parts[1]) != class_id or int(relative.stem) != int(record["signer_id"]):
            raise ValueError(f"Class/signer mismatch: {relative}")
        if record["label"].strip() != labels.loc[class_id, "label"].strip() or record["category"].strip() != labels.loc[class_id, "category"].strip():
            raise ValueError(f"Label/category mismatch: {relative}")
        original = (dataset / relative).resolve()
        if not original.is_relative_to(dataset) or not original.is_file():
            raise ValueError(f"Missing source clip: {relative}")
        destination = output / "clips" / record["split"] / str(class_id) / relative.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(original)
        if destination.exists():
            if sha256(destination) != digest:
                raise ValueError(f"Existing partition file differs: {destination}")
        else:
            shutil.copy2(original, destination)
        if sha256(destination) != digest:
            raise ValueError(f"Copy verification failed: {destination}")
        # Drop the friend's machine-specific absolute path entirely.
        record.update(vid_path=relative.as_posix(), video_path=str(destination.resolve()),
                      source_relative_path=relative.as_posix(), sha256=digest, exists=True)
        records.append(record)
    local = pd.DataFrame(records)
    hashes = local.groupby("sha256").split.nunique()
    if (hashes > 1).any():
        raise ValueError("Identical video bytes occur across splits")
    local.to_csv(output / "manifest_local.csv", index=False)
    portable = local.drop(columns=["video_path"])
    portable["partition_path"] = [f"clips/{r.split}/{r.id_label}/{Path(r.vid_path).name}" for r in local.itertuples()]
    portable.to_csv(output / "manifest_portable.csv", index=False)
    report = {"source_manifest_sha256": sha256(source), "samples": len(local),
              "classes": int(local.id_label.nunique()), "all_copies_hash_verified": True,
              "cross_split_signer_overlap": False, "cross_split_duplicate_bytes": False,
              "splits": {name: {"samples": len(part), "classes": int(part.id_label.nunique()),
                                  "signer_ids": sorted(part.signer_id.unique().tolist())}
                         for name, part in local.groupby("split")}}
    (output / "split_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


_EXTRACTOR = None


def init_extractor():
    global _EXTRACTOR
    import mediapipe  # Main thread of each isolated worker, before Streamlit or OpenCV.
    from senyas.features import LandmarkExtractor
    _EXTRACTOR = LandmarkExtractor()
    _EXTRACTOR.cv2.setNumThreads(1)


def extract_one(record, cache_dir):
    import numpy as np
    from senyas.features import temporal_feature_names
    source_hash = hashlib.sha256((ROOT / "senyas/features.py").read_bytes()).hexdigest()
    signature = hashlib.sha256((record["sha256"] + source_hash + "mediapipe-0.10.21-frames32-side640").encode()).hexdigest()
    target = Path(cache_dir) / f"{signature}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as bundle:
            return {**record, **json.loads(str(bundle["info"])), "success": True,
                    "cache_file": str(target), "cache_hit": True, "error": ""}
    try:
        vector, info = _EXTRACTOR.extract_video(record["video_path"])
        np.savez_compressed(target, vector=vector, feature_names=np.array(temporal_feature_names()),
                            info=json.dumps(asdict(info)))
        return {**record, **asdict(info), "success": True, "cache_file": str(target), "cache_hit": False, "error": ""}
    except Exception as exc:
        return {**record, "success": False, "cache_file": "", "cache_hit": False,
                "error": f"{type(exc).__name__}: {exc}"}


def extract(output, splits, workers):
    import pandas as pd
    frame = pd.read_csv(output / "manifest_local.csv", keep_default_na=False)
    frame = frame[frame.split.isin(splits)]
    if "test" in splits:
        raise ValueError("This validation runner deliberately leaves test clips unprocessed")
    cache = output / "feature_cache"
    cache.mkdir(exist_ok=True)
    rows, started = [], time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_extractor) as pool:
        futures = [pool.submit(extract_one, row, cache) for row in frame.to_dict("records")]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if len(rows) % 25 == 0 or not row["success"] or len(rows) == len(futures):
                pd.DataFrame(rows).to_csv(output / "extraction_checkpoint.csv", index=False)
                print(f"Extracted {len(rows)}/{len(futures)}; failures {sum(not r['success'] for r in rows)}; elapsed {(time.perf_counter()-started)/60:.1f} min", flush=True)
    result = pd.DataFrame(rows).sort_values("vid_path")
    result.to_csv(output / "feature_index.csv", index=False)
    print(json.dumps({"extracted": len(result), "success": int(result.success.sum()),
                      "failed": int((~result.success).sum()), "splits": splits}), flush=True)


def evaluate(output, report_dir):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from senyas.features import temporal_feature_names
    from senyas.model import select_decision_tree, decision_trace
    from senyas.evaluation import recognition_metrics, prediction_latency_ms
    from senyas.explanations import describe_trace, TRACKING_FEATURES
    original = ROOT / 'artifacts/hybrid_recognizer.joblib'
    original_hash = sha256(original)
    index = pd.read_csv(output / 'feature_index.csv', keep_default_na=False)
    manifest = pd.read_csv(output / 'manifest_local.csv', keep_default_na=False)
    expected = manifest[manifest.split.isin(['train', 'validation'])]
    if index.vid_path.duplicated().any() or set(index.vid_path) != set(expected.vid_path):
        raise ValueError('Extraction index does not account for every training/validation sample exactly once')
    if set(index.split) != {'train', 'validation'}:
        raise ValueError('Test samples must not enter this preliminary run')
    success = index[index.success.eq(True)].sort_values('vid_path').reset_index(drop=True)
    names = temporal_feature_names()
    vectors = []
    for row in success.itertuples():
        with np.load(row.cache_file, allow_pickle=False) as bundle:
            if bundle['feature_names'].tolist() != names:
                raise ValueError('Feature schema drift')
            vectors.append(bundle['vector'])
    X = np.vstack(vectors).astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite model features')
    y = success.id_label.to_numpy(dtype=int)
    train = success.split.eq('train').to_numpy()
    validation = success.split.eq('validation').to_numpy()
    if set(y[train]) != set(range(105)) or set(y[validation]) != set(range(105)):
        raise ValueError('A split is missing a class')
    report_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, candidates = select_decision_tree(X[train], y[train], X[validation], y[validation], seed=42)
    fit_seconds = time.perf_counter() - started
    predicted = model.predict(X[validation]).astype(int)
    metrics = recognition_metrics(y[validation], predicted)
    table = success.loc[validation, ['vid_path', 'id_label', 'label', 'category', 'signer_id', 'split', 'hand_detection_rate', 'extraction_ms']].copy()
    table['predicted_id'] = predicted
    lookup = manifest.drop_duplicates('id_label').set_index('id_label').label.to_dict()
    table['predicted_label'] = table.predicted_id.map(lookup)
    table['correct'] = table.predicted_id.eq(table.id_label)
    table['leaf_vote_fraction'] = model.predict_proba(X[validation]).max(axis=1)
    full = manifest[manifest.split.eq('validation')][['vid_path', 'id_label', 'label', 'category', 'signer_id']].merge(
        table.drop(columns=['id_label', 'label', 'category', 'signer_id', 'split']), on='vid_path', how='left', validate='one_to_one')
    full = full.merge(index[['vid_path', 'success', 'error']], on='vid_path', how='left', validate='one_to_one')
    full['correct'] = full.correct.eq(True)
    full.to_csv(report_dir / 'validation_predictions.csv', index=False)
    per_class = pd.DataFrame(metrics['classification_report']).T.iloc[:105].reset_index(names='class_id')
    per_class['class_id'] = per_class.class_id.astype(int)
    per_class['label'] = per_class.class_id.map(lookup)
    per_class.to_csv(report_dir / 'validation_per_class.csv', index=False)
    pd.DataFrame(metrics['confusion_matrix'], index=range(105), columns=range(105)).to_csv(report_dir / 'validation_confusion_matrix.csv', index_label='true_class')
    pd.DataFrame([asdict(c) for c in candidates]).to_csv(report_dir / 'decision_tree_candidates.csv', index=False)
    failure_rows = index[~index.success.eq(True)].drop(columns=['video_path', 'cache_file'])
    failure_rows.to_csv(report_dir / 'extraction_failures.csv', index=False)
    artifact = {'model': model, 'explainer_model': model, 'feature_names': names,
                'label_lookup': {int(i): {'label': label} for i, label in lookup.items()},
                'confidence_threshold': None, 'metadata': {'fit_split': 'train', 'selection_split': 'validation', 'seed': 42}}
    model_path = report_dir / 'preliminary_decision_tree.joblib'
    joblib.dump(artifact, model_path, compress=3)
    traces = []
    for idx in np.where(validation)[0]:
        trace = decision_trace(artifact, X[idx])
        if not all(rule['condition_holds'] for rule in trace['rules']) or not trace['agrees_with_primary']:
            raise ValueError('Decision-path verification failed')
        traces.append(trace)
    example_index = next((i for i, ok in enumerate(predicted == y[validation]) if ok), 0)
    actual_index = np.where(validation)[0][example_index]
    example = traces[example_index]
    observations = {name: float(X[actual_index, i]) for i, name in enumerate(names) if name in TRACKING_FEATURES or any(r['feature'] == name for r in example['rules'])}
    example.update(vid_path=success.iloc[actual_index].vid_path, true_id=int(y[actual_index]),
                   descriptions=describe_trace(example, observations))
    (report_dir / 'example_decision_path.json').write_text(json.dumps(example, indent=2), encoding='utf-8')
    group_reports = {}
    for column in ['category', 'signer_id']:
        group_reports[column] = [{'group': str(name), 'attempted': len(part), 'extracted': int(part.success.eq(True).sum()),
                                  'correct': int(part.correct.sum()), 'accuracy_all_attempted': float(part.correct.mean())}
                                 for name, part in full.groupby(column)]
    all_labels = list(range(105))
    all_report = classification_report(full.id_label, full.predicted_id.fillna(-1).astype(int), labels=all_labels, output_dict=True, zero_division=0)
    report = {'status': 'preliminary_validation_model_selection', 'date': '2026-10-26',
        'model': 'DecisionTreeClassifier', 'seed': 42, 'candidate_count': len(candidates),
        'selection': f'{len(candidates)} candidates fitted on training only; ranked by validation macro F1, accuracy, then fewer nodes',
        'selected_parameters': {k: v for k, v in asdict(candidates[0]).items() if not k.startswith('validation_')},
        'class_weight': 'balanced', 'train_samples': int(train.sum()), 'training_accuracy': float(accuracy_score(y[train], model.predict(X[train]))),
        'validation_attempted': int(manifest.split.eq('validation').sum()), 'validation_extracted': int(validation.sum()),
        'validation_failed': int(((index.split == 'validation') & ~index.success.eq(True)).sum()),
        'validation_metrics_extracted': metrics, 'correct': int((predicted == y[validation]).sum()),
        'accuracy_all_attempted': float(full.correct.mean()), 'all_attempted_macro': all_report['macro avg'],
        'weighted_metrics': metrics['classification_report']['weighted avg'], 'groups': group_reports,
        'tree_nodes': int(model.tree_.node_count), 'tree_leaves': int(model.get_n_leaves()), 'tree_depth': int(model.get_depth()),
        'used_features': int(np.count_nonzero(model.feature_importances_)), 'feature_count': len(names),
        'trace_checks': len(traces), 'valid_traces': len(traces), 'candidate_fit_and_selection_seconds': fit_seconds,
        'prediction_latency_ms': prediction_latency_ms(model, X[validation]),
        'source_manifest_sha256': json.loads((output / 'split_audit.json').read_text())['source_manifest_sha256'],
        'features_source_sha256': sha256(ROOT / 'senyas/features.py'), 'training_source_sha256': sha256(ROOT / 'senyas/model.py'),
        'preliminary_model_sha256': sha256(model_path), 'app_model_sha256_before': original_hash,
        'app_model_sha256_after': sha256(original), 'test_samples_processed': 0,
        'interpretation': 'Validation informed parameter selection, so these are preliminary selection scores, not an independent final test estimate.'}
    (report_dir / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name in ['split_audit.json', 'manifest_portable.csv']:
        shutil.copy2(output / name, report_dir / name)
    print(json.dumps({k: report[k] for k in ['validation_attempted','validation_extracted','correct','accuracy_all_attempted','selected_parameters','test_samples_processed']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("extract")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--splits", nargs="+", choices=["train", "validation"], default=["train", "validation"])
    p.add_argument("--workers", type=int, default=3)
    p = sub.add_parser('evaluate')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.manifest, args.dataset, args.output)
    elif args.command == 'extract':
        extract(args.output, args.splits, args.workers)
    else:
        evaluate(args.output, args.report_dir)


if __name__ == "__main__":
    main()
