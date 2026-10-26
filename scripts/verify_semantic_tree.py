"""Independently reproduce a completed semantic tree's validation predictions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
    from senyas.sequences import load_sequence
    from senyas.semantic_tree import prepare_tree_input
    from senyas.model import decision_trace
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output
    metrics=json.loads((root/'metrics.json').read_text())
    artifact=joblib.load(root/'semantic_decision_tree.joblib')
    index=pd.read_csv(root/'feature_index.csv',keep_default_na=False)
    accepted=index[index.success.eq(True)].sort_values('vid_path')
    validation=accepted[accepted.split.eq('validation')]
    training=accepted[accepted.split.eq('train')]
    if set(index.split)!={'train','validation'}:raise AssertionError('Unexpected processed split')
    if set(training.signer_id)&set(validation.signer_id):raise AssertionError('Signer group leakage')
    if artifact['feature_preparation'].training_rows!=len(training):raise AssertionError('Preprocessing training sample count differs')
    if int(artifact['model'].tree_.n_node_samples[0])!=len(training):raise AssertionError('Classifier fitting sample count differs')
    vectors=[];feature_times=[]
    for row in validation.itertuples():
        sequence=load_sequence(row.cache_file)
        aspect=getattr(row,'source_frame_aspect_ratio',None)
        if aspect is not None and str(aspect)!='':sequence.metadata['source_frame_aspect_ratio']=float(aspect)
        started=time.perf_counter();vectors.append(prepare_tree_input(artifact,sequence));feature_times.append((time.perf_counter()-started)*1000)
    vectors=np.vstack(vectors)
    predictions=artifact['model'].predict(vectors).astype(int)
    recorded=pd.read_csv(root/'validation_predictions.csv').set_index('vid_path')
    if not np.array_equal(predictions,recorded.loc[validation.vid_path,'predicted_id'].to_numpy(dtype=int)):
        raise AssertionError('Saved model predictions do not reproduce the recorded results')
    truth=validation.id_label.to_numpy(dtype=int)
    measured={'accuracy':float(accuracy_score(truth,predictions)),
              'macro_precision':float(precision_score(truth,predictions,average='macro',zero_division=0)),
              'macro_recall':float(recall_score(truth,predictions,average='macro',zero_division=0)),
              'macro_f1':float(f1_score(truth,predictions,average='macro',zero_division=0))}
    for key,value in measured.items():
        if not np.isclose(value,metrics['validation_metrics'][key],atol=1e-12):raise AssertionError('Metric differs: '+key)
    matrix=pd.read_csv(root/'validation_confusion_matrix.csv',index_col=0).to_numpy()
    if matrix.sum()!=len(validation) or np.trace(matrix)!=np.count_nonzero(predictions==truth):raise AssertionError('Confusion-matrix totals differ')
    valid_paths=0
    for vector,prediction in zip(vectors,predictions):
        trace=decision_trace(artifact,vector)
        if trace['prediction']['class_id']!=int(prediction) or not all(rule['condition_holds'] for rule in trace['rules']):
            raise AssertionError('A saved tree path is inconsistent')
        valid_paths+=1
    original_hash=hashlib.sha256((ROOT/'artifacts/hybrid_recognizer.joblib').read_bytes()).hexdigest()
    if original_hash!=metrics['app_model_sha256']:raise AssertionError('Released app model changed')
    if hashlib.sha256((root/'semantic_decision_tree.joblib').read_bytes()).hexdigest()!=metrics['model_sha256']:
        raise AssertionError('Saved model hash differs from its metrics receipt')
    prototype_audit=None
    if artifact.get('augmentation')=='dtw':
        projector=artifact['dtw_projector'];training_keys=set(training.vid_path);validation_keys=set(validation.vid_path)
        if set(projector.fit_keys)!=training_keys or set(projector.fit_keys)&validation_keys:
            raise AssertionError('DTW projector fitting clips differ from training split')
        if not set(projector.prototype_keys)<=training_keys:raise AssertionError('A DTW reference is outside training')
        for group,bank in projector.excluded_banks.items():
            if any(projector.prototype_groups[i]==str(group) for i in bank):raise AssertionError('DTW training reference group leaked')
        audit=json.loads((root.parent/'cv_folds.json').read_text())
        for fold in audit:
            fold_projector=joblib.load(root.parent/f'dtw-fold-{fold["fold"]}.joblib')
            if set(fold_projector.fit_keys)!=set(fold['fit_keys']) or set(fold_projector.fit_keys)&set(fold['held_out_keys']):
                raise AssertionError('A CV DTW projector includes held-out training groups')
            for group,bank in fold_projector.excluded_banks.items():
                if any(fold_projector.prototype_groups[i]==str(group) for i in bank):raise AssertionError('A CV training reference group leaked')
        prototype_audit={'fitting_clips':len(projector.fit_keys),'prototype_pool':len(projector.prototype_keys),
                         'full_bank_references':len(projector.full_bank),'training_group_exclusion_verified':True,
                         'outer_cv_fitting_boundaries_verified':True}
    verification={'validation_predictions_reproduced':len(predictions),'valid_paths':valid_paths,'training_samples_verified':len(training),
                  'metrics':measured,'correct':int(np.count_nonzero(predictions==truth)),'original_model_unchanged':True,'test_samples_processed':0,
                  'saved_model_hash_verified':True,'feature_derivation_ms':{'mean':float(np.mean(feature_times)),'p95':float(np.percentile(feature_times,95))},
                  'prototype_audit':prototype_audit}
    (root/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
