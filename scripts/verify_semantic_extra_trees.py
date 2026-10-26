"""Reproduce saved ensemble predictions, probability averaging, and individual paths."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from senyas.semantic_tree import prepare_tree_input
from senyas.sequences import load_sequence


def verify_ensemble(model,vectors,training_samples,progress=False):
    if not isinstance(model,ExtraTreesClassifier) or model.bootstrap:raise ValueError('An unbootstrapped Extra Trees model is required')
    vectors=np.asarray(vectors,dtype=np.float32);averaged=np.zeros((len(vectors),len(model.classes_)),np.float64);valid_paths=0
    for number,tree in enumerate(model.estimators_,1):
        if int(tree.tree_.n_node_samples[0])!=training_samples:raise AssertionError('A tree was fitted on unexpected sample rows')
        paths=tree.decision_path(vectors);leaves=tree.apply(vectors)
        probabilities=tree.tree_.value[leaves,0,:].copy()
        probabilities/=probabilities.sum(axis=1,keepdims=True)
        if not np.allclose(probabilities,tree.predict_proba(vectors),atol=1e-12):raise AssertionError('Leaf probabilities differ')
        averaged+=probabilities
        for row,leaf in enumerate(leaves):
            nodes=paths.indices[paths.indptr[row]:paths.indptr[row+1]]
            if nodes[0]!=0 or nodes[-1]!=leaf:raise AssertionError('A tree path has invalid endpoints')
            for source,target in zip(nodes[:-1],nodes[1:]):
                feature=tree.tree_.feature[source];threshold=tree.tree_.threshold[source]
                expected=tree.tree_.children_left[source] if vectors[row,feature]<=threshold else tree.tree_.children_right[source]
                if expected!=target:raise AssertionError('An individual ensemble path differs from its comparisons')
            valid_paths+=1
        if progress and number%50==0:print(f'Verified {number}/{len(model.estimators_)} ensemble trees',flush=True)
    averaged/=len(model.estimators_)
    if not np.allclose(averaged,model.predict_proba(vectors),atol=1e-12):raise AssertionError('Ensemble probability averaging differs')
    if not np.array_equal(model.classes_[np.argmax(averaged,axis=1)],model.predict(vectors)):
        raise AssertionError('Ensemble class prediction differs from averaged probabilities')
    return valid_paths


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args();root=args.output
    artifact=joblib.load(root/'semantic_extra_trees.joblib');model=artifact['model'];metrics=json.loads((root/'metrics.json').read_text())
    index=pd.read_csv(root/'feature_index.csv',keep_default_na=False);accepted=index[index.success.eq(True)].sort_values('vid_path')
    training=accepted[accepted.split.eq('train')];validation=accepted[accepted.split.eq('validation')]
    if set(index.split)!={'train','validation'} or set(training.signer_id)&set(validation.signer_id):raise AssertionError('Split/group leakage')
    if artifact['feature_preparation'].training_rows!=len(training):raise AssertionError('Feature preparation fitting rows differ')
    vectors=[];feature_times=[]
    for count,row in enumerate(validation.itertuples(),1):
        sequence=load_sequence(row.cache_file);started=time.perf_counter();vectors.append(prepare_tree_input(artifact,sequence));feature_times.append((time.perf_counter()-started)*1000)
        if count%100==0:print(f'Rebuilt {count}/{len(validation)} validation feature vectors',flush=True)
    vectors=np.vstack(vectors);predictions=model.predict(vectors).astype(int)
    recorded=pd.read_csv(root/'validation_predictions.csv').set_index('vid_path')
    if not np.array_equal(predictions,recorded.loc[validation.vid_path,'predicted_id'].to_numpy(dtype=int)):
        raise AssertionError('Saved model does not reproduce validation predictions')
    recorded_proba=pd.read_csv(root/'validation_probabilities.csv',index_col='vid_path').loc[validation.vid_path].to_numpy()
    if not np.allclose(recorded_proba,model.predict_proba(vectors),atol=1e-12):raise AssertionError('Recorded ensemble probabilities differ')
    truth=validation.id_label.to_numpy(dtype=int)
    measured={'accuracy':float(accuracy_score(truth,predictions)),'macro_precision':float(precision_score(truth,predictions,average='macro',zero_division=0)),
        'macro_recall':float(recall_score(truth,predictions,average='macro',zero_division=0)),
        'macro_f1':float(f1_score(truth,predictions,average='macro',zero_division=0))}
    for key,value in measured.items():
        if not np.isclose(value,metrics['validation_metrics'][key],atol=1e-12):raise AssertionError('Metric differs: '+key)
    confusion=pd.read_csv(root/'validation_confusion_matrix.csv',index_col=0).to_numpy()
    if confusion.sum()!=len(validation) or np.trace(confusion)!=np.count_nonzero(predictions==truth):raise AssertionError('Confusion matrix totals differ')
    paths=verify_ensemble(model,vectors,len(training),progress=True)
    protected=json.loads((root/'preserved_files.json').read_text())
    for path,expected in protected.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise AssertionError('Preserved file changed: '+path)
    if hashlib.sha256((root/'semantic_extra_trees.joblib').read_bytes()).hexdigest()!=metrics['model_sha256']:raise AssertionError('Saved model hash differs')
    audit=json.loads((root/'cv_folds.json').read_text())
    for fold in audit:
        if set(fold['fit_groups'])&set(fold['held_out_groups']):raise AssertionError('A CV group overlaps')
        if fold['preprocessing_training_samples']!=len(fold['fit_rows']):raise AssertionError('CV preprocessing row count differs')
    result={'validation_predictions_reproduced':len(predictions),'ensemble_trees':len(model.estimators_),
        'individual_paths_verified':paths,'ensemble_probability_averaging_verified':True,'training_samples_verified':len(training),
        'metrics':measured,'correct':int(np.count_nonzero(predictions==truth)),'protected_files_verified':len(protected),
        'test_samples_processed':0,'model_hash_verified':True,'feature_derivation_ms':{'mean':float(np.mean(feature_times)),'p95':float(np.percentile(feature_times,95))}}
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
