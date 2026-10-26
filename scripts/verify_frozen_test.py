"""Independent saved-model replay of the fixed final-test predictions and metrics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scripts.evaluate_frozen_test import assert_frozen,digest,load_models,test_metrics as calculate_metrics
from scripts.verify_semantic_extra_trees import verify_ensemble
from senyas.sequences import load_sequence
from senyas.semantic_tree import prepare_tree_input
from senyas.model import decision_trace


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output
    protocol=json.loads((root/'frozen_protocol.json').read_text());identity=protocol['identity']
    artifacts=load_models(protocol);index=pd.read_csv(root/'feature_index.csv',keep_default_na=False)
    if set(index.split)!={'test'} or len(index)!=identity['test_samples']:raise AssertionError('Unexpected processed split or test size')
    if set(index.signer_id)!=set(identity['test_groups']):raise AssertionError('Unexpected test groups')
    for row in index.itertuples():
        if digest(row.video_path)!=row.sha256:raise AssertionError('Test clip bytes changed: '+row.vid_path)
    accepted=index[index.success.eq(True)].sort_values('vid_path').reset_index(drop=True)
    vectors={name:[] for name in artifacts}
    for count,row in enumerate(accepted.itertuples(),1):
        sequence=load_sequence(row.cache_file)
        for name,artifact in artifacts.items():vectors[name].append(prepare_tree_input(artifact,sequence))
        if count%50==0 or count==len(accepted):print(f'Replayed {count}/{len(accepted)} test feature vectors',flush=True)
    vectors={name:np.vstack(rows) for name,rows in vectors.items()}
    if not np.array_equal(vectors['decision_tree'],vectors['extra_trees']):raise AssertionError('Model inputs differ')
    with np.load(root/'test_features.npz',allow_pickle=False) as cached:
        if list(cached['keys'])!=accepted.vid_path.tolist() or not np.array_equal(cached['vectors'],vectors['decision_tree']):
            raise AssertionError('Recorded test vectors differ from saved-model preprocessing')
    verified={};labels=list(range(105));truth=accepted.id_label.to_numpy(dtype=int)
    for name,artifact in artifacts.items():
        folder=root/name;metrics=json.loads((folder/'metrics.json').read_text());model=artifact['model'];X=vectors[name]
        predicted=model.predict(X).astype(int);probabilities=model.predict_proba(X)
        recorded=pd.read_csv(folder/'test_predictions.csv').set_index('vid_path')
        if not np.array_equal(predicted,recorded.loc[accepted.vid_path,'predicted_id'].to_numpy(dtype=int)):
            raise AssertionError('Saved test predictions differ: '+name)
        saved_proba=pd.read_csv(folder/'test_probabilities.csv',index_col='vid_path').loc[accepted.vid_path].to_numpy()
        if not np.allclose(saved_proba,probabilities,atol=1e-12):raise AssertionError('Saved class probabilities differ: '+name)
        conditional=calculate_metrics(truth,predicted,labels)
        # Rebuild rejection outcomes from the extraction index, not from the score receipt.
        predictions=dict(zip(accepted.vid_path,predicted))
        full_pred=np.asarray([predictions.get(path,-1) for path in index.vid_path],dtype=int)
        all_attempted=calculate_metrics(index.id_label.to_numpy(dtype=int),full_pred,labels)
        for field,measured in [('accepted_test_metrics',conditional),('all_attempted_test_metrics',all_attempted)]:
            for metric in ('accuracy','macro_precision','macro_recall','macro_f1'):
                if not np.isclose(measured[metric],metrics[field][metric],atol=1e-12):raise AssertionError('Metric differs: '+name+' '+field+' '+metric)
        matrix=pd.read_csv(folder/'all_attempted_confusion_matrix.csv',index_col=0).to_numpy()
        correct=int(np.count_nonzero(predicted==truth));rejected=len(index)-len(accepted)
        if matrix.sum()!=len(index) or np.trace(matrix[:,:105])!=correct or matrix[:,-1].sum()!=rejected:
            raise AssertionError('All-attempted confusion totals differ')
        if name=='decision_tree':
            saved_traces={trace['vid_path']:trace for trace in json.loads((folder/'test_decision_paths.json').read_text())}
            paths=0
            for row_number,(vector,prediction) in enumerate(zip(X,predicted)):
                trace=decision_trace(artifact,vector)
                if trace['prediction']['class_id']!=int(prediction) or not all(rule['condition_holds'] for rule in trace['rules']):raise AssertionError('Decision Tree test path is invalid')
                if trace['rules']!=saved_traces[accepted.iloc[row_number].vid_path]['rules']:raise AssertionError('Saved test path differs')
                paths+=1
        else:paths=verify_ensemble(model,X,identity['training_samples'],progress=True)
        verified[name]={'test_predictions_reproduced':len(predicted),'test_attempted':len(index),'test_rejected':rejected,
            'correct':correct,'paths_verified':paths,'accepted_metrics':{key:conditional[key] for key in ('accuracy','macro_precision','macro_recall','macro_f1')},
            'all_attempted_metrics':{key:all_attempted[key] for key in ('accuracy','macro_precision','macro_recall','macro_f1')}}
    assert_frozen(identity['protected_sha256'])
    result={'status':'verified','models':verified,'protected_files_verified':len(identity['protected_sha256']),
        'model_weights_unchanged':True,'preprocessing_unchanged':True,'all_test_video_hashes_verified':len(index),
        'training_or_tuning_performed':False,'test_duplicate_bytes':identity['data_audit']['test_duplicate_bytes'],
        'cross_split_duplicate_bytes':identity['data_audit']['cross_split_duplicate_bytes']}
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
