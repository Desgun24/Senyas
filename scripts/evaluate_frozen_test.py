"""One fixed held-out test evaluation: no training, tuning, or model replacement."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(key,'1')
os.environ.setdefault('GLOG_minloglevel','2');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save_json(path,value):
    target=Path(path);temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(target)


def assert_frozen(hashes):
    for path,expected in hashes.items():
        if digest(path)!=expected:raise AssertionError('Frozen input changed: '+str(path))


def freeze_protocol(path,identity):
    path=Path(path)
    if path.exists():
        existing=json.loads(path.read_text())
        if existing['identity']!=identity:raise ValueError('Refusing to change a frozen test protocol')
        return existing
    value={'frozen_at':datetime.now(timezone.utc).isoformat(),'identity':identity}
    save_json(path,value);return value


def audit_manifest(manifest):
    if manifest.vid_path.duplicated().any():raise ValueError('Duplicate manifest clip paths')
    if (manifest.groupby('sha256').split.nunique()>1).any():raise ValueError('Duplicate video bytes cross dataset splits')
    if (manifest.groupby('signer_id').split.nunique()>1).any():raise ValueError('Manifest group leakage')
    duplicate=manifest[manifest.sha256.duplicated(keep=False)]
    return {'cross_split_duplicate_bytes':False,'cross_split_group_overlap':False,
        'test_duplicate_bytes':bool(manifest[manifest.split.eq('test')].sha256.duplicated().any()),
        'within_split_duplicate_byte_groups':[{'sha256':str(key),'conflicting_labels':part.id_label.nunique()>1,
            'clips':part[['vid_path','id_label','label','split','signer_id']].to_dict('records')}
            for key,part in duplicate.groupby('sha256')]}


def test_metrics(truth,predicted,labels):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
    return {'samples':len(truth),'accuracy':float(accuracy_score(truth,predicted)),
        'macro_precision':float(precision_score(truth,predicted,labels=labels,average='macro',zero_division=0)),
        'macro_recall':float(recall_score(truth,predicted,labels=labels,average='macro',zero_division=0)),
        'macro_f1':float(f1_score(truth,predicted,labels=labels,average='macro',zero_division=0)),
        'labels':labels,'confusion_matrix':confusion_matrix(truth,predicted,labels=labels).astype(int).tolist(),
        'classification_report':classification_report(truth,predicted,labels=labels,output_dict=True,zero_division=0)}


def status(root,stage,**values):
    save_json(root/'status.json',{'stage':stage,**values});print(stage+' '+json.dumps(values),flush=True)


def load_models(protocol):
    import joblib
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    identity=protocol['identity'];assert_frozen(identity['protected_sha256'])
    artifacts={name:joblib.load(record['path']) for name,record in identity['models'].items()}
    if not isinstance(artifacts['decision_tree']['model'],DecisionTreeClassifier):raise ValueError('Wrong Decision Tree artifact')
    if not isinstance(artifacts['extra_trees']['model'],ExtraTreesClassifier):raise ValueError('Wrong Extra Trees artifact')
    first=artifacts['decision_tree'];second=artifacts['extra_trees']
    for artifact in artifacts.values():
        if artifact['representation']!='phases' or artifact.get('summary_feature_policy','all')!='all':raise ValueError('Expected the original full three-phase model')
        if artifact['metadata']['fit_split']!='train' or artifact['metadata']['refit_on_validation']:raise ValueError('Model was refitted beyond training')
        if artifact['feature_preparation'].training_rows!=identity['training_samples']:raise ValueError('Preprocessing fitting row count differs')
        expected_extractor={key:identity['extraction_configuration'][key] for key in ('target_fps','max_image_side','model_complexity')}
        if artifact['metadata']['extractor']!=expected_extractor:raise ValueError('Extraction settings differ from training')
        if artifact['model'].classes_.astype(int).tolist()!=list(range(105)):raise ValueError('Classifier vocabulary differs')
    if first['feature_names']!=second['feature_names'] or len(first['feature_names'])!=3392:raise ValueError('Model input features differ')
    a=first['feature_preparation'];b=second['feature_preparation']
    if a.raw_feature_names!=b.raw_feature_names or not np.array_equal(a.enabled,b.enabled) or not np.array_equal(a.medians,b.medians):
        raise ValueError('Frozen preprocessors differ')
    expected=identity['training_samples']
    if int(first['model'].tree_.n_node_samples[0])!=expected:raise ValueError('Decision Tree fitting sample count differs')
    if second['model'].bootstrap or any(int(tree.tree_.n_node_samples[0])!=expected for tree in second['model'].estimators_):
        raise ValueError('Extra Trees fitting sample count differs')
    return artifacts


def extract_test(root,manifest,configuration,workers):
    import pandas as pd
    from scripts.train_semantic_tree import initialize_worker,extract_clip
    source=manifest[manifest.split.eq('test')].sort_values('vid_path')
    cache=root/'sequence_cache';cache.mkdir(exist_ok=True);rows=[];started=time.perf_counter()
    signature=configuration['configuration_hash']
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=(configuration,)) as pool:
        futures=[pool.submit(extract_clip,record,str(cache),signature) for record in source.to_dict('records')]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows)%10==0 or not rows[-1]['success'] or len(rows)==len(source):
                elapsed=time.perf_counter()-started;failed=sum(not row['success'] for row in rows)
                pd.DataFrame(rows).to_csv(root/'extraction_checkpoint.csv',index=False)
                status(root,'test_landmark_extraction',completed=len(rows),total=len(source),failed=failed,elapsed_seconds=round(elapsed,1))
    index=pd.DataFrame(rows).sort_values('vid_path').reset_index(drop=True)
    index.to_csv(root/'feature_index.csv',index=False);return index


def evaluate(root,protocol,index,artifacts):
    import numpy as np
    import pandas as pd
    from senyas.sequences import load_sequence
    from senyas.semantic_tree import prepare_tree_input,summarize_semantic_sequence
    from senyas.model import decision_trace
    accepted=index[index.success.eq(True)].sort_values('vid_path').reset_index(drop=True)
    if not len(accepted):raise ValueError('No test clip passed the frozen extraction rules')
    vectors=[];missing=[];first=artifacts['decision_tree'];preparation=first['feature_preparation']
    for count,row in enumerate(accepted.itertuples(),1):
        sequence=load_sequence(row.cache_file)
        first_vector=prepare_tree_input(first,sequence);second_vector=prepare_tree_input(artifacts['extra_trees'],sequence)
        if not np.array_equal(first_vector,second_vector):raise AssertionError('Model input vectors differ')
        vectors.append(first_vector)
        raw=summarize_semantic_sequence(sequence,'phases');missing.append(~np.isfinite(raw[preparation.enabled]))
        if count%50==0 or count==len(accepted):status(root,'test_feature_preparation',completed=count,total=len(accepted))
    vectors=np.vstack(vectors);missing=np.asarray(missing)
    np.savez_compressed(root/'test_features.npz',vectors=vectors,keys=accepted.vid_path.to_numpy(dtype=str),feature_names=np.asarray(first['feature_names']))
    labels=list(range(105));summary={};truth=accepted.id_label.to_numpy(dtype=int)
    for name,artifact in artifacts.items():
        folder=root/name;folder.mkdir(exist_ok=True);model=artifact['model'];predicted=model.predict(vectors).astype(int)
        probabilities=model.predict_proba(vectors)
        conditional=test_metrics(truth,predicted,labels)
        rows=accepted[['vid_path','id_label','label','category','signer_id']].copy()
        rows['predicted_id']=predicted;rows['correct']=predicted==truth;rows['model_probability']=probabilities.max(axis=1)
        full=index[['vid_path','id_label','label','category','signer_id','success','error']].merge(
            rows[['vid_path','predicted_id','correct','model_probability']],on='vid_path',how='left',validate='one_to_one')
        full['correct']=full.correct.eq(True)
        all_predicted=full.predicted_id.fillna(-1).to_numpy(dtype=int)
        end_to_end=test_metrics(full.id_label.to_numpy(dtype=int),all_predicted,labels)
        full.to_csv(folder/'test_predictions.csv',index=False)
        pd.DataFrame(probabilities,index=rows.vid_path,columns=labels).to_csv(folder/'test_probabilities.csv',index_label='vid_path')
        pd.DataFrame(conditional['confusion_matrix'],index=labels,columns=labels).to_csv(folder/'test_confusion_matrix.csv',index_label='true_class')
        # An additional rejection column accounts for every attempted test clip.
        rejection_matrix=np.zeros((105,106),dtype=int)
        for target,prediction in zip(full.id_label.to_numpy(dtype=int),all_predicted):rejection_matrix[target,prediction if prediction>=0 else 105]+=1
        end_to_end['confusion_matrix']=rejection_matrix.tolist();end_to_end['prediction_labels']=labels+['REJECTED']
        pd.DataFrame(rejection_matrix,index=labels,columns=labels+['REJECTED']).to_csv(folder/'all_attempted_confusion_matrix.csv',index_label='true_class')
        pd.DataFrame(end_to_end['classification_report']).T.to_csv(folder/'test_per_class.csv',index_label='class_or_average')
        groups={column:[{'group':str(group),'attempted':len(part),'accepted':int(part.success.eq(True).sum()),'correct':int(part.correct.sum()),'accuracy_all_attempted':float(part.correct.mean())}
            for group,part in full.groupby(column)] for column in ('category','signer_id')}
        result={'status':'final_frozen_test','role':protocol['identity']['models'][name]['role'],
            'model_sha256':protocol['identity']['models'][name]['sha256'],'training_samples':protocol['identity']['training_samples'],
            'test_attempted':len(full),'test_accepted':len(accepted),'test_rejected':int((~full.success.eq(True)).sum()),
            'correct':int(full.correct.sum()),'accepted_test_metrics':conditional,'all_attempted_test_metrics':end_to_end,
            'input_features':vectors.shape[1],'models_refitted':False,'hyperparameters_changed':False,'groups':groups}
        if name=='decision_tree':
            traces=[]
            for position,vector in enumerate(vectors):
                trace=decision_trace(artifact,vector);trace['vid_path']=accepted.iloc[position].vid_path
                if trace['prediction']['class_id']!=int(predicted[position]) or not all(rule['condition_holds'] for rule in trace['rules']):raise AssertionError('Decision Tree test path differs')
                trace['imputed_path_features']=[rule['feature'] for rule in trace['rules'] if missing[position,artifact['feature_names'].index(rule['feature'])]]
                traces.append(trace)
            save_json(folder/'test_decision_paths.json',traces)
        save_json(folder/'metrics.json',result)
        summary[name]={key:result[key] for key in ('role','test_attempted','test_accepted','test_rejected','correct')}
        summary[name]['accepted_metrics']={key:conditional[key] for key in ('accuracy','macro_precision','macro_recall','macro_f1')}
        summary[name]['all_attempted_metrics']={key:end_to_end[key] for key in ('accuracy','macro_precision','macro_recall','macro_f1')}
        print(name+' '+json.dumps(summary[name]),flush=True)
    assert_frozen(protocol['identity']['protected_sha256'])
    receipt={'status':'complete','frozen_at':protocol['frozen_at'],'test_split':'supplied manifest test only',
        'models':summary,'model_weights_unchanged':True,'preprocessing_unchanged':True,'no_fitting_or_tuning':True}
    save_json(root/'test_results.json',receipt);status(root,'complete',results_file=str(root/'test_results.json'))
    return receipt


def main():
    import pandas as pd
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision-tree',type=Path,required=True);parser.add_argument('--extra-trees',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--extraction-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();root=args.output
    if not 1<=args.workers<=8:raise ValueError('Use one to eight extraction workers')
    root.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(args.manifest,keep_default_na=False)
    data_audit=audit_manifest(manifest)
    test=manifest[manifest.split.eq('test')]
    if len(test)!=315 or set(test.id_label)!=set(range(105)):raise ValueError('Test split differs from the frozen 315-clip vocabulary')
    if set(test.signer_id)!={1,5,10}:raise ValueError('Unexpected held-out test groups')
    configuration=json.loads(args.extraction_config.read_text())
    from importlib.metadata import version
    if version('mediapipe')!=configuration['mediapipe']:raise ValueError('MediaPipe version differs from training extraction')
    if digest(args.manifest)!=configuration['manifest_sha256']:raise ValueError('Manifest differs from model development')
    assert_frozen({str(ROOT/path):value for path,value in configuration['source_sha256'].items()})
    models={
        'decision_tree':{'path':str(args.decision_tree.resolve()),'sha256':digest(args.decision_tree),'role':'interpretable primary model'},
        'extra_trees':{'path':str(args.extra_trees.resolve()),'sha256':digest(args.extra_trees),'role':'data-driven baseline'},
    }
    for name,record in models.items():
        source_metrics=json.loads((Path(record['path']).parent/'metrics.json').read_text())
        if source_metrics['model_sha256']!=record['sha256']:raise ValueError('Saved model differs from its validation receipt')
        if source_metrics['test_samples_processed']!=0 or source_metrics['refit_on_validation']:raise ValueError('A selected model is not the untouched-test training artifact')
    sources=[ROOT/'senyas/semantic_features.py',ROOT/'senyas/sequences.py',ROOT/'senyas/semantic_tree.py',
        ROOT/'senyas/model.py',ROOT/'scripts/train_semantic_tree.py',Path(__file__).resolve()]
    protected={str(path.resolve()):digest(path) for path in sources+[args.manifest,args.extraction_config,ROOT/'artifacts/hybrid_recognizer.joblib']}
    protected.update({record['path']:record['sha256'] for record in models.values()})
    previous=Path(models['extra_trees']['path']).parent/'preserved_files.json'
    if previous.exists():
        for path,value in json.loads(previous.read_text()).items():
            if digest(path)!=value:raise ValueError('An earlier preserved model/file changed: '+path)
            protected[str(Path(path).resolve())]=value
    identity={'models':models,'training_samples':int(manifest.split.eq('train').sum()),'test_samples':315,'test_groups':[1,5,10],
        'feature_count':3392,'representation':'three phases','extraction_configuration':configuration,
        'minimum_hand_frames':3,'maximum_clip_seconds':30,'test_failure_handling':'report rejected clips and count them as incorrect in all-attempted metrics',
        'data_audit':data_audit,'no_model_changes_after_test':True,'protected_sha256':protected}
    protocol=freeze_protocol(root/'frozen_protocol.json',identity);artifacts=load_models(protocol)
    snapshot=root/'frozen-snapshots';snapshot.mkdir(exist_ok=True)
    for name,record in models.items():
        target=snapshot/(name+'.joblib')
        if not target.exists():shutil.copy2(record['path'],target)
        if digest(target)!=record['sha256']:raise ValueError('Frozen backup model differs')
    for path in sources:
        target=snapshot/path.name
        if not target.exists():shutil.copy2(path,target)
        if digest(target)!=digest(path):raise ValueError('Frozen source backup differs')
    save_json(root/'extraction_configuration.json',configuration)
    if (root/'test_results.json').exists():
        print((root/'test_results.json').read_text());return
    index=extract_test(root,manifest,configuration,args.workers)
    assert_frozen(protocol['identity']['protected_sha256']);evaluate(root,protocol,index,artifacts)


if __name__=='__main__':main()
