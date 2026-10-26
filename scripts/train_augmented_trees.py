"""Run separate training-group-CV ablations of three inspectable feature additions."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from senyas.augmented_features import BASE_NAMES,extra_names,transition_features,proximity_features
from senyas.dtw_features import DTWProjector,sampled_profile
from senyas.evaluation import recognition_metrics
from senyas.model import decision_trace
from senyas.semantic_tree import TreeFeaturePreparation,summarize_semantic_sequence
from senyas.sequences import SEQUENCE_NAMES,load_sequence

VARIANTS=('transitions','proximity','dtw')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    target=Path(path);temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(target)


def status(root,stage,**kwargs):
    save_json(root/'status.json',{'stage':stage,**kwargs});print(stage+' '+json.dumps(kwargs),flush=True)


def build_features(root,index):
    cache=root/'derived_features.npz';accepted=index[index.success.eq(True)].sort_values('vid_path').reset_index(drop=True)
    enriched=root/'feature_index.csv'
    if cache.exists() and enriched.exists():
        loaded=pd.read_csv(enriched,keep_default_na=False);accepted=loaded[loaded.success.eq(True)].sort_values('vid_path').reset_index(drop=True)
        with np.load(cache,allow_pickle=False) as data:
            if list(data['keys'])!=accepted.vid_path.tolist():raise ValueError('Feature cache rows differ')
            return loaded,accepted,{name:data[name] for name in ('base','transitions','proximity','profiles')}
    import cv2
    cv2.setNumThreads(1);matrices={name:[] for name in ('base','transitions','proximity','profiles')};aspects={}
    for count,row in enumerate(accepted.itertuples(),1):
        sequence=load_sequence(row.cache_file)
        video=cv2.VideoCapture(row.video_path)
        decoded,frame=video.read();video.release()
        if not decoded:raise ValueError('Missing decoded frame dimensions: '+row.vid_path)
        height,width=frame.shape[:2];scale=min(1.,480/max(width,height))
        if scale<1:frame=cv2.resize(frame,None,fx=scale,fy=scale)
        height,width=frame.shape[:2]
        sequence.metadata['source_frame_aspect_ratio']=float(width/height);aspects[row.vid_path]=float(width/height)
        # Confirm the decoded aspect agrees with the existing cached body coordinates.
        pose=sequence.landmarks['pose'][:,:,:2]*[width/height,-1.]
        centre=(pose[:,11]+pose[:,12])/2;shoulder=np.linalg.norm(pose[:,11]-pose[:,12],axis=1)
        for side in ('left','right'):
            expected_x=(sequence.landmarks[side][:,0,0]*(width/height)-centre[:,0])/np.where(shoulder>.02,shoulder,np.nan)
            observed=sequence.values[:,SEQUENCE_NAMES.index(f'{side}.wrist.body_x')]
            common=np.isfinite(expected_x)&np.isfinite(observed)
            if common.any() and not np.allclose(expected_x[common],observed[common],atol=2e-4):
                raise ValueError('Decoded geometry differs from cached extraction: '+row.vid_path)
        base=summarize_semantic_sequence(sequence,'phases')
        matrices['base'].append(base);matrices['transitions'].append(transition_features(sequence,base))
        matrices['proximity'].append(proximity_features(sequence));matrices['profiles'].append(sampled_profile(sequence))
        if count%100==0 or count==len(accepted):status(root,'derive_cached_landmarks',completed=count,total=len(accepted))
    matrices={name:np.asarray(rows,dtype=np.float32) for name,rows in matrices.items()}
    index=index.copy();index['source_frame_aspect_ratio']=index.vid_path.map(aspects)
    index.to_csv(enriched,index=False);accepted=index[index.success.eq(True)].sort_values('vid_path').reset_index(drop=True)
    np.savez_compressed(cache,**matrices,keys=accepted.vid_path.to_numpy(dtype=str))
    return index,accepted,matrices


def dtw_fold(root,fold,fit_rows,held_rows,accepted,profiles):
    path=root/f'dtw-fold-{fold}.npz';model_path=root/f'dtw-fold-{fold}.joblib'
    if path.exists() and model_path.exists():
        projector=joblib.load(model_path)
        if tuple(accepted.iloc[fit_rows].vid_path)!=projector.fit_keys:raise ValueError('DTW fold cache fitting keys differ')
        with np.load(path,allow_pickle=False) as data:return data['fit'],data['held_out'],projector
    fit=accepted.iloc[fit_rows]
    projector=DTWProjector.fit(profiles[fit_rows],fit.id_label.to_numpy(),fit.signer_id.to_numpy(),fit.vid_path.to_numpy())
    def progress(part):return lambda done,total:status(root,'dtw_training_fold',fold=fold,part=part,completed=done,total=total)
    X_fit=projector.transform(profiles[fit_rows],fit.signer_id.to_numpy(),progress('group_excluded_fit'))
    X_held=projector.transform(profiles[held_rows],progress=progress('held_out_groups'))
    np.savez_compressed(path,fit=X_fit,held_out=X_held);joblib.dump(projector,model_path,compress=3)
    return X_fit,X_held,projector


def evaluate_settings(settings,folds):
    measured=[]
    for number,(X_fit,y_fit,X_held,y_held) in enumerate(folds):
        model=DecisionTreeClassifier(random_state=42,class_weight='balanced',**settings).fit(X_fit,y_fit)
        prediction=model.predict(X_held)
        measured.append({'fold':number,'accuracy':float(accuracy_score(y_held,prediction)),
                         'macro_f1':float(f1_score(y_held,prediction,average='macro',zero_division=0)),
                         'nodes':int(model.tree_.node_count)})
    return {**settings,'cv_mean_macro_f1':float(np.mean([row['macro_f1'] for row in measured])),
            'cv_mean_accuracy':float(np.mean([row['accuracy'] for row in measured])),
            'cv_mean_nodes':float(np.mean([row['nodes'] for row in measured])),'fold_scores':measured}


def rank(row):return row['cv_mean_macro_f1'],row['cv_mean_accuracy'],-row['cv_mean_nodes']


def select(root,accepted,matrices,workers):
    train_rows=np.flatnonzero(accepted.split.eq('train'));training=accepted.iloc[train_rows]
    truth=training.id_label.to_numpy();groups=training.signer_id.to_numpy()
    folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(matrices['base'][train_rows],truth,groups))
    parameters=[{'criterion':'entropy','max_depth':depth,'min_samples_leaf':leaf,'ccp_alpha':alpha}
                for depth,leaf,alpha in itertools.product([10,14,None],[1,2],[0.,.001])]
    feature_names={'baseline':BASE_NAMES,**{name:BASE_NAMES+extra_names(name) for name in ('transitions','proximity')}}
    prepared={name:[] for name in ('baseline',)+VARIANTS};audit=[]
    for fold,(fit,held) in enumerate(folds):
        fit_rows=train_rows[fit];held_rows=train_rows[held]
        if set(groups[fit])&set(groups[held]):raise AssertionError('Training groups crossed CV boundary')
        if set(truth[fit])!=set(range(105)) or set(truth[held])!=set(range(105)):raise ValueError('CV fold lacks a word')
        for variant in ('baseline','transitions','proximity'):
            raw=matrices['base'] if variant=='baseline' else np.column_stack([matrices['base'],matrices[variant]])
            preparation=TreeFeaturePreparation.fit(raw[fit_rows],feature_names[variant])
            prepared[variant].append((preparation.transform(raw[fit_rows]),truth[fit],preparation.transform(raw[held_rows]),truth[held]))
        X_fit,X_held,projector=dtw_fold(root,fold,fit_rows,held_rows,accepted,matrices['profiles'])
        feature_names['dtw']=BASE_NAMES+projector.feature_names
        raw_fit=np.column_stack([matrices['base'][fit_rows],X_fit]);raw_held=np.column_stack([matrices['base'][held_rows],X_held])
        preparation=TreeFeaturePreparation.fit(raw_fit,feature_names['dtw'])
        prepared['dtw'].append((preparation.transform(raw_fit),truth[fit],preparation.transform(raw_held),truth[held]))
        audit.append({'fold':fold,'fit_groups':sorted(str(group) for group in set(groups[fit])),
                      'held_out_groups':sorted(str(group) for group in set(groups[held])),
                      'fit_keys':training.iloc[fit].vid_path.tolist(),'held_out_keys':training.iloc[held].vid_path.tolist(),
                      'preprocessing_training_samples':len(fit),'dtw_training_references_exclude_query_group':True})
    baseline_settings={'criterion':'entropy','max_depth':10,'min_samples_leaf':1,'ccp_alpha':.001}
    selections={'baseline':evaluate_settings(baseline_settings,prepared['baseline'])};all_results=[]
    for variant in VARIANTS:
        results=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(evaluate_settings,settings,prepared[variant]) for settings in parameters]
            for future in as_completed(futures):
                results.append(future.result());status(root,'training_group_cv',variant=variant,completed=len(results),total=len(parameters),best_macro_f1=max(row['cv_mean_macro_f1'] for row in results))
        ordered=[next(row for row in results if all(row[key]==value for key,value in setting.items())) for setting in parameters]
        selections[variant]=max(ordered,key=rank);all_results.extend({'variant':variant,**row} for row in ordered)
    save_json(root/'cv_folds.json',audit);save_json(root/'frozen_selection.json',selections)
    pd.DataFrame(all_results).to_csv(root/'cv_candidates.csv',index=False)
    return selections


def finish_variant(root,variant,accepted,index,matrices,selections,manifest,baseline,projector=None,dtw_extra=None):
    folder=root/variant;folder.mkdir(exist_ok=True)
    if (folder/'metrics.json').exists():return json.loads((folder/'metrics.json').read_text())
    train=accepted.split.eq('train').to_numpy();validation=accepted.split.eq('validation').to_numpy();truth=accepted.id_label.to_numpy(dtype=int)
    extra=dtw_extra if variant=='dtw' else matrices[variant]
    names=BASE_NAMES+(projector.feature_names if variant=='dtw' else extra_names(variant))
    raw=np.column_stack([matrices['base'],extra]);preparation=TreeFeaturePreparation.fit(raw[train],names)
    X_fit=preparation.transform(raw[train]);X_held=preparation.transform(raw[validation])
    chosen=selections[variant];settings={key:chosen[key] for key in ('criterion','max_depth','min_samples_leaf','ccp_alpha')}
    model=DecisionTreeClassifier(random_state=42,class_weight='balanced',**settings).fit(X_fit,truth[train])
    lookup={int(row.id_label):{'label':row.label,'category':row.category} for row in manifest.drop_duplicates('id_label').itertuples()}
    artifact={'model':model,'explainer_model':model,'feature_names':list(preparation.feature_names),
        'raw_sequence_feature_names':list(SEQUENCE_NAMES),'feature_preparation':preparation,'representation':'phases',
        'augmentation':variant,'schema':'semantic-augmented-tree-v1','label_lookup':lookup,'confidence_threshold':None,
        'trained_at':datetime.now(timezone.utc).isoformat(),'metadata':{'fit_split':'train','selection_split':'training_group_cv',
        'refit_on_validation':False,'selected_candidate':chosen,'training_features_exclude_query_reference_group':variant=='dtw',
        'extractor':{'target_fps':30.,'max_image_side':480,'model_complexity':1}}}
    if projector is not None and variant=='dtw':artifact['dtw_projector']=projector
    model_path=folder/'semantic_decision_tree.joblib';joblib.dump(artifact,model_path,compress=3)
    predicted=model.predict(X_held).astype(int);measured=recognition_metrics(truth[validation],predicted)
    predictions=accepted.loc[validation,['vid_path','id_label','label','category','signer_id']].copy()
    predictions['predicted_id']=predicted;predictions['correct']=predicted==truth[validation]
    full=manifest[manifest.split.eq('validation')][['vid_path','id_label','label','category','signer_id']].merge(
        predictions[['vid_path','predicted_id','correct']],on='vid_path',how='left',validate='one_to_one')
    full=full.merge(index[['vid_path','success','error']],on='vid_path',how='left',validate='one_to_one');full['correct']=full.correct.eq(True)
    full.to_csv(folder/'validation_predictions.csv',index=False);index.to_csv(folder/'feature_index.csv',index=False)
    pd.DataFrame(measured['confusion_matrix'],index=range(105),columns=range(105)).to_csv(folder/'validation_confusion_matrix.csv',index_label='true_class')
    pd.DataFrame(measured['classification_report']).T.to_csv(folder/'validation_per_class.csv',index_label='class_or_average')
    traces=[];base_set=set(BASE_NAMES);missing=~np.isfinite(raw[validation][:,preparation.enabled])
    for position,vector in enumerate(X_held):
        trace=decision_trace(artifact,vector);trace['vid_path']=predictions.iloc[position].vid_path
        if not trace['agrees_with_primary'] or not all(rule['condition_holds'] for rule in trace['rules']):raise AssertionError('Invalid tree path')
        trace['imputed_path_features']=[rule['feature'] for rule in trace['rules'] if missing[position,artifact['feature_names'].index(rule['feature'])]]
        if variant=='dtw':
            labels=sorted({int(rule['feature'].split('.')[1].split('_')[1]) for rule in trace['rules'] if rule['feature'].startswith('dtw.')})
            source_row=np.flatnonzero(validation)[position]
            trace['dtw_evidence']=[projector.explain(matrices['profiles'][source_row],label) for label in labels]
        traces.append(trace)
    save_json(folder/'validation_decision_paths.json',traces)
    paired=full.merge(pd.read_csv(baseline/'validation_predictions.csv')[['vid_path','correct']],on='vid_path',suffixes=('_new','_baseline'))
    paired=paired[paired.success.eq(True)];new=paired.correct_new.eq(True);old=paired.correct_baseline.eq(True)
    used=np.flatnonzero(model.feature_importances_)
    final={'status':'complete_preliminary_validation','augmentation':variant,'selected_candidate':chosen,'candidate_count':12,
        'train_accepted':int(train.sum()),'validation_accepted':int(validation.sum()),'validation_attempted':len(full),
        'validation_failed':int((~full.success.eq(True)).sum()),'correct':int((predicted==truth[validation]).sum()),
        'validation_metrics':measured,'accuracy_all_attempted':float(full.correct.mean()),
        'training_accuracy':float(accuracy_score(truth[train],model.predict(X_fit))),
        'raw_summary_features':len(names),'enabled_summary_features':len(preparation.feature_names),
        'nodes':int(model.tree_.node_count),'depth':int(model.get_depth()),'leaves':int(model.get_n_leaves()),
        'distinct_extra_features_used':sum(artifact['feature_names'][i] not in base_set for i in used),
        'distinct_base_features_used':sum(artifact['feature_names'][i] in base_set for i in used),
        'fixed_baseline_errors':int((new&~old).sum()),'new_errors':int((~new&old).sum()),
        'valid_decision_paths':len(traces),'model_sha256':digest(model_path),'app_model_sha256':digest(ROOT/'artifacts/hybrid_recognizer.joblib'),
        'test_samples_processed':0,'refit_on_validation':False,
        'interpretation':'Features and settings assessed with training-group CV; validation is a preliminary comparison after repeated development, not final test performance.'}
    save_json(folder/'metrics.json',final);status(root,'validation_complete',variant=variant,correct=final['correct'],accuracy=measured['accuracy'],macro_f1=measured['macro_f1'])
    return final


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();root=args.output
    if not 1<=args.workers<=4:raise ValueError('Use one to four tree-training workers')
    if root.resolve()==args.baseline.resolve():raise ValueError('Experiment output must differ from the baseline')
    root.mkdir(parents=True,exist_ok=True)
    source_paths=[ROOT/'senyas/augmented_features.py',ROOT/'senyas/dtw_features.py',ROOT/'senyas/semantic_tree.py',Path(__file__).resolve()]
    configuration={'source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in source_paths},
        'manifest_sha256':digest(args.manifest),'variants':list(VARIANTS),'tree_settings':{'criterion':'entropy','depth':[10,14,None],'leaf':[1,2],'pruning':[0.,.001]},
        'seed':42,'cv_folds':3,'dtw_samples':24,'dtw_window_fraction':.25,'dtw_prototypes_per_word':1,
        'dtw_training_bank':'exclude entire query group; outer fold scaler and references fit only within its training groups'}
    config_path=root/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:raise ValueError('Existing experiment configuration differs')
    save_json(config_path,configuration)
    sources=root/'source-snapshot';sources.mkdir(exist_ok=True)
    for path in source_paths:shutil.copy2(path,sources/path.name)
    protected={str(args.baseline/name):digest(args.baseline/name) for name in ('semantic_decision_tree.joblib','metrics.json','validation_predictions.csv','feature_index.csv')}
    protected[str(ROOT/'artifacts/hybrid_recognizer.joblib')]=digest(ROOT/'artifacts/hybrid_recognizer.joblib')
    save_json(root/'preserved_files.json',protected)
    backup=root/'baseline-backup';backup.mkdir(exist_ok=True)
    for name in ('semantic_decision_tree.joblib','metrics.json','validation_predictions.csv'):shutil.copy2(args.baseline/name,backup/name)
    extraction=json.loads((args.baseline/'extraction_configuration.json').read_text())
    if configuration['manifest_sha256']!=extraction['manifest_sha256']:raise ValueError('Manifest differs from original extraction')
    for relative,expected in extraction['source_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Cached extractor implementation differs')
    manifest=pd.read_csv(args.manifest,keep_default_na=False);index=pd.read_csv(args.baseline/'feature_index.csv',keep_default_na=False)
    expected=manifest[manifest.split.isin(['train','validation'])].set_index('vid_path').sort_index();actual=index.set_index('vid_path').sort_index()
    if set(index.split)!={'train','validation'} or index.vid_path.duplicated().any():raise ValueError('Unexpected cached clips')
    for column in ('id_label','split','signer_id','sha256'):
        if not expected[column].astype(str).equals(actual[column].astype(str)):raise ValueError('Manifest/cache mismatch: '+column)
    if (manifest.groupby('signer_id').split.nunique()>1).any():raise ValueError('Group leakage across supplied splits')
    index['cache_file']=[str(Path(value).resolve()) if value else '' for value in index.cache_file]
    index,accepted,matrices=build_features(root,index)
    selection_path=root/'frozen_selection.json'
    selections=json.loads(selection_path.read_text()) if selection_path.exists() else select(root,accepted,matrices,args.workers)
    # All variant and parameter choices are now frozen before validation scoring.
    results={}
    for variant in ('transitions','proximity'):
        results[variant]=finish_variant(root,variant,accepted,index,matrices,selections,manifest,args.baseline)
    train=np.flatnonzero(accepted.split.eq('train'));validation=np.flatnonzero(accepted.split.eq('validation'));training=accepted.iloc[train]
    projector_path=root/'dtw-final-projector.joblib';extra_path=root/'dtw-final-features.npz'
    if projector_path.exists() and extra_path.exists():
        projector=joblib.load(projector_path)
        with np.load(extra_path,allow_pickle=False) as cached:dtw_extra=cached['extra']
    else:
        projector=DTWProjector.fit(matrices['profiles'][train],training.id_label.to_numpy(),training.signer_id.to_numpy(),training.vid_path.to_numpy())
        dtw_extra=np.full((len(accepted),len(projector.feature_names)),np.nan,np.float32)
        dtw_extra[train]=projector.transform(matrices['profiles'][train],training.signer_id.to_numpy(),lambda done,total:status(root,'dtw_final_training',completed=done,total=total))
        dtw_extra[validation]=projector.transform(matrices['profiles'][validation],progress=lambda done,total:status(root,'dtw_final_validation_features',completed=done,total=total))
        joblib.dump(projector,projector_path,compress=3);np.savez_compressed(extra_path,extra=dtw_extra)
    results['dtw']=finish_variant(root,'dtw',accepted,index,matrices,selections,manifest,args.baseline,projector,dtw_extra)
    for path,expected_hash in protected.items():
        if digest(path)!=expected_hash:raise AssertionError('Preserved baseline/app file changed: '+path)
    chosen=max(selections,key=lambda variant:rank(selections[variant]))
    summary={'status':'complete','training_cv_preferred_variant':chosen,'baseline_cv':selections['baseline'],
        'variants':{variant:{key:value[key] for key in ('correct','validation_accepted','validation_metrics','distinct_extra_features_used','fixed_baseline_errors','new_errors')} for variant,value in results.items()},
        'baseline_and_app_unchanged':True,'test_samples_processed':0}
    # Compact scalar metrics only in the experiment-wide receipt.
    for value in summary['variants'].values():value['validation_metrics']={key:value['validation_metrics'][key] for key in ('accuracy','macro_precision','macro_recall','macro_f1')}
    save_json(root/'experiment_results.json',summary);status(root,'complete',results_file=str(root/'experiment_results.json'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
