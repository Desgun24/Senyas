"""Benchmark Extra Trees on exactly the preserved three-phase Decision Tree inputs."""
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import StratifiedGroupKFold
from senyas.evaluation import recognition_metrics
from senyas.semantic_tree import TreeFeaturePreparation,summary_names,summarize_semantic_sequence
from senyas.sequences import SEQUENCE_NAMES,load_sequence


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)


def make_folds(raw,truth,groups,names):
    """All imputation and feature filtering are refitted inside each training fold."""
    folds=[];audit=[]
    for number,(fit,held) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(raw,truth,groups)):
        if set(groups[fit])&set(groups[held]):raise AssertionError('A group crossed a CV boundary')
        if set(truth[fit])!=set(truth) or set(truth[held])!=set(truth):raise ValueError('A grouped fold is missing a class')
        preparation=TreeFeaturePreparation.fit(raw[fit],names)
        folds.append((preparation.transform(raw[fit]),truth[fit],preparation.transform(raw[held]),truth[held]))
        audit.append({'fold':number,'fit_rows':fit.tolist(),'held_out_rows':held.tolist(),
            'fit_groups':sorted(str(group) for group in set(groups[fit])),
            'held_out_groups':sorted(str(group) for group in set(groups[held])),
            'preprocessing_training_samples':preparation.training_rows,'enabled_features':len(preparation.feature_names)})
    return folds,audit


def evaluate_candidate(settings,folds,tree_workers,n_estimators=300):
    scores=[];started=time.perf_counter()
    for number,(X_fit,y_fit,X_held,y_held) in enumerate(folds):
        model=ExtraTreesClassifier(n_estimators=n_estimators,criterion='gini',max_depth=None,
            class_weight='balanced',random_state=42,n_jobs=tree_workers,bootstrap=False,**settings)
        model.fit(X_fit,y_fit);predicted=model.predict(X_held)
        scores.append({'fold':number,'accuracy':float(accuracy_score(y_held,predicted)),
            'macro_f1':float(f1_score(y_held,predicted,average='macro',zero_division=0)),
            'total_nodes':sum(int(tree.tree_.node_count) for tree in model.estimators_)})
        print(f'Extra Trees max_features={settings["max_features"]}, leaf={settings["min_samples_leaf"]}: fold {number+1}/3 macro F1 {scores[-1]["macro_f1"]:.4f}',flush=True)
    return {**settings,'n_estimators':n_estimators,'cv_mean_accuracy':float(np.mean([row['accuracy'] for row in scores])),
        'cv_mean_macro_f1':float(np.mean([row['macro_f1'] for row in scores])),
        'cv_std_macro_f1':float(np.std([row['macro_f1'] for row in scores])),
        'cv_mean_total_nodes':float(np.mean([row['total_nodes'] for row in scores])),
        'elapsed_seconds':time.perf_counter()-started,'fold_scores':scores}


def candidate_rank(row):return row['cv_mean_macro_f1'],row['cv_mean_accuracy'],-row['cv_mean_total_nodes']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--summary-cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate-workers',type=int,default=2);parser.add_argument('--tree-workers',type=int,default=3)
    args=parser.parse_args();root=args.output
    if args.candidate_workers<1 or args.tree_workers<1 or args.candidate_workers*args.tree_workers>6:
        raise ValueError('Use at most six combined candidate/tree workers')
    if root.resolve()==args.baseline.resolve():raise ValueError('Output must differ from the baseline')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'semantic_extra_trees.joblib').exists():raise ValueError('Refusing to overwrite an existing model')
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000)
    settings=[{'max_features':features,'min_samples_leaf':leaf} for features,leaf in itertools.product(['sqrt',.1,.2,.35],[1,2])]
    configuration={'n_estimators':300,'criterion':'gini','max_depth':None,'class_weight':'balanced','bootstrap':False,
        'seed':42,'cv_folds':3,'settings':settings,'representation':'phases','summary_feature_policy':'all',
        'manifest_sha256':digest(args.manifest),'summary_cache_sha256':digest(args.summary_cache),
        'source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in [Path(__file__).resolve(),ROOT/'senyas/semantic_tree.py']}}
    config_path=root/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:raise ValueError('Existing experiment configuration differs')
    save_json(config_path,configuration)
    snapshot=root/'source-snapshot';snapshot.mkdir(exist_ok=True)
    for relative in configuration['source_sha256']:shutil.copy2(ROOT/relative,snapshot/Path(relative).name)
    protected={str(path):digest(path) for path in [ROOT/'artifacts/hybrid_recognizer.joblib',
        args.baseline/'semantic_decision_tree.joblib',args.baseline/'metrics.json',args.baseline/'feature_index.csv',args.baseline/'validation_predictions.csv']}
    ablation=ROOT/'outputs/semantic-tree-tracking-ablation-20261027/ablation_protocol.json'
    if ablation.exists():
        prior=json.loads(ablation.read_text())['preserved_models']
        for path in prior:protected[path]=digest(path)
        for policy in ('no_observation_fractions','no_tracking_flags','geometry_motion_only'):
            path=ablation.parent/policy/'semantic_decision_tree.joblib'
            if path.exists():protected[str(path)]=digest(path)
    save_json(root/'preserved_files.json',protected)
    backup=root/'baseline-backup';backup.mkdir(exist_ok=True)
    for name in ('semantic_decision_tree.joblib','metrics.json','validation_predictions.csv'):shutil.copy2(args.baseline/name,backup/name)
    extraction=json.loads((args.baseline/'extraction_configuration.json').read_text())
    if configuration['manifest_sha256']!=extraction['manifest_sha256']:raise ValueError('Manifest differs from cached extraction')
    for relative,expected_hash in extraction['source_sha256'].items():
        if digest(ROOT/relative)!=expected_hash:raise ValueError('Cached extractor source differs')
    manifest=pd.read_csv(args.manifest,keep_default_na=False);index=pd.read_csv(args.baseline/'feature_index.csv',keep_default_na=False)
    source=manifest[manifest.split.isin(['train','validation'])].set_index('vid_path').sort_index()
    actual=index.set_index('vid_path').sort_index()
    if index.vid_path.duplicated().any() or set(index.split)!={'train','validation'}:raise ValueError('Unexpected cached split')
    for column in ('id_label','split','signer_id','sha256'):
        if not source[column].astype(str).equals(actual[column].astype(str)):raise ValueError('Cached metadata differs: '+column)
    if (manifest.groupby('signer_id').split.nunique()>1).any():raise ValueError('Groups cross manifest splits')
    index['cache_file']=[str(Path(value).resolve()) if value else '' for value in index.cache_file]
    index.to_csv(root/'feature_index.csv',index=False)
    accepted=index[index.success.eq(True)].sort_values('vid_path').reset_index(drop=True)
    with np.load(args.summary_cache,allow_pickle=False) as data:
        if list(data['keys'])!=accepted.vid_path.tolist():raise ValueError('Summary cache rows differ')
        raw=data['base'].astype(np.float32)
    names=summary_names('phases')
    if raw.shape!=(len(accepted),len(names)):raise ValueError('Summary cache feature width differs')
    for row_number in np.linspace(0,len(accepted)-1,12,dtype=int):
        expected=summarize_semantic_sequence(load_sequence(accepted.iloc[row_number].cache_file),'phases')
        if not np.allclose(raw[row_number],expected,atol=1e-7,rtol=0,equal_nan=True):raise ValueError('Cached summary does not reproduce')
    train=accepted.split.eq('train').to_numpy();validation=accepted.split.eq('validation').to_numpy();truth=accepted.id_label.to_numpy(dtype=int)
    if set(truth[train])!=set(range(105)) or set(truth[validation])!=set(range(105)):raise ValueError('Missing word class')
    baseline=joblib.load(args.baseline/'semantic_decision_tree.joblib')
    preparation=TreeFeaturePreparation.fit(raw[train],names)
    reference=baseline['feature_preparation']
    if preparation.raw_feature_names!=reference.raw_feature_names or not np.array_equal(preparation.enabled,reference.enabled) or not np.array_equal(preparation.medians,reference.medians):
        raise AssertionError('Final feature preparation differs from the preserved Decision Tree')
    X_fit=preparation.transform(raw[train]);X_validation=preparation.transform(raw[validation])
    baseline_prediction=baseline['model'].predict(X_validation).astype(int)
    previous=pd.read_csv(args.baseline/'validation_predictions.csv').set_index('vid_path')
    if not np.array_equal(baseline_prediction,previous.loc[accepted.loc[validation,'vid_path'],'predicted_id'].to_numpy(dtype=int)):
        raise AssertionError('Identical inputs did not reproduce the baseline predictions')
    print(f'Verified identical {X_fit.shape[1]} inputs; training {int(train.sum())}, validation {int(validation.sum())}; baseline correct {int((baseline_prediction==truth[validation]).sum())}',flush=True)
    training=accepted.loc[train];folds,audit=make_folds(raw[train],truth[train],training.signer_id.to_numpy(),names)
    save_json(root/'cv_folds.json',audit)
    checkpoint=root/'cv_candidate_checkpoint.json';results=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    missing=[setting for setting in settings if not any(all(row[key]==value for key,value in setting.items()) for row in results)]
    started=time.perf_counter()
    save_json(root/'status.json',{'stage':'training_group_cv','completed':len(results),'total':len(settings)})
    with ThreadPoolExecutor(max_workers=args.candidate_workers) as pool:
        futures=[pool.submit(evaluate_candidate,setting,folds,args.tree_workers) for setting in missing]
        for future in as_completed(futures):
            results.append(future.result());save_json(checkpoint,results)
            best=max(results,key=candidate_rank)
            save_json(root/'status.json',{'stage':'training_group_cv','completed':len(results),'total':len(settings),'best_cv_macro_f1':best['cv_mean_macro_f1']})
            print(f'Completed {len(results)}/{len(settings)} ensemble settings; best training CV macro F1 {best["cv_mean_macro_f1"]:.4f}',flush=True)
    ordered=[next(row for row in results if all(row[key]==value for key,value in setting.items())) for setting in settings]
    chosen=max(ordered,key=candidate_rank);save_json(root/'frozen_selection.json',chosen)
    pd.DataFrame(ordered).to_csv(root/'candidate_metrics.csv',index=False)
    save_json(root/'status.json',{'stage':'final_training','trees':300,'training_samples':int(train.sum())})
    model=ExtraTreesClassifier(n_estimators=300,criterion='gini',max_depth=None,class_weight='balanced',random_state=42,
        n_jobs=args.tree_workers,bootstrap=False,max_features=chosen['max_features'],min_samples_leaf=chosen['min_samples_leaf']).fit(X_fit,truth[train])
    artifact={'model':model,'feature_names':list(preparation.feature_names),'raw_sequence_feature_names':list(SEQUENCE_NAMES),
        'feature_preparation':preparation,'representation':'phases','summary_feature_policy':'all','schema':'semantic-extra-trees-v1',
        'label_lookup':baseline['label_lookup'],'confidence_threshold':None,'trained_at':datetime.now(timezone.utc).isoformat(),
        'metadata':{'fit_split':'train','selection_split':'training_group_cv','refit_on_validation':False,'selected_candidate':chosen,
        'baseline_model_sha256':digest(args.baseline/'semantic_decision_tree.joblib'),'seed':42,'handshape_labels_used':False,
        'extractor':{'target_fps':30.,'max_image_side':480,'model_complexity':1},
        'explanation':'Extra Trees ensemble on the same engineered inputs; no single Decision Tree path represents its full decision.'}}
    model_path=root/'semantic_extra_trees.joblib';joblib.dump(artifact,model_path,compress=3)
    predicted=model.predict(X_validation).astype(int);probabilities=model.predict_proba(X_validation);metrics=recognition_metrics(truth[validation],predicted)
    rows=accepted.loc[validation,['vid_path','id_label','label','category','signer_id']].copy()
    rows['predicted_id']=predicted;rows['correct']=predicted==truth[validation];rows['ensemble_probability']=probabilities.max(axis=1)
    full=manifest[manifest.split.eq('validation')][['vid_path','id_label','label','category','signer_id']].merge(
        rows[['vid_path','predicted_id','correct','ensemble_probability']],on='vid_path',how='left',validate='one_to_one')
    full=full.merge(index[['vid_path','success','error']],on='vid_path',how='left',validate='one_to_one');full['correct']=full.correct.eq(True)
    full.to_csv(root/'validation_predictions.csv',index=False)
    pd.DataFrame(probabilities,index=rows.vid_path,columns=model.classes_.astype(int)).to_csv(root/'validation_probabilities.csv',index_label='vid_path')
    pd.DataFrame(metrics['confusion_matrix'],index=range(105),columns=range(105)).to_csv(root/'validation_confusion_matrix.csv',index_label='true_class')
    pd.DataFrame(metrics['classification_report']).T.to_csv(root/'validation_per_class.csv',index_label='class_or_average')
    pd.DataFrame({'feature':artifact['feature_names'],'impurity_importance':model.feature_importances_}).sort_values('impurity_importance',ascending=False).to_csv(root/'feature_importance.csv',index=False)
    old_correct=baseline_prediction==truth[validation];new_correct=predicted==truth[validation]
    result={'status':'complete_preliminary_validation','model':'ExtraTreesClassifier','n_estimators':300,
        'train_accepted':int(train.sum()),'validation_attempted':len(full),'validation_accepted':int(validation.sum()),'validation_failed':int((~full.success.eq(True)).sum()),
        'correct':int(new_correct.sum()),'accuracy_all_attempted':float(full.correct.mean()),'validation_metrics':metrics,
        'training_accuracy':float(accuracy_score(truth[train],model.predict(X_fit))),'selected_candidate':chosen,'candidate_count':len(results),
        'training_cv_ensemble_fits':len(results)*3,'input_features':len(artifact['feature_names']),'raw_summary_features':len(names),
        'same_feature_preparation_as_baseline':True,'baseline_accuracy':float(old_correct.mean()),
        'accuracy_difference_percentage_points':float((new_correct.mean()-old_correct.mean())*100),
        'fixed_baseline_errors':int((new_correct&~old_correct).sum()),'new_errors':int((~new_correct&old_correct).sum()),
        'model_sha256':digest(model_path),'app_model_sha256':digest(ROOT/'artifacts/hybrid_recognizer.joblib'),
        'training_seconds':time.perf_counter()-started,'test_samples_processed':0,'refit_on_validation':False,
        'interpretation':'Extra Trees settings selected within grouped training folds; validation is preliminary and has informed earlier development, not a final test score.'}
    save_json(root/'metrics.json',result)
    for path,expected_hash in protected.items():
        if digest(path)!=expected_hash:raise AssertionError('A preserved model or baseline file changed: '+path)
    save_json(root/'preservation_verified.json',{'protected_files_verified':len(protected),'app_model_unchanged':True,'test_samples_processed':0})
    save_json(root/'status.json',{'stage':'complete','metrics_file':str(root/'metrics.json')})
    print(json.dumps({key:result[key] for key in ('correct','validation_accepted','validation_failed','accuracy_all_attempted','baseline_accuracy','accuracy_difference_percentage_points','selected_candidate')},indent=2),flush=True)


if __name__=='__main__':main()
