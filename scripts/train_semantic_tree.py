"""Train on the supplied training split and evaluate semantic trees on validation only."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('GLOG_minloglevel','2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save_json(path,value):
    target=Path(path);temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(target)


EXTRACTOR=None
CONFIG=None


def initialize_worker(config):
    global EXTRACTOR,CONFIG
    import mediapipe  # Main thread in each isolated worker.
    from senyas.sequences import SequenceExtractor
    CONFIG=config
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000)  # Below normal; keep the desktop responsive.
    EXTRACTOR=SequenceExtractor(target_fps=config['target_fps'],max_image_side=config['max_image_side'],model_complexity=config['model_complexity'])
    EXTRACTOR.cv2.setNumThreads(1)


def extract_clip(record,cache,configuration_hash):
    from senyas.sequences import save_sequence,load_sequence
    key=hashlib.sha256((record['sha256']+configuration_hash).encode()).hexdigest()
    sequence_path=Path(cache)/(key+'.npz');failure_path=Path(cache)/(key+'.failure.json')
    if digest(record['video_path'])!=record['sha256']:
        raise ValueError('Dataset bytes changed: '+record['vid_path'])
    if sequence_path.exists():
        sequence=load_sequence(sequence_path)
        return {**record,'success':True,'error':'','cache_file':str(sequence_path),'cache_hit':True,**sequence.metadata}
    if failure_path.exists():
        return {**record,**json.loads(failure_path.read_text()),'cache_hit':True}
    try:
        capture=EXTRACTOR.cv2.VideoCapture(record['video_path'])
        fps=float(capture.get(EXTRACTOR.cv2.CAP_PROP_FPS));frames=int(capture.get(EXTRACTOR.cv2.CAP_PROP_FRAME_COUNT));capture.release()
        if fps<=0 or frames<=0:raise ValueError('Clip has invalid duration metadata')
        if frames/fps>30:raise ValueError('Clip exceeds the thirty-second extraction budget')
        sequence=EXTRACTOR.extract_video(record['video_path'],max_seconds=frames/fps+.01)
        if sequence.metadata['hand_frames']<3:
            raise ValueError(f"Only {sequence.metadata['hand_frames']} frames contained a detected hand; at least three required")
        temporary=sequence_path.with_suffix('.pending.npz')
        save_sequence(temporary,sequence)
        temporary.replace(sequence_path)
        return {**record,'success':True,'error':'','cache_file':str(sequence_path),'cache_hit':False,**sequence.metadata}
    except Exception as exc:
        failure={'success':False,'error':f'{type(exc).__name__}: {exc}','cache_file':''}
        save_json(failure_path,failure)
        return {**record,**failure,'cache_hit':False}


def extract(args):
    import pandas as pd
    from senyas.sequences import SEQUENCE_NAMES
    manifest=pd.read_csv(args.manifest,keep_default_na=False)
    required={'vid_path','video_path','id_label','label','category','split','signer_id','sha256'}
    if not required.issubset(manifest):raise ValueError('Manifest is missing required fields')
    if manifest.vid_path.duplicated().any():raise ValueError('Duplicate paths in manifest')
    if (manifest.groupby('signer_id').split.nunique()>1).any():raise ValueError('Signer groups cross split boundaries')
    source=manifest[manifest.split.isin(['train','validation'])].copy()
    if set(source.split)!={'train','validation'}:raise ValueError('Both training and validation data are required')
    config={'target_fps':30.,'max_image_side':480,'model_complexity':1,
            'features':len(SEQUENCE_NAMES),'feature_names':list(SEQUENCE_NAMES),
            'source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in [ROOT/'senyas/semantic_features.py',ROOT/'senyas/sequences.py']},
            'manifest_sha256':digest(args.manifest),'mediapipe':'0.10.21'}
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache=args.output/'sequence_cache';cache.mkdir(parents=True,exist_ok=True)
    save_json(args.output/'extraction_configuration.json',{**config,'configuration_hash':signature})
    rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker,initargs=(config,)) as pool:
        futures=[pool.submit(extract_clip,record,cache,signature) for record in source.to_dict('records')]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            if len(rows)%25==0 or not row['success'] or len(rows)==len(source):
                elapsed=time.perf_counter()-started
                status={'stage':'extraction','completed':len(rows),'total':len(source),'failed':sum(not item['success'] for item in rows),
                        'elapsed_seconds':elapsed,'eta_seconds':elapsed/max(len(rows),1)*(len(source)-len(rows)),
                        'by_split':{split:{'completed':sum(item['split']==split for item in rows),'failed':sum(item['split']==split and not item['success'] for item in rows)} for split in ['train','validation']}}
                save_json(args.output/'status.json',status)
                pd.DataFrame(rows).to_csv(args.output/'extraction_checkpoint.csv',index=False)
                print(f"Extracted {len(rows)}/{len(source)}; failures {status['failed']}; elapsed {elapsed/60:.1f} min; ETA {status['eta_seconds']/60:.1f} min",flush=True)
    index=pd.DataFrame(rows).sort_values('vid_path').reset_index(drop=True)
    index.to_csv(args.output/'feature_index.csv',index=False)
    return index


def grouped_candidates(matrix,truth,groups,names,parameters,representation,workers,progress):
    """Choose settings entirely within training; fit preprocessing inside each fold."""
    import numpy as np
    from sklearn.metrics import accuracy_score,f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.tree import DecisionTreeClassifier
    from senyas.semantic_tree import TreeFeaturePreparation
    folds=[];audit=[]
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42)
    for fold,(fit,held_out) in enumerate(splitter.split(matrix,truth,groups)):
        fit_groups=set(groups[fit]);held_out_groups=set(groups[held_out])
        if fit_groups&held_out_groups:raise AssertionError('A training group crossed a CV boundary')
        if set(truth[fit])!=set(truth) or set(truth[held_out])!=set(truth):
            raise ValueError('A grouped training fold is missing a word class')
        preparation=TreeFeaturePreparation.fit(matrix[fit],names)
        folds.append((preparation.transform(matrix[fit]),truth[fit],preparation.transform(matrix[held_out]),truth[held_out]))
        audit.append({'fold':fold,'fit_samples':len(fit),'held_out_samples':len(held_out),
                      'fit_groups':sorted(str(value) for value in fit_groups),
                      'held_out_groups':sorted(str(value) for value in held_out_groups),
                      'preprocessing_training_samples':preparation.training_rows,
                      'enabled_summary_features':len(preparation.feature_names)})
    def evaluate(settings):
        scores=[]
        for fold,(X_fit,y_fit,X_held_out,y_held_out) in enumerate(folds):
            model=DecisionTreeClassifier(random_state=42,class_weight='balanced',**settings)
            model.fit(X_fit,y_fit);predicted=model.predict(X_held_out)
            scores.append({'fold':fold,'accuracy':float(accuracy_score(y_held_out,predicted)),
                           'macro_f1':float(f1_score(y_held_out,predicted,average='macro',zero_division=0)),
                           'nodes':int(model.tree_.node_count)})
        return {'representation':representation,**settings,
                'cv_mean_accuracy':float(np.mean([item['accuracy'] for item in scores])),
                'cv_mean_macro_f1':float(np.mean([item['macro_f1'] for item in scores])),
                'cv_std_macro_f1':float(np.std([item['macro_f1'] for item in scores])),
                'cv_mean_nodes':float(np.mean([item['nodes'] for item in scores])),
                'fold_scores':scores}
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(evaluate,settings) for settings in parameters]
        for future in as_completed(futures):
            results.append(future.result());progress(results)
    # Explicit parameter order breaks exact ties deterministically despite parallel completion.
    ordered=[next(row for row in results if all(row[key]==value for key,value in settings.items())) for settings in parameters]
    best=max(ordered,key=lambda row:(row['cv_mean_macro_f1'],row['cv_mean_accuracy'],-row['cv_mean_nodes']))
    return ordered,best,audit


def train(args,index):
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score,f1_score
    from sklearn.tree import DecisionTreeClassifier
    from senyas.evaluation import recognition_metrics,prediction_latency_ms
    from senyas.sequences import load_sequence,SEQUENCE_NAMES
    from senyas.semantic_tree import summarize_semantic_sequence,summary_names,summary_feature_mask,TreeFeaturePreparation
    from senyas.model import decision_trace
    manifest=pd.read_csv(args.manifest,keep_default_na=False)
    source=manifest[manifest.split.isin(['train','validation'])]
    if index.vid_path.duplicated().any() or set(index.vid_path)!=set(source.vid_path):raise ValueError('Extraction index does not account for all requested clips')
    if set(index.split)!={'train','validation'}:raise ValueError('Unexpected split in feature index')
    if (manifest.groupby('signer_id').split.nunique()>1).any():raise ValueError('Signer groups cross split boundaries')
    expected=source.set_index('vid_path').sort_index();actual=index.set_index('vid_path').sort_index()
    for column in ('id_label','split','signer_id','sha256'):
        if not expected[column].astype(str).equals(actual[column].astype(str)):
            raise ValueError('Cached clip metadata differs from manifest: '+column)
    accepted=index[index.success.eq(True)].sort_values('vid_path').reset_index(drop=True)
    if args.summary_cache is not None:
        if args.representations!=['phases']:raise ValueError('The supplied summary cache is for three phases only')
        with np.load(args.summary_cache,allow_pickle=False) as cached:
            if list(cached['keys'])!=accepted.vid_path.tolist():raise ValueError('Summary cache row ordering differs')
            matrix=cached['base'].astype(np.float32)
        if matrix.shape!=(len(accepted),len(summary_names('phases'))):raise ValueError('Summary cache width differs')
        # Reproduce representative cached summaries before reusing the matrix.
        for row_number in np.linspace(0,len(accepted)-1,12,dtype=int):
            original=summarize_semantic_sequence(load_sequence(accepted.iloc[row_number].cache_file),'phases')
            if not np.allclose(original,matrix[row_number],equal_nan=True,atol=1e-7,rtol=0):raise ValueError('A cached summary differs from current derivation')
        matrices={'phases':matrix};print(f'Reused {len(accepted)} three-phase summaries',flush=True)
    else:
        matrices={name:[] for name in args.representations}
        for count,row in enumerate(accepted.itertuples(),1):
            sequence=load_sequence(row.cache_file)
            for name in matrices:matrices[name].append(summarize_semantic_sequence(sequence,name))
            if count%100==0 or count==len(accepted):
                save_json(args.output/'status.json',{'stage':'temporal_summaries','completed':count,'total':len(accepted)})
                print(f'Summarized {count}/{len(accepted)} cached sequences',flush=True)
        matrices={name:np.vstack(rows).astype(np.float32) for name,rows in matrices.items()}
    names_by_representation={};excluded={}
    for representation in matrices:
        names=summary_names(representation);keep=summary_feature_mask(names,args.summary_feature_policy)
        names_by_representation[representation]=tuple(name for name,enabled in zip(names,keep) if enabled)
        excluded[representation]=[name for name,enabled in zip(names,keep) if not enabled]
        matrices[representation]=matrices[representation][:,keep]
    save_json(args.output/'excluded_summary_features.json',excluded)
    train_mask=accepted.split.eq('train').to_numpy();validation_mask=accepted.split.eq('validation').to_numpy()
    y=accepted.id_label.to_numpy(dtype=int)
    if set(y[train_mask])!=set(range(105)) or set(y[validation_mask])!=set(range(105)):raise ValueError('An accepted split is missing a class')
    parameters=[dict(criterion=criterion,max_depth=depth,min_samples_leaf=leaf,ccp_alpha=alpha)
                for criterion,depth,leaf,alpha in itertools.product(['gini','entropy'],[10,14,None],[1,2,4],[0.,.001])]
    if args.candidate_set=='focused':
        parameters=[dict(criterion='entropy',max_depth=depth,min_samples_leaf=leaf,ccp_alpha=alpha)
                    for depth,leaf,alpha in itertools.product([10,14,None],[1,2],[0.,.001])]
    if args.fixed_settings_from is not None:
        fixed_model=joblib.load(args.fixed_settings_from)['model']
        settings=fixed_model.get_params()
        if settings['random_state']!=42 or settings['class_weight']!='balanced':
            raise ValueError('Fixed baseline must use the same seed and class weighting')
        parameters=[{key:settings[key] for key in ('criterion','max_depth','min_samples_leaf','ccp_alpha')}]
    results=[];best=None;started=time.perf_counter();completed=0
    for representation,matrix in matrices.items():
        preparation=TreeFeaturePreparation.fit(matrix[train_mask],names_by_representation[representation])
        X_train=preparation.transform(matrix[train_mask]);X_validation=preparation.transform(matrix[validation_mask])
        if args.selection=='train-group-cv':
            def progress(rows):
                chosen_so_far=max(rows,key=lambda row:(row['cv_mean_macro_f1'],row['cv_mean_accuracy'],-row['cv_mean_nodes']))
                save_json(args.output/'status.json',{'stage':'grouped_training_cv','completed':len(rows),'total':len(parameters),
                    'folds':3,'elapsed_seconds':time.perf_counter()-started,'best_cv_macro_f1':chosen_so_far['cv_mean_macro_f1']})
                print(f'Grouped CV {len(rows)}/{len(parameters)} settings (three training folds each); best macro F1 {chosen_so_far["cv_mean_macro_f1"]:.4f}',flush=True)
            candidate_rows,selected,fold_audit=grouped_candidates(matrix[train_mask],y[train_mask],
                accepted.loc[train_mask,'signer_id'].to_numpy(),names_by_representation[representation],parameters,representation,args.cv_workers,progress)
            settings={key:selected[key] for key in ('criterion','max_depth','min_samples_leaf','ccp_alpha')}
            model=DecisionTreeClassifier(random_state=42,class_weight='balanced',**settings).fit(X_train,y[train_mask])
            result={**selected,'nodes':int(model.tree_.node_count),'actual_depth':int(model.get_depth()),'input_features':X_train.shape[1]}
            results.extend(candidate_rows)
            rank=(selected['cv_mean_macro_f1'],selected['cv_mean_accuracy'],-selected['cv_mean_nodes'],-X_train.shape[1])
            if best is None or rank>best[0]:best=(rank,model,preparation,representation,result,X_train,X_validation)
            save_json(args.output/f'{representation}_cv_folds.json',fold_audit)
            continue
        for parameters_for_tree in parameters:
            model=DecisionTreeClassifier(random_state=42,class_weight='balanced',**parameters_for_tree)
            model.fit(X_train,y[train_mask]);predicted=model.predict(X_validation)
            accuracy=float(accuracy_score(y[validation_mask],predicted));macro_f1=float(f1_score(y[validation_mask],predicted,average='macro',zero_division=0))
            result={'representation':representation,**parameters_for_tree,'validation_accuracy':accuracy,'validation_macro_f1':macro_f1,
                    'nodes':int(model.tree_.node_count),'actual_depth':int(model.get_depth()),'input_features':X_train.shape[1]}
            results.append(result);completed+=1
            rank=(macro_f1,accuracy,-model.tree_.node_count,-X_train.shape[1])
            if best is None or rank>best[0]:best=(rank,model,preparation,representation,result,X_train,X_validation)
            save_json(args.output/'status.json',{'stage':'training','completed':completed,'total':len(parameters)*len(matrices),
                'elapsed_seconds':time.perf_counter()-started,'best_validation_accuracy':best[4]['validation_accuracy'],'best_validation_macro_f1':best[4]['validation_macro_f1']})
            if completed%6==0:print(f'Trained {completed}/{len(parameters)*len(matrices)} candidates; best accuracy {best[4]["validation_accuracy"]:.4f}; macro F1 {best[4]["validation_macro_f1"]:.4f}',flush=True)
    _,model,preparation,representation,chosen,X_train,X_validation=best
    lookup={int(row.id_label):{'label':row.label,'category':row.category} for row in manifest.drop_duplicates('id_label').itertuples()}
    artifact={'model':model,'explainer_model':model,'feature_names':list(preparation.feature_names),'raw_sequence_feature_names':list(SEQUENCE_NAMES),
        'feature_preparation':preparation,'representation':representation,'summary_feature_policy':args.summary_feature_policy,'label_lookup':lookup,'confidence_threshold':None,
        'trained_at':datetime.now(timezone.utc).isoformat(),'schema':'semantic-tree-v1',
        'metadata':{'fit_split':'train','selection_split':('fixed_baseline_settings' if args.fixed_settings_from is not None else 'training_group_cv' if args.selection=='train-group-cv' else 'validation'),'refit_on_validation':False,'handshape_labels_used':False,
                    'extractor':{'target_fps':30.,'max_image_side':480,'model_complexity':1},'seed':42,'selected_candidate':chosen}}
    model_path=args.output/'semantic_decision_tree.joblib';joblib.dump(artifact,model_path,compress=3)
    predictions=model.predict(X_validation).astype(int);metrics=recognition_metrics(y[validation_mask],predictions)
    predictions_table=accepted.loc[validation_mask,['vid_path','id_label','label','category','signer_id']].copy()
    predictions_table['predicted_id']=predictions;predictions_table['correct']=predictions==y[validation_mask]
    predictions_table['leaf_vote_fraction']=model.predict_proba(X_validation).max(axis=1)
    full=manifest[manifest.split.eq('validation')][['vid_path','id_label','label','category','signer_id']].merge(
        predictions_table.drop(columns=['id_label','label','category','signer_id']),on='vid_path',how='left',validate='one_to_one')
    full=full.merge(index[['vid_path','success','error']],on='vid_path',how='left',validate='one_to_one');full['correct']=full.correct.eq(True)
    full.to_csv(args.output/'validation_predictions.csv',index=False)
    ranking_columns=['cv_mean_macro_f1','cv_mean_accuracy'] if args.selection=='train-group-cv' else ['validation_macro_f1','validation_accuracy']
    pd.DataFrame(results).sort_values(ranking_columns,ascending=False).to_csv(args.output/'candidate_metrics.csv',index=False)
    pd.DataFrame(metrics['classification_report']).T.to_csv(args.output/'validation_per_class.csv',index_label='class_or_average')
    pd.DataFrame(metrics['confusion_matrix'],index=range(105),columns=range(105)).to_csv(args.output/'validation_confusion_matrix.csv',index_label='true_class')
    traces=[decision_trace(artifact,vector) for vector in X_validation]
    valid=sum(all(rule['condition_holds'] for rule in trace['rules']) and trace['agrees_with_primary'] for trace in traces)
    if valid!=len(traces):raise AssertionError('A decision path failed consistency checks')
    # Include masks so imputed measurements cannot be described as observed poses.
    raw=matrices[representation][validation_mask]
    for i,trace in enumerate(traces):
        trace['vid_path']=predictions_table.iloc[i].vid_path
        missing=~np.isfinite(raw[i,preparation.enabled])
        trace['imputed_path_features']=[rule['feature'] for rule in trace['rules'] if missing[artifact['feature_names'].index(rule['feature'])]]
    save_json(args.output/'validation_decision_paths.json',traces)
    comparison=None
    baseline_path=ROOT/'outputs/preliminary-validation-20261026/validation_predictions.csv'
    if baseline_path.exists():
        baseline=pd.read_csv(baseline_path)
        shared=full.merge(baseline[['vid_path','predicted_id','success']],on='vid_path',suffixes=('_new','_old'))
        common=shared[shared.success_new.eq(True)&shared.success_old.eq(True)]
        comparison={'common_accepted_samples':len(common),'new_accuracy_common':float(common.predicted_id_new.eq(common.id_label).mean()),
                    'old_accuracy_common':float(common.predicted_id_old.eq(common.id_label).mean()),
                    'old_all_attempted_accuracy':float(baseline.correct.eq(True).mean()),'new_all_attempted_accuracy':float(full.correct.mean())}
    preserved_comparison=None
    if args.baseline is not None:
        previous=pd.read_csv(args.baseline/'validation_predictions.csv')
        previous_metrics=json.loads((args.baseline/'metrics.json').read_text())
        paired=full.merge(previous[['vid_path','predicted_id','success']],on='vid_path',suffixes=('_new','_previous'),validate='one_to_one')
        paired=paired[paired.success_new.eq(True)&paired.success_previous.eq(True)]
        new_correct=paired.predicted_id_new.eq(paired.id_label);old_correct=paired.predicted_id_previous.eq(paired.id_label)
        preserved_comparison={'baseline_directory':str(args.baseline),'common_accepted_samples':len(paired),
            'previous_accuracy':float(old_correct.mean()),'new_accuracy':float(new_correct.mean()),
            'previous_macro_f1':previous_metrics['validation_metrics']['macro_f1'],'new_macro_f1':metrics['macro_f1'],
            'fixed_errors':int((new_correct&~old_correct).sum()),'new_errors':int((~new_correct&old_correct).sum()),
            'unchanged_errors':int((~new_correct&~old_correct).sum())}
    groups={column:[{'group':str(group),'attempted':len(part),'accepted':int(part.success.eq(True).sum()),'correct':int(part.correct.sum()),'all_attempted_accuracy':float(part.correct.mean())}
                    for group,part in full.groupby(column)] for column in ['category','signer_id']}
    final={'status':'complete_preliminary_validation','model':'DecisionTreeClassifier','schema':'semantic-tree-v1',
        'train_attempted':int(manifest.split.eq('train').sum()),'train_accepted':int(train_mask.sum()),
        'validation_attempted':len(full),'validation_accepted':int(validation_mask.sum()),'validation_failed':int((~full.success.eq(True)).sum()),
        'correct':int((predictions==y[validation_mask]).sum()),'accuracy_all_attempted':float(full.correct.mean()),
        'validation_metrics':metrics,'training_accuracy':float(accuracy_score(y[train_mask],model.predict(X_train))),
        'selected_candidate':chosen,'candidate_count':len(results),'raw_frame_features':len(SEQUENCE_NAMES),
        'enabled_summary_features':len(artifact['feature_names']),'raw_summary_features':len(preparation.raw_feature_names),
        'summary_feature_policy':args.summary_feature_policy,'excluded_summary_features':len(excluded[representation]),
        'split_features_used':int(np.count_nonzero(model.feature_importances_)),'leaves':int(model.get_n_leaves()),
        'valid_decision_paths':valid,'candidate_training_seconds':time.perf_counter()-started,
        'prediction_latency_ms':prediction_latency_ms(model,X_validation),'comparison':comparison,'groups':groups,
        'preserved_model_comparison':preserved_comparison,'selection':'fixed_baseline_settings' if args.fixed_settings_from is not None else args.selection,'training_cv_fits':len(results)*3 if args.selection=='train-group-cv' else 0,
        'model_sha256':digest(model_path),'app_model_sha256':digest(ROOT/'artifacts/hybrid_recognizer.joblib'),
        'test_samples_processed':0,'refit_on_validation':False,'handshape_labels_used':False,
        'interpretation':('Tree settings copied unchanged from the preserved three-phase baseline; phase count is the experimental change. Validation comparison is preliminary, not final test performance.' if args.fixed_settings_from is not None else
            'Tree settings selected by three grouped folds within training; validation evaluated after selection. Feature experiments were motivated by earlier validation results; preliminary comparison, not final test performance.'
            if args.selection=='train-group-cv' else 'Validation selected representation and hyperparameters; preliminary model-selection scores, not final test performance.')}
    save_json(args.output/'metrics.json',final);save_json(args.output/'status.json',{'stage':'complete','metrics_file':str(args.output/'metrics.json')})
    print(json.dumps({key:final[key] for key in ['train_accepted','validation_accepted','validation_failed','correct','accuracy_all_attempted','selected_candidate','comparison']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--train-only',action='store_true',help='Reuse the completed extraction index')
    parser.add_argument('--representations',nargs='+',choices=['global','phases','phases5'],default=['global','phases'])
    parser.add_argument('--selection',choices=['validation','train-group-cv'],default='validation')
    parser.add_argument('--cv-workers',type=int,default=3)
    parser.add_argument('--reuse-extraction',type=Path,help='Read a completed experiment cache without extracting any video')
    parser.add_argument('--baseline',type=Path,help='Preserve and compare against a completed semantic model experiment')
    parser.add_argument('--fixed-settings-from',type=Path,help='Use exactly the saved baseline tree settings for a controlled phase comparison')
    parser.add_argument('--summary-feature-policy',choices=['all','no_observation_fractions','no_tracking_flags','geometry_motion_only'],default='all')
    parser.add_argument('--summary-cache',type=Path,help='Reuse verified three-phase summaries from a previous experiment')
    parser.add_argument('--candidate-set',choices=['full','focused'],default='full')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not 1<=args.workers<=8:raise ValueError('Use between one and eight extraction workers')
    if not 1<=args.cv_workers<=4:raise ValueError('Use between one and four CV workers')
    if args.fixed_settings_from is not None and (args.selection!='validation' or len(args.representations)!=1):
        raise ValueError('A fixed-settings comparison requires exactly one representation and no CV search')
    if (args.output/'semantic_decision_tree.joblib').exists():raise ValueError('Refusing to overwrite an existing trained model')
    save_json(args.output/'training_configuration.json',{'representations':args.representations,'selection':args.selection,
        'cv_folds':3 if args.selection=='train-group-cv' else 0,'seed':42,'cv_workers':args.cv_workers,
        'manifest':str(args.manifest.resolve()),'manifest_sha256':digest(args.manifest),
        'source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in [ROOT/'senyas/semantic_tree.py',Path(__file__).resolve()]},
        'fixed_settings_from':str(args.fixed_settings_from.resolve()) if args.fixed_settings_from is not None else None,
        'fixed_settings_model_sha256':digest(args.fixed_settings_from) if args.fixed_settings_from is not None else None,
        'summary_feature_policy':args.summary_feature_policy,'candidate_set':args.candidate_set,
        'summary_cache':str(args.summary_cache.resolve()) if args.summary_cache is not None else None,
        'summary_cache_sha256':digest(args.summary_cache) if args.summary_cache is not None else None})
    app_hash=digest(ROOT/'artifacts/hybrid_recognizer.joblib')
    protected={}
    if args.baseline is not None:
        if args.baseline.resolve()==args.output.resolve():raise ValueError('Baseline and experiment folders must differ')
        for name in ('semantic_decision_tree.joblib','metrics.json','feature_index.csv','validation_predictions.csv'):
            path=args.baseline/name;protected[str(path)]=digest(path)
        save_json(args.output/'preserved_baseline.json',protected)
    if args.reuse_extraction is not None:
        import pandas as pd
        if args.reuse_extraction.resolve()==args.output.resolve():raise ValueError('Reused cache must remain in a separate source folder')
        config=json.loads((args.reuse_extraction/'extraction_configuration.json').read_text())
        if digest(args.manifest)!=config['manifest_sha256']:raise ValueError('Manifest differs from the extraction cache')
        for relative,expected in config['source_sha256'].items():
            if digest(ROOT/relative)!=expected:raise ValueError('Extractor code differs from the cache: '+relative)
        index=pd.read_csv(args.reuse_extraction/'feature_index.csv',keep_default_na=False)
        # Resolve cache paths before writing a standalone experiment index.
        for row_index in index.index[index.success.eq(True)]:
            index.loc[row_index,'cache_file']=str(Path(index.loc[row_index,'cache_file']).resolve())
        index.to_csv(args.output/'feature_index.csv',index=False)
        save_json(args.output/'extraction_configuration.json',{**config,'reused_from':str(args.reuse_extraction.resolve())})
    elif args.train_only:
        import pandas as pd
        index=pd.read_csv(args.output/'feature_index.csv',keep_default_na=False)
    else:index=extract(args)
    train(args,index)
    if digest(ROOT/'artifacts/hybrid_recognizer.joblib')!=app_hash:raise AssertionError('The app model changed unexpectedly')
    for path,expected in protected.items():
        if digest(path)!=expected:raise AssertionError('Preserved baseline changed: '+path)
    save_json(args.output/'preservation_verified.json',{'app_model_unchanged':True,'baseline_files_verified':protected,'test_samples_processed':0})


if __name__=='__main__':main()
