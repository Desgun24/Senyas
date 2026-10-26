import numpy as np
import pytest
from senyas.sequences import FeatureSequence,SEQUENCE_NAMES
from senyas.semantic_tree import summarize_semantic_sequence,summary_names,summary_feature_mask,TreeFeaturePreparation,prepare_tree_input


def sequence():
    values=np.full((30,len(SEQUENCE_NAMES)),np.nan,dtype=np.float32)
    values[:,SEQUENCE_NAMES.index('left.present')]=1
    values[:,SEQUENCE_NAMES.index('right.present')]=0
    values[:,SEQUENCE_NAMES.index('left.index.middle_flexion')]=np.r_[np.zeros(10),np.ones(10)*.5,np.ones(10)]
    return FeatureSequence(values,np.arange(30)/30)


def test_phase_summaries_preserve_order_and_missing_observations():
    sample=sequence();vector=summarize_semantic_sequence(sample);names=summary_names()
    assert vector[names.index('left.index.middle_flexion.opening.median')]==0
    assert vector[names.index('left.index.middle_flexion.middle.median')]==.5
    assert vector[names.index('left.index.middle_flexion.ending.median')]==1
    assert np.isnan(vector[names.index('right.index.middle_flexion.opening.median')])
    assert vector[names.index('right.index.middle_flexion.opening.observed_fraction')]==0


def test_training_preparation_uses_only_training_medians_and_drops_missing_constants():
    train=np.array([[1,np.nan,5],[3,np.nan,5],[5,np.nan,5]],dtype=np.float32)
    fitted=TreeFeaturePreparation.fit(train,['observed','missing','constant'])
    assert fitted.feature_names==('observed',) and fitted.training_rows==3
    assert fitted.transform([np.nan,100,100])[0,0]==3
    assert fitted.transform([999,np.nan,5])[0,0]==999
    assert fitted.medians[0]==3


def test_experimental_directions_are_excluded_and_no_hand_sequence_rejected():
    assert not any('palm_normal_winding' in name or 'body_depth_proxy' in name for name in summary_names())
    sample=sequence();sample.values[:,SEQUENCE_NAMES.index('left.present')]=0
    with pytest.raises(ValueError,match='three frames'):summarize_semantic_sequence(sample)


def test_saved_model_summary_order_must_match_before_predicting():
    raw=summarize_semantic_sequence(sequence(),'global')
    matrix=np.vstack([raw,raw.copy()])
    column=summary_names('global').index('left.index.middle_flexion.global.median')
    matrix[1,column]+=.25
    preparation=TreeFeaturePreparation.fit(matrix,summary_names('global'))
    artifact={'raw_sequence_feature_names':SEQUENCE_NAMES,'representation':'phases','feature_preparation':preparation}
    with pytest.raises(ValueError,match='ordering'):prepare_tree_input(artifact,sequence())


def test_five_phases_use_active_timestamps_and_preserve_internal_tracking_gaps():
    values=np.full((31,len(SEQUENCE_NAMES)),np.nan,dtype=np.float32)
    left=SEQUENCE_NAMES.index('left.present');right=SEQUENCE_NAMES.index('right.present')
    feature=SEQUENCE_NAMES.index('left.index.middle_flexion')
    values[:,[left,right]]=0
    # Leading/trailing off-camera padding must not shift the active phase boundaries.
    values[5:26,left]=1
    values[5:26,feature]=np.arange(21)/20
    values[15,left]=0;values[15,feature]=np.nan
    sample=FeatureSequence(values,np.arange(31)/16)
    vector=summarize_semantic_sequence(sample,'phases5');names=summary_names('phases5')
    phases=('opening','early_middle','middle','late_middle','ending')
    for phase in phases:
        value=vector[names.index(f'left.index.middle_flexion.{phase}.median')]
        assert np.isfinite(value)
    medians=[vector[names.index(f'left.index.middle_flexion.{phase}.median')] for phase in phases]
    assert np.all(np.diff(medians)>0)
    assert vector[names.index('left.index.middle_flexion.global.observed_fraction')]==pytest.approx(20/21)
    assert vector[names.index('left.index.middle_flexion.middle.observed_fraction')]<1
    trimmed=FeatureSequence(values[5:26],np.arange(21)/16)
    assert np.allclose(vector,summarize_semantic_sequence(trimmed,'phases5'),equal_nan=True)


def test_five_phase_artifact_reconstructs_its_own_feature_schema():
    sample=sequence();raw=summarize_semantic_sequence(sample,'phases5')
    matrix=np.vstack([raw,raw.copy()])
    column=summary_names('phases5').index('left.index.middle_flexion.global.median')
    matrix[1,column]+=.25
    preparation=TreeFeaturePreparation.fit(matrix,summary_names('phases5'))
    artifact={'raw_sequence_feature_names':SEQUENCE_NAMES,'representation':'phases5','feature_preparation':preparation}
    expected=preparation.transform(raw)[0]
    assert np.array_equal(prepare_tree_input(artifact,sample),expected)
    assert len(summary_names('phases5'))==len(summary_names('global'))*3


def test_training_group_cv_keeps_groups_separate_and_fits_preprocessing_per_fold():
    from scripts.train_semantic_tree import grouped_candidates
    groups=np.repeat(np.arange(6),2);truth=np.tile([0,1],6)
    matrix=np.column_stack([truth.astype(float),groups.astype(float)])
    settings=[{'criterion':'gini','max_depth':2,'min_samples_leaf':1,'ccp_alpha':0.}]
    rows,best,folds=grouped_candidates(matrix,truth,groups,('shape','position'),settings,'phases5',1,lambda rows:None)
    assert len(rows)==1 and best['cv_mean_accuracy']==1
    held_out_groups=[]
    for fold in folds:
        assert not set(fold['fit_groups'])&set(fold['held_out_groups'])
        assert fold['preprocessing_training_samples']==fold['fit_samples']<len(truth)
        held_out_groups.extend(fold['held_out_groups'])
    assert sorted(held_out_groups)==[str(value) for value in range(6)]


def test_availability_ablation_removes_only_the_declared_feature_families():
    names=summary_names('phases')
    assert summary_feature_mask(names).all()
    for policy in ('no_observation_fractions','no_tracking_flags','geometry_motion_only'):
        kept=[name for name,keep in zip(names,summary_feature_mask(names,policy)) if keep]
        if policy!='no_tracking_flags':assert not any(name.endswith('.observed_fraction') for name in kept)
        if policy!='no_observation_fractions':assert not any('.present.' in name for name in kept)
        assert 'left.index.middle_flexion.opening.median' in kept
        assert 'left.wrist.body_x.global.end_minus_start' in kept
    with pytest.raises(ValueError,match='policy'):summary_feature_mask(names,'unknown')


@pytest.mark.parametrize('policy',['no_observation_fractions','no_tracking_flags','geometry_motion_only'])
def test_ablated_artifacts_rebuild_the_exact_saved_feature_schema(policy,tmp_path):
    import joblib
    from sklearn.tree import DecisionTreeClassifier
    samples=[];raw=[]
    names=summary_names('phases');keep=summary_feature_mask(names,policy)
    for value in (.1,.5,.9):
        sample=sequence();sample.values[:,SEQUENCE_NAMES.index('left.index.middle_flexion')]=value
        samples.append(sample);raw.append(summarize_semantic_sequence(sample)[keep])
    raw=np.asarray(raw);preparation=TreeFeaturePreparation.fit(raw,tuple(name for name,enabled in zip(names,keep) if enabled))
    model=DecisionTreeClassifier(max_depth=1,random_state=42).fit(preparation.transform(raw),[0,1,1])
    artifact={'raw_sequence_feature_names':SEQUENCE_NAMES,'representation':'phases','summary_feature_policy':policy,
              'feature_preparation':preparation,'model':model,'schema':'semantic-tree-v1'}
    target=tmp_path/'ablation.joblib';joblib.dump(artifact,target);loaded=joblib.load(target)
    restored=np.vstack([prepare_tree_input(loaded,sample) for sample in samples])
    np.testing.assert_array_equal(restored,preparation.transform(raw))
    np.testing.assert_array_equal(loaded['model'].predict(restored),[0,1,1])
