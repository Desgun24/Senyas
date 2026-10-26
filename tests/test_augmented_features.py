import numpy as np
import pytest
from senyas.sequences import FeatureSequence,SEQUENCE_NAMES
from senyas.augmented_features import transition_features,transition_names,finger_body_distances,PROXIMITY_CHANNELS,proximity_features
from senyas.dtw_features import DTWProjector,CHANNEL_NAMES,SAMPLES,batched_alignment


def sample_sequence(length=12):
    values=np.full((length,len(SEQUENCE_NAMES)),np.nan,np.float32)
    values[:,SEQUENCE_NAMES.index('left.present')]=1
    values[:,SEQUENCE_NAMES.index('right.present')]=0
    values[:,SEQUENCE_NAMES.index('left.wrist.body_x')]=np.linspace(0,1,length)
    values[:,SEQUENCE_NAMES.index('left.wrist.body_y')]=0
    return FeatureSequence(values,np.arange(length)/30)


def test_transition_travel_never_connects_tracking_gaps():
    sequence=sample_sequence(7)
    x=SEQUENCE_NAMES.index('left.wrist.body_x');y=SEQUENCE_NAMES.index('left.wrist.body_y')
    sequence.values[:,x]=[0,.1,.2,np.nan,100,100.1,100.2]
    sequence.values[3,y]=np.nan;sequence.values[3,SEQUENCE_NAMES.index('left.present')]=0
    measured=dict(zip(transition_names(),transition_features(sequence)))
    assert measured['left.trajectory.path_length']==pytest.approx(.4,abs=1e-5)
    assert measured['left.trajectory.horizontal_reversals']==0


def test_finger_identity_and_aspect_correct_body_normalization_are_preserved():
    sequence=sample_sequence();length=len(sequence.timestamps)
    pose=np.full((length,33,3),np.nan,np.float32);face=np.full((length,468,3),np.nan,np.float32)
    hand=np.full((length,21,3),np.nan,np.float32)
    pose[:,11]=[.4,.7,0];pose[:,12]=[.6,.7,0];face[:,152]=[.5,.5,0]
    hand[:,8]=[.55,.5,0];hand[:,4]=[.8,.5,0]
    sequence.landmarks={'pose':pose,'face':face,'left':hand,'right':np.full_like(hand,np.nan)}
    with pytest.raises(ValueError,match='aspect ratio'):proximity_features(sequence)
    sequence.metadata['source_frame_aspect_ratio']=2.
    matrix=finger_body_distances(sequence)
    assert matrix[0,PROXIMITY_CHANNELS.index('left.index.near.chin')]==pytest.approx(.25)
    assert matrix[0,PROXIMITY_CHANNELS.index('left.thumb.near.chin')]==pytest.approx(1.5)
    assert np.isnan(matrix[:,PROXIMITY_CHANNELS.index('right.index.near.chin')]).all()
    pose[:,11]=np.nan
    assert np.isnan(finger_body_distances(sequence)).all()


def profiles():
    result=[];truth=[];groups=[];keys=[]
    for group in range(4):
        for label in range(2):
            data=np.full((SAMPLES,len(CHANNEL_NAMES)),np.nan,np.float32)
            data[:,0]=(np.linspace(0,1,SAMPLES) if label==0 else np.linspace(1,0,SAMPLES))+.01*group
            result.append(data);truth.append(label);groups.append(str(group));keys.append(f'{group}/{label}')
    return np.array(result),np.array(truth),np.array(groups),np.array(keys)


def test_dtw_training_banks_exclude_entire_query_group_and_roundtrip(tmp_path):
    import joblib
    data,truth,groups,keys=profiles();projector=DTWProjector.fit(data,truth,groups,keys)
    for group,bank in projector.excluded_banks.items():
        assert all(projector.prototype_groups[index]!=group for index in bank)
    transformed=projector.transform(data,training_groups=groups)
    assert transformed.shape==(len(data),len(projector.feature_names))
    target=tmp_path/'projector.joblib';joblib.dump(projector,target);loaded=joblib.load(target)
    np.testing.assert_allclose(loaded.transform(data),projector.transform(data),equal_nan=True)
    explanation=loaded.explain(data[0],0)
    assert explanation['alignment_path'][0]==(0,0)
    assert explanation['alignment_path'][-1]==(SAMPLES-1,SAMPLES-1)
    assert sum(explanation[part] for part in ('finger_shape_cost','orientation_cost','location_cost','movement_cost','missing_cost'))==pytest.approx(explanation['total_cost'],abs=1e-5)


def test_dtw_preserves_order_and_does_not_treat_missing_as_a_match():
    data,truth,groups,keys=profiles();projector=DTWProjector.fit(data,truth,groups,keys)
    normalized=projector.normalize(data)
    costs,_=batched_alignment(normalized[0],normalized[:2],projector.enabled)
    assert costs[0,0]<costs[1,0]
    missing=np.full_like(normalized[0],np.nan)
    costs,_=batched_alignment(missing,normalized[:2],projector.enabled)
    assert np.isnan(costs[:,:-1]).all() and np.all(costs[:,-1]==0)
