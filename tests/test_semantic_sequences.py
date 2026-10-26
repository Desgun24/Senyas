from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from senyas.semantic_features import FEATURE_NAMES, FINGER_POINTS, frame_features, extract_result
from senyas.sequences import FeatureSequence, SEQUENCE_NAMES, add_motion, timed_derivative, save_sequence, load_sequence, circle_measurements
from senyas.sequence_matching import TrainingScale, align_sequences


def landmarks():
    hand=np.zeros((21,3));hand[0]=[.5,.8,0]
    for finger,x in zip(['index','middle','ring','pinky'],[.4,.47,.54,.61]):
        for j,y in zip(FINGER_POINTS[finger],[.6,.45,.35,.25]):hand[j]=[x,y,0]
    hand[1:5]=[[.43,.73,0],[.36,.64,0],[.3,.57,0],[.24,.5,0]]
    pose=np.full((33,3),np.nan)
    for j,point in {11:[.3,.55,0],12:[.7,.55,0],7:[.4,.25,0],8:[.6,.25,0],0:[.5,.28,0],
                    13:[.2,.75,0],14:[.8,.75,0],15:[.5,.8,-.1],16:[.7,.8,-.1],23:[.35,.9,0],24:[.65,.9,0]}.items():pose[j]=point
    return hand,pose,np.full((468,3),np.nan)


def vector(hand=None,pose=None):
    default,p,f=landmarks()
    return frame_features(default if hand is None else hand,np.full((21,3),np.nan),p if pose is None else pose,f,16/9)


def get(values,key):return values[FEATURE_NAMES.index(key)]


def test_geometry_translation_and_scale_invariance():
    hand,pose,face=landmarks();missing=np.full_like(hand,np.nan)
    a=frame_features(hand,missing,pose,face,16/9)
    def change(points):
        out=points*.65;out[:,:2]+=[.1,.12];return out
    b=frame_features(change(hand),missing,change(pose),face,16/9)
    np.testing.assert_allclose(a,b,atol=1e-5,equal_nan=True)


def test_folded_finger_differs_from_straight_finger():
    hand,_,_=landmarks();straight=vector(hand)
    hand[7]=[.4,.49,0];hand[8]=[.4,.57,0]
    folded=vector(hand)
    assert get(straight,'left.index.middle_flexion')<.01
    assert get(folded,'left.index.middle_flexion')>.8
    assert get(folded,'left.index.tip_to_root')<get(straight,'left.index.tip_to_root')


def test_all_ten_tip_pairs_and_missing_hand_stays_unknown():
    values=vector()
    assert sum(name.startswith('left.tips.') for name in FEATURE_NAMES)==10
    assert get(values,'right.present')==0
    assert np.isnan(get(values,'right.index.base_flexion'))
    assert np.isnan(get(values,'hands.wrist_gap'))


def test_missing_shoulders_do_not_become_fake_body_positions():
    values=vector(pose=np.full((33,3),np.nan))
    assert get(values,'pose.present')==0
    assert np.isnan(get(values,'left.wrist.body_x'))
    assert np.isnan(get(values,'left.near.chest.palm'))
    assert np.isfinite(get(values,'left.index.middle_flexion'))


def test_independent_hand_z_origins_never_determine_interhand_distance():
    hand,pose,face=landmarks();right=hand.copy();right[:,0]+=.1
    a=frame_features(hand,right,pose,face,16/9)
    right[:,2]+=100
    b=frame_features(hand,right,pose,face,16/9)
    for name in FEATURE_NAMES:
        if name.startswith('hands.') or 'near_other_' in name:
            np.testing.assert_allclose(get(a,name),get(b,name),atol=1e-5,equal_nan=True)


def test_low_visibility_pose_points_are_missing():
    points=[SimpleNamespace(x=.1,y=.2,z=0,visibility=.1) for _ in range(33)]
    values,arrays=extract_result(SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=points)),16/9)
    assert np.isnan(arrays['pose']).all()
    assert get(values,'pose.present')==0


def test_motion_uses_real_timestamps_and_does_not_bridge_gaps():
    times=np.array([0,.04,.08,.5])
    derivative=timed_derivative(np.array([0,.08,.16,1.]),times)
    np.testing.assert_allclose(derivative[1:3],[2,2])
    assert np.isnan(derivative[0]) and np.isnan(derivative[-1])


def test_relative_finger_sway_is_separate_from_shape_change():
    times=np.arange(31)/30
    base=np.vstack([vector()]*31)
    base[:,FEATURE_NAMES.index('left.index.local_tip_x')]=.15*np.sin(times*6*np.pi)
    values=add_motion(base,times)
    assert values[-1,SEQUENCE_NAMES.index('left.index.sway_extent')]>.2
    assert values[-1,SEQUENCE_NAMES.index('left.index.sway_reversals')]>0
    assert values[-1,SEQUENCE_NAMES.index('left.index.shape_variation')]<1e-6


def test_sequence_roundtrip_keeps_masks_timing_and_raw_landmarks(tmp_path):
    times=np.arange(15)/30
    sequence=FeatureSequence(add_motion(np.vstack([vector()]*15),times),times,{'test':True},{'left':np.zeros((15,21,3))})
    path=tmp_path/'sequence.npz';save_sequence(path,sequence);loaded=load_sequence(path)
    np.testing.assert_allclose(loaded.values,sequence.values,equal_nan=True)
    np.testing.assert_array_equal(loaded.timestamps,times)
    assert loaded.landmarks['left'].shape==(15,21,3)
    with pytest.raises(ValueError):FeatureSequence(sequence.values,np.zeros(15))


def test_circle_direction_changes_when_trajectory_reverses():
    times=np.arange(61)/30;theta=np.linspace(0,2*np.pi,61)
    base=np.vstack([vector()]*61)
    base[:,FEATURE_NAMES.index('left.wrist.body_x')]=.4*np.cos(theta)
    base[:,FEATURE_NAMES.index('left.wrist.body_y')]=.4*np.sin(theta)
    forward=FeatureSequence(add_motion(base,times),times)
    backward=FeatureSequence(add_motion(base[::-1],times),times)
    f,b=circle_measurements(forward,'left'),circle_measurements(backward,'left')
    assert f['available'] and b['available']
    assert f['signed_turns']==pytest.approx(1)
    assert b['signed_turns']==pytest.approx(-1)
    assert f['radius_relative_std']<.001


def example_sequence(repeats=1,reverse=False):
    base=np.vstack([vector()]*12)
    base[:,FEATURE_NAMES.index('left.index.middle_flexion')]=np.r_[np.zeros(4),np.ones(4),np.zeros(4)]
    base[:,FEATURE_NAMES.index('left.wrist.body_x')]=np.linspace(0,1,12)
    if reverse:base=base[::-1].copy()
    base=np.repeat(base,repeats,axis=0)
    times=np.arange(len(base))/30
    return FeatureSequence(add_motion(base,times),times)


def test_dtw_aligns_stretched_sequence_and_retains_real_costs():
    first,second=example_sequence(),example_sequence(2)
    scale=TrainingScale.fit([first,second])
    result=align_sequences(first,second,scale,window_fraction=1)
    assert result.path[0]==(0,0) and result.path[-1]==(11,23)
    assert sum(result.group_costs.values())==pytest.approx(result.score)
    assert all(0<=j2-j1<=1 and 0<=i2-i1<=1 for (i1,j1),(i2,j2) in zip(result.path,result.path[1:]))


def test_dtw_is_sensitive_to_event_order():
    first=example_sequence();reverse=example_sequence(reverse=True)
    scale=TrainingScale.fit([first,reverse])
    assert align_sequences(first,first,scale).score<align_sequences(first,reverse,scale).score


def test_dtw_rejects_missing_observations_instead_of_matching_zero_hands():
    first=example_sequence();scale=TrainingScale.fit([first])
    missing=FeatureSequence(np.full_like(first.values,np.nan),first.timestamps)
    with pytest.raises(ValueError,match='No supported alignment'):align_sequences(first,missing,scale)
