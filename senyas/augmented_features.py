"""Additional inspectable measurements; the original three-phase schema stays fixed."""
from __future__ import annotations
import numpy as np
from .semantic_features import BODY_REGIONS,FINGERS,FINGER_POINTS,SIDES
from .semantic_tree import SUPPORTED_INDICES,summary_names,summarize_semantic_sequence
from .sequences import SEQUENCE_NAMES,SEQUENCE_SPECS,FeatureSequence

LOOKUP={name:i for i,name in enumerate(SEQUENCE_NAMES)}
BASE_NAMES=summary_names('phases')
BASE_LOOKUP={name:i for i,name in enumerate(BASE_NAMES)}
TRANSITION_CHANNELS=tuple(SEQUENCE_NAMES[i] for i in SUPPORTED_INDICES
    if SEQUENCE_SPECS[i].group not in ('tracking','movement','finger_motion'))
MOVEMENT_STATS=('path_length','horizontal_travel','vertical_travel','horizontal_reversals',
                'vertical_reversals','stationary_fraction','stable_pose_fraction','longest_stable_pose_seconds')
PROXIMITY_CHANNELS=tuple(f'{side}.{finger}.near.{region}' for side in SIDES for finger in FINGERS for region in BODY_REGIONS)
PROXIMITY_STATS=('global.median','global.p10','global.observed_fraction','opening.median','middle.median','ending.median')


def active_bounds(sequence):
    hands=sequence.values[:,[LOOKUP['left.present'],LOOKUP['right.present']]]
    observed=np.flatnonzero(np.any(hands>.5,axis=1))
    if len(observed)<3:raise ValueError('At least three hand observations are required')
    return int(observed[0]),int(observed[-1])+1


def transition_names():
    return tuple([f'{name}.transition.{a}_to_{b}' for name in TRANSITION_CHANNELS
        for a,b in [('opening','middle'),('middle','ending')]]+
        [f'{side}.trajectory.{stat}' for side in SIDES for stat in MOVEMENT_STATS])


def _reversals(displacement,dt,valid):
    count=0;previous=0
    for change,elapsed,observed in zip(displacement,dt,valid):
        if not observed:previous=0;continue
        sign=int(np.sign(change)) if abs(change)/elapsed>.1 else 0
        if sign and previous and sign!=previous:count+=1
        if sign:previous=sign
    return float(count)


def transition_features(sequence,base=None):
    if base is None:base=summarize_semantic_sequence(sequence,'phases')
    output=[]
    for name in TRANSITION_CHANNELS:
        medians=[base[BASE_LOOKUP[f'{name}.{phase}.median']] for phase in ('opening','middle','ending')]
        output.extend([medians[1]-medians[0],medians[2]-medians[1]])
    start,end=active_bounds(sequence);values=sequence.values[start:end];times=sequence.timestamps[start:end]
    dt=np.diff(times)
    for side in SIDES:
        points=values[:,[LOOKUP[f'{side}.wrist.body_{axis}'] for axis in 'xy']]
        changes=np.diff(points,axis=0)
        valid=np.isfinite(points[:-1]).all(axis=1)&np.isfinite(points[1:]).all(axis=1)&(dt<=.16)
        if not valid.any():output.extend([np.nan]*len(MOVEMENT_STATS));continue
        distances=np.linalg.norm(changes,axis=1)
        stationary=valid&(distances/dt<.08)
        joints=[LOOKUP[f'{side}.{finger}.{joint}_flexion'] for finger in FINGERS
                for joint in (('base','tip') if finger=='thumb' else ('base','middle','tip'))]
        shape=values[:,joints]
        shape_valid=np.isfinite(shape[:-1]).all(axis=1)&np.isfinite(shape[1:]).all(axis=1)
        shape_speed=np.mean(np.abs(np.diff(shape,axis=0)),axis=1)/dt
        pose_valid=valid&shape_valid
        stable=stationary&shape_valid&(shape_speed<.15)
        longest=running=0.
        for elapsed,held in zip(dt,stable):
            running=running+elapsed if held else 0.;longest=max(longest,running)
        output.extend([float(distances[valid].sum()),float(np.abs(changes[valid,0]).sum()),
                       float(np.abs(changes[valid,1]).sum()),_reversals(changes[:,0],dt,valid),
                       _reversals(changes[:,1],dt,valid),float(dt[stationary].sum()/dt[valid].sum()),
                       float(dt[stable].sum()/dt[pose_valid].sum()) if pose_valid.any() else np.nan,
                       longest if pose_valid.any() else np.nan])
    return np.asarray(output,dtype=np.float32)


def proximity_names():
    return tuple(f'{name}.{stat}' for name in PROXIMITY_CHANNELS for stat in PROXIMITY_STATS)


def finger_body_distances(sequence):
    """Finger identity is retained; every distance is 2D proximity, not touch."""
    aspect=sequence.metadata.get('source_frame_aspect_ratio')
    if aspect is None or not np.isfinite(aspect) or aspect<=0:
        raise ValueError('Finger-body geometry requires the original frame aspect ratio')
    arrays=sequence.landmarks
    if not {'left','right','pose','face'}.issubset(arrays):raise ValueError('Cached raw landmarks are required')
    body=arrays['pose'][:,:,:2]*[aspect,-1.];face=arrays['face'][:,:,:2]*[aspect,-1.]
    width=np.linalg.norm(body[:,11]-body[:,12],axis=1)
    width[~np.isfinite(width)|(width<.02)]=np.nan
    centre=(body[:,11]+body[:,12])/2
    anchors={'left_ear':body[:,7],'right_ear':body[:,8],'nose':body[:,0],
             'left_shoulder':body[:,11],'right_shoulder':body[:,12],
             'left_elbow':body[:,13],'right_elbow':body[:,14],
             'chest':.7*centre+.3*(body[:,23]+body[:,24])/2,
             'forehead':face[:,10],'lips':(face[:,13]+face[:,14])/2,'chin':face[:,152]}
    # Face landmarks supersede the coarse pose nose only when available.
    anchors['nose']=np.where(np.isfinite(face[:,1]).all(axis=1)[:,None],face[:,1],anchors['nose'])
    cheeks=face[:,[234,454]]
    choose=np.argmin(np.linalg.norm(cheeks-body[:,7,None,:],axis=2),axis=1)
    rows=np.arange(len(body));anchors['left_cheek']=cheeks[rows,choose]
    anchors['right_cheek']=cheeks[rows,1-choose]
    ears_ok=np.isfinite(body[:,[7,8]]).all(axis=(1,2))
    anchors['left_cheek'][~ears_ok]=np.nan;anchors['right_cheek'][~ears_ok]=np.nan
    columns=[]
    for side in SIDES:
        hand=arrays[side][:,:,:2]*[aspect,-1.]
        for finger in FINGERS:
            tip=hand[:,FINGER_POINTS[finger][-1]]
            for region in BODY_REGIONS:columns.append(np.linalg.norm(tip-anchors[region],axis=1)/width)
    return np.column_stack(columns).astype(np.float32)


def proximity_features(sequence):
    start,end=active_bounds(sequence);values=finger_body_distances(sequence)[start:end]
    times=sequence.timestamps[start:end];progress=(times-times[0])/(times[-1]-times[0]);output=[]
    masks=[(progress>=i/3)&((progress<(i+1)/3) if i<2 else progress<=1) for i in range(3)]
    for column in values.T:
        finite=column[np.isfinite(column)]
        output.extend([float(np.median(finite)) if len(finite) else np.nan,
                       float(np.percentile(finite,10)) if len(finite) else np.nan,
                       float(np.isfinite(column).mean())])
        for mask in masks:
            part=column[mask];part=part[np.isfinite(part)]
            output.append(float(np.median(part)) if len(part) else np.nan)
    return np.asarray(output,dtype=np.float32)


def extra_names(variant):
    if variant=='transitions':return transition_names()
    if variant=='proximity':return proximity_names()
    raise ValueError('Unknown deterministic feature addition')


def prepare_augmented_input(artifact,sequence):
    base=summarize_semantic_sequence(sequence,'phases');variant=artifact['augmentation']
    if variant=='transitions':extra=transition_features(sequence,base);names=extra_names(variant)
    elif variant=='proximity':extra=proximity_features(sequence);names=extra_names(variant)
    elif variant=='dtw':
        from .dtw_features import sampled_profile
        projector=artifact['dtw_projector'];extra=projector.transform(sampled_profile(sequence)[None,:])[0];names=projector.feature_names
    else:raise ValueError('Unknown saved feature addition')
    if BASE_NAMES+tuple(names)!=artifact['feature_preparation'].raw_feature_names:
        raise ValueError('Saved augmented feature ordering differs')
    return artifact['feature_preparation'].transform(np.r_[base,extra])[0]
