"""Fixed, named summaries of semantic sequences for an inspectable Decision Tree."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from .sequences import FeatureSequence, SEQUENCE_NAMES, SEQUENCE_SPECS

SUPPORTED_INDICES = tuple(i for i,spec in enumerate(SEQUENCE_SPECS) if not spec.experimental)
GLOBAL_STATISTICS = ('median','p10','p90','end_minus_start','observed_fraction')
PHASES = ('opening','middle','ending')
PHASE_LAYOUTS = {
    'global': (),
    'phases': PHASES,
    'phases5': ('opening','early_middle','middle','late_middle','ending'),
}
FEATURE_POLICIES=('all','no_observation_fractions','no_tracking_flags','geometry_motion_only')


def summary_feature_mask(names,policy='all'):
    """Remove explicit availability proxies without changing geometric measurements."""
    if policy not in FEATURE_POLICIES:raise ValueError('Unknown summary feature policy')
    keep=[]
    for name in names:
        fraction=name.endswith('.observed_fraction')
        tracking='.present.' in name
        drop=(policy in ('no_observation_fractions','geometry_motion_only') and fraction)
        drop|=(policy in ('no_tracking_flags','geometry_motion_only') and tracking)
        keep.append(not drop)
    return np.asarray(keep,dtype=bool)


def summary_names(representation='phases'):
    if representation not in PHASE_LAYOUTS:
        raise ValueError('Unknown temporal representation')
    phases=PHASE_LAYOUTS[representation]
    names=[]
    for index in SUPPORTED_INDICES:
        feature=SEQUENCE_NAMES[index]
        names.extend(f'{feature}.global.{statistic}' for statistic in GLOBAL_STATISTICS)
        if phases:
            for phase in phases:
                names.extend([f'{feature}.{phase}.median',f'{feature}.{phase}.observed_fraction'])
    return tuple(names)


def summarize_semantic_sequence(sequence: FeatureSequence,representation='phases'):
    names=summary_names(representation)
    phases=PHASE_LAYOUTS[representation]
    if sequence.feature_names!=SEQUENCE_NAMES:
        raise ValueError('Semantic sequence schema differs')
    hands=sequence.values[:,[SEQUENCE_NAMES.index(side+'.present') for side in ('left','right')]]
    observed=np.where(np.any(hands>.5,axis=1))[0]
    if len(observed)<3:
        raise ValueError('At least three frames with hand observations are required')
    # Trim only leading/trailing frames without either hand. Retain internal gaps.
    start,end=int(observed[0]),int(observed[-1])+1
    values=sequence.values[start:end][:,SUPPORTED_INDICES]
    timestamps=sequence.timestamps[start:end]
    duration=timestamps[-1]-timestamps[0]
    if duration<=0:
        raise ValueError('The active sequence needs a positive duration')
    progress=(timestamps-timestamps[0])/duration
    output=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for column in values.T:
            finite=np.isfinite(column)
            if finite.any():
                median=float(np.nanmedian(column));low,high=np.nanpercentile(column,[10,90])
            else:
                median=low=high=np.nan
            first=column[progress<=.15];last=column[progress>=.85]
            delta=float(np.nanmedian(last)-np.nanmedian(first)) if np.isfinite(first).any() and np.isfinite(last).any() else np.nan
            output.extend([median,low,high,delta,float(finite.mean())])
            if phases:
                for phase in range(len(phases)):
                    mask=(progress>=phase/len(phases))&((progress<(phase+1)/len(phases)) if phase<len(phases)-1 else (progress<=1))
                    part=column[mask]
                    output.extend([float(np.nanmedian(part)) if np.isfinite(part).any() else np.nan,
                                   float(np.isfinite(part).mean()) if len(part) else 0.])
    vector=np.asarray(output,dtype=np.float32)
    if vector.shape!=(len(names),) or np.isinf(vector).any():
        raise AssertionError('Invalid semantic summary')
    return vector


@dataclass
class TreeFeaturePreparation:
    raw_feature_names: tuple[str,...]
    enabled: np.ndarray
    medians: np.ndarray
    training_rows: int

    @classmethod
    def fit(cls,matrix,names,minimum_observed_fraction=.05):
        matrix=np.asarray(matrix,dtype=np.float32)
        if matrix.ndim!=2 or matrix.shape[1]!=len(names) or not len(matrix):
            raise ValueError('A matching nonempty training matrix is required')
        if np.isinf(matrix).any():
            raise ValueError('Infinite training values are invalid')
        observed=np.isfinite(matrix).mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            medians=np.nanmedian(matrix,axis=0)
            minimum=np.nanmin(matrix,axis=0);maximum=np.nanmax(matrix,axis=0)
        enabled=np.isfinite(medians)&(observed>=minimum_observed_fraction)&((maximum-minimum)>1e-6)
        if not enabled.any():
            raise ValueError('No varying supported training features')
        return cls(tuple(names),enabled,medians[enabled],len(matrix))

    @property
    def feature_names(self):
        return tuple(name for name,keep in zip(self.raw_feature_names,self.enabled) if keep)

    def transform(self,matrix):
        values=np.asarray(matrix,dtype=np.float32)
        if values.ndim==1:values=values[None,:]
        if values.ndim!=2 or values.shape[1]!=len(self.raw_feature_names) or np.isinf(values).any():
            raise ValueError('Input does not match fitted feature preparation')
        values=values[:,self.enabled].copy()
        return np.where(np.isfinite(values),values,self.medians).astype(np.float32)


def prepare_tree_input(artifact,sequence):
    if tuple(artifact['raw_sequence_feature_names'])!=sequence.feature_names:
        raise ValueError('Sequence does not match the saved model feature schema')
    if artifact.get('schema')=='semantic-augmented-tree-v1':
        from .augmented_features import prepare_augmented_input
        return prepare_augmented_input(artifact,sequence)
    names=summary_names(artifact['representation'])
    keep=summary_feature_mask(names,artifact.get('summary_feature_policy','all'))
    if tuple(name for name,enabled in zip(names,keep) if enabled)!=artifact['feature_preparation'].raw_feature_names:
        raise ValueError('Temporal summary ordering differs from the saved model')
    summary=summarize_semantic_sequence(sequence,artifact['representation'])
    return artifact['feature_preparation'].transform(summary[keep])[0]
