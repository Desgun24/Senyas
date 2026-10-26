"""Training-only prototype distances with inspectable multivariate DTW paths."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from .augmented_features import LOOKUP,active_bounds
from .semantic_features import FINGERS,SIDES

CHANNEL_GROUPS=[]
for side in SIDES:
    for finger in FINGERS:
        for joint in (('base','tip') if finger=='thumb' else ('base','middle','tip')):
            CHANNEL_GROUPS.append((f'{side}.{finger}.{joint}_flexion','finger_shape'))
    for i,first in enumerate(FINGERS):
        for second in FINGERS[i+1:]:CHANNEL_GROUPS.append((f'{side}.tips.{first}_{second}_gap','finger_shape'))
    for axis in 'xyz':CHANNEL_GROUPS.append((f'{side}.hand_axis_{axis}','orientation'))
    for axis in 'xy':
        CHANNEL_GROUPS.extend([(f'{side}.wrist.body_{axis}','location'),(f'{side}.motion.velocity_{axis}','movement')])
for name in ('wrist_gap','wrist_horizontal_offset','wrist_vertical_offset'):
    CHANNEL_GROUPS.append((f'hands.{name}','location'))
CHANNEL_NAMES=tuple(name for name,_ in CHANNEL_GROUPS)
GROUPS=('finger_shape','orientation','location','movement')
PARTS=('total_cost',)+tuple(group+'_cost' for group in GROUPS)+('missing_cost','shared_observation_fraction')
SAMPLES=24


def sampled_profile(sequence):
    start,end=active_bounds(sequence);times=sequence.timestamps[start:end]
    targets=np.linspace(times[0],times[-1],SAMPLES)
    right=np.clip(np.searchsorted(times,targets),0,len(times)-1);left=np.maximum(right-1,0)
    selected=np.where(np.abs(times[left]-targets)<=np.abs(times[right]-targets),left,right)
    # Select actual observations; never interpolate across missing readings.
    return sequence.values[start:end][selected][:,[LOOKUP[name] for name in CHANNEL_NAMES]].astype(np.float32)


def _costs(query,references,enabled):
    """Each available feature family has equal weight, irrespective of its width."""
    count,length,_=references.shape
    sums=np.zeros((count,length,length),np.float32);weights=np.zeros_like(sums);shared_sum=np.zeros_like(sums)
    raw=[];availability=[]
    for group in GROUPS:
        selected=enabled&np.array([family==group for _,family in CHANNEL_GROUPS])
        a=query[:,selected];b=references[:,:,selected]
        va=np.isfinite(a).astype(np.float32);vb=np.isfinite(b).astype(np.float32)
        a=np.nan_to_num(a);b=np.nan_to_num(b)
        shared=np.einsum('if,rjf->rij',va,vb,optimize=False)
        union=va.sum(axis=1)[None,:,None]+vb.sum(axis=2)[:,None,:]-shared
        squared=np.einsum('if,rjf->rij',a*a,vb,optimize=False)+np.einsum('if,rjf->rij',va,b*b,optimize=False)-2*np.einsum('if,rjf->rij',a,b,optimize=False)
        fraction=np.divide(shared,union,out=np.zeros_like(shared),where=union>0)
        cost=np.divide(np.maximum(squared,0),shared,out=np.zeros_like(shared),where=shared>0)+(1-fraction)*.5
        available=union>0;raw.append(cost*available);availability.append(available)
        sums+=cost*available;weights+=available;shared_sum+=fraction*available
    missing=np.where(weights==0,2.,0.).astype(np.float32)
    cost=np.divide(sums,weights,out=np.zeros_like(sums),where=weights>0)+missing
    parts=[np.divide(value,weights,out=np.zeros_like(value),where=weights>0) for value in raw]+[missing]
    fraction=np.divide(shared_sum,weights,out=np.zeros_like(sums),where=weights>0)
    return cost,np.stack(parts),fraction


def batched_alignment(query,references,enabled,return_path=False):
    """Exact band-constrained cumulative-cost DTW, batched across prototypes."""
    cost,parts,shared=_costs(query,references,enabled)
    count,length,_=cost.shape;radius=int(np.ceil(length*.25));rows=np.arange(count)
    accumulated=np.full((length+1,length+1,count),np.inf,np.float32);accumulated[0,0]=0
    predecessor=np.zeros((length,length,count),np.uint8)
    for i in range(length):
        for j in range(max(0,i-radius),min(length,i+radius+1)):
            prior=np.stack([accumulated[i,j],accumulated[i,j+1],accumulated[i+1,j]])
            choice=np.argmin(prior,axis=0);predecessor[i,j]=choice
            accumulated[i+1,j+1]=cost[:,i,j]+prior[choice,rows]
    i=np.full(count,length-1);j=i.copy();active=np.ones(count,bool);path_lengths=np.zeros(count,int)
    contributions=np.zeros((len(GROUPS)+1,count),np.float64);coverage=np.zeros(count,np.float64)
    paths=[[] for _ in range(count)] if return_path else None
    while active.any():
        ids=rows[active];ii=i[active];jj=j[active]
        contributions[:,ids]+=parts[:,ids,ii,jj];coverage[ids]+=shared[ids,ii,jj];path_lengths[ids]+=1
        if paths is not None:
            for ref,first,second in zip(ids,ii,jj):paths[ref].append((int(first),int(second)))
        choice=predecessor[ii,jj,ids]
        i[ids]-=(choice!=2);j[ids]-=(choice!=1)
        active=(i>=0)&(j>=0)
    fractions=coverage/path_lengths
    result=np.column_stack([accumulated[-1,-1]/path_lengths,(contributions/path_lengths).T,fractions]).astype(np.float32)
    # Unsupported alignments remain missing, never an apparent perfect match.
    result[fractions<.15,:-1]=np.nan
    if paths is not None:
        for path in paths:path.reverse()
    return result,paths


@dataclass
class DTWProjector:
    medians: np.ndarray
    scales: np.ndarray
    enabled: np.ndarray
    prototypes: np.ndarray
    prototype_keys: tuple
    prototype_groups: tuple
    labels: tuple
    full_bank: np.ndarray
    excluded_banks: dict
    fit_keys: tuple
    fit_groups: tuple

    @property
    def feature_names(self):
        return tuple(f'dtw.reference_{label:03d}.{part}' for label in self.labels for part in PARTS)

    def normalize(self,profiles):
        profiles=np.asarray(profiles,dtype=np.float32)
        if profiles.shape[-2:]!=(SAMPLES,len(CHANNEL_NAMES)):raise ValueError('DTW channel schema differs')
        return np.clip((profiles-self.medians)/self.scales,-6,6)

    @classmethod
    def fit(cls,profiles,truth,groups,keys):
        profiles=np.asarray(profiles,dtype=np.float32);truth=np.asarray(truth);groups=np.asarray(groups).astype(str)
        if len(profiles)!=len(truth) or len(profiles)!=len(groups) or len(profiles)!=len(keys):raise ValueError('Prototype fitting rows differ')
        if len(set(keys))!=len(keys):raise ValueError('Duplicate prototype sample keys')
        flat=profiles.reshape(-1,len(CHANNEL_NAMES))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            medians=np.nanmedian(flat,axis=0);q25,q75=np.nanpercentile(flat,[25,75],axis=0)
        scales=q75-q25;enabled=np.isfinite(medians)&np.isfinite(scales)&(scales>1e-4)
        if not enabled.any():raise ValueError('No supported varying DTW channels')
        medians=np.nan_to_num(medians);scales=np.where(enabled,scales,1.)
        normalized=np.clip((profiles-medians)/scales,-6,6)
        labels=tuple(int(value) for value in sorted(set(truth)));bank_groups=[None]+sorted(set(groups));banks={key:[] for key in bank_groups}
        for label in labels:
            members=np.flatnonzero(truth==label);data=normalized[members]
            # Select an observed central training exemplar; no synthetic word recipe.
            distances=np.zeros((len(members),len(members)),np.float64)
            for first in range(len(members)):
                a=data[first][:,enabled];b=data[:,:,enabled]
                valid_a=np.isfinite(a);valid_b=np.isfinite(b);shared=valid_a[None]&valid_b
                squared=np.where(shared,(np.nan_to_num(b)-np.nan_to_num(a))**2,0)
                counts=shared.sum(axis=(1,2));union=(valid_a[None]|valid_b).sum(axis=(1,2))
                distances[first]=np.divide(squared.sum(axis=(1,2)),counts,out=np.full(len(members),2.),where=counts>0)+.5*(1-np.divide(counts,union,out=np.zeros(len(members)),where=union>0))
            for excluded in bank_groups:
                eligible=np.flatnonzero(groups[members]!=excluded) if excluded is not None else np.arange(len(members))
                if not len(eligible):raise ValueError('A word has no prototype outside the excluded training group')
                central=eligible[np.argmin(distances[np.ix_(eligible,eligible)].mean(axis=1))]
                banks[excluded].append(int(members[central]))
        union=sorted({index for bank in banks.values() for index in bank});lookup={source:position for position,source in enumerate(union)}
        remap=lambda bank:np.asarray([lookup[source] for source in bank],dtype=int)
        result=cls(medians,scales,enabled,normalized[union],tuple(str(keys[i]) for i in union),
                   tuple(str(groups[i]) for i in union),labels,remap(banks[None]),
                   {str(group):remap(bank) for group,bank in banks.items() if group is not None},tuple(keys),tuple(groups))
        for group,bank in result.excluded_banks.items():
            if any(result.prototype_groups[index]==group for index in bank):raise AssertionError('Prototype group exclusion failed')
        return result

    def transform(self,profiles,training_groups=None,progress=None):
        normalized=self.normalize(profiles);rows=[]
        if training_groups is not None and len(training_groups)!=len(normalized):raise ValueError('Training group rows differ')
        for i,query in enumerate(normalized):
            bank=self.full_bank if training_groups is None else self.excluded_banks[str(training_groups[i])]
            # Compute only prototypes used for this query's bank.
            distances,_=batched_alignment(query,self.prototypes[bank],self.enabled)
            rows.append(distances.reshape(-1))
            if progress is not None and ((i+1)%100==0 or i+1==len(normalized)):progress(i+1,len(normalized))
        return np.asarray(rows,dtype=np.float32)

    def explain(self,profile,label):
        position=self.labels.index(int(label));reference=int(self.full_bank[position])
        distances,paths=batched_alignment(self.normalize(profile),self.prototypes[reference:reference+1],self.enabled,True)
        measured={part:float(value) if np.isfinite(value) else None for part,value in zip(PARTS,distances[0])}
        return {'reference_class_id':int(label),'reference_clip':self.prototype_keys[reference],
                'reference_group':self.prototype_groups[reference],'alignment_path':paths[0],**measured}
