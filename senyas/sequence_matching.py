"""Inspectable DTW alignment on engineered features, without word-specific rules."""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np

from .sequences import FeatureSequence, SEQUENCE_SPECS, SEQUENCE_NAMES


@dataclass
class TrainingScale:
    median: np.ndarray
    scale: np.ndarray
    enabled: np.ndarray
    feature_names: tuple = SEQUENCE_NAMES

    @classmethod
    def fit(cls, training_sequences: list[FeatureSequence]):
        """Fit only on training observations; validation must never enter this list."""
        if not training_sequences or any(seq.feature_names != SEQUENCE_NAMES for seq in training_sequences):
            raise ValueError("Training sequences with a matching schema are required")
        values = np.vstack([seq.values for seq in training_sequences])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(values, axis=0)
            q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
        scale = q75-q25
        enabled = np.isfinite(medians) & np.isfinite(scale) & (scale > 1e-5)
        enabled &= np.sum(np.isfinite(values), axis=0) >= 5
        enabled &= np.array([not spec.experimental and spec.group != 'tracking' for spec in SEQUENCE_SPECS])
        if not enabled.any():
            raise ValueError("No varying, supported features in the training observations")
        return cls(np.nan_to_num(medians), np.where(enabled, scale, 1.), enabled)

    def transform(self, sequence: FeatureSequence):
        if sequence.feature_names != self.feature_names:
            raise ValueError("Sequence and training scaler schemas differ")
        return np.clip((sequence.values-self.median)/self.scale, -6, 6)


@dataclass
class Alignment:
    score: float
    total_cost: float
    path: list[tuple[int, int]]
    group_costs: dict[str, float]
    shared_observation_fraction: float


def _group_cost(first, second):
    valid_a, valid_b = np.isfinite(first).astype(float), np.isfinite(second).astype(float)
    a, b = np.nan_to_num(first), np.nan_to_num(second)
    shared = valid_a @ valid_b.T
    union = valid_a.sum(axis=1)[:, None] + valid_b.sum(axis=1)[None, :] - shared
    sums = (a*a) @ valid_b.T + valid_a @ (b*b).T - 2*a@b.T
    fractions = np.divide(shared, union, out=np.zeros_like(shared), where=union>0)
    cost = np.divide(np.maximum(sums,0), shared, out=np.zeros_like(shared), where=shared>0)
    # Penalize missing disagreement. Both-missing dimensions provide no evidence.
    cost += (1-fractions)*.5
    available = union>0
    return cost, fractions, available


def align_sequences(first: FeatureSequence, second: FeatureSequence, scaling: TrainingScale,
                    window_fraction=.25, group_weights: dict[str,float] | None=None) -> Alignment:
    """Minimum cumulative-cost path, reported with mean path cost for comparison.

    Feature groups contribute equally by default, regardless of their widths.
    Exact costs are retained so a description cannot invent a matching event.
    """
    if not 0 <= window_fraction <= 1:
        raise ValueError("Warping window must be between zero and one")
    a, b = scaling.transform(first), scaling.transform(second)
    names = sorted({spec.group for spec, enabled in zip(SEQUENCE_SPECS, scaling.enabled) if enabled})
    matrices, fractions, availability, weights = {}, {}, {}, {}
    for group in names:
        weight = float((group_weights or {}).get(group,1.))
        if not np.isfinite(weight) or weight<0:
            raise ValueError("Group weights must be finite and nonnegative")
        if weight == 0:
            continue
        selected = scaling.enabled & np.array([spec.group==group for spec in SEQUENCE_SPECS])
        matrices[group], fractions[group], availability[group] = _group_cost(a[:,selected], b[:,selected])
        weights[group] = weight
    if not matrices:
        raise ValueError("No enabled feature groups")
    n,m = len(a),len(b)
    weight_sum = sum(availability[group]*weights[group] for group in matrices)
    shared = np.divide(sum(fractions[group]*availability[group]*weights[group] for group in matrices), weight_sum,
                       out=np.zeros((n,m)),where=weight_sum>0)
    cost = np.divide(sum(matrices[group]*availability[group]*weights[group] for group in matrices),weight_sum,
                     out=np.full((n,m),np.inf),where=weight_sum>0)
    cost[shared<.25] = np.inf
    accumulated = np.full((n+1,m+1),np.inf)
    accumulated[0,0] = 0.
    predecessor = np.zeros((n,m),dtype=np.uint8)
    radius = max(abs(n-m),int(np.ceil(max(n,m)*window_fraction)))
    for i in range(n):
        for j in range(max(0,i-radius),min(m,i+radius+1)):
            prior = [accumulated[i,j], accumulated[i,j+1], accumulated[i+1,j]]
            choice = int(np.argmin(prior))
            predecessor[i,j] = choice
            accumulated[i+1,j+1] = cost[i,j]+prior[choice]
    if not np.isfinite(accumulated[n,m]):
        raise ValueError("No supported alignment: tracking gaps, insufficient shared observations, or restrictive warping window")
    path, i,j = [],n-1,m-1
    while i>=0 and j>=0:
        path.append((i,j))
        choice = predecessor[i,j]
        if choice==0:
            i-=1; j-=1
        elif choice==1:
            i-=1
        else:
            j-=1
    path.reverse()
    if path[0]!=(0,0) or path[-1]!=(n-1,m-1):
        raise AssertionError("Incomplete DTW path")
    ix,jx = np.array(path).T
    contributions = {group: float(np.mean(np.divide(matrices[group][ix,jx]*availability[group][ix,jx]*weights[group],
                      weight_sum[ix,jx],out=np.zeros(len(path)),where=weight_sum[ix,jx]>0))) for group in matrices}
    total = float(accumulated[n,m])
    return Alignment(total/len(path),total,path,contributions,float(shared[ix,jx].mean()))
