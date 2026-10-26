import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from scripts.train_semantic_extra_trees import make_folds
from scripts.verify_semantic_extra_trees import verify_ensemble


def test_ensemble_cv_refits_preparation_without_held_out_group_rows():
    groups=np.repeat(np.arange(6),2);truth=np.tile([0,5],6)
    raw=np.column_stack([truth.astype(float),groups.astype(float)])
    raw[0,1]=np.nan
    folds,audit=make_folds(raw,truth,groups,('finger_flexion','position'))
    seen=[]
    for (X_fit,y_fit,X_held,y_held),fold in zip(folds,audit):
        assert not set(fold['fit_groups'])&set(fold['held_out_groups'])
        assert fold['preprocessing_training_samples']==len(X_fit)<len(raw)
        assert np.isfinite(X_fit).all() and np.isfinite(X_held).all()
        assert set(y_fit)==set(y_held)=={0,5}
        seen.extend(fold['held_out_rows'])
    assert sorted(seen)==list(range(len(raw)))


def test_extra_trees_prediction_uses_average_leaf_probabilities_with_class_mapping():
    rng=np.random.default_rng(12);X=rng.normal(size=(36,3)).astype(np.float32)
    y=np.where(X[:,0]+X[:,1]>.1,5,0)
    model=ExtraTreesClassifier(n_estimators=7,max_features=1,min_samples_leaf=3,random_state=42,bootstrap=False,n_jobs=1).fit(X,y)
    assert verify_ensemble(model,X,len(X))==7*len(X)
    assert set(model.classes_)=={0,5}
