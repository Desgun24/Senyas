import ast
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.evaluate_frozen_test import assert_frozen,audit_manifest,digest,freeze_protocol,test_metrics as calculate_test_metrics


def test_frozen_inputs_reject_modified_model_bytes(tmp_path):
    model=tmp_path/'model.joblib';model.write_bytes(b'frozen-model')
    hashes={str(model):digest(model)};assert_frozen(hashes)
    model.write_bytes(b'changed-model')
    with pytest.raises(AssertionError,match='Frozen input changed'):assert_frozen(hashes)


def test_existing_test_protocol_cannot_be_replaced_by_new_settings(tmp_path):
    target=tmp_path/'protocol.json';identity={'model_sha256':'abc','image_size':480}
    first=freeze_protocol(target,identity);second=freeze_protocol(target,identity)
    assert first==second
    with pytest.raises(ValueError,match='frozen test protocol'):freeze_protocol(target,{'model_sha256':'def','image_size':640})
    assert json.loads(target.read_text())==first


def test_test_metrics_count_rejected_predictions_as_errors():
    measured=calculate_test_metrics(np.array([0,1,1]),np.array([0,1,-1]),[0,1])
    assert measured['samples']==3 and measured['accuracy']==pytest.approx(2/3)
    assert measured['macro_recall']==pytest.approx(.75)


def test_final_evaluation_contains_no_training_or_parameter_update_calls():
    source=Path(__file__).resolve().parents[1]/'scripts/evaluate_frozen_test.py'
    parsed=ast.parse(source.read_text())
    forbidden={'fit','fit_transform','partial_fit','set_params'}
    calls=[node.func.attr for node in ast.walk(parsed) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    assert not forbidden.intersection(calls)


def test_manifest_audit_distinguishes_within_split_conflicts_from_cross_split_leakage():
    import pandas as pd
    manifest=pd.DataFrame([
        {'vid_path':'a','id_label':0,'label':'FOUR','split':'validation','signer_id':6,'sha256':'same'},
        {'vid_path':'b','id_label':1,'label':'SEVEN','split':'validation','signer_id':7,'sha256':'same'},
        {'vid_path':'c','id_label':0,'label':'FOUR','split':'test','signer_id':1,'sha256':'other'},
    ])
    audit=audit_manifest(manifest)
    assert not audit['cross_split_duplicate_bytes'] and not audit['test_duplicate_bytes']
    assert len(audit['within_split_duplicate_byte_groups'])==1
    assert audit['within_split_duplicate_byte_groups'][0]['conflicting_labels']
    manifest.loc[2,'sha256']='same'
    with pytest.raises(ValueError,match='cross dataset splits'):audit_manifest(manifest)
