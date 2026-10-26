import json
import uuid
import numpy as np
import pytest
from senyas.handshape_samples import save_labelled_sample
from senyas.sequences import FeatureSequence,SEQUENCE_NAMES


def sample():
    times=np.arange(10)/30
    values=np.full((10,len(SEQUENCE_NAMES)),np.nan,dtype=np.float32)
    values[:,SEQUENCE_NAMES.index('left.present')]=1
    sequence=FeatureSequence(values,times)
    frames=[np.zeros((48,64,3),dtype=np.uint8) for _ in times]
    return sequence,frames


def test_explicit_save_has_clip_group_side_labels_and_exact_timing(tmp_path):
    sequence,frames=sample();capture_id=str(uuid.uuid4())
    paths=save_labelled_sample(tmp_path,capture_id,sequence,frames,{'left':'open_tight'},(0,.25),'session-a')
    data=json.loads(paths['json'].read_text())
    assert data['clip_group']==capture_id and data['labels_by_hand']=={'left':'open_tight'}
    assert data['audio_tracks']==0 and paths['mp4'].stat().st_size>100 and paths['npz'].is_file()
    with pytest.raises(ValueError,match='already saved'):
        save_labelled_sample(tmp_path,capture_id,sequence,frames,{'left':'open_tight'},(0,.25),'session-a')


def test_multiple_shape_intervals_reuse_one_clip_without_group_leakage(tmp_path):
    sequence,frames=sample();capture_id=str(uuid.uuid4())
    first=save_labelled_sample(tmp_path,capture_id,sequence,frames,{'left':'fist'},(0,.1),'a')
    size=first['mp4'].stat().st_size
    save_labelled_sample(tmp_path,capture_id,sequence,frames,{'left':'open_tight'},(.15,.29),'a')
    data=json.loads(first['json'].read_text())
    assert len(data['annotations'])==2 and first['mp4'].stat().st_size==size
    assert len(list(tmp_path.glob('*.mp4')))==1
    with pytest.raises(ValueError,match='overlap'):
        save_labelled_sample(tmp_path,capture_id,sequence,frames,{'left':'open_wide'},(.15,.29),'a')


def test_labels_cannot_assert_an_unseen_hand_or_escape_output_directory(tmp_path):
    sequence,frames=sample()
    with pytest.raises(ValueError,match='right hand'):
        save_labelled_sample(tmp_path,str(uuid.uuid4()),sequence,frames,{'right':'fist'},(0,.25),'a')
    with pytest.raises(ValueError,match='short lowercase'):
        save_labelled_sample(tmp_path,str(uuid.uuid4()),sequence,frames,{'left':'../fist'},(0,.25),'a')
    assert not list(tmp_path.iterdir())
