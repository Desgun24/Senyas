"""Save explicitly labelled development clips without fitting any model."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid
import numpy as np
from .sequences import FeatureSequence, save_sequence

HAND_SHAPES = ("fist", "open_tight", "open_wide", "curved_tight", "curved_wide", "pointing", "pinch")


def save_labelled_sample(destination: Path, capture_id: str, sequence: FeatureSequence, frames,
                         labels: dict[str,str], interval: tuple[float,float], signer_session: str):
    import cv2
    # Names always come from a UUID, never an untrusted label or user path fragment.
    sample_id = str(uuid.UUID(capture_id))
    if len(frames) != len(sequence.timestamps) or len(frames)<2:
        raise ValueError("Sample frames and sequence timestamps must correspond")
    if not labels or any(side not in ['left','right'] or not re.fullmatch(r'[a-z][a-z0-9_]{0,39}', label)
                         for side,label in labels.items()):
        raise ValueError("Choose at least one hand and a short lowercase hand-shape label")
    start,end = map(float,interval)
    if not np.isfinite([start,end]).all() or not 0<=start<end<=sequence.timestamps[-1]+1e-6:
        raise ValueError("Choose a nonempty label interval within the recording")
    selected = (sequence.timestamps>=start)&(sequence.timestamps<=end)
    for side in labels:
        present = sequence.values[:,sequence.feature_names.index(f'{side}.present')]>.5
        if (present&selected).sum()<3:
            raise ValueError(f'The {side} hand needs at least three detected frames in the label interval')
    root=Path(destination).resolve()
    root.mkdir(parents=True,exist_ok=True)
    paths = {suffix: root/f'{sample_id}.{suffix}' for suffix in ['mp4','npz','json']}
    fingerprint=hashlib.sha256(sequence.values.tobytes()+sequence.timestamps.tobytes()).hexdigest()
    annotation={'labels_by_hand':labels,'label_start_seconds':start,'label_end_seconds':end}
    if any(path.exists() for path in paths.values()):
        if not all(path.is_file() for path in paths.values()):
            raise ValueError('A previous save is incomplete. Record a new sample rather than overwriting it.')
        stored=json.loads(paths['json'].read_text(encoding='utf-8'))
        if stored.get('recording_fingerprint')!=fingerprint:
            raise ValueError('The existing clip differs from this recording')
        for previous in stored['annotations']:
            if previous==annotation:
                raise ValueError('This label interval was already saved.')
            overlap=min(end,previous['label_end_seconds'])-max(start,previous['label_start_seconds'])
            if overlap>1e-6 and any(side in previous['labels_by_hand'] and label!=previous['labels_by_hand'][side] for side,label in labels.items()):
                raise ValueError('Different shape labels overlap on the same hand. Choose separate time intervals.')
        stored['annotations'].append(annotation)
        temporary=paths['json'].with_suffix('.json.tmp')
        temporary.write_text(json.dumps(stored,indent=2,ensure_ascii=False),encoding='utf-8')
        temporary.replace(paths['json'])
        return paths
    height,width=frames[0].shape[:2]
    if any(frame.shape!=(height,width,3) for frame in frames):
        raise ValueError('All recorded frames must have the same dimensions')
    fps=(len(frames)-1)/(sequence.timestamps[-1]-sequence.timestamps[0])
    writer=cv2.VideoWriter(str(paths['mp4']),cv2.VideoWriter_fourcc(*'mp4v'),float(fps),(width,height))
    if not writer.isOpened():
        raise ValueError('The local video encoder could not start')
    try:
        for frame in frames:writer.write(frame)
    finally:
        writer.release()
    if not paths['mp4'].is_file() or paths['mp4'].stat().st_size<100:
        raise ValueError('The labelled clip could not be saved')
    save_sequence(paths['npz'],sequence)
    metadata={'sample_id':sample_id,'clip_group':sample_id,'signer_session':signer_session.strip(),
        'recorded_at':datetime.now(timezone.utc).isoformat(),'source':'explicit_feature_lab_capture',
        'video':paths['mp4'].name,'sequence':paths['npz'].name,'audio_tracks':0,
        'labels_by_hand':labels,'label_start_seconds':start,'label_end_seconds':end,
        'recording_fingerprint':fingerprint,'annotations':[annotation],
        'timing':'MP4 uses average frame rate; exact capture timestamps are preserved in NPZ',
        'training_status':'labelled_example_only; no classifier fitted',
        'partition_rule':'Keep every frame of this clip in one split; prefer grouping by signer/session'}
    paths['json'].write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    return paths
