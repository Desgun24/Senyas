import base64
import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from senyas.camera import FRAME_COUNT, FRAME_SIZE, decode_camera_attempt
from senyas.features import LandmarkExtractor, frame_feature_names, temporal_feature_names


@pytest.fixture
def camera_payload():
    # Asymmetric colours make channel-order and accidental mirroring detectable.
    pixels = np.zeros((360, 640, 3), dtype=np.uint8)
    pixels[:, :320] = [240, 10, 20]
    pixels[:, 320:] = [10, 20, 240]
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "capture_id": "test-capture", "context": "0:Decision Tree",
        "frames": [encoded] * FRAME_COUNT,
        "timestamps": np.linspace(0, 4, FRAME_COUNT).tolist(),
    }


def test_camera_decodes_unmirrored_bgr_frames(camera_payload):
    attempt = decode_camera_attempt(camera_payload, "0:Decision Tree")
    assert len(attempt.frames) == FRAME_COUNT
    assert attempt.frames[0].shape == (FRAME_SIZE[1], FRAME_SIZE[0], 3)
    assert attempt.duration_seconds == 4
    np.testing.assert_allclose(attempt.frames[0][100, 100], [20, 10, 240], atol=3)
    np.testing.assert_allclose(attempt.frames[0][100, 500], [240, 20, 10], atol=3)


def test_development_camera_accepts_actual_frame_count_and_preserves_timing(camera_payload):
    camera_payload['frames']=[camera_payload['frames'][0]]*81
    camera_payload['timestamps']=np.linspace(0,4,81).tolist()
    attempt=decode_camera_attempt(camera_payload,'0:Decision Tree',expected_frame_count=(16,121))
    assert len(attempt.frames)==81 and attempt.timestamps[-1]==4
    with pytest.raises(ValueError,match='incomplete'):
        decode_camera_attempt(camera_payload,'0:Decision Tree')


@pytest.mark.parametrize("problem", ["stale", "incomplete", "timing", "paused", "corrupt", "size"])
def test_camera_rejects_unusable_captures(camera_payload, problem):
    if problem == "stale":
        camera_payload["context"] = "1:Decision Tree"
    elif problem == "incomplete":
        camera_payload["frames"].pop()
    elif problem == "timing":
        camera_payload["timestamps"][4] = camera_payload["timestamps"][3]
    elif problem == "paused":
        camera_payload["timestamps"][16:] = [t + 1.5 for t in camera_payload["timestamps"][16:]]
    elif problem == "corrupt":
        camera_payload["frames"][0] = "not-base64!"
    else:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buffer, format="JPEG")
        camera_payload["frames"][0] = base64.b64encode(buffer.getvalue()).decode("ascii")
    with pytest.raises(ValueError):
        decode_camera_attempt(camera_payload, "0:Decision Tree")


def fake_extractor(has_hand=True):
    extractor = LandmarkExtractor.__new__(LandmarkExtractor)
    extractor.cv2 = cv2
    extractor.max_frames = FRAME_COUNT
    extractor.min_hand_frames = 3
    extractor.max_image_side = 640
    extractor.base_names = frame_feature_names()
    extractor.feature_names = temporal_feature_names()

    def process(rgb):
        # Synthetic one-hand landmarks depend on the input RGB pixels.
        hand = SimpleNamespace(landmark=[
            SimpleNamespace(x=float(rgb[0, 0, 0]) / 255 + i * .01, y=.2 + i * .02, z=i * .001)
            for i in range(21)
        ]) if has_hand else None
        return SimpleNamespace(left_hand_landmarks=hand, right_hand_landmarks=None, pose_landmarks=None)

    extractor.holistic = SimpleNamespace(process=process)
    return extractor


def test_camera_and_video_use_identical_features_for_one_hand(monkeypatch):
    frames = [np.full((360, 640, 3), index, dtype=np.uint8) for index in range(FRAME_COUNT)]
    extractor = fake_extractor()
    camera_vector, camera_info = extractor.extract_frames(frames, 4)
    remaining = iter(frames)
    capture = SimpleNamespace(
        isOpened=lambda: True,
        get=lambda prop: FRAME_COUNT if prop == cv2.CAP_PROP_FRAME_COUNT else 8,
        read=lambda: (True, next_frame) if (next_frame := next(remaining, None)) is not None else (False, None),
        release=lambda: None,
    )
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    video_vector, video_info = extractor.extract_video("synthetic.mp4")
    np.testing.assert_array_equal(camera_vector, video_vector)
    assert camera_vector.shape == (1630,)
    assert np.isfinite(camera_vector).all()
    assert camera_info.hand_frames == video_info.hand_frames == FRAME_COUNT
    assert camera_info.hand_detection_rate == 1


def test_camera_rejects_attempts_without_detected_hands():
    with pytest.raises(ValueError, match="Only 0 sampled frames"):
        fake_extractor(has_hand=False).extract_frames([np.zeros((360, 640, 3), dtype=np.uint8)] * FRAME_COUNT, 4)
