"""Browser-camera input for a complete, short signing attempt."""
from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit.components.v1 as components


FRAME_COUNT = 32
FRAME_SIZE = (640, 360)
CAPTURE_SECONDS = 4
_camera = components.declare_component(
    "senyas_live_camera", path=str(Path(__file__).parent / "components" / "live_camera")
)


@dataclass(frozen=True)
class CameraAttempt:
    capture_id: str
    context: str
    frames: list[np.ndarray]
    duration_seconds: float


def live_camera(context: str, processed_capture_id: str | None = None) -> dict | None:
    return _camera(
        context=context,
        processed_capture_id=processed_capture_id,
        duration_seconds=CAPTURE_SECONDS,
        frame_count=FRAME_COUNT,
        key="senyas_live_camera",
        default=None,
    )


def decode_camera_attempt(payload: dict, expected_context: str) -> CameraAttempt:
    """Validate a complete capture and decode its unmirrored JPEG samples."""
    if not isinstance(payload, dict) or payload.get("context") != expected_context:
        raise ValueError("The target changed. Record a new signing attempt.")
    capture_id = payload.get("capture_id")
    if not isinstance(capture_id, str) or not 1 <= len(capture_id) <= 80:
        raise ValueError("The camera capture has no valid identifier. Please record again.")
    encoded = payload.get("frames")
    times = payload.get("timestamps")
    if not isinstance(encoded, list) or len(encoded) != FRAME_COUNT:
        raise ValueError("The camera capture was incomplete. Please record the full four seconds.")
    if not isinstance(times, list) or len(times) != FRAME_COUNT:
        raise ValueError("The camera capture has incomplete timing information.")
    timestamps = np.asarray(times, dtype=float)
    if timestamps.shape != (FRAME_COUNT,) or not np.isfinite(timestamps).all():
        raise ValueError("The camera capture has invalid timing information.")
    gaps = np.diff(timestamps)
    duration = float(timestamps[-1] - timestamps[0])
    if not (gaps > 0).all() or gaps.max() > 1.0 or not 3.0 <= duration <= 6.0:
        raise ValueError("The camera paused during recording. Keep this tab active and try again.")
    if any(not isinstance(frame, str) for frame in encoded) or sum(map(len, encoded)) > 8_000_000:
        raise ValueError("The camera capture is invalid or too large. Please record again.")

    frames: list[np.ndarray] = []
    for encoded_frame in encoded:
        try:
            data = base64.b64decode(encoded_frame, validate=True)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG" or image.size != FRAME_SIZE:
                    raise ValueError("The camera frame has an unexpected size or format.")
                # The extractor expects OpenCV's BGR channel order.
                frames.append(np.asarray(image.convert("RGB"))[:, :, ::-1].copy())
        except (binascii.Error, OSError) as exc:
            raise ValueError("A camera frame could not be read. Please record again.") from exc
    return CameraAttempt(capture_id, expected_context, frames, duration)
