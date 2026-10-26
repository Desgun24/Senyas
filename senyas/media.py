"""Bundled silent video player with a native replay control."""
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

_player = components.declare_component("senyas_lesson_video", path=str(Path(__file__).parent / "components" / "lesson_video"))


@st.cache_data
def video_data(path: str) -> str:
    return "data:video/mp4;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def play_lesson(path: Path, context: str) -> None:
    _player(source=video_data(str(path)), context=context, key="lesson_video", default=None)
