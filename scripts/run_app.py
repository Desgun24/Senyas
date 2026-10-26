from __future__ import annotations

import sys
from pathlib import Path

import mediapipe as _mediapipe  # Load Windows DLLs on the main thread.  # noqa: F401
from streamlit.web import cli


if __name__ == "__main__":
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    cli.main()
