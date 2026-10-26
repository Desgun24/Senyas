from __future__ import annotations

import sys
from pathlib import Path

import mediapipe as _mediapipe  # Load Windows DLLs on the main thread.  # noqa: F401
from streamlit.web import cli


if __name__ == "__main__":
    entrypoint = "feature_lab.py" if "--feature-lab" in sys.argv else "app.py"
    if "--feature-lab" in sys.argv:
        sys.argv.remove("--feature-lab")
    app_path = Path(__file__).resolve().parents[1] / entrypoint
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    cli.main()
