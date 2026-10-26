# Senyas

Senyas is a Filipino Sign Language learning app with **105 words across 10 category levels**, silent lesson videos, live camera practice, quizzes, and saved progress. Lessons show English words with Filipino equivalents and follow **Watch → Copy → Replay → Next**.

**Project credits:** Jeff Barrientos, Hannah Dimaculangan, and Trixie Mendoza.

## Install on Windows

[Download Senyas 1.1.0 for Windows x64](https://github.com/Desgun24/Senyas/releases/download/v1.1.0/Senyas-1.1.0-Windows-x64.zip) · [Release notes and checksums](https://github.com/Desgun24/Senyas/releases/tag/v1.1.0)

1. Get `Senyas-1.1.0-Windows-x64.zip` and extract it into a permanent writable folder, such as `Documents/Senyas`.
2. Double-click **Install Senyas.cmd** inside the extracted folder.
3. Wait for setup to download Python and the libraries, verify the app, and open it in your browser.
4. Allow camera access when starting a signing activity.

Use the **Senyas** Desktop or Start menu shortcut, or **Start Senyas.cmd**, afterward. Keep the extracted folder in place because the shortcuts use it. Re-running the installer repairs missing packages without resetting progress.

The installer targets **Windows x64** and needs internet for the first setup. Edge or Chrome and a working webcam are recommended. Python, Git, and the original dataset are not prerequisites. Setup uses a private `.runtime` folder without changing the system Python installation or PATH. Libraries are downloaded from their distribution services and verified against pinned SHA-256 hashes. Once setup finishes, lesson playback and recognition work locally without internet.

## Learn and practice

| Page | Activity |
| --- | --- |
| Home | Continue learning, open quizzes, or take the final exam. |
| Lessons | Watch a labeled video, copy the sign using the camera, replay, and move to the next word. Each category ends with a quiz. |
| Quiz | Select one or more categories from a checkbox grid. Questions mix signing a word and typing the meaning of a video. |
| Final Exam | Practice all 105 words, randomly divided into 52 and 53 questions of the two types. |
| Progress | Resume words and levels, review completed material, and inspect assessment results. |

An assessment uses every selected word once. Its question order and submitted answers persist across refreshes. Category completion requires finishing the words and category quiz; the score is recorded separately.

The sidebar **Settings** popup selects Decision Tree or Extra Trees. Decision Tree is the interpretable primary model and the default. Developer mode describes its actual decision path for the latest applicable camera attempt, including whether a branch used an unavailable measurement's training-set replacement. Extra Trees is the data-driven ensemble baseline and has no single decision path displayed. Explanations remain hidden during an unfinished assessment to avoid revealing answers.

## Camera and local data

Camera practice captures a four-second attempt after a countdown. The preview is live; recognition runs after capture. Microphone access is not requested. If an embedded browser blocks the camera, use Edge or Chrome with the local address shown by Senyas.

Progress is saved in `data/progress.json` inside the app folder. Back up this file before moving to a newer release, then place it in the new release's `data` folder to keep your history. Setup and release archives contain no personal progress. Recordings are processed in memory; prediction metadata and measured features for the latest explanation are saved locally without video files.

The installed launcher serves only on `127.0.0.1`, normally using port 8765. The local service may stay running after the browser tab closes, allowing a shortcut to reopen it. Restarting Windows stops that service. Startup diagnostics are in `.runtime/server.log`.

## Run from source

The Windows installer also works from a source checkout. For development with an existing **Python 3.11 x64** installation:

```powershell
py -3.11 -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
& .venv/Scripts/python.exe scripts/run_app.py
```

The launcher imports MediaPipe on the main thread before Streamlit starts, which avoids the Windows native-library startup issue. The VS Code run task uses this launcher. The application needs `artifacts/frozen/`, `config/lessons.csv`, and `assets/lessons/`, all included in the release package.

## Content and features

`config/labels.csv` preserves the source class IDs and categories. `config/lessons.csv` adds display labels, Filipino equivalents, accepted answers, ordering, and bundled video paths. `assets/lessons/` contains one silent H.264 MP4 per class and a manifest of original relative clip names and output checksums.

Recognition measures 278 geometric features and 50 motion features per frame. It excludes 14 experimental measurements, leaving 314 supported measurements. Five whole-sign summaries plus a median and availability summary in each of three phases produce 314 × 11 = 3,454 candidates. Training-only filtering removes 62 constant, missing, or rarely observed columns, leaving **3,392 model inputs**. Leading and trailing frames with no tracked hands are trimmed; internal gaps remain missing, and the saved training medians replace unavailable summaries.

The frozen Decision Tree uses 135 distinct input columns and at most 10 branch decisions per prediction. Both models were fitted on 1,494 training clips, with no validation refit, and share the same engineered features. Validation accuracy was **84.69%** for Decision Tree and **98.13%** for Extra Trees on 320 accepted clips; one additional validation recording had no usable hands. On the fixed held-out test set, Decision Tree scored **262/315 (83.17%)**, and Extra Trees scored **311/315 (98.73%)**. Test macro F1 was **82.78%** and **98.69%**, respectively, with no rejected test clips. Portable predictions, confusion matrices, verification receipts, and model hashes are in `artifacts/frozen/`.

Weights and preprocessing are unchanged after the held-out test. The runtime uses full-complexity MediaPipe tracking, a 480-pixel maximum image side, and timestamp-aware motion. Camera capture targets 30 FPS; actual capture and processing rates depend on hardware. Tree thresholds are inspectable; the upstream landmark detector is a learned model. Proximity is not proof of touch, estimated depth does not establish palm-facing direction, and facial expressions are not engineered. Dataset scores do not establish accuracy for every webcam environment, and vocabulary translations may require instructor review.

## Verification and research

The separate camera-based feature lab can inspect ordered measurements and save labelled hand-shape clips:

```powershell
& .venv/Scripts/python.exe scripts/run_app.py --feature-lab --server.address 127.0.0.1 --server.port 8502 --server.headless true
```

The lab implements the feature schema and experimental DTW alignment engine; it is a measurement tool rather than the app's trained recognizer. The learning app uses the frozen three-phase models. See [semantic feature development](docs/SEMANTIC_FEATURE_DEVELOPMENT.md) for supported measurements, deferred labels, clip annotation, and measured frame rates.

```powershell
& .venv/Scripts/python.exe -c "import mediapipe; import pytest; raise SystemExit(pytest.main(['-q']))"
& .venv/Scripts/python.exe scripts/build_release.py
```

The release builder writes a clean ZIP and `SHA256SUMS.txt` under `dist/`; installed libraries, raw datasets, caches, and learner history are excluded. Dataset preparation, extraction, training, and evaluation remain separate research scripts. Running the learning app requires none of those stages.

See [camera controls](docs/LIVE_CAMERA.md) and [current release notes](docs/RELEASE_NOTES_1.1.0.md). The [original explanation audit](docs/EXPLANATION_AUDIT.md) and [original validation checks](docs/LEARNING_APP_VALIDATION.md) document version 1.0.0. Third-party components retain their own notices, documented in [third-party notices](THIRD_PARTY_NOTICES.md).
