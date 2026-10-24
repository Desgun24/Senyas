# Senyas

Senyas is a beginner Filipino Sign Language (FSL) learning prototype built around the FSL105 video dataset. It combines replayable lessons, bilingual vocabulary practice, video-based sign recognition, uncertainty feedback, an accuracy-focused Extra Trees ensemble, and exact rule traces from a retained Decision Tree baseline.

The project follows the presentation requirements:

- preprocess an unstructured collection of sign-language video clips;
- extract landmark, hand-shape, position, and motion features;
- create a 70/15/15 train/validation/test split;
- train an Extra Trees recognizer while retaining an explainable Decision Tree baseline;
- connect recognition to lessons and practice activities;
- evaluate accuracy, robustness, decision traces, efficiency, and usability functions; and
- provide a basic learner-facing UI/UX.

## Important data decision

The supplied `train.csv` and `test.csv` are not reused as the final experimental split. FSL105 clip names follow `clips/<class>/<signer-or-session>.MOV`. Senyas keeps that filename-derived signer/session in exactly one split, then searches for the closest sample-level 70/15/15 allocation. This prevents clips from the same person/session appearing in both training and testing.

Confirm this filename interpretation against the dataset's original documentation before reporting the result as confirmed unseen-*signer* performance. Until confirmed, the UI and report use the more careful term "signer/session."

## Open and run in VS Code

1. Open the project folder in VS Code.
2. Keep the existing dataset beside this project as `dataset/`, or copy `.env.example` to `.env` and set `SENYAS_DATASET_DIR`.
3. Open **Terminal > Run Task > Senyas: setup**. The task uses Python on PATH or the Codex bundled runtime on this computer. If neither is available, install Python 3.11 or 3.12 and reopen VS Code.
4. Run **Senyas: prepare manifest**.
5. Run **Senyas: extract features**. Processing all 2,130 videos can take a while because MediaPipe examines up to 32 frames per clip.
6. Run **Senyas: train and evaluate**.
7. Run **Senyas: run app**, then open the local Streamlit address shown in the terminal.

Equivalent terminal commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.venv\Scripts\python.exe scripts/prepare_dataset.py --dataset dataset
.venv\Scripts\python.exe scripts/extract_features.py
.venv\Scripts\python.exe scripts/train_model.py
.venv\Scripts\streamlit.exe run app.py
```

To prepare the manifest from the folder supplied with this project without moving the dataset:

```powershell
.venv\Scripts\python.exe scripts/prepare_dataset.py --dataset "C:\Users\Hannah Dimaculangan\OneDrive\Documents\Foregin_Sign_Language\dataset"
```

## Pipeline outputs

| Output | Purpose |
|---|---|
| `data/processed/manifest.csv` | Validated video metadata and leakage-safe split assignment |
| `data/processed/features.npz` | Compressed fixed-length feature matrix |
| `data/processed/feature_index.csv` | Per-video extraction status, hand detection rate, and timing |
| `artifacts/hybrid_recognizer.joblib` | Extra Trees recognizer, Decision Tree explainer, feature names, labels, and validation-selected uncertainty threshold |
| `artifacts/evaluation.json` | Validation/test metrics, robustness groups, latency, and trace checks |
| `artifacts/test_predictions.csv` | Auditable clip-level test predictions |

Generated data and model files are ignored by Git.

## What the features mean

For each sampled frame, the extractor records:

- 21 three-dimensional landmarks for each hand;
- wrist position in the image and relative to the shoulder midpoint;
- palm scale for size normalization;
- five finger-bend angles and fingertip-extension distances per hand;
- shoulder width, hand presence, distance between hands, and vertical hand offset.

Coordinates describing hand shape are centered at the wrist and divided by palm size. Position features are divided by shoulder width. Each frame descriptor is converted to a fixed vector with mean, standard deviation, minimum, maximum, range, start, end, change, mean movement, and maximum movement. Extra Trees provides the primary recognition result, while the retained Decision Tree provides an exact learner-facing rule trace that is explicitly labeled as a baseline explanation.

## Evaluation design

- **Recognition:** accuracy, macro precision, macro recall, macro F1, per-class report, and confusion matrix.
- **Generalization:** the held-out test split contains signer/session groups not used to fit or tune the tree.
- **Robustness:** metrics can be grouped by signer/session and, when tagged, lighting, background, distance, and signing speed.
- **Explainability:** up to 50 Decision Tree baseline traces are replayed programmatically to confirm every displayed comparison agrees with the branch taken; agreement between baseline and Extra Trees predictions is reported separately.
- **Efficiency:** feature-extraction time and mean/p95/max classifier latency.
- **Usability functions:** lesson playback, bilingual answers, retries, uncertain results, progress tracking, and optional explanations.

Condition-specific testing needs human-reviewed tags. Copy `config/conditions_template.csv`, add one row per clip, and pass it to `prepare_dataset.py` with `--conditions`.

## Testing

```powershell
.venv\Scripts\python.exe -m pytest
```

For a fast extraction smoke test before processing the full dataset:

```powershell
.venv\Scripts\python.exe scripts/extract_features.py --limit 5
```

## Responsible use and review

The software is a research prototype and learning aid, not an authoritative interpreter. FSL is a natural language with regional and community variation. Before a formal study or deployment:

- have the selected vocabulary, Filipino meanings, and demonstrations reviewed by qualified FSL signers or educators;
- obtain consent for any newly recorded participants;
- document signer demographics and recording conditions without exposing identities;
- report the filename-derived grouping assumption clearly;
- do not treat a low-confidence or incorrect model result as a judgment of a learner's signing ability; and
- test accessibility with Deaf and hard-of-hearing users.

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the report-ready methodology and [docs/DATASET_CARD.md](docs/DATASET_CARD.md) for the current data audit.
