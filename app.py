from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mediapipe as _mediapipe  # Preload before pandas to avoid a Windows DLL conflict.  # noqa: F401
import pandas as pd
import plotly.express as px
import streamlit as st

from senyas.config import PROJECT_ROOT, artifact_dir, dataset_dir
from senyas.curriculum import check_meaning, find_demo_video, load_lessons
from senyas.features import LandmarkExtractor
from senyas.model import decision_trace, friendly_feature_name, load_artifact, predict_vector
from senyas.progress import load_progress, record_attempt, save_progress


st.set_page_config(page_title="Senyas · Learn FSL", page_icon="🤟", layout="wide")

st.markdown(
    """
    <style>
      :root { --ink:#18323b; --teal:#0f766e; --mint:#e6f4f1; --coral:#ef8354; --paper:#fbfaf6; }
      .stApp { background: linear-gradient(180deg, #fbfaf6 0%, #f2f8f6 100%); color: var(--ink); }
      [data-testid="stSidebar"] { background: #163a43; }
      [data-testid="stSidebar"] * { color: #f7fbfa !important; }
      .hero { padding: 2.2rem; border-radius: 24px; background: linear-gradient(130deg,#143b43,#0f766e); color:white; margin-bottom:1.2rem; }
      .hero h1 { font-size:3rem; margin:0 0 .35rem 0; letter-spacing:-.04em; color:#ffffff !important; }
      .hero p { font-size:1.08rem; max-width:740px; opacity:.9; margin:0; }
      .eyebrow { text-transform:uppercase; letter-spacing:.14em; font-size:.75rem; font-weight:700; color:#f6c6ac; }
      .card { background:white; border:1px solid #dbe7e4; border-radius:18px; padding:1.2rem; min-height:150px; box-shadow:0 8px 28px rgba(20,59,67,.06); }
      .step { border-left:4px solid var(--coral); padding:.15rem 0 .15rem 1rem; margin:.65rem 0; }
      .feedback-ok { background:#e7f8ef; border:1px solid #85d5aa; border-radius:16px; padding:1rem; }
      .feedback-warn { background:#fff6dd; border:1px solid #e7c463; border-radius:16px; padding:1rem; }
      .feedback-no { background:#fff0ea; border:1px solid #efad91; border-radius:16px; padding:1rem; }
      div[data-testid="stMetric"] { background:white; border:1px solid #dbe7e4; padding:1rem; border-radius:16px; }
      .stButton > button { border-radius:999px; border:0; background:#0f766e; color:white; font-weight:700; padding:.55rem 1.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


DATASET = dataset_dir()
ARTIFACTS = artifact_dir()
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
PROGRESS_PATH = PROJECT_ROOT / "data" / "progress.json"
MODEL_PATH = ARTIFACTS / "hybrid_recognizer.joblib"
EVALUATION_PATH = ARTIFACTS / "evaluation.json"


@st.cache_data
def lessons_table() -> pd.DataFrame:
    return load_lessons()


@st.cache_data
def manifest_table() -> pd.DataFrame | None:
    return pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else None


@st.cache_resource
def recognition_artifact() -> dict | None:
    return load_artifact(MODEL_PATH) if MODEL_PATH.exists() else None


def demo_video(class_id: int) -> Path | None:
    return find_demo_video(DATASET, class_id)


def lesson_picker(lessons: pd.DataFrame, key: str, reveal_label: bool = True) -> pd.Series:
    options = lessons.index.tolist()
    selected = st.selectbox(
        "Choose a sign",
        options,
        format_func=lambda index: (
            f"{lessons.loc[index, 'label'].title()} · {lessons.loc[index, 'filipino'].title()}"
            if reveal_label
            else f"Practice sign {int(lessons.loc[index, 'beginner_order']):02d}"
        ),
        key=key,
    )
    return lessons.loc[selected]


def show_video(path: Path | None, empty_message: str = "Demonstration video is not available.") -> None:
    if path and path.exists():
        st.video(str(path))
    else:
        st.info(empty_message + " Check the dataset path in `.env` or place `dataset/` beside the app.")


def page_home(lessons: pd.DataFrame) -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Beginner Filipino Sign Language</div>
          <h1>Learn with your hands.<br/>Understand the feedback.</h1>
          <p>Senyas pairs short visual lessons with two focused activities: reproduce a sign for recognition feedback, then recall its Filipino or English meaning.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    manifest = manifest_table()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Starter lessons", len(lessons))
    metric_columns[1].metric("Dataset signs", int(manifest["id_label"].nunique()) if manifest is not None else "105")
    metric_columns[2].metric("Video clips", len(manifest) if manifest is not None else "2,130")
    metric_columns[3].metric("Recognition model", "Ready" if MODEL_PATH.exists() else "Train first")

    st.subheader("A simple learning loop")
    columns = st.columns(3)
    cards = (
        ("01 · Watch", "Replay a reviewed FSL demonstration and read its Filipino and English meanings."),
        ("02 · Practice", "Upload a signing attempt and receive matching, different, or uncertain feedback."),
        ("03 · Understand", "Open the decision trace to see which hand shape, position, or motion feature mattered."),
    )
    for column, (title, body) in zip(columns, cards):
        column.markdown(f'<div class="card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)

    st.caption("Scope: selected, isolated beginner FSL signs. This prototype is a learning aid, not a replacement for instruction from the Deaf community or qualified FSL educators.")


def page_lessons(lessons: pd.DataFrame) -> None:
    st.title("Visual lessons")
    st.write("Study one sign at a time. Replay the clip and focus on hand shape, placement, orientation, and movement.")
    lesson = lesson_picker(lessons, "lesson_picker")
    left, right = st.columns([1.45, 1])
    with left:
        show_video(demo_video(int(lesson["id"])))
    with right:
        st.caption(str(lesson["category"]).title())
        st.header(str(lesson["label"]).title())
        st.subheader(str(lesson["filipino"]).title())
        st.markdown(
            """
            <div class="step"><strong>Observe</strong><br/>Where are both hands relative to the shoulders?</div>
            <div class="step"><strong>Notice</strong><br/>Which fingers bend or extend?</div>
            <div class="step"><strong>Replay</strong><br/>Follow the direction and speed of the motion.</div>
            """,
            unsafe_allow_html=True,
        )
        progress = load_progress(PROGRESS_PATH)
        if st.button("Mark lesson complete", use_container_width=True):
            lesson_id = int(lesson["id"])
            completed = set(progress.get("completed_lessons", []))
            completed.add(lesson_id)
            progress["completed_lessons"] = sorted(completed)
            save_progress(PROGRESS_PATH, progress)
            st.success("Lesson added to your progress.")


def render_trace(trace: dict) -> None:
    explainer = trace.get("explainer_prediction")
    if explainer:
        agreement = "agrees" if trace.get("agrees_with_primary") else "does not agree"
        st.caption(
            f"Transparent Decision Tree vote: {str(explainer['label']).title()} — "
            f"{agreement} with the Extra Trees prediction."
        )
    rules = trace["rules"]
    if not rules:
        st.write("The model reached a leaf without an additional split.")
        return
    for index, rule in enumerate(rules, start=1):
        st.markdown(
            f"**{index}. {friendly_feature_name(rule['feature']).title()}**  \n"
            f"Observed `{rule['value']:.3f}` {rule['operator']} learned cutoff `{rule['threshold']:.3f}`"
        )


def page_copy_sign(lessons: pd.DataFrame) -> None:
    st.title("Copy the sign")
    st.write("Watch the target, record a short attempt, then upload the clip for recognition feedback.")
    lesson = lesson_picker(lessons, "copy_picker")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("1 · Watch the target")
        show_video(demo_video(int(lesson["id"])))
    with right:
        st.subheader("2 · Upload your attempt")
        upload = st.file_uploader("MOV or MP4, ideally 2–5 seconds", type=["mov", "mp4", "avi", "mkv"], key="copy_upload")
        if upload:
            st.video(upload)

    artifact = recognition_artifact()
    if artifact is None:
        st.info("The recognition model has not been trained yet. Run the three pipeline tasks in VS Code, then refresh this page.")
        return
    if upload and st.button("Check my sign", use_container_width=True):
        suffix = Path(upload.name).suffix or ".mp4"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as stream:
                stream.write(upload.getbuffer())
                temporary_path = Path(stream.name)
            with st.spinner("Finding hand landmarks and motion patterns…"):
                with LandmarkExtractor(max_frames=32) as extractor:
                    vector, extraction = extractor.extract_video(temporary_path)
                result = predict_vector(artifact, vector)
                trace = decision_trace(artifact, vector)

            matches_target = result["class_id"] == int(lesson["id"])
            if result["uncertain"]:
                style, title = "feedback-warn", "Uncertain — please try again"
                guidance = "Keep both hands in frame, face the camera, and use even lighting."
                outcome = "uncertain"
            elif matches_target:
                style, title = "feedback-ok", "Matching sign"
                guidance = "Your attempt matched the target. Try once more for consistency."
                outcome = "matching"
            else:
                style, title = "feedback-no", "Different sign detected"
                guidance = f"The model read this as **{result['label'].title()}**. Rewatch the target and compare hand placement and movement."
                outcome = "different"
            st.markdown(
                f'<div class="{style}"><h3>{title}</h3><p>Confidence: {result["confidence"]:.0%}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown(guidance)
            st.caption(
                f"Feature extraction {extraction.extraction_ms:.0f} ms · model decision {result['prediction_ms']:.2f} ms · "
                f"hands found in {extraction.hand_detection_rate:.0%} of sampled frames"
            )
            with st.expander("Show the explainable Decision Tree baseline trace"):
                render_trace(trace)
                st.caption(
                    "Primary recognition uses the higher-accuracy Extra Trees ensemble. "
                    "These exact rules belong to the separate transparent Decision Tree baseline."
                )

            progress = load_progress(PROGRESS_PATH)
            progress = record_attempt(
                progress,
                "copy",
                {
                    "target_id": int(lesson["id"]),
                    "predicted_id": result["class_id"],
                    "confidence": result["confidence"],
                    "outcome": outcome,
                },
            )
            save_progress(PROGRESS_PATH, progress)
        except Exception as exc:
            st.error(f"This clip could not be analyzed: {exc}")
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)


def page_type_meaning(lessons: pd.DataFrame) -> None:
    st.title("Type the meaning")
    st.write("Watch the sign without its title, then answer in Filipino or English.")
    lesson = lesson_picker(lessons, "meaning_picker", reveal_label=False)
    show_video(demo_video(int(lesson["id"])))
    with st.form("meaning_form"):
        answer = st.text_input("What does this sign mean?", placeholder="Type a Filipino or English answer")
        submitted = st.form_submit_button("Check answer", use_container_width=True)
    if submitted:
        correct = check_meaning(answer, lesson)
        if correct:
            st.success(f"Correct — {lesson['label'].title()} / {lesson['filipino'].title()}")
        else:
            st.error(f"Not quite. Accepted meaning: {lesson['label'].title()} / {lesson['filipino'].title()}")
        st.caption("Accepted alternatives: " + ", ".join(lesson["accepted_answers"]))
        progress = load_progress(PROGRESS_PATH)
        progress = record_attempt(
            progress,
            "meaning",
            {"target_id": int(lesson["id"]), "answer": answer, "correct": bool(correct)},
        )
        save_progress(PROGRESS_PATH, progress)


def page_pipeline() -> None:
    st.title("From video clips to an explainable decision")
    st.write("This page documents the exact preprocessing and feature-engineering path used by the prototype.")
    steps = (
        ("1", "Validate & label", "Combine the source CSVs, verify every clip, standardize paths, and infer the FSL105 signer/session ID from the filename."),
        ("2", "Split by signer", "Search for the closest 70/15/15 sample ratio while keeping every signer in exactly one split."),
        ("3", "Extract landmarks", "Sample up to 32 frames; detect 21 landmarks for each hand plus shoulder references with MediaPipe."),
        ("4", "Normalize", "Express hand shape relative to the wrist and palm size, and hand position relative to the shoulders."),
        ("5", "Engineer motion", "Summarize mean, spread, range, start/end change, and frame-to-frame movement for every descriptor."),
        ("6", "Train & explain", "Tune a class-balanced decision tree on validation data, lock it, test once, and expose its actual rule path."),
    )
    for number, title, body in steps:
        st.markdown(f'<div class="step"><strong>{number} · {title}</strong><br/>{body}</div>', unsafe_allow_html=True)

    manifest = manifest_table()
    if manifest is None:
        st.info("Run `scripts/prepare_dataset.py` to generate the split manifest and see its measured distribution here.")
        return
    split_counts = manifest.groupby("split").agg(samples=("vid_path", "size"), signers=("signer_id", "nunique"), classes=("id_label", "nunique")).reset_index()
    split_counts["ratio"] = split_counts["samples"] / len(manifest)
    st.subheader("Measured split")
    st.dataframe(split_counts, hide_index=True, use_container_width=True)
    fig = px.bar(split_counts, x="split", y="samples", color="split", text="samples", color_discrete_sequence=["#0f766e", "#ef8354", "#6f8f96"])
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0), height=330)
    st.plotly_chart(fig, use_container_width=True)


def page_evaluation() -> None:
    st.title("Evaluation")
    if not EVALUATION_PATH.exists():
        st.info("No evaluation report yet. Train the model to populate accuracy, robustness, decision-trace, and latency results.")
        return
    report = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    test = report["test"]
    columns = st.columns(4)
    columns[0].metric("Test accuracy", f"{test['accuracy']:.1%}")
    columns[1].metric("Macro F1", f"{test['macro_f1']:.1%}")
    columns[2].metric("Macro precision", f"{test['macro_precision']:.1%}")
    columns[3].metric("Macro recall", f"{test['macro_recall']:.1%}")

    baseline = report.get("baseline_test")
    comparison = report.get("comparison", {})
    if baseline:
        baseline_columns = st.columns(3)
        baseline_columns[0].metric("Decision Tree baseline", f"{baseline['accuracy']:.1%}")
        baseline_columns[1].metric("Extra Trees accuracy", f"{test['accuracy']:.1%}")
        baseline_columns[2].metric(
            "Accuracy gain",
            f"{comparison.get('test_accuracy_gain_percentage_points', 0.0):+.1f} points",
        )

    tabs = st.tabs(["Recognition", "Robustness", "Explainability", "Efficiency & usability"])
    with tabs[0]:
        st.caption("Confusion matrix for the primary Extra Trees recognizer on the untouched test split.")
        labels = test["labels"]
        matrix = test["confusion_matrix"]
        figure = px.imshow(matrix, x=labels, y=labels, color_continuous_scale="Teal", labels={"x": "Predicted class", "y": "True class", "color": "Clips"})
        figure.update_layout(height=650)
        st.plotly_chart(figure, use_container_width=True)
        st.caption("A 105-class confusion matrix is dense; use hover to inspect individual class pairs.")
    with tabs[1]:
        robustness = report.get("robustness", {})
        if robustness:
            for group_name, rows in robustness.items():
                st.subheader(group_name.replace("_", " ").title())
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.warning("Condition-specific results are waiting for lighting, background, distance, and speed tags. Use `config/conditions_template.csv`.")
    with tabs[2]:
        explanation = report["explainability"]
        st.metric("Decision traces verified", f"{explanation['valid_traces']} / {explanation['traces_checked']}")
        st.write(f"Tree depth: **{explanation['tree_depth']}** · Nodes: **{explanation['tree_nodes']}**")
        st.write(
            "Extra Trees performs the primary recognition. The displayed trace is the exact path "
            "through the retained Decision Tree baseline, and is labeled when it disagrees with the ensemble."
        )
        st.write(f"Trace agreement with primary predictions: **{explanation.get('trace_primary_agreement', 0.0):.1%}**")
    with tabs[3]:
        efficiency = report["efficiency"]
        prediction = efficiency["prediction_ms"]
        st.metric("Mean classifier latency", f"{prediction['mean']:.2f} ms")
        extraction = efficiency.get("feature_extraction_ms_mean")
        if extraction is not None:
            st.metric("Mean end-to-end feature extraction", f"{extraction:.0f} ms")
        st.write("Usability checklist: demonstration playback, bilingual answers, retry flow, uncertainty feedback, progress recording, and optional rule explanations.")


def page_progress(lessons: pd.DataFrame) -> None:
    st.title("Your progress")
    progress = load_progress(PROGRESS_PATH)
    completed = len(set(progress.get("completed_lessons", [])))
    copy_attempts = progress.get("practice_attempts", [])
    meaning_attempts = progress.get("meaning_attempts", [])
    columns = st.columns(3)
    columns[0].metric("Lessons complete", f"{completed} / {len(lessons)}")
    columns[1].metric("Copy attempts", len(copy_attempts))
    columns[2].metric("Meaning attempts", len(meaning_attempts))
    if meaning_attempts:
        correct = sum(bool(item.get("correct")) for item in meaning_attempts)
        st.progress(correct / len(meaning_attempts), text=f"Meaning accuracy: {correct}/{len(meaning_attempts)}")
    if copy_attempts:
        st.subheader("Recent recognition practice")
        st.dataframe(pd.DataFrame(copy_attempts[-10:]).iloc[::-1], hide_index=True, use_container_width=True)


lessons = lessons_table()
st.sidebar.markdown("## 🤟 Senyas")
st.sidebar.caption("Beginner FSL learning prototype")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Lessons", "Copy the Sign", "Type the Meaning", "Data Pipeline", "Evaluation", "Progress"],
)
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {'found' if DATASET.exists() else 'not found'}")
st.sidebar.caption(f"Model: {'ready' if MODEL_PATH.exists() else 'not trained'}")

if page == "Home":
    page_home(lessons)
elif page == "Lessons":
    page_lessons(lessons)
elif page == "Copy the Sign":
    page_copy_sign(lessons)
elif page == "Type the Meaning":
    page_type_meaning(lessons)
elif page == "Data Pipeline":
    page_pipeline()
elif page == "Evaluation":
    page_evaluation()
else:
    page_progress(lessons)
