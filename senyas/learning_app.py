"""Senyas learner experience: portable content, guided practice and assessments."""
from __future__ import annotations

import html
import os
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from .assessments import create_assessment, submit_answer
from .camera import decode_camera_attempt, live_camera
from .config import PROJECT_ROOT, artifact_dir
from .curriculum import bundled_video, categories, category_title, check_meaning, load_lessons
from .explanations import DESCRIPTION_VERSION, TRACKING_FEATURES, describe_trace
from .features import LandmarkExtractor
from .media import play_lesson
from .model import decision_trace, load_artifact, predict_vector, select_recognizer
from .progress import category_status, lesson_stage, load_progress, now, record_attempt, save_progress, set_lesson_stage

PAGES = ["Home", "Lessons", "Quiz", "Final Exam", "Progress"]
MODELS = ["Decision Tree", "Extra Trees"]


def progress_path() -> Path:
    return Path(os.environ.get("SENYAS_PROGRESS_PATH", PROJECT_ROOT / "data" / "progress.json"))


def commit(progress: dict) -> None:
    save_progress(progress_path(), progress)


@st.cache_data
def curriculum():
    return load_lessons()


@st.cache_resource
def recognizer(model_name: str):
    path = artifact_dir() / "hybrid_recognizer.joblib"
    return select_recognizer(load_artifact(path), model_name) if path.exists() else None


def esc(value) -> str:
    return html.escape(str(value))


def word(lesson) -> str:
    return str(lesson["display_label"]).capitalize()


def go(page: str, **params) -> None:
    st.query_params.clear()
    st.query_params["page"] = page
    for key, value in params.items():
        st.query_params[key] = str(value)
    st.rerun()


def open_lesson(progress: dict, class_id: int) -> None:
    progress["current_lesson"] = int(class_id)
    if str(class_id) not in progress["lesson_states"]:
        set_lesson_stage(progress, int(class_id), "watch")
    commit(progress)
    go("Lessons", word=class_id)


@st.dialog("Settings")
def settings_dialog() -> None:
    progress = load_progress(progress_path())
    settings = progress["settings"]
    selected = st.selectbox("Recognition model", MODELS, index=MODELS.index(settings["model"]), key="settings_model")
    developer = st.toggle("Developer mode", value=settings["developer_mode"], key="settings_developer")
    st.caption("Developer mode shows what the model noticed below the page content.")
    if progress.get("active_assessment"):
        st.info("The current assessment keeps the model it started with. Changes apply to your next assessment and lesson practice.")
    if st.button("Save settings", type="primary", use_container_width=True):
        progress = load_progress(progress_path())
        progress["settings"] = {"model": selected, "developer_mode": developer}
        commit(progress)
        st.rerun()


def navigation(page: str) -> None:
    with st.sidebar:
        st.markdown('<div class="brand">🤟 Senyas</div>', unsafe_allow_html=True)
        st.caption("Learn Filipino Sign Language")
        st.divider()
        for title in PAGES:
            if st.button(title, key=f"nav_{title}", type="primary" if page == title else "secondary", use_container_width=True):
                go(title)
        st.divider()
        settings_clicked = st.button("⚙ Settings", help="Settings", key="open_settings", use_container_width=True)
    if settings_clicked:
        settings_dialog()


def show_video(lesson, context: str) -> None:
    path = bundled_video(lesson)
    if path.exists():
        play_lesson(path, context)
    else:
        st.error("This lesson's video is missing from the installed app.")


def begin_assessment(progress: dict, lessons, chosen: list[str], kind: str) -> None:
    if not progress.get("active_assessment"):
        progress["active_assessment"] = create_assessment(lessons, chosen, kind, progress["settings"]["model"])
        commit(progress)
    target = "Final Exam" if progress["active_assessment"]["kind"] == "final" else "Quiz"
    go(target)


def resume_id(progress: dict, lessons) -> int:
    current = int(progress["current_lesson"])
    if current in lessons["id"].tolist() and lesson_stage(progress, current) != "complete":
        return current
    return next((int(i) for i in lessons["id"] if int(i) not in progress["completed_lessons"]), current)


def home(progress: dict, lessons) -> None:
    st.markdown('''<section class="hero"><div class="eyebrow">Filipino Sign Language</div>
    <h1>A new word.<br>A new way to connect.</h1>
    <p>Watch, sign, and build your confidence one topic at a time.</p></section>''', unsafe_allow_html=True)
    metrics = st.columns(3)
    metrics[0].metric("Words to discover", len(lessons))
    metrics[1].metric("Category levels", len(categories(lessons)))
    metrics[2].metric("Words completed", len(set(progress["completed_lessons"]) & set(lessons["id"])))
    cols = st.columns(3)
    cards = [
        ("Lessons", "Watch a demonstration, practise with your camera, then watch it again.", "Continue learning"),
        ("Quiz", "Choose your topics and practise signing and remembering their meanings.", "Choose quiz topics"),
        ("Final Exam", "Bring all the topics together in a complete mixed assessment.", "Open final exam"),
    ]
    for col, (title, description, action) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card"><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)
            if st.button(action, key=f"home_{title}", use_container_width=True, type="primary" if title == "Lessons" else "secondary"):
                open_lesson(progress, resume_id(progress, lessons)) if title == "Lessons" else go(title)
    active = progress.get("active_assessment")
    if active:
        st.info(f"You have a saved assessment: {len(active['answers'])} of {len(active['questions'])} answers submitted.")
        if st.button("Resume saved assessment"):
            go("Final Exam" if active["kind"] == "final" else "Quiz")
    st.caption("Learn at your own pace. You can replay demonstrations and revisit completed words at any time.")


def capture_sign(progress: dict, lesson, context: str, model_name: str, assessment_id=None) -> dict | None:
    artifact = recognizer(model_name)
    if artifact is None:
        st.error("The recognition model is missing from this installation.")
        return None
    payload = live_camera(context, st.session_state.get("camera_processed_capture_id"))
    if isinstance(payload, dict) and payload.get("context") == context and payload.get("capture_id") != st.session_state.get("camera_processed_capture_id"):
        st.session_state.camera_processed_capture_id = payload.get("capture_id")
        try:
            attempt = decode_camera_attempt(payload, context)
            with st.spinner("Checking your sign…"):
                with LandmarkExtractor(max_frames=32) as extractor:
                    vector, extraction = extractor.extract_frames(attempt.frames, attempt.duration_seconds)
                prediction = predict_vector(artifact, vector)
                trace = decision_trace(artifact, vector)
            if not all(rule["condition_holds"] for rule in trace["rules"]):
                raise ValueError("The model's explanation could not be verified. Please record again.")
            observations = {name: float(value) for name, value in zip(artifact["feature_names"], vector)
                            if name in {rule["feature"] for rule in trace["rules"]} | TRACKING_FEATURES}
            correct = prediction["class_id"] == int(lesson["id"]) and not prediction["uncertain"]
            result = {"context": context, "capture_id": attempt.capture_id, "model": model_name,
                      "prediction": prediction, "trace": trace, "extraction": asdict(extraction), "correct": correct}
            progress = record_attempt(progress, "copy", {
                "target_id": int(lesson["id"]), "model": model_name, "predicted_id": prediction["class_id"],
                "confidence": prediction["confidence"], "outcome": "uncertain" if prediction["uncertain"] else "matching" if correct else "different",
                "input_source": "live_camera", "capture_id": attempt.capture_id,
            })
            progress["last_diagnostic"] = {
                "context": context, "class_id": int(lesson["id"]), "model": model_name,
                "assessment_id": assessment_id, "trace": trace, "descriptions": describe_trace(trace, observations),
                "observations": observations, "description_version": DESCRIPTION_VERSION,
            }
            commit(progress)
            st.session_state.camera_last_result = result
        except Exception as exc:
            st.session_state.camera_last_result = {"context": context, "error": str(exc)}
        st.rerun()
    result = st.session_state.get("camera_last_result", {})
    if result.get("context") != context:
        return None
    if "error" in result:
        st.error(f"This camera attempt could not be analyzed: {result['error']}")
        st.caption("Keep your hands and shoulders in view, use even lighting, and record again.")
        return None
    return result


def lesson_page(progress: dict, lessons) -> None:
    st.title("Your next word")
    try:
        class_id = int(st.query_params.get("word", progress["current_lesson"]))
        lesson = lessons.loc[lessons["id"].eq(class_id)].iloc[0]
    except (ValueError, IndexError):
        class_id = resume_id(progress, lessons)
        lesson = lessons.loc[lessons["id"].eq(class_id)].iloc[0]
    group = lessons.loc[lessons["category"].eq(lesson["category"])]
    group_ids = group["id"].astype(int).tolist()
    level = categories(lessons).index(lesson["category"]) + 1
    completed = len(set(group_ids) & set(progress["completed_lessons"]))
    if st.button(f"Level {level} · {category_title(lesson['category'])} · {completed}/{len(group)} words · View progress ↗", use_container_width=True, key="lesson_progress_link"):
        go("Progress")
    st.progress(completed / len(group))
    stage = lesson_stage(progress, class_id)
    st.markdown(f'<div class="phase">Learn → Quiz &nbsp; · &nbsp; Word {group_ids.index(class_id) + 1} of {len(group)} &nbsp; · &nbsp; {dict(watch="Watch",copy="Copy",replay="Replay",complete="Review")[stage]}</div>', unsafe_allow_html=True)
    st.header(word(lesson))
    st.subheader(str(lesson["filipino"]).capitalize())
    if stage in {"watch", "complete"}:
        st.write("Watch the hand shape and movement. Replay as often as you need.")
        show_video(lesson, f"learn:{class_id}:{stage}")
        if st.button("Next · Copy the sign", type="primary", use_container_width=True):
            set_lesson_stage(progress, class_id, "copy")
            commit(progress)
            st.rerun()
    elif stage == "copy":
        st.write("Start your camera, wait for the countdown, and copy this sign.")
        context = f"lesson:{class_id}:{progress['settings']['model']}"
        result = capture_sign(progress, lesson, context, progress["settings"]["model"])
        if result:
            if result["prediction"]["uncertain"]:
                st.warning("The model is unsure. Try again or watch the demonstration once more.")
            elif result["correct"]:
                st.success("The model matched your sign to this word.")
            else:
                predicted = lessons.loc[lessons["id"].eq(result["prediction"]["class_id"])].iloc[0]
                st.info(f"The model read this as {word(predicted)}. You can retry or compare it with the replay.")
        cols = st.columns(3)
        if cols[0].button("Replay demonstration", use_container_width=True):
            set_lesson_stage(progress, class_id, "watch")
            commit(progress)
            st.rerun()
        if cols[1].button("Retry", disabled=result is None, use_container_width=True):
            st.session_state.pop("camera_last_result", None)
            st.rerun()
        if cols[2].button("Next · Replay", disabled=result is None, type="primary", use_container_width=True):
            set_lesson_stage(progress, class_id, "replay")
            commit(progress)
            st.rerun()
    else:
        st.write("Watch it again and compare the movement with your attempt.")
        show_video(lesson, f"replay:{class_id}")
        last = class_id == group_ids[-1]
        remaining = [i for i in group_ids if i != class_id and i not in progress["completed_lessons"]]
        start_quiz = last and not remaining
        if st.button("Next · Category quiz" if start_quiz else "Next word", type="primary", use_container_width=True):
            set_lesson_stage(progress, class_id, "complete")
            commit(progress)
            if start_quiz:
                begin_assessment(progress, lessons, [lesson["category"]], "category")
            else:
                open_lesson(progress, remaining[0] if last else group_ids[group_ids.index(class_id) + 1])


def after_submission(progress: dict, assessment: dict, index: int, answer: dict) -> None:
    if submit_answer(progress, assessment["id"], index, answer):
        commit(progress)
    if not progress["active_assessment"]:
        go("Final Exam" if assessment["kind"] == "final" else "Quiz", result=assessment["id"])
    st.rerun()


def assessment_review(progress: dict, lessons, assessment: dict) -> None:
    st.subheader("Your results")
    total = len(assessment["questions"])
    st.metric("Score", f"{assessment['correct']} / {total}")
    st.progress(assessment["correct"] / total)
    st.caption("Camera answers are scored using the model's prediction. Review any unexpected recognition results.")
    for mode, title in [("sign", "Sign the word"), ("meaning", "Type the meaning")]:
        answers = [row for row in assessment["answers"] if row["mode"] == mode]
        with st.expander(f"{title} · {sum(bool(row['correct']) for row in answers)} / {len(answers)}", expanded=True):
            for row in answers:
                lesson = lessons.loc[lessons["id"].eq(row["class_id"])].iloc[0]
                supplied = str(row.get("answer", ""))
                if mode == "sign":
                    match = lessons.loc[lessons["id"].eq(row.get("predicted_id"))]
                    supplied = word(match.iloc[0]) if not match.empty else "Unrecognized"
                st.write(f"{'✓' if row['correct'] else '↻'} {word(lesson)} / {str(lesson['filipino']).capitalize()} — Your answer: {supplied}")
    cols = st.columns(2)
    if cols[0].button("Practise these topics again", use_container_width=True):
        begin_assessment(progress, lessons, assessment["categories"], assessment["kind"])
    if assessment["kind"] == "category":
        index = categories(lessons).index(assessment["categories"][0])
        if index + 1 < len(categories(lessons)):
            next_category = categories(lessons)[index + 1]
            if cols[1].button("Continue to next level", type="primary", use_container_width=True):
                open_lesson(progress, int(lessons.loc[lessons["category"].eq(next_category)].iloc[0]["id"]))
        elif cols[1].button("Open final exam", type="primary", use_container_width=True):
            go("Final Exam")
    elif cols[1].button("View progress", use_container_width=True):
        go("Progress")


def assessment_page(progress: dict, lessons, final=False) -> None:
    st.title("Final Exam" if final else "Quiz")
    result_id = st.query_params.get("result")
    review = next((item for item in progress["assessments"] if item["id"] == result_id), None)
    if review:
        assessment_review(progress, lessons, review)
        return
    active = progress.get("active_assessment")
    if active and (active["kind"] == "final") == final:
        index = len(active["answers"])
        question = active["questions"][index]
        lesson = lessons.loc[lessons["id"].eq(question["class_id"])].iloc[0]
        st.caption(" · ".join(category_title(cat) for cat in active["categories"]))
        if active["kind"] == "category":
            category = active["categories"][0]
            level = categories(lessons).index(category) + 1
            if st.button(f"Level {level} · {category_title(category)} · Quiz · View progress ↗", use_container_width=True):
                go("Progress")
            st.markdown('<div class="phase">Learn ✓ → Quiz &nbsp; · &nbsp; Next: results and the next level</div>', unsafe_allow_html=True)
        st.progress(index / len(active["questions"]), text=f"Question {index + 1} of {len(active['questions'])} · Quiz phase")
        if question["mode"] == "meaning":
            st.subheader("What does this sign mean?")
            show_video(lesson, f"quiz:{active['id']}:{index}")
            with st.form(f"meaning_{active['id']}_{index}"):
                answer = st.text_input("Type its meaning", placeholder="Filipino or English", key=f"answer_{active['id']}_{index}")
                submitted = st.form_submit_button("Submit answer", type="primary", use_container_width=True)
            if submitted:
                if not answer.strip():
                    st.warning("Enter an answer before continuing.")
                else:
                    correct = check_meaning(answer, lesson)
                    updated = record_attempt(progress, "meaning", {"target_id": int(lesson["id"]), "answer": answer, "correct": correct})
                    after_submission(updated, active, index, {"answer": answer.strip(), "correct": correct})
        else:
            st.subheader("Sign this word")
            st.header(word(lesson))
            st.write(str(lesson["filipino"]).capitalize())
            context = f"quiz:{active['id']}:{index}:{active['model']}"
            result = capture_sign(progress, lesson, context, active["model"], active["id"])
            if result:
                st.success("Recording analyzed. Submit it or record another attempt.")
            if st.button("Submit sign", type="primary", disabled=result is None, use_container_width=True):
                after_submission(progress, active, index, {
                    "correct": result["correct"], "predicted_id": result["prediction"]["class_id"],
                    "confidence": result["prediction"]["confidence"], "capture_id": result["capture_id"],
                })
        st.caption("Your answers and question order are saved. Results appear after the last question.")
        if st.button("Save and return home"):
            go("Home")
        with st.expander("End this attempt"):
            st.write("Submitted answers will be kept in your saved history. Start a new assessment after ending this one.")
            if st.button("End assessment", key="end_assessment"):
                progress.setdefault("abandoned_assessments", []).append({**active, "ended_at": now()})
                progress["active_assessment"] = None
                commit(progress)
                go("Final Exam" if final else "Quiz")
        return
    if active:
        st.info("Another assessment is saved. Resume or end that attempt before starting a new one.")
        if st.button("Resume saved assessment", type="primary"):
            go("Final Exam" if active["kind"] == "final" else "Quiz")
    if final:
        st.markdown('<div class="card"><h3>All the words, together</h3><p>105 questions across 10 topics. Sign a given word or type the meaning of a demonstration. Each word appears once.</p></div>', unsafe_allow_html=True)
        st.write("The two question types are split as evenly as possible: 52 of one type and 53 of the other.")
        if st.button("Start final exam", type="primary", disabled=bool(active), use_container_width=True):
            begin_assessment(progress, lessons, categories(lessons), "final")
    else:
        st.write("Select one or more topics. Every selected word appears once, with a balanced mix of camera and meaning questions.")
        st.subheader("Choose topics")
        chosen = []
        topic_list = categories(lessons)
        for offset in range(0, len(topic_list), 3):
            columns = st.columns(3)
            for col, category in zip(columns, topic_list[offset:offset + 3]):
                with col, st.container(border=True):
                    if st.checkbox(category_title(category), key=f"quiz_topic_{category}"):
                        chosen.append(category)
                    count = int(lessons["category"].eq(category).sum())
                    st.caption(f"{count} words")
        if chosen:
            st.caption(f"{len(lessons.loc[lessons['category'].isin(chosen)])} words selected")
        if st.button("Start quiz", type="primary", disabled=not chosen or bool(active), use_container_width=True):
            begin_assessment(progress, lessons, chosen, "custom")


def progress_page(progress: dict, lessons) -> None:
    st.title("Your progress")
    st.write("Choose a topic or word to continue where you left off, or revisit something you have finished.")
    completed = set(progress["completed_lessons"])
    st.progress(len(completed & set(lessons["id"])) / len(lessons), text=f"{len(completed & set(lessons['id']))} of {len(lessons)} words completed")
    for level, category in enumerate(categories(lessons), 1):
        group = lessons.loc[lessons["category"].eq(category)]
        ids = group["id"].astype(int).tolist()
        done = len(completed & set(ids))
        status = category_status(progress, ids, category)
        label = {"complete": "Finished", "started": "In progress", "untouched": "Not started"}[status]
        st.markdown(f'<div class="word-tile {status}"><h4>Level {level} · {esc(category_title(category))}</h4><p>{label} · {done}/{len(ids)} words · Category quiz {"complete" if category in progress["category_quizzes"] else "to do"}</p></div>', unsafe_allow_html=True)
        cols = st.columns(2)
        if cols[0].button(f"{'Review' if status == 'complete' else 'Continue'} {category_title(category)}", key=f"level_{category}", use_container_width=True):
            if done == len(ids) and category not in progress["category_quizzes"]:
                begin_assessment(progress, lessons, [category], "category")
            else:
                current = int(progress["current_lesson"])
                class_id = current if current in ids and current not in completed else next((class_id for class_id in ids if class_id not in completed), ids[0])
                open_lesson(progress, class_id)
        if cols[1].button(f"{'Review' if category in progress['category_quizzes'] else 'Take'} category quiz", key=f"category_quiz_{category}", disabled=done != len(ids), use_container_width=True):
            if category in progress["category_quizzes"]:
                go("Quiz", result=progress["category_quizzes"][category])
            else:
                begin_assessment(progress, lessons, [category], "category")
        with st.expander(f"Words in {category_title(category)}"):
            for offset in range(0, len(group), 3):
                columns = st.columns(3)
                for col, (_, lesson) in zip(columns, group.iloc[offset:offset + 3].iterrows()):
                    class_id = int(lesson["id"])
                    state = "complete" if class_id in completed else "started" if str(class_id) in progress["lesson_states"] else "untouched"
                    text = {"complete": "✓ Finished", "started": "◐ In progress", "untouched": "○ Not started"}[state]
                    with col:
                        st.markdown(f'<div class="word-tile {state}"><h4>{esc(word(lesson))}</h4><p>{text}</p></div>', unsafe_allow_html=True)
                        if st.button(f"{'Review' if state == 'complete' else 'Open'} {word(lesson)}", key=f"word_{class_id}", use_container_width=True):
                            open_lesson(progress, class_id)
    if progress["assessments"]:
        st.subheader("Assessment history")
        for assessment in reversed(progress["assessments"]):
            title = "Final exam" if assessment["kind"] == "final" else " + ".join(category_title(cat) for cat in assessment["categories"])
            if st.button(f"{title} · {assessment['correct']}/{len(assessment['questions'])} · {assessment['completed_at'][:10]}", key=f"history_{assessment['id']}", use_container_width=True):
                go("Final Exam" if assessment["kind"] == "final" else "Quiz", result=assessment["id"])


def developer_panel(progress: dict, lessons, page: str) -> None:
    if not progress["settings"]["developer_mode"]:
        return
    st.divider()
    st.subheader("What the model noticed")
    if progress.get("active_assessment"):
        st.info("Assessment explanations will be available after you finish the assessment.")
        return
    diagnostic = progress.get("last_diagnostic")
    if not diagnostic:
        st.info("Record a sign to see its explanation here.")
        return
    # Do not reveal a hidden quiz word through diagnostics, even via another page.
    assessment_id = diagnostic.get("assessment_id")
    if assessment_id and not any(item["id"] == assessment_id for item in progress["assessments"]):
        st.info("The explanation for this assessment will be available after you finish it.")
        return
    if diagnostic["model"] != progress["settings"]["model"]:
        st.info("Record a sign with the selected model to see its explanation.")
        return
    latest = st.session_state.get("camera_last_result", {})
    if latest.get("context") == diagnostic["context"] and "error" in latest:
        st.info("The latest recording could not be analyzed. Record again to see its explanation.")
        return
    if page == "Lessons":
        try:
            current_id = int(st.query_params.get("word", progress["current_lesson"]))
        except ValueError:
            st.info("Choose a word from Progress to see its explanation.")
            return
        if current_id != diagnostic["class_id"] or assessment_id:
            st.info("Record this word to see the explanation for this lesson.")
            return
    trace = diagnostic["trace"]
    target = lessons.loc[lessons["id"].eq(diagnostic["class_id"])].iloc[0]
    st.caption(f"Latest analyzed attempt · {word(target)} · {category_title(target['category'])}")
    label_row = lessons.loc[lessons["id"].eq(trace["explainer_prediction"]["class_id"])].iloc[0]
    if trace["trace_source"] == "decision_tree_primary":
        st.write(f"Actual Decision Tree path · {word(label_row)}")
    else:
        st.write(f"Separate Decision Tree baseline · {word(label_row)}")
        st.caption("Extra Trees made the recognition decision. These observations belong to the separate baseline tree.")
    st.caption("Descriptions follow the recorded features and learned branches. Positions use the unmirrored image; changes between sampled frames are not speed measurements.")
    observations = {rule["feature"]: rule["value"] for rule in trace["rules"]}
    observations.update(diagnostic.get("observations", {}))
    try:
        descriptions = describe_trace(trace, observations)
    except ValueError:
        st.warning("This saved explanation could not be verified. Record a new sign to see its explanation.")
        return
    if "observations" not in diagnostic:
        st.info("This older recording did not save full tracking coverage. Record again for verified hand-state descriptions.")
    if not descriptions:
        st.write("The tree reached its answer without another hand-feature check.")
    for observation in descriptions:
        note = f'<p class="tracking-note">{esc(observation["note"])}</p>' if observation.get("note") else ""
        st.markdown(f'<div class="observation"><strong>{esc(observation["title"])}</strong><p>{esc(observation["description"])}</p>{note}</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Senyas · Learn FSL", page_icon="🤟", layout="wide", initial_sidebar_state="expanded")
    css = Path(__file__).with_name("theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    lessons = curriculum()
    progress = load_progress(progress_path())
    page = st.query_params.get("page", "Home")
    if page not in PAGES:
        page = "Home"
    navigation(page)
    if page == "Home":
        home(progress, lessons)
    elif page == "Lessons":
        lesson_page(progress, lessons)
    elif page == "Quiz":
        assessment_page(progress, lessons)
    elif page == "Final Exam":
        assessment_page(progress, lessons, final=True)
    else:
        progress_page(progress, lessons)
    developer_panel(progress, lessons, page)
