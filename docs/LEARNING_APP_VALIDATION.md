# Learning app verification

Date: 26 October 2026

The learning redesign is implemented locally. The app uses 105 words in the original 10 categories, five navigation destinations, a Settings dialog, a guided Watch → Copy → Replay lesson loop, and shared assessment logic for category quizzes, custom quizzes, and the final exam.

## Content

- The original numeric IDs and category assignments are retained in `config/labels.csv` and mapped to `config/lessons.csv`.
- One demonstration per word is bundled in `assets/lessons/`: 105 H.264 MP4s, approximately 15.07 MiB in total.
- Each exported video was inspected for audio streams and fully decoded with FFmpeg. All 105 have no audio tracks.
- Output hashes and source-relative clip names are recorded in `assets/lessons/manifest.json`. Original dataset files were read without modification.
- Filipino equivalents and accepted answers cover every word. Display corrections retain their original model IDs.

## Automated checks

The complete Python suite passed: 52 tests. It covers bundled content and hashes, original ID/category mappings, bilingual answers, assessment balance and unique word coverage, persisted question order, duplicate submission prevention, progress migration and atomic saves, fixed feature width, camera validation, actual tree paths, qualitative descriptions, navigation, quiz answer concealment, category transitions, stale captures after a model change, and the feature-to-language audit documented in `EXPLANATION_AUDIT.md`.

The tests use synthetic camera features and separate progress files. They do not train or refit models, and they do not run a new dataset evaluation.

## Browser checks

Headless Edge with a simulated camera verified the following:

- model selection and Developer mode save through the real Settings dialog;
- learner navigation has no radio controls or exposed model selector;
- the expanded sidebar contains all destinations and Settings, quiz topics use a checkbox grid, and display text cannot be drag-selected;
- bundled video playback, mute state, and Replay work;
- camera capture reaches MediaPipe, no-hand recordings show a retry message, and Stop releases the stream;
- the progress link resumes the saved copy stage;
- selecting Greetings and Colors creates a 23-word quiz whose question order survives refresh;
- ending an assessment is explicit and retains submitted history;
- the final exam covers all 105 words in a 52/53 split;
- meaning prompts and unfinished assessments conceal answer labels and developer diagnostics;
- Filipino answers are graded correctly;
- a narrow viewport has no horizontal overflow, with no uncaught browser errors.

Sampled contrast ratios were approximately 9.11:1 for the hovered primary button, 14.56:1 for body text, and 7.13:1 for hero text against the lighter green endpoint. These checks cover the sampled styles and do not constitute an exhaustive accessibility audit.

## Portability and preservation

A fresh standalone copy containing only app code, configuration, bundled media, and the saved model opened Home, Lessons, Quiz, Final Exam, and Progress successfully. It contained no dataset directory or processed feature data, and its dataset environment variable pointed to a nonexistent location. All 105 video paths resolved within that copy.

The trained model hash remained unchanged. The existing learner progress file matched the backup taken before implementation; application use subsequently upgrades the schema while preserving historical completions and attempts.

Physical-webcam recognition accuracy was not established by these checks. Camera questions are scored from model predictions; Developer mode descriptions summarize real learned branches without displaying their numeric comparisons. Extra Trees continues to label its separate Decision Tree baseline explicitly. Assessment diagnostics appear after completion.
