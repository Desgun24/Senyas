# Windows release validation

Version 1.0.0 was prepared and validated locally on 26 October 2026.

## Verified

- A clean release ZIP was extracted into a separate folder on D:. It contains the app, model, and 105 silent lesson videos, with no installed libraries or learner progress.
- The installer built a private CPython 3.11.9 x64 runtime and installed 64 pinned runtime packages using SHA-256 hashes. The Python archive and pip wheel were downloaded and verified before testing; existing pip download caches were reused. This is a fresh-runtime installation check, not a measurement of download time on an empty machine.
- `pip check` reported no broken requirements. The startup verifier loaded the Decision Tree model, initialized MediaPipe, and checked every bundled video against its manifest hash.
- The desktop launcher started the app on loopback. A second invocation reused the running instance.
- Headless Edge opened Home, Lessons, Quiz, Final Exam, Progress, and Settings without a Streamlit exception. Quiz displayed all ten category checkboxes.
- The existing application suite passed all 52 tests.

The automated installer run used `-NoLaunch -NoShortcuts`; the launcher was tested separately. Desktop and Start menu shortcut creation is implemented but was not exercised by this test, so the user's existing shortcuts were not changed. Real camera recognition remains subject to device permissions and recording conditions; these packaging checks do not establish recognition accuracy.

## Repository cleanup

The old tracked `.venv-broken-20261024-133313` environment and personal `data/progress.json` were removed from the Git index. Their local files remain intact. Ignore rules cover environments, runtime downloads, build output, caches, and personal progress. Repository cleanup preserves existing commit history.
