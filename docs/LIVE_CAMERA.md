# Live camera practice

Open **Lessons**, watch a word's demonstration, and press **Next · Copy the sign**. Press **Start camera**, allow camera access, then press **Record 4-second sign**. Wait for the three-second countdown and perform one sign while keeping your hands and shoulders visible. The app automatically checks the attempt; you can retry or continue to the replay. Quiz and final-exam camera questions use the same controls and reveal their results after the assessment.

The preview is live; recognition runs after each four-second recording. Version 1.1.0 captures up to 121 timestamped frames, targets 30 FPS, and uses the frozen 3,392-feature models with the same full-complexity landmark extractor used in evaluation. Slow devices may capture or process fewer frames per second. **Mirror preview only** changes the display without reversing the frames used for recognition. **Stop camera** releases the webcam; no microphone access is requested. If camera permission is blocked in an embedded browser, open the launcher's local address in Edge or Chrome and allow camera access there.

Use the gear icon to open **Settings**. **Decision Tree** is the default interpretable primary model; **Extra Trees** is the ensemble baseline. Enable **Developer mode** for qualitative descriptions of the actual Decision Tree branches below the page content. Missing measurements used through training-median imputation are identified; proximity is not described as physical contact. Extra Trees has no single-tree explanation displayed. Assessment explanations stay hidden until completion.

To start the app again from the Senyas folder in PowerShell:

```powershell
& .venv\Scripts\python.exe scripts/run_app.py
```

Camera frames are processed in memory, and successful analyses save the predicted class and outcome to the existing progress history. Dataset video extraction remains available through the pipeline scripts. The saved evaluation results describe the original dataset; they do not establish accuracy for live webcam captures.
