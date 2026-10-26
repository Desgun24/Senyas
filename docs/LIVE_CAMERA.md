# Live camera practice

Open **Lessons**, watch a word's demonstration, and press **Next · Copy the sign**. Press **Start camera**, allow camera access, then press **Record 4-second sign**. Wait for the three-second countdown and perform one sign while keeping your hands and shoulders visible. The app automatically checks the attempt; you can retry or continue to the replay. Quiz and final-exam camera questions use the same controls and reveal their results after the assessment.

The preview is live; recognition runs after each four-second recording. The app samples 32 frames, uses the existing landmark feature extractor, and keeps the trained model unchanged. **Mirror preview only** changes the display without reversing the frames used for recognition. **Stop camera** releases the webcam; no microphone access is requested. If camera permission is blocked in an embedded browser, open [the local app](http://127.0.0.1:8501) in Edge or Chrome and allow camera access there.

Use the gear icon to open **Settings**. **Decision Tree** is the default recognition model; **Extra Trees** is also available. Enable **Developer mode** for plain-language observations below the page content. The actual Decision Tree path uses its own features; Extra Trees labels the separate baseline tree explicitly. Assessment explanations stay hidden until completion.

To start the app again from the Senyas folder in PowerShell:

```powershell
& .venv\Scripts\python.exe scripts/run_app.py
```

Camera frames are processed in memory, and successful analyses save the predicted class and outcome to the existing progress history. Dataset video extraction remains available through the pipeline scripts. The saved evaluation results describe the original dataset; they do not establish accuracy for live webcam captures.
