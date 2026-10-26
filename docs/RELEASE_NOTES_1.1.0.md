# Senyas 1.1.0

The learning app now uses the frozen three-phase models with 3,392 engineered inputs. Decision Tree remains the interpretable default; Extra Trees serves as the data-driven baseline. The exact weights and preprocessing used for the held-out test are bundled, with model hashes and portable results under `artifacts/frozen/`.

| Model | Validation accuracy | Test accuracy | Test macro F1 |
| --- | --- | --- | --- |
| Decision Tree | 271/320 (84.69%) | 262/315 (83.17%) | 82.78% |
| Extra Trees | 314/320 (98.13%) | 311/315 (98.73%) | 98.69% |

Both models were fitted on 1,494 training clips, without refitting on validation data. One additional validation recording had no usable hands; all 315 test recordings were processed. These dataset results do not establish webcam accuracy in every environment.

Camera attempts use timestamped capture targeting 30 FPS and the full landmark tracker used in evaluation. Developer mode describes actual Decision Tree branches, labels imputed measurements, and avoids claiming that image proximity proves touch. Extra Trees has no single-tree path displayed. Existing lesson progress and assessment history can be retained by copying `data/progress.json` into the new release's `data` folder; explanations saved by the old pipeline require a new recording.

Download and extract **Senyas-1.1.0-Windows-x64.zip**, then run **Install Senyas.cmd**. Setup installs pinned Python and libraries into the private `.runtime` folder, checks the models and 105 silent videos, creates shortcuts, and launches the app. Internet is needed for initial installation; Git, system Python, and the external dataset are unnecessary. `SHA256SUMS.txt` accompanies the download.

Project credits: Jeff Barrientos, Hannah Dimaculangan, and Trixie Mendoza. Third-party notices are retained.
