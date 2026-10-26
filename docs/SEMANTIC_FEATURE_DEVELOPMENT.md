# Interpretable sequence feature development

The measurement pipeline retains temporal order and derives 278 geometric measurements plus 50 motion measurements, totaling 328 named features per frame. Missing measurements remain missing. Version 1.1.0 uses the frozen three-phase Decision Tree and Extra Trees models: 314 supported measurements become 3,454 summary candidates, then training-only filtering retains 3,392 inputs. The older 1,630-input artifact remains available for historical research; it is not used by the current learning app.

## Run the feature lab

```powershell
& .venv/Scripts/python.exe scripts/run_app.py --feature-lab --server.address 127.0.0.1 --server.port 8502 --server.headless true
```

Open http://127.0.0.1:8502/. The lab records a four-second camera attempt, measures the sequence after capture, displays individual feature values and movement, and can explicitly save labelled hand-shape clips. It does not claim to recognize a word or to identify a trained hand-shape class. The ordinary learning app remains at port 8501.

The lab requests a 30 FPS camera stream and captures up to 121 freshly presented samples over approximately four seconds. A slower camera yields fewer samples instead of stretching the recording to meet a promised frame count. It reports actual timestamp-based sampling separately from processing throughput. Hardware, browser scheduling, camera exposure, and tracking difficulty can lower either rate. Processing runs after recording; this is not a streaming 30 FPS recognition service. A rate of 24 FPS or higher meets the current accepted performance target.

## Feature decisions

| Proposed feature | Implemented measurement or decision | Interpretation limits |
| --- | --- | --- |
| Fist axis and hand tilt | Hand longitudinal unit vector and image tilt from vertical | Replaces overlapping names; vertical is zero and horizontal is ninety degrees |
| Fist direction | Hand-axis direction is retained independently of palm orientation | Knuckle-facing and forearm direction must not be conflated; no ambiguous fist-direction class |
| Palm direction | Estimated palm-plane normal from ordered landmark cross-products | Experimental; front/back sign needs review/calibration, so it is excluded from default DTW scaling |
| Finger state for all fingers | Separate joint flexions, tip-to-root distance, tip-to-palm distance, and longitudinal projection | Extended, curved, folded, and curled are future supervised labels; numerical fallbacks never assert a state |
| All fingertip-touch combinations | Ten pairwise tip gaps per hand | Supports examining groups of two to five without a large set of redundant binary combinations; small gap does not prove contact |
| Finger direction | A base-to-tip unit ray per finger | Suppressed for a very short folded ray; estimated hand depth remains approximate |
| Fist and open or curved tight/wide shapes | Joint geometry and neighbouring-finger spread | Shared shape labels can be learned from annotated clips; no shape classifier is trained yet |
| Thumb with fingers and thumb position | Thumb tip in local palm coordinates, thumb-to-tip gaps, and joint geometry | Replaces an ambiguous yes/no field; no front-of-palm label without orientation calibration |
| Clockwise and anticlockwise circles | Timestamped wrist trajectory, signed turning rate, fitted-radius consistency, closure gap, and signed turns | Descriptive image-plane measures, not an automatic circle label; positive turns are anticlockwise in the unmirrored camera image |
| Finger flailing | Palm-relative tip speed, half-second sway extent, reversal rate, and joint-shape variation | Separates whole-hand translation from relative finger motion; a trained wiggle/flailing label still needs examples |
| Hands touching and placement | Image wrist/palm gaps, minimum landmark gap, horizontal/vertical offsets, and fingertip-to-other-palm gaps | Proximity and above/beside relationships are supported; physical touch is not confirmed |
| Right or left hand in front | Pose-wrist depth difference | Experimental proxy, excluded from default matching; independent hand z origins are never compared |
| Touch forehead, lips, nose, cheeks, ears, chin, chest, shoulders, elbows | Minimum image-plane palm, fingertip, and knuckle distance to supported body anchors | Relative to shoulder scale; precise contact and palm-versus-back surface cannot be established |
| Same or other elbow and opposite wrist | Anatomical left/right elbow proximity and palm-to-other-wrist gap | Same/other can be derived from the active hand; inner elbow surface lacks a reliable landmark |
| Finger wrapped around another hand | Retain local finger geometry and hand-to-hand proximity | Wrapping and palm/back contact are deferred because occlusion and monocular depth are insufficient for a reliable assertion |
| Hand forward/backward, sideways, or stationary | Body-relative wrist x/y and timestamped velocity; pose depth proxy retained separately | Up/down and image-left/right have explicit axes; forward/back is a noisy experimental depth measurement |
| Head nod and shake | Nose displacement relative to ears, ear-line roll, and motion derivatives | Experimental head-motion proxies; no verified yaw/pitch or yes/no nod/shake class |
| Hand around or in front of head and chest | Continuous head-relative and shoulder-relative positions; chest proximity | Above/left/right have a measurable image reference; in-front depth is not asserted |

Body proximity currently aggregates palm, fingertips, and knuckles. The exact nearest individual finger and physical contact surface are not separate classifier inputs. Raw landmarks are saved in sequence caches so these details can be reviewed or derived later without extracting landmarks again.

## Engineering formulas and axes

Image-normalized x and within-hand z are multiplied by the image aspect ratio before geometry is computed. The camera convention is x positive to unmirrored image-right, y positive upward, and estimated z positive toward the camera. Each hand has its own depth origin. The body depth proxy uses pose-wrist coordinates in the pose coordinate system instead of comparing separate hand z origins.

- Joint flexion: one minus the angle between the two joint segments divided by pi. Straight segments approach zero; more flexion increases the value. Zero-length geometry returns missing.
- Palm scale: the three-dimensional within-hand wrist-to-middle-knuckle gap. Distances used for shape are divided by this scale.
- Palm coordinates: subtract the palm centre, project onto across-palm, longitudinal, and normal axes, then divide by palm scale. This reduces sensitivity to whole-hand translation and rotation for finger-motion measurements.
- Body-relative wrist position: wrist displacement from the shoulder midpoint divided by the visible shoulder gap. Missing or unreliable shoulders produce missing body features, without an image-centre fallback.
- Tip proximity: within-hand fingertip gap divided by palm scale. Hand-to-body and between-hand proximity use image-plane gaps divided by shoulder width.
- Velocity: displacement between two consecutive observed samples divided by their actual timestamp interval. A gap longer than 0.16 seconds is not bridged.
- Finger sway: extent and significant velocity-direction reversals in a trailing half-second palm-relative trajectory. Joint-flexion variability is measured separately so stable-shape movement need not be confused with finger bending.
- Circle diagnostics: fit a circle to a tracked two-dimensional wrist trajectory, then retain signed angular travel, radial variability, and closure gap relative to path length. A line, too little movement, or a tracking gap does not yield a valid circle diagnostic.

The full per-frame registry includes feature name, group, unit, explanation, and experimental status. Presence flags accompany missing values, preventing an absent hand from becoming a zero-valued closed hand. Face and pose detection provide anchors; facial expression is not engineered in this implementation.

## Sequences and DTW

No per-word event script is needed. Each source video already has a word label. Frame measurements and actual timestamps form an ordered recording of the hand configuration while it moves. A DTW comparison can align a longer held pose with a shorter one while preserving progression through the recording.

`senyas/sequence_matching.py` implements the comparison engine, not a fitted 105-word recognizer. Its training-scaler API learns feature medians and interquartile ranges from supplied training observations only. Constant or unsupported dimensions are disabled. Experimental groups and tracking flags are excluded from distance features by default.

Within each feature group, the engine compares shared observed dimensions, penalizes missing disagreement, and balances groups so that a large proximity group cannot dominate solely through its feature count. A constrained dynamic-programming path minimizes cumulative cost. Its mean path cost, exact path indices, shared-observation fraction, and actual feature-group contributions are returned. Mean path cost is a comparison score rather than a calibrated probability; path selection optimizes cumulative cost, not minimum average cost.

The DTW engine has no deployed word reference bank or calibrated recognition scores. The deployed Decision Tree and Extra Trees use the separately trained, frozen three-phase summaries, not DTW; their evaluation results and model hashes are in `artifacts/frozen/`.

## Collecting three to five clips

Start with three to five clips that include repeated examples of the shared shapes. Several shapes can occur in one clip: label a separate interval for each and save again to append its annotation to the existing clip. Hold the same shape during each labelled interval, use a natural position, and vary the angle or distance modestly between clips. Label left and right hands independently; leave an unlabelled or absent hand unlabelled. Conflicting overlapping labels on the same hand are rejected.

Examples such as fist, open_tight, open_wide, curved_tight, curved_wide, pointing, and pinch are reusable across words. The label describes hand shape, not palm orientation, body location, or a particular word. Additional shape names can be introduced after reviewing actual demonstrations.

The Save labelled clip button writes a silent MP4, an NPZ containing exact timestamps and landmarks/features, and JSON labels to `data/handshape_samples/`. Saving is explicit; captures otherwise remain in session memory. The MP4 uses the average capture frame rate, while exact nonuniform timestamps remain in the NPZ. Every frame from a recording shares a clip-group identifier. Signer/session grouping should also be preserved during later evaluation; nearby frames from the same clip must never be treated as independent train/validation examples.

Three to five clips are a starting set for checking and initial concept training, not evidence of robust generalization across signers. Model accuracy cannot be claimed until held-out samples are evaluated.

## Performance and validation

The initial benchmark used three existing training clips: clips/0/0.MOV, clips/50/0.MOV, and clips/100/0.MOV. Each was sampled at 30 FPS. The 480-pixel lite configuration processed approximately 24.1 to 28.3 frames per second; the 360-pixel lite comparison processed approximately 34.9 to 36.0; and the 480-pixel full pose comparison processed approximately 28.9 to 30.0. A later full-pose run during browser testing ranged from 20.3 to 24.7 FPS, showing sensitivity to load. A simulated-camera browser run delivered about 20 FPS even when the camera constraint requested thirty. These are short throughput and capture checks, not a webcam guarantee or recognition evaluation. Tracking coverage was approximately 32 to 59 percent across the complete prerecorded clips, which include frames without detected hands.

Unit tests cover scale/translation invariance, separate finger flexions, all ten fingertip pairs, missing tracking, independent hand-depth origins, timestamped derivatives, stable-shape sway, circle direction, cache round trips, DTW ordering and cost accounting, explicit sample saving, and rejection of labels on unseen hands. Learner tests use the current frozen-model adapter; compatibility tests also retain coverage of the older extractor.

## References

- MediaPipe Holistic: https://chuoling.github.io/mediapipe/solutions/holistic.html
- Hand landmark coordinate conventions: https://chuoling.github.io/mediapipe/solutions/hands.html
- Pose landmark conventions: https://chuoling.github.io/mediapipe/solutions/pose.html
- DTW and constrained alignment: https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html
- Camera frame-rate constraints: https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackConstraints/frameRate
