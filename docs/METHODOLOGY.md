# Senyas methodology

## 1. Scope and research question

Senyas recognizes selected isolated beginner Filipino Sign Language signs and connects the result to short lessons and practice activities. The experiment asks whether normalized hand-landmark and motion features can support an accurate Extra Trees ensemble while retaining a separate interpretable Decision Tree baseline for learner-facing traces on held-out signer/session groups.

Continuous signing, sentence translation, facial grammar, and production use are outside the current scope.

## 2. Dataset preparation

The supplied FSL105 collection contains 2,130 `.MOV` clips in 105 label folders. Source tables provide the relative clip path, numeric class, English label, and semantic category. The preprocessing script:

1. combines the source train and test tables;
2. removes duplicate paths and fails if any referenced file is missing;
3. standardizes Windows and POSIX path separators;
4. verifies supported video extensions;
5. derives a signer/session group from the clip filename; and
6. optionally joins human-reviewed recording-condition tags.

The group interpretation must be confirmed using the original dataset documentation. If it is not confirmed, evaluation claims should say "unseen filename-derived sessions" rather than "unseen signers."

## 3. Leakage-safe 70/15/15 split

All clips from a signer/session group remain together. A deterministic random search (seed 42) evaluates candidate group orders and chooses the partition closest to 70% training, 15% validation, and 15% test by sample count. Candidate partitions receive a strong penalty when a class is absent from any split.

The training split fits model parameters. The validation split independently selects Extra Trees ensemble settings and Decision Tree baseline settings. After selection, both models are refit on training plus validation clips. Extra Trees is the primary recognizer; the test split is evaluated once as the held-out generalization result.

Because indivisible signer groups have unequal numbers of clips, exact percentages may be impossible. The generated manifest records the measured ratio, and the code verifies that no group crosses split boundaries.

## 4. Video preprocessing

OpenCV decodes each clip. Up to 32 frames are sampled uniformly from start to end, preserving coverage of short and long clips while bounding computation. Frames are resized only when their longest side exceeds 640 pixels; aspect ratio is preserved. BGR pixels are converted to RGB for MediaPipe.

A clip is rejected and logged when it cannot be decoded or fewer than three sampled frames contain a detected hand. Failed clips remain in `feature_index.csv` with the exception type and message, so exclusions are auditable.

## 5. Landmark extraction and feature engineering

MediaPipe Holistic estimates 21 landmarks per hand and the shoulder reference. Each frame includes:

- hand-presence indicators;
- wrist image coordinates;
- wrist coordinates relative to the shoulder midpoint and shoulder width;
- hand landmarks centered on the wrist and divided by palm size;
- five bend angles and five wrist-to-fingertip distances for each hand;
- shoulder width;
- inter-hand distance and vertical offset.

Missing detections are interpolated between valid observations. A feature that is missing for the full clip is set to zero, while explicit presence indicators retain the missingness signal.

For every frame-level descriptor, ten temporal statistics create a fixed-length feature vector: mean, standard deviation, minimum, maximum, range, first value, last value, net change, mean absolute frame-to-frame movement, and maximum absolute movement.

This design captures hand shape, body-relative placement, and coarse motion while keeping every feature nameable. The ensemble can combine many nonlinear feature decisions, while the baseline tree preserves a direct rule trace. Sequence models may recognize temporal order better but would not provide the same simple explanation.

## 6. Model training and uncertainty

A class-balanced `ExtraTreesClassifier` is the primary recognition model. Validation searches the number of randomly considered features and minimum leaf size using 300 trees. An independent class-balanced `DecisionTreeClassifier` is retained and tuned over:

- criterion: Gini or entropy;
- maximum depth: 6, 10, or 14;
- minimum samples per leaf: 2, 4, or 8; and
- cost-complexity pruning alpha: 0 or 0.001.

Both searches use validation macro F1 as the primary selection measure, followed by validation accuracy. The final artifact stores the ensemble, baseline tree, feature-name order, class map, and selected parameters. The uncertainty threshold is selected on validation data to maximize coverage while targeting at least 95% accuracy among accepted validation predictions. A prediction below that threshold is shown as *uncertain* instead of forcing matching/different feedback.

## 7. Decision traces

For one clip, the application obtains the exact node path through the retained Decision Tree baseline. Each non-leaf node is displayed as the measured named feature, comparison operator, and learned cutoff. The interface explicitly reports whether this transparent baseline agrees with the primary Extra Trees prediction. The trace does not claim to be the full ensemble computation, a linguistic definition, or a guarantee that the sign was produced correctly.

The evaluation replays up to 50 held-out baseline paths, verifies every displayed condition, and reports baseline-to-ensemble prediction agreement.

## 8. Learning activities

### Copy the Sign

The learner watches a target demonstration and uploads a short attempt. The model returns:

- **matching** when the confident predicted class equals the target;
- **different** when a confident prediction names another class; or
- **uncertain** when confidence is below threshold.

The learner can inspect the decision trace and retry.

### Type the Meaning

The learner watches an unnamed sign and types its meaning. Text is case-folded, punctuation is removed, and spaces are normalized. The answer is checked against reviewed English labels, Filipino meanings, and explicit synonyms. The accepted meanings are revealed after submission.

## 9. Evaluation

The locked test set reports accuracy, macro precision, macro recall, macro F1, classification report, and confusion matrix. Macro measures are emphasized because all 105 classes should matter equally.

Robustness tables group test predictions by held-out signer/session. Lighting, background, camera distance, and signing-speed results appear only when those conditions have been human-tagged. Efficiency includes feature-extraction time and per-sample model latency. Functional checks cover playback, uploads, answer variants, retries, uncertainty, progress saving, and explanations.

## 10. Threats to validity

- Filename-derived groups may represent recording sessions rather than unique people.
- MediaPipe accuracy may differ by skin tone, camera quality, occlusion, mobility, and signing style.
- Temporal aggregation loses some order information.
- Ensemble confidence is validation-thresholded but is not a perfectly calibrated probability.
- Decision traces describe the transparent baseline, not all 300 ensemble trees.
- Dataset demonstrations and bilingual meanings require review by FSL experts and Deaf community members.
- A curated 105-class dataset does not measure performance in continuous, real-world signing.

