# Explanation audit

Reviewed on 26 October 2026 against `senyas/features.py`, the saved model, and the latest saved Decision Tree log. All ten recorded branches matched their actual feature indices, comparisons, and next nodes. Reconstructing a sample from the recorded path features reproduced the saved leaf and tree prediction. The path itself was consistent; several plain-language descriptions needed correction.

| Feature or statistic | Actual meaning | Correction |
|---|---|---|
| Finger bend | Normalized angle from the configured landmark triple; larger means straighter | Describe the relative finger angle at the indicated start, end, average, or extreme. Do not infer a universal bent/straight cutoff. |
| Finger coordinates | Wrist-relative coordinates divided by a palm reference | Keep the wrist and palm reference in the description; a fingertip-coordinate change is not automatically whole-hand movement. |
| Absolute wrist coordinates | Position in the unmirrored image | Use image directions and identify the relevant temporal statistic. |
| Body-relative wrist coordinates | Wrist offset from shoulder midpoint divided by visible shoulder gap | Preserve the shoulder reference and flag missing shoulder tracking. |
| Hand separation | Wrist separation divided by the shoulder reference | Describe the normalized separation, not an absolute physical distance. Require simultaneous detection of both hands. |
| Standard deviation | Spread across sampled values | Describe spread, without inventing motion direction. |
| Range | Largest minus smallest sampled value | Describe the extent of the recorded range. |
| Average frame difference | Average absolute change between adjacent sampled feature values | Describe average changes between samples; it is not speed per second. |
| Largest frame difference | Largest absolute change between adjacent sampled feature values | Describe the largest change; it does not establish overall steadiness. |
| Start-to-end difference | Last minus first value | Describe the endpoint difference. A zero difference does not mean the finger stayed still in between. |
| Detection flags | Binary per-frame detection, summarized separately | Distinguish detection at an endpoint, at least once, throughout, or during only part of the sign. |
| Missing measurements | Hand measurements are interpolated or zero-filled; missing pose uses a default reference | Show tracking limitations and filled values instead of asserting a measured hand pose. |

Joint names such as CMC, MCP, PIP, and DIP are rendered as thumb base, base knuckle, middle joint, or end joint. Depth remains an estimated value relative to the wrist and palm reference. No explicit facial-expression or palm-facing labels are computed by the current feature extractor, so the log does not claim those states.

New camera diagnostics save the detection coverage and endpoint flags needed to qualify each description. Older logs are regenerated from their numeric trace rather than reusing their old wording; when the necessary tracking coverage was not saved, they ask for a fresh recording instead of asserting an unverified pose. Historical attempts are retained.

An invalid branch can no longer be silently skipped: unverified branches, inconsistent comparisons, or non-finite comparisons produce a request for a fresh explanation. Extra Trees continues to identify its separate Decision Tree baseline. These descriptions report the classifier's measured inputs and learned comparisons; they are not validated linguistic rules for Filipino Sign Language.

The complete Python suite passed 52 tests, including angle direction, distinct temporal statistics, zero-net-change motion, coordinate references, simultaneous-hand requirements, missing shoulders, endpoint tracking, legacy logs, and branch consistency. These checks do not train or change the saved recognition models.
