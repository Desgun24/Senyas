# FSL105 local dataset card

## Local audit

| Field | Observed value |
|---|---:|
| Video clips | 2,130 |
| Video format | `.MOV` |
| Classes | 105 |
| Class folders | `0` through `104` |
| Semantic categories | Greetings, survival, numbers, calendar, days, family, relationships, colors, food, drinks |
| Source training rows | 1,704 |
| Source test rows | 426 |
| Filename IDs | `0` through `21` |

Filename IDs `0`–`17` occur once in all 105 classes. IDs `18`–`21` are incomplete. This pattern supports using the filename as a signer/session grouping key, but it does not prove that every ID is a unique person. Confirm the interpretation with the dataset creators or original publication.

## Labels requiring review

The source label table includes punctuation variants and possible spelling issues, including `DON’T UNDERSTAND`, `DON’T KNOW`, `IM FINE`, `YOURE WELCOME`, and `WEELCHAIR PERSON`. Numeric IDs are treated as canonical model classes so display-text corrections will not silently change targets.

## Recommended additional fields

For a defensible robustness study, add human-reviewed metadata for:

- participant or signer ID;
- consent/status for newly collected data;
- dominant hand;
- lighting level;
- background complexity;
- camera distance and angle;
- signing speed;
- occlusion;
- device/camera; and
- reviewer notes.

Do not infer sensitive participant attributes from video. Store research metadata separately from public model artifacts.

