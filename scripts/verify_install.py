"""Fail setup early on missing native libraries, content, or incompatible models."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mediapipe  # noqa: E402,F401
from senyas.curriculum import load_lessons, bundled_video  # noqa: E402
from senyas.features import LandmarkExtractor, temporal_feature_names  # noqa: E402
from senyas.model import load_artifact, select_recognizer  # noqa: E402


def main():
    lessons = load_lessons()
    assert len(lessons) == 105 and lessons.category.nunique() == 10
    manifest = json.loads((ROOT / 'assets/lessons/manifest.json').read_text(encoding='utf-8'))
    for _, lesson in lessons.iterrows():
        assert bundled_video(lesson).is_file(), lesson['id_label']
    for clip in manifest['videos']:
        assert hashlib.sha256((ROOT / clip['video']).read_bytes()).hexdigest() == clip['sha256']
        assert clip['audio_tracks'] == 0
    artifact = load_artifact(ROOT / 'artifacts/hybrid_recognizer.joblib')
    assert artifact['feature_names'] == temporal_feature_names()
    assert len(select_recognizer(artifact, 'Decision Tree')['model'].classes_) == 105
    with LandmarkExtractor():
        pass
    print('Startup check passed: 105 lessons, 10 levels, camera libraries, and recognition models.')


if __name__ == '__main__':
    main()
