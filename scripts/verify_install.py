"""Fail setup early on missing native libraries, content, or incompatible models."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mediapipe  # noqa: E402,F401
from senyas.curriculum import load_lessons, bundled_video  # noqa: E402
from senyas.frozen_recognizer import load_recognizer  # noqa: E402
from senyas.sequences import SequenceExtractor  # noqa: E402


def main():
    lessons = load_lessons()
    assert len(lessons) == 105 and lessons.category.nunique() == 10
    manifest = json.loads((ROOT / 'assets/lessons/manifest.json').read_text(encoding='utf-8'))
    for _, lesson in lessons.iterrows():
        assert bundled_video(lesson).is_file(), lesson['id_label']
    for clip in manifest['videos']:
        assert hashlib.sha256((ROOT / clip['video']).read_bytes()).hexdigest() == clip['sha256']
        assert clip['audio_tracks'] == 0
    for name in ('Decision Tree', 'Extra Trees'):
        artifact = load_recognizer(ROOT / 'artifacts/frozen', name)
        assert len(artifact['feature_names']) == 3392
    # Source checkouts can use Git's CRLF conversion; weights always use exact bytes.
    for filename, expected in json.loads((ROOT / 'artifacts/frozen/manifest.json').read_text())['runtime_source_lf_sha256'].items():
        assert hashlib.sha256((ROOT / filename).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected, filename
    with SequenceExtractor(target_fps=30, max_image_side=480, model_complexity=1):
        pass
    print('Startup check passed: 105 silent lessons, 10 levels, full camera tracker, both frozen 3,392-feature models and preprocessing hashes.')


if __name__ == '__main__':
    main()
