"""Build a clean Windows setup ZIP with no installed libraries or learner history."""
from __future__ import annotations
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    version = (ROOT / 'VERSION').read_text().strip()
    folder = f'Senyas-{version}-Windows-x64'
    files = [ROOT / name for name in ['app.py', 'README.md', 'VERSION', 'requirements.txt',
             'requirements-windows.lock', 'Install Senyas.cmd', 'Start Senyas.cmd', 'THIRD_PARTY_NOTICES.md']]
    for name in ['senyas', 'config', 'assets', 'scripts', 'docs', '.streamlit']:
        files.extend(path for path in (ROOT / name).rglob('*') if path.is_file()
                     and '__pycache__' not in path.parts and path.name != 'secrets.toml')
    files.append(ROOT / 'artifacts/hybrid_recognizer.joblib')
    files.append(ROOT / 'artifacts/model_summary.json')
    output = ROOT / 'dist'
    output.mkdir(exist_ok=True)
    archive = output / f'{folder}.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(set(files)):
            if not path.is_file():
                raise FileNotFoundError(path)
            relative = path.relative_to(ROOT).as_posix()
            if any(part in relative.split('/') for part in ('.git', '.venv', 'node_modules', '.runtime', 'dataset')):
                raise ValueError(f'Unexpected dependency/data directory: {relative}')
            info = zipfile.ZipInfo(f'{folder}/{relative}', date_time=(2026, 10, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, path.read_bytes())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / 'SHA256SUMS.txt').write_text(f'{digest}  {archive.name}\n')
    print(f'{archive}\n{archive.stat().st_size:,} bytes\nSHA256 {digest}')


if __name__ == '__main__':
    main()
