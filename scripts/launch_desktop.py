"""Launch the installed local app and open its browser window."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.runtime'
KERNEL = ctypes.WinDLL('kernel32', use_last_error=True)
KERNEL.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
KERNEL.OpenProcess.restype = wintypes.HANDLE
KERNEL.CloseHandle.argtypes = [wintypes.HANDLE]
KERNEL.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
KERNEL.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
KERNEL.CreateMutexW.restype = wintypes.HANDLE
KERNEL.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
KERNEL.ReleaseMutex.argtypes = [wintypes.HANDLE]


def alive(pid):
    handle = KERNEL.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    try:
        return bool(KERNEL.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        KERNEL.CloseHandle(handle)


def healthy(url):
    try:
        with urllib.request.urlopen(url + '/_stcore/health', timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    state_file = STATE / 'server.json'
    # Avoid two simultaneous shortcut clicks launching two servers.
    mutex = KERNEL.CreateMutexW(None, False, 'Local\\SenyasLauncher' + hashlib.sha256(str(ROOT).encode()).hexdigest()[:16])
    if not mutex or KERNEL.WaitForSingleObject(mutex, 120000) not in (0, 128):
        raise RuntimeError('Another Senyas launch is still starting. Try again shortly.')
    try:
        if state_file.exists():
            previous = json.loads(state_file.read_text())
            if alive(previous['pid']) and healthy(previous['url']):
                if not args.no_browser:
                    webbrowser.open(previous['url'])
                print(previous['url'])
                return
        port = None
        for candidate in range(8765, 8796):
            with socket.socket() as probe:
                try:
                    probe.bind(('127.0.0.1', candidate))
                    port = candidate
                    break
                except OSError:
                    continue
        if port is None:
            raise RuntimeError('No available local port. Close other Senyas instances and try again.')
        url = f'http://127.0.0.1:{port}'
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment['SENYAS_PROGRESS_PATH'] = str(ROOT / 'data/progress.json')
        with (STATE / 'server.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_app.py'),
                '--server.address', '127.0.0.1', '--server.port', str(port),
                '--server.headless', 'true', '--server.fileWatcherType', 'none',
                '--browser.gatherUsageStats', 'false'], cwd=ROOT, env=environment,
                stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError('Senyas could not start. See .runtime/server.log or run the installer again.')
            if healthy(url):
                state_file.write_text(json.dumps({'pid': process.pid, 'url': url}))
                if not args.no_browser:
                    webbrowser.open(url)
                print(url)
                return
            time.sleep(1)
        process.terminate()
        raise RuntimeError('Startup timed out. See .runtime/server.log.')
    finally:
        KERNEL.ReleaseMutex(mutex)
        KERNEL.CloseHandle(mutex)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if '--no-browser' in sys.argv:
            raise
        ctypes.windll.user32.MessageBoxW(0, str(exc), 'Senyas', 0x10)
        sys.exit(1)
