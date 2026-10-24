$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if ($pythonCommand) {
    $pythonExecutable = $pythonCommand.Source
} elseif (Test-Path -LiteralPath $bundledPython) {
    $pythonExecutable = $bundledPython
} else {
    throw "Python 3.11 or 3.12 was not found on PATH. Install Python, reopen VS Code, and run this task again."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $pythonExecutable -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install --default-timeout 180 -r requirements.txt

Write-Host "Setup complete. Select .venv\Scripts\python.exe as the VS Code interpreter."
