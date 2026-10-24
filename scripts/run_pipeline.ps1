$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$python = ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Run scripts/setup.ps1 first."
}

& $python scripts/prepare_dataset.py @args
& $python scripts/extract_features.py
& $python scripts/train_model.py

Write-Host "Pipeline complete. Start the UI with: .venv\Scripts\streamlit.exe run app.py"

