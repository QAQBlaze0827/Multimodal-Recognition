$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
    Write-Host "[test] .venv not found. Run scripts/setup_windows.ps1 first."
    exit 1
}

& .\.venv\Scripts\Activate.ps1
python app.py --vision-only --no-display --max-frames 10
