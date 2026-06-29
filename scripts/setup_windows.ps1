param(
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

Write-Host "[setup] Creating Python $PythonVersion virtual environment..."
py -$PythonVersion -m venv .venv

Write-Host "[setup] Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "[setup] Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "[setup] Installing runtime dependencies..."
pip install -r requirements.txt

Write-Host "[setup] Done."
Write-Host "Run: .\.venv\Scripts\Activate.ps1"
Write-Host "Then: python app.py --vision-only"
