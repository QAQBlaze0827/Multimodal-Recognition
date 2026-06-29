#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${1:-3.11}"

echo "[setup] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev \
    portaudio19-dev \
    libgl1 libglib2.0-0

echo "[setup] Creating Python virtual environment..."
python3 -m venv .venv

echo "[setup] Activating virtual environment..."
source .venv/bin/activate

echo "[setup] Upgrading pip..."
python -m pip install --upgrade pip

echo "[setup] Installing runtime dependencies..."
pip install -r requirements.txt

echo "[setup] Done."
echo "Run: source .venv/bin/activate"
echo "Then: python app.py --vision-only"
