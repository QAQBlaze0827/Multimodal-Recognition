#!/usr/bin/env bash
set -euo pipefail

if [ ! -f .venv/bin/activate ]; then
    echo "[test] .venv not found. Run scripts/setup_wsl.sh first."
    exit 1
fi

source .venv/bin/activate
python app.py --vision-only --no-display --max-frames 10
