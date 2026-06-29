#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python3 -u scripts/download_audio_datasets.py --datasets crema-d tess > data/logs/download_audio_cremad_tess.log 2>&1
