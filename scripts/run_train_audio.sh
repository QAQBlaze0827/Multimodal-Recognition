#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python3 -u scripts/train_audio_tiny_cnn.py --data-dir data/datasets/audio > data/logs/train_audio.log 2>&1
