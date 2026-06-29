#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python3 -u scripts/train_video_mini_xception.py --data-dir data/datasets/fer > data/logs/train_vision.log 2>&1
