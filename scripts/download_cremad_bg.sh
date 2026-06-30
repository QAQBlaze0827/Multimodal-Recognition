#!/bin/bash
cd /home/vmp010/Multimodal-Recognition
source .venv/bin/activate
nohup python scripts/download_audio_datasets.py --datasets crema-d > crema_d_download.log 2>&1 &
echo "PID: $!"
