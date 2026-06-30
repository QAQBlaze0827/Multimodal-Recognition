from __future__ import annotations
import re

with open("logs/train_audio_phase3_v3.log") as f:
    lines = f.readlines()

for line in lines:
    if "val_accuracy" in line:
        parts = re.findall(r"val_accuracy: [\d.]+|val_loss: [\d.]+", line)
        if parts:
            print(" | ".join(parts))
