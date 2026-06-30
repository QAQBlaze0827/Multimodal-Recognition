from __future__ import annotations
import re
import sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/train_audio_baseline_all.log"
with open(log_file) as f:
    lines = f.readlines()

count = 0
for line in lines:
    if "val_accuracy" in line and "Epoch" in line:
        parts = re.findall(r"Epoch (\d+)/\d+|val_accuracy: [\d.]+|val_loss: [\d.]+", line)
        if parts:
            print(" | ".join(parts))
            count += 1
            if count >= 5:
                break

print(f"\nTotal val lines: {sum(1 for l in lines if 'val_accuracy' in l)}")
