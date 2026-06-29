from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image


EMOTIONS_MAP = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and extract FER2013 dataset.")
    parser.add_argument("--output", default="data/datasets/fer")
    parser.add_argument("--csv-url", default="https://raw.githubusercontent.com/anomalyco/FER2013/master/fer2013.csv")
    return parser


def download_csv(url: str, dest: Path) -> Path:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Downloading FER2013 from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[download] Saved to {dest}")
    return dest


def extract_images(csv_path: Path, output_dir: Path) -> None:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["emotion", "pixels", "Usage"], f"Unexpected CSV header: {header}"

        for row in reader:
            emotion_code = int(row[0])
            pixels = np.array(row[1].split(), dtype=np.uint8).reshape(48, 48)
            usage = row[2].strip()

            if usage == "Training":
                split = "train"
            elif usage in ("PublicTest", "PrivateTest"):
                split = "val"
            else:
                continue

            label = EMOTIONS_MAP[emotion_code]
            img_dir = output_dir / split / label
            img_dir.mkdir(parents=True, exist_ok=True)
            existing = len(list(img_dir.iterdir()))
            img_path = img_dir / f"{existing:05d}.png"
            Image.fromarray(pixels, mode="L").save(img_path)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output)

    csv_path = output_dir / "fer2013.csv"
    if not csv_path.exists():
        try:
            import kagglehub
            print("[download] Using kagglehub...")
            path = kagglehub.dataset_download("msambare/fer2013")
            import glob
            matches = glob.glob(str(Path(path) / "*.csv"))
            if matches:
                csv_path = Path(matches[0])
            else:
                raise FileNotFoundError("No CSV found in kagglehub download")
        except ImportError:
            download_csv(args.csv_url, csv_path)
        except Exception as e:
            print(f"[download] kagglehub failed ({e}), falling back to direct download...")
            download_csv(args.csv_url, csv_path)
    else:
        print(f"[download] CSV already exists: {csv_path}")

    extract_images(csv_path, output_dir)

    train_count = sum(len(list(d.iterdir())) for d in (output_dir / "train").iterdir())
    val_count = sum(len(list(d.iterdir())) for d in (output_dir / "val").iterdir())
    print(f"[download] Done: {train_count} training, {val_count} validation images")


if __name__ == "__main__":
    main()
