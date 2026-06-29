from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


RAVDESS_URL = (
    "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
)
CREMA_D_URL = (
    "https://github.com/CheyneyComputerScience/CREMA-D/raw/master/Audio%20WAV%20Files.zip"
)
TESS_URL = (
    "https://www.kaggle.com/api/v1/datasets/ejlok1/toronto-emotional-speech-set-tess/download"
)

EMOTION_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "disgust",
    "calm": "neutral",
    "pleasant_surprise": "surprise",
    "pleasant_surprised": "surprise",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download audio emotion datasets.")
    parser.add_argument("--output", default="data/datasets/audio")
    parser.add_argument("--datasets", nargs="+", default=["ravdess", "crema-d", "tess"],
                        choices=["ravdess", "crema-d", "tess"])
    return parser


def _download(url: str, dest: Path, desc: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[download] {dest.name} already exists, skipping")
        return
    print(f"[download] Downloading {desc} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"[download] Saved to {dest}")


def extract_ravdess(zip_path: Path, output_dir: Path) -> None:
    output_dir = output_dir / "ravdess"
    extract_dir = output_dir / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    count = 0
    for wav in extract_dir.rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) < 3:
            continue
        try:
            emotion_code = int(parts[2])
        except ValueError:
            continue
        ravdess_map = {
            1: "neutral", 2: "calm", 3: "happy", 4: "sad",
            5: "angry", 6: "fear", 7: "disgust", 8: "surprise",
        }
        label = ravdess_map.get(emotion_code)
        if label is None:
            continue
        mapped = EMOTION_MAP.get(label)
        if mapped is None:
            continue
        dest_dir = output_dir / mapped
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, dest_dir / wav.name)
        count += 1

    shutil.rmtree(extract_dir)
    print(f"[ravdess] Extracted {count} files")


def extract_cremad(zip_path: Path, output_dir: Path) -> None:
    output_dir = output_dir / "crema_d"
    extract_dir = output_dir / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    count = 0
    for wav in extract_dir.rglob("*.wav"):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        raw_emotion = parts[2].lower()
        cremad_map = {
            "ang": "angry", "dis": "disgust", "fea": "fear",
            "hap": "happy", "neu": "neutral", "sad": "sad",
        }
        label = cremad_map.get(raw_emotion)
        if label is None:
            continue
        mapped = EMOTION_MAP.get(label)
        if mapped is None:
            continue
        dest_dir = output_dir / mapped
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, dest_dir / wav.name)
        count += 1

    shutil.rmtree(extract_dir)
    print(f"[crema-d] Extracted {count} files")


def extract_tess(zip_path: Path, output_dir: Path) -> None:
    output_dir = output_dir / "tess"
    extract_dir = output_dir / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    count = 0
    for wav in extract_dir.rglob("*.wav"):
        folder = wav.parent.name.lower()
        tess_map = {
            "neutral": "neutral", "happy": "happy", "sad": "sad",
            "angry": "angry", "fear": "fear", "surprise": "surprise",
            "disgust": "disgust", "pleasant_surprise": "surprise",
        }
        label = tess_map.get(folder)
        if label is None:
            continue
        mapped = EMOTION_MAP.get(label)
        if mapped is None:
            continue
        dest_dir = output_dir / mapped
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, dest_dir / wav.name)
        count += 1

    shutil.rmtree(extract_dir)
    print(f"[tess] Extracted {count} files")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets

    if "ravdess" in datasets:
        zip_path = output_dir / "ravdess.zip"
        _download(RAVDESS_URL, zip_path, desc="RAVDESS")
        extract_ravdess(zip_path, output_dir)

    if "crema-d" in datasets:
        zip_path = output_dir / "crema_d.zip"
        _download(CREMA_D_URL, zip_path, desc="CREMA-D")
        extract_cremad(zip_path, output_dir)

    if "tess" in datasets:
        zip_path = output_dir / "tess.zip"
        _download(TESS_URL, zip_path, desc="TESS")
        extract_tess(zip_path, output_dir)

    print("[download] Audio datasets complete!")


if __name__ == "__main__":
    main()
