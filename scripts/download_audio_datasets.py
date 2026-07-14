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
    "https://bj.bcebos.com/paddleaudio/datasets/TESS_Toronto_emotional_speech_set.zip"
)

EMOTION_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "anger",
    "calm": "neutral",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download audio emotion datasets.")
    parser.add_argument("--output", default="data/datasets/audio")
    parser.add_argument("--datasets", nargs="+", default=["ravdess", "tess"],
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
            5: "angry",
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


def download_cremad(output_dir: Path) -> None:
    output_dir = output_dir / "crema_d"
    if output_dir.exists():
        print(f"[crema-d] Already exists, skipping")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print(f"[crema-d] Cannot download: {exc}")
        print("  Run: pip install huggingface_hub")
        return

    repo_id = "razahtet/crema-d-audio"
    print(f"[crema-d] Downloading snapshot from {repo_id} ...")
    cache_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    audio_root = Path(cache_dir) / "AudioWAV"
    if not audio_root.exists():
        print(f"[crema-d] ERROR: AudioWAV directory not found in {cache_dir}")
        # fallback: search for wav files
        wav_files = list(Path(cache_dir).rglob("*.wav"))
        if wav_files:
            audio_root = wav_files[0].parent
        else:
            print(f"[crema-d] No wav files found, aborting")
            return

    emotion_map = {
        "ANG": "anger", "DIS": "disgust", "FEA": "fear",
        "HAP": "happy", "NEU": "neutral", "SAD": "sad",
    }
    count = 0
    for wav in Path(audio_root).glob("*.wav"):
        stem = wav.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        emotion_code = parts[-2]  # e.g., 1001_DFA_ANG_XX -> ANG
        label = emotion_map.get(emotion_code)
        if label is None:
            continue
        dest_dir = output_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / wav.name
        if not dest_path.exists():
            shutil.copy2(wav, dest_path)
        count += 1

    print(f"[crema-d] Extracted {count} files")


def extract_tess(zip_path: Path, output_dir: Path) -> None:
    output_dir = output_dir / "tess"
    extract_dir = output_dir / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    tess_map = {
        "angry": "angry",
        "happy": "happy", "neutral": "neutral", "sad": "sad",
    }

    count = 0
    for wav in extract_dir.rglob("*.wav"):
        parent = wav.parent.name
        parts = parent.split("_")
        if len(parts) < 2:
            continue
        raw_emotion = "_".join(parts[1:]).lower()
        label = tess_map.get(raw_emotion)
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
        download_cremad(output_dir)

    if "tess" in datasets:
        zip_path = output_dir / "tess.zip"
        _download(TESS_URL, zip_path, desc="TESS")
        extract_tess(zip_path, output_dir)

    print("[download] Audio datasets complete!")


if __name__ == "__main__":
    main()
