"""
Download CREMA-D from Zenodo (fast mirror).
URLs: https://zenodo.org/records/14646870
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


EMOTION_MAP = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

# All tar files from Zenodo record 14646870
TAR_FILES = [
    f"wds-audio-{split}-{i:06d}.tar"
    for split in ["train", "test", "valid"]
    for i in range(20)
]
BASE_URL = "https://zenodo.org/records/14646870/files"


def download_file(url: str, dest: Path, desc: str = "") -> None:
    if dest.exists():
        return
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_tar(tar_path: Path, output_dir: Path) -> int:
    count = 0
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".wav"):
                continue
            name = Path(member.name).stem
            parts = name.split("_")
            if len(parts) < 3:
                continue
            emotion_code = parts[-2]
            label = EMOTION_MAP.get(emotion_code)
            if label is None:
                continue
            dest_dir = output_dir / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / Path(member.name).name
            if not dest_path.exists():
                with tar.extractfile(member) as src:
                    with open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/datasets/audio")
    args = parser.parse_args()

    output_dir = Path(args.output) / "crema_d"
    if output_dir.exists():
        print("[crema-d] Already exists, skipping")
        return

    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download all tar files in parallel
    print("[crema-d] Downloading from Zenodo...")
    urls = [(f"{BASE_URL}/{f}", cache_dir / f) for f in TAR_FILES]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(download_file, url, dest, desc=dest.name)
            for url, dest in urls
        ]
        for f in as_completed(futures):
            f.result()  # raise if error

    print("[crema-d] Extracting WAV files...")
    total = 0
    for f in sorted(cache_dir.iterdir()):
        if f.suffix == ".tar":
            n = extract_tar(f, output_dir)
            total += n
            print(f"  {f.name}: {n} files")

    # Cleanup cache
    shutil.rmtree(cache_dir)
    print(f"[crema-d] Extracted {total} files")


if __name__ == "__main__":
    main()
