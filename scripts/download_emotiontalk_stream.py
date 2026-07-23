from __future__ import annotations

import argparse
from pathlib import Path

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream-download EmotionTalk Audio.tar with resume support.")
    parser.add_argument("--repo-id", default="BAAI/Emotiontalk")
    parser.add_argument("--filename", default="Audio.tar")
    parser.add_argument("--output", default="data/datasets/emotiontalk_raw_stream")
    parser.add_argument("--chunk-mb", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / args.filename
    partial_path = output_dir / f"{args.filename}.download"
    if final_path.exists():
        print(f"[emotiontalk] Already downloaded: {final_path}")
        return

    token = get_token()
    if not token:
        raise SystemExit("Hugging Face token not found. Run: hf auth login")

    info = HfApi().repo_info(args.repo_id, repo_type="dataset", files_metadata=True)
    expected_size = None
    for sibling in info.siblings:
        if sibling.rfilename == args.filename:
            expected_size = sibling.size
            break
    if expected_size is None:
        raise SystemExit(f"{args.filename} not found in {args.repo_id}")

    downloaded = partial_path.stat().st_size if partial_path.exists() else 0
    if downloaded > expected_size:
        raise SystemExit(f"Partial file is larger than expected: {partial_path}")

    if downloaded == expected_size:
        partial_path.rename(final_path)
        print(f"[emotiontalk] Downloaded: {final_path}")
        return

    url = f"https://huggingface.co/datasets/{args.repo_id}/resolve/main/{args.filename}"
    headers = {"Authorization": f"Bearer {token}"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
        print(f"[emotiontalk] Resuming from {downloaded / 1024 / 1024:.1f} MB")
    else:
        print(f"[emotiontalk] Starting download, expected {expected_size / 1024 / 1024:.1f} MB")

    chunk_size = args.chunk_mb * 1024 * 1024
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as response:
        if downloaded and response.status_code != 206:
            raise SystemExit(f"Server did not accept resume request: HTTP {response.status_code}")
        response.raise_for_status()
        with partial_path.open("ab") as fh:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                print(f"[emotiontalk] {downloaded / 1024 / 1024:.1f} / {expected_size / 1024 / 1024:.1f} MB", flush=True)

    actual_size = partial_path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(f"Downloaded size mismatch: {actual_size} != {expected_size}")

    if final_path.exists():
        raise SystemExit(f"Final file already exists, not overwriting: {final_path}")
    partial_path.rename(final_path)
    print(f"[emotiontalk] Downloaded: {final_path}")


if __name__ == "__main__":
    main()
