from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download EmotionTalk Audio.tar from Hugging Face.")
    parser.add_argument("--repo-id", default="BAAI/Emotiontalk")
    parser.add_argument("--filename", default="Audio.tar")
    parser.add_argument("--output", default="data/datasets/emotiontalk_raw")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated dataset access.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: pip install -r requirements_train.txt") from exc

    print("[emotiontalk] This dataset is gated and licensed CC BY-NC-SA 4.0.")
    print("[emotiontalk] Accept the dataset terms on Hugging Face before running this script.")
    path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=args.filename,
        local_dir=str(output_dir),
        token=args.token,
    )
    print(f"[emotiontalk] Downloaded: {path}")


if __name__ == "__main__":
    main()
