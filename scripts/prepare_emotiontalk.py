from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 16000

EMOTION_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "surprise": "happy",
    "surprised": "happy",
    "sad": "sad",
    "anger": "anger",
    "angry": "anger",
    "disgust": "anger",
    "disgusted": "anger",
    "fear": "anger",
    "fearful": "anger",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare EmotionTalk audio files for 4-class training.")
    parser.add_argument("--input", default="data/datasets/emotiontalk_raw/Audio.tar")
    parser.add_argument("--output", default="data/datasets/emotiontalk")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of exported wav files.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    return EMOTION_MAP.get(text)


def find_emotion(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("emotion_result", "emotion", "label"):
            label = normalize_label(value.get(key))
            if label:
                return label
        for item in value.values():
            label = find_emotion(item)
            if label:
                return label
    elif isinstance(value, list):
        for item in value:
            label = find_emotion(item)
            if label:
                return label
    else:
        return normalize_label(value)
    return None


def sample_keys(path: str) -> set[str]:
    p = Path(path)
    return {
        p.stem.lower(),
        p.name.lower(),
        path.replace("\\", "/").lower(),
    }


def iter_tar_files(tar_path: Path, suffixes: tuple[str, ...]):
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            lower = name.lower()
            if lower.endswith(".tar"):
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                tmp_name = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                        tmp_name = tmp.name
                        tmp.write(extracted.read())
                    yield from iter_tar_files(Path(tmp_name), suffixes)
                finally:
                    if tmp_name:
                        Path(tmp_name).unlink(missing_ok=True)
            elif lower.endswith(suffixes):
                extracted = tar.extractfile(member)
                if extracted is not None:
                    yield name, extracted.read()


def collect_labels(tar_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for name, data in iter_tar_files(tar_path, (".json", ".jsonl", ".csv")):
        lower = name.lower()
        try:
            if lower.endswith(".json"):
                payload = json.loads(data.decode("utf-8"))
                label = find_emotion(payload)
                file_name = payload.get("file_name") if isinstance(payload, dict) else None
                if label:
                    for key in sample_keys(str(file_name or name)):
                        labels[key] = label
            elif lower.endswith(".jsonl"):
                for line in data.decode("utf-8").splitlines():
                    payload = json.loads(line)
                    label = find_emotion(payload)
                    file_name = payload.get("file_name") if isinstance(payload, dict) else None
                    if label:
                        for key in sample_keys(str(file_name or name)):
                            labels[key] = label
            else:
                rows = csv.DictReader(io.StringIO(data.decode("utf-8")))
                for row in rows:
                    label = find_emotion(row)
                    file_name = row.get("file_name") or row.get("path") or row.get("audio")
                    if label:
                        for key in sample_keys(str(file_name or name)):
                            labels[key] = label
        except Exception as exc:
            print(f"[emotiontalk] Warning: skipped metadata {name}: {exc}")
    return labels


def infer_label_from_path(path: str) -> str | None:
    parts = Path(path).parts
    for part in parts:
        label = normalize_label(part)
        if label:
            return label
    return None


def label_for_wav(path: str, labels: dict[str, str]) -> str | None:
    for key in sample_keys(path):
        if key in labels:
            return labels[key]
    return infer_label_from_path(path)


def resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        return audio
    from scipy import signal

    new_len = int(len(audio) * SAMPLE_RATE / sample_rate)
    return signal.resample(audio, new_len)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"EmotionTalk archive not found: {input_path}")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("Install training dependencies first: pip install -r requirements_train.txt") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = collect_labels(input_path)
    print(f"[emotiontalk] Loaded {len(labels)} metadata labels")

    exported = 0
    skipped = 0
    for name, data in iter_tar_files(input_path, (".wav",)):
        label = label_for_wav(name, labels)
        if label is None:
            skipped += 1
            continue
        try:
            audio, sr = sf.read(io.BytesIO(data))
        except Exception as exc:
            print(f"[emotiontalk] Warning: skipped audio {name}: {exc}")
            skipped += 1
            continue
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = resample_to_16k(np.asarray(audio, dtype=np.float32), int(sr))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-8:
            audio = audio / peak

        dest_dir = output_dir / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"emotiontalk_{exported:06d}.wav"
        if dest.exists() and not args.overwrite:
            skipped += 1
            continue
        sf.write(dest, audio.astype(np.float32), SAMPLE_RATE)
        exported += 1
        if args.limit and exported >= args.limit:
            break

    print(f"[emotiontalk] Exported {exported} wav files to {output_dir}")
    print(f"[emotiontalk] Skipped {skipped} files")


if __name__ == "__main__":
    main()
