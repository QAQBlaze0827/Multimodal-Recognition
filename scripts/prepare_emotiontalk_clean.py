from __future__ import annotations

import argparse
import io
import json
import random
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16000
EMOTION_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "anger",
    "anger": "anger",
}


@dataclass(frozen=True)
class SampleRecord:
    wav_name: str
    label: str
    source_label: str
    votes: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare clean balanced EmotionTalk data for 4-class training.")
    parser.add_argument("--input", default="data/datasets/emotiontalk_raw_stream/Audio.tar")
    parser.add_argument("--output", default="data/datasets/emotiontalk_clean")
    parser.add_argument("--max-per-class", type=int, default=1110)
    parser.add_argument("--min-votes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    return EMOTION_MAP.get(str(value).strip().lower().replace("-", "_"))


def wav_name_for_json(json_name: str) -> str:
    return json_name.replace("/json/", "/wav/").removesuffix(".json") + ".wav"


def matching_votes(payload: dict, label: str) -> int:
    votes = 0
    data = payload.get("data", {})
    if not isinstance(data, dict):
        return votes
    for item in data.values():
        if isinstance(item, dict) and normalize_label(item.get("emotion")) == label:
            votes += 1
    return votes


def collect_records(tar_path: Path, min_votes: int) -> dict[str, SampleRecord]:
    records: dict[str, SampleRecord] = {}
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".json"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            try:
                payload = json.loads(extracted.read().decode("utf-8"))
            except Exception as exc:
                print(f"[emotiontalk-clean] Warning: skipped metadata {member.name}: {exc}")
                continue

            source_label = str(payload.get("emotion_result", "")).strip().lower()
            label = normalize_label(source_label)
            if label is None:
                continue
            votes = matching_votes(payload, label)
            if min_votes and votes < min_votes:
                continue
            wav_name = wav_name_for_json(member.name)
            records[wav_name] = SampleRecord(
                wav_name=wav_name,
                label=label,
                source_label=source_label,
                votes=votes,
            )
    return records


def choose_balanced(records: dict[str, SampleRecord], max_per_class: int, seed: int) -> dict[str, SampleRecord]:
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records.values():
        grouped[record.label].append(record)

    rng = random.Random(seed)
    selected: dict[str, SampleRecord] = {}
    for label, items in sorted(grouped.items()):
        rng.shuffle(items)
        limit = len(items) if max_per_class <= 0 else min(max_per_class, len(items))
        for record in items[:limit]:
            selected[record.wav_name] = record
        print(f"[emotiontalk-clean] selected {label}: {limit} / {len(items)}")
    return selected


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
    records = collect_records(input_path, min_votes=args.min_votes)
    selected = choose_balanced(records, max_per_class=args.max_per_class, seed=args.seed)

    exported = defaultdict(int)
    skipped = 0
    with tarfile.open(input_path) as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".wav"):
                continue
            record = selected.get(member.name)
            if record is None:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                skipped += 1
                continue
            try:
                audio, sr = sf.read(io.BytesIO(extracted.read()))
            except Exception as exc:
                print(f"[emotiontalk-clean] Warning: skipped audio {member.name}: {exc}")
                skipped += 1
                continue
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = resample_to_16k(np.asarray(audio, dtype=np.float32), int(sr))
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1e-8:
                audio = audio / peak

            dest_dir = output_dir / record.label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"emotiontalk_clean_{record.label}_{exported[record.label]:04d}.wav"
            if dest.exists() and not args.overwrite:
                skipped += 1
                continue
            sf.write(dest, audio.astype(np.float32), SAMPLE_RATE)
            exported[record.label] += 1

    for label in ("neutral", "happy", "sad", "anger"):
        print(f"[emotiontalk-clean] exported {label}: {exported[label]}")
    print(f"[emotiontalk-clean] skipped: {skipped}")
    print(f"[emotiontalk-clean] output: {output_dir}")


if __name__ == "__main__":
    main()
