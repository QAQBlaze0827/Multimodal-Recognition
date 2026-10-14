from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
import wave
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = ("neutral", "happy", "sad", "anger")
LABEL_MAP = {"neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "anger"}
EXCLUDED_LABELS = {"surprised", "disgusted", "fearful"}
# Keep whole recording groups together, including groups sharing a speaker.
SPLIT_GROUPS = {
    "train": ("G00002", "G00004", "G00005", "G00006", "G00007", "G00008",
              "G00009", "G00010", "G00011", "G00015"),
    "val": ("G00001", "G00003", "G00012"),
    "test": ("G00013", "G00014"),
}
GROUP_SPLIT = {group: split for split, groups in SPLIT_GROUPS.items() for group in groups}


def scan_archive(tar: tarfile.TarFile) -> tuple[list[dict], dict]:
    records = []
    audio = {}
    names = set()
    labels = Counter()
    group_speakers = defaultdict(set)
    for member in tar:
        if not member.isfile():
            continue
        if member.name in names:
            raise ValueError(f"Duplicate archive path: {member.name}")
        names.add(member.name)
        path = PurePosixPath(member.name)
        if path.suffix not in (".json", ".wav"):
            continue
        with tar.extractfile(member) as stream:
            if path.suffix == ".wav":
                with wave.open(stream, "rb") as wav:
                    if wav.getnframes() <= 0 or wav.getframerate() <= 0:
                        raise ValueError(f"Empty WAV: {member.name}")
                    audio[member.name] = {
                        "sample_rate": wav.getframerate(),
                        "channels": wav.getnchannels(),
                        "sample_width_bytes": wav.getsampwidth(),
                        "duration_seconds": wav.getnframes() / wav.getframerate(),
                    }
                continue
            raw = stream.read()
        payload = json.loads(raw.decode("utf-8"))
        relative = PurePosixPath(payload["file_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source path: {relative}")
        if str(path) != f"Audio/json/{relative.with_suffix('.json')}":
            raise ValueError(f"Metadata/source path mismatch: {path}")
        group = relative.parts[0]
        if group not in GROUP_SPLIT:
            raise ValueError(f"Unknown recording group: {group}; review the split policy first")
        speaker = payload["speaker_id"]
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"Missing speaker ID: {path}")
        group_speakers[group].add(speaker)
        source_label = payload["emotion_result"]
        if source_label not in LABEL_MAP and source_label not in EXCLUDED_LABELS:
            raise ValueError(f"Unknown emotion label: {source_label!r} in {path}")
        labels[source_label] += 1
        if source_label not in LABEL_MAP:
            continue
        annotations = list(payload["data"].values())
        votes = sum(item["emotion"] == source_label for item in annotations)
        times = payload["paragraphs"]
        start, end = float(times["startTime"]), float(times["endTime"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid timestamps: {path}")
        records.append({
            "split": GROUP_SPLIT[group],
            "source_json": str(path),
            "source_wav": f"Audio/wav/{relative}",
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "source_label": source_label,
            "label": LABEL_MAP[source_label],
            "speaker_id": speaker,
            "group_id": group,
            "conversation_id": str(relative.parent),
            "start_seconds": start,
            "end_seconds": end,
            "annotation_duration_seconds": float(times["duration"]),
            "matching_votes": votes,
            "annotator_count": len(annotations),
            "review_recommended": votes < 4,
        })

    if len(audio) != sum(labels.values()):
        raise ValueError("WAV and annotation counts differ; inspect the archive")
    for record in records:
        name = record["source_wav"]
        if name not in audio:
            raise ValueError(f"Missing WAV: {name}")
        record.update(audio[name])
    # Check all source speakers, not only speakers in the retained four classes.
    speakers = {
        split: set().union(*(group_speakers[g] for g in groups))
        for split, groups in SPLIT_GROUPS.items()
    }
    for left, right in combinations(speakers, 2):
        overlap = speakers[left] & speakers[right]
        if overlap:
            raise ValueError(f"Speaker overlap between {left} and {right}: {sorted(overlap)}")
    validate_records(records)
    return sorted(records, key=lambda row: row["source_wav"]), {
        "source_label_counts": dict(sorted(labels.items())),
        "wav_count": len(audio),
        "excluded_label_counts": {label: labels[label] for label in sorted(EXCLUDED_LABELS)},
        "source_group_speakers": {g: sorted(s) for g, s in sorted(group_speakers.items())},
    }


def validate_records(records: list[dict]) -> None:
    if len({r["source_wav"] for r in records}) != len(records):
        raise ValueError("Duplicate retained WAV path")
    for split in SPLIT_GROUPS:
        labels = {r["label"] for r in records if r["split"] == split}
        if labels != set(EMOTIONS):
            raise ValueError(f"Split {split} does not contain all four classes")
    for field in ("speaker_id", "group_id", "conversation_id", "source_wav"):
        sets = {s: {r[field] for r in records if r["split"] == s} for s in SPLIT_GROUPS}
        for left, right in combinations(sets, 2):
            if sets[left] & sets[right]:
                raise ValueError(f"{field} overlap between {left} and {right}")


def build_summary(records: list[dict], audit: dict, archive: Path) -> dict:
    splits = {}
    for split in SPLIT_GROUPS:
        rows = [r for r in records if r["split"] == split]
        counts = Counter(r["label"] for r in rows)
        splits[split] = {
            "count": len(rows),
            "percentage": 100 * len(rows) / len(records),
            "class_counts": {label: counts[label] for label in EMOTIONS},
            "speaker_ids": sorted({r["speaker_id"] for r in rows}),
            "group_ids": sorted({r["group_id"] for r in rows}),
            "review_recommended": sum(r["review_recommended"] for r in rows),
        }
    return {
        "policy": "emotiontalk-audio-four-class-speaker-disjoint-v1",
        "archive_path": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "retained_count": len(records),
        "label_order": list(EMOTIONS),
        "split_groups": SPLIT_GROUPS,
        "splits": splits,
        **audit,
        "checks": {
            "matching_wav_headers_readable": True,
            "unique_archive_member_paths": True,
            "speaker_group_conversation_path_disjoint": True,
            "all_four_classes_in_each_split": True,
            "full_audio_decoding_checked": False,
            "audio_content_duplicates_checked": False,
            "unseen_by_legacy_models": False,
        },
    }


def report_markdown(summary: dict) -> str:
    lines = [
        "# EmotionTalk 四類音訊資料切分檢查", "",
        "本次只建立資料清單，沒有訓練、解壓、轉檔或修改原始資料。", "",
        f"- 切分版本：`{summary['policy']}`",
        f"- 原始標註：{sum(summary['source_label_counts'].values()):,} 筆。",
        f"- 保留原始四類：{summary['retained_count']:,} 筆；僅將 angry 統一命名為 anger。",
        "- 驚訝、恐懼、厭惡只排除於新清單之外，原始檔案仍保留。", "",
        "| 集合 | 中性 | 開心 | 難過 | 生氣 | 合計 | 比例 | 說話者 | 待複核 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, split in summary["splits"].items():
        values = " | ".join(str(split["class_counts"][label]) for label in EMOTIONS)
        lines.append(f"| {name} | {values} | {split['count']} | {split['percentage']:.2f}% | "
                     f"{len(split['speaker_ids'])} | {split['review_recommended']} |")
    lines.extend(["", "## 切分方法", "",
                  "依來源組別和說話者固定分組，不隨機打散片段。這是自訂切分，不是官方切分。",
                  "先保留全部四類樣本；類別平衡應在後續訓練階段處理，不改測試集分布。"])
    for name, split in summary["splits"].items():
        lines.append(f"- {name}：組別 {', '.join(split['group_ids'])}；"
                     f"說話者 {', '.join(split['speaker_ids'])}。")
    lines.extend([
        "", "## 已檢查", "",
        "- 三組之間沒有重複 speaker_id、錄音組別、對話來源或 WAV 路徑。",
        "- 保留樣本有對應 WAV，WAV 檔頭可讀，各集合都包含四類。",
        "- 清單保留原始標籤、來源、時間、音訊格式與標註票數。",
        "- 少於 4 位標註者同意最終標籤時標為待複核，不自動刪除。", "",
        "## 使用與限制", "",
        "- train.csv、val.csv、test.csv 是清單，不是已解壓音檔；source_wav 是 Audio.tar 內部路徑。",
        "- 現有訓練腳本尚未接入這些清單，不能直接用舊資料夾重訓來代替此切分。",
        "- speaker_id 依原始標註判斷，未用生物辨識核實身分；未完整解碼或做音訊內容去重。",
        "- 舊模型曾使用此來源資料；本測試集不能宣稱對舊模型完全未見。",
        "- 不要依 test 的結果反覆調參；模型選擇用 val，最終才評估 test。",
        "- 此報告不是模型準確率報告，也未驗證真人麥克風、影像或樹莓派效果。",
        "- 後續仍須決定錄音切片、聲道與重取樣處理，並對齊訓練和正式推論。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit EmotionTalk and create speaker-disjoint CSV manifests; no training.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/datasets/emotiontalk_raw_stream/Audio.tar")
    parser.add_argument("--output", type=Path, help="New output directory; existing paths are never overwritten.")
    args = parser.parse_args()
    output = args.output or ROOT / "data/reports" / datetime.now().strftime("emotiontalk_four_class_%Y%m%d_%H%M%S_%f")
    if output.exists():
        raise SystemExit(f"Output already exists; no files changed: {output}")
    print(f"[manifest] Reading archive: {args.input}", flush=True)
    try:
        with tarfile.open(args.input, "r:*") as tar:
            records, audit = scan_archive(tar)
        summary = build_summary(records, audit, args.input)
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, wave.Error) as exc:
        raise SystemExit(f"Audit failed; no output written: {exc}") from exc
    output.mkdir(parents=True, exist_ok=False)
    for split in SPLIT_GROUPS:
        with (output / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(r for r in records if r["split"] == split)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(report_markdown(summary))
    for split, stats in summary["splits"].items():
        print(f"[manifest] {split}: {stats['count']} samples, {len(stats['speaker_ids'])} speakers")
    print(f"[manifest] Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
