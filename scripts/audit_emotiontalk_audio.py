from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
import sys
import tarfile
import wave
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_emotiontalk_manifests import validate_records


def inspect_wav(raw: bytes) -> dict:
    with wave.open(io.BytesIO(raw), "rb") as wav:
        rate, channels, width, frames = (
            wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()
        )
        if width != 2 or channels not in (1, 2) or frames == 0:
            raise ValueError("Expected non-empty mono/stereo PCM16 WAV")
        pcm = wav.readframes(frames)
    if len(pcm) != frames * channels * width:
        raise ValueError("Truncated PCM payload")
    values = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels)
    samples = values.astype(np.float64) / 32768.0
    rms = np.sqrt(np.mean(samples ** 2, axis=0))
    mono = samples.mean(axis=1)
    mono_rms = float(np.sqrt(np.mean(mono ** 2)))
    strongest = float(rms.max())
    correlation = None
    if channels == 2:
        centered = samples - samples.mean(axis=0)
        denominator = float(np.sqrt(np.sum(centered[:, 0] ** 2) * np.sum(centered[:, 1] ** 2)))
        if denominator > 0:
            correlation = float(np.sum(centered[:, 0] * centered[:, 1]) / denominator)
    # Hash decoded PCM and its format, excluding non-audio RIFF metadata.
    digest = hashlib.sha256(struct.pack("<III", rate, channels, width) + pcm).hexdigest()
    return {
        "sample_rate": rate,
        "channels": channels,
        "duration_seconds": frames / rate,
        "pcm_sha256": digest,
        "left_rms": float(rms[0]),
        "right_rms": float(rms[1]) if channels == 2 else None,
        "mono_rms": mono_rms,
        "peak": float(np.max(np.abs(samples))),
        "full_scale_fraction": float(np.mean((values == 32767) | (values == -32768))),
        "channel_correlation": correlation,
        "identical_stereo_channels": channels == 2 and bool(np.array_equal(values[:, 0], values[:, 1])),
        "channel_imbalance": channels == 2 and strongest > 0 and float(rms.min()) < 0.1 * strongest,
        "mono_cancellation": channels == 2 and strongest > 0 and mono_rms < 0.5 * strongest,
        "exact_silence": bool(np.all(values == 0)),
        "low_whole_clip_rms": mono_rms < 0.01,
    }


def duplicate_groups(rows: list[dict]) -> list[dict]:
    hashes = defaultdict(list)
    for row in rows:
        hashes[row["pcm_sha256"]].append(row)
    return [
        {"pcm_sha256": digest,
         "cross_split": len({r["split"] for r in matches}) > 1,
         "conflicting_labels": len({r["label"] for r in matches}) > 1,
         "samples": [{k: r[k] for k in ("split", "label", "source_wav")} for r in matches]}
        for digest, matches in sorted(hashes.items()) if len(matches) > 1
    ]


def read_manifests(folder: Path) -> tuple[list[dict], dict]:
    rows = []
    hashes = {}
    for split in ("train", "val", "test"):
        path = folder / f"{split}.csv"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8", newline="") as stream:
            subset = list(csv.DictReader(stream))
        if any(r["split"] != split for r in subset):
            raise ValueError(f"Incorrect split column: {path}")
        rows.extend(subset)
    validate_records(rows)
    return rows, hashes


def audit_archive(archive: Path, records: list[dict]) -> tuple[list[dict], list[dict]]:
    wanted = {r["source_wav"]: r for r in records}
    metadata = {r["source_json"]: r for r in records}
    seen_audio, seen_metadata = set(), set()
    results, errors = [], []
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            name = member.name
            if not member.isfile() or (name not in wanted and name not in metadata):
                continue
            try:
                with tar.extractfile(member) as stream:
                    raw = stream.read()
                if name in metadata:
                    if name in seen_metadata:
                        raise ValueError("Duplicate metadata path")
                    seen_metadata.add(name)
                    if hashlib.sha256(raw).hexdigest() != metadata[name]["metadata_sha256"]:
                        raise ValueError("Metadata differs from manifest")
                    continue
                if name in seen_audio:
                    raise ValueError("Duplicate WAV path")
                seen_audio.add(name)
                metrics = inspect_wav(raw)
                source = wanted[name]
                if metrics["sample_rate"] != int(source["sample_rate"]) or metrics["channels"] != int(source["channels"]):
                    raise ValueError("WAV format differs from manifest")
                if abs(metrics["duration_seconds"] - float(source["duration_seconds"])) > 1 / metrics["sample_rate"]:
                    raise ValueError("WAV duration differs from manifest")
                results.append({**{k: source[k] for k in ("split", "label", "speaker_id", "source_wav")}, **metrics})
                if len(results) % 2000 == 0:
                    print(f"[audit] Decoded {len(results)}/{len(records)} WAV files", flush=True)
            except (OSError, EOFError, ValueError, wave.Error) as exc:
                errors.append({"path": name, "reason": str(exc)})
    for name in sorted(set(wanted) - seen_audio):
        errors.append({"path": name, "reason": "Missing WAV"})
    for name in sorted(set(metadata) - seen_metadata):
        errors.append({"path": name, "reason": "Missing metadata"})
    return sorted(results, key=lambda r: r["source_wav"]), errors


def summarize(rows: list[dict], errors: list[dict], manifest_hashes: dict) -> dict:
    duplicates = duplicate_groups(rows)
    flags = ("channel_imbalance", "mono_cancellation", "exact_silence", "low_whole_clip_rms", "identical_stereo_channels")
    splits = {}
    for split in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split]
        durations = [r["duration_seconds"] for r in subset]
        splits[split] = {
            "decoded": len(subset),
            "formats": dict(Counter(f"{r['sample_rate']}Hz/{r['channels']}ch" for r in subset)),
            "mean_duration_seconds": float(np.mean(durations)) if durations else None,
            "median_duration_seconds": float(np.median(durations)) if durations else None,
            "shorter_than_1_5s": sum(d < 1.5 for d in durations),
            "longer_than_1_5s": sum(d > 1.5 for d in durations),
            "full_scale_over_0_1_percent": sum(r["full_scale_fraction"] > 0.001 for r in subset),
            **{flag: sum(r[flag] for r in subset) for flag in flags},
        }
    return {
        "status": "blocked" if errors or any(d["cross_split"] for d in duplicates) else "decoded_review_required",
        "manifest_sha256": manifest_hashes,
        "decoded": len(rows),
        "splits": splits,
        "errors": errors,
        "exact_duplicate_groups": duplicates,
        "limits": ["No perceptual/near-duplicate detection", "No microphone recording or listening test", "No model training or accuracy evaluation"],
    }


def report_markdown(summary: dict) -> str:
    lines = ["# EmotionTalk 音訊完整讀取與前處理檢查", "",
             "本次只在記憶體中解碼音訊並計算統計，沒有解壓寫入、轉檔、刪資料或訓練。", "",
             f"- 成功解碼：{summary['decoded']:,} 筆。",
             f"- 錯誤：{len(summary['errors'])} 筆；狀態：`{summary['status']}`。",
             f"- 完全相同的 PCM 群組：{len(summary['exact_duplicate_groups'])}；"
             f"其中跨集合：{sum(d['cross_split'] for d in summary['exact_duplicate_groups'])}。", "",
             "| 集合 | 成功讀取 | 平均秒數 | 短於 1.5 秒 | 長於 1.5 秒 | 聲道不平衡 | 混音抵銷 | 全零 | 低 RMS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in summary["splits"].items():
        average = f"{s['mean_duration_seconds']:.2f}" if s["decoded"] else "無資料"
        lines.append(f"| {name} | {s['decoded']} | {average} | {s['shorter_than_1_5s']} | "
                     f"{s['longer_than_1_5s']} | {s['channel_imbalance']} | {s['mono_cancellation']} | "
                     f"{s['exact_silence']} | {s['low_whole_clip_rms']} |")
    lines.extend(["", "## 音訊格式", ""])
    for name, s in summary["splits"].items():
        lines.append(f"- {name}：{s['formats']}；雙聲道完全相同 {s['identical_stereo_channels']} 筆；"
                     f"滿刻度樣本超過 0.1%：{s['full_scale_over_0_1_percent']} 筆。")
    lines.extend(["", "## 指標如何解讀", "",
                  "- 聲道不平衡：弱聲道 RMS 小於強聲道的 10%；不能據此認定哪邊是人聲或雜訊。",
                  "- 混音抵銷：左右平均後的 RMS 小於強聲道的一半；需要聽音複核，不自動選聲道。",
                  "- 低 RMS：整段左右平均後 RMS 小於 0.01，只是提示，不等同於沒有說話或正式 VAD 判定。",
                  "- 滿刻度比例：PCM16 樣本等於 -32768 或 32767 的比例，是削波線索，不是聽感品質評分。",
                  "- 重複檢查只比對相同取樣率、聲道數、位元深度的完全相同 PCM；不涵蓋近似、重取樣或片段重疊。", "",
                  "## 下一步前處理方案（尚未實作）", "",
                  "- 先在 train 複核低音量、聲道或削波提示；不因 test 指標反覆調整模型或資料規則。",
                  "- 原始資料保留，統一的聲道處理與帶抗混疊的 16 kHz 重取樣需同時供訓練、評估與部署使用。",
                  "- 保留來源切分後再切片，避免同一錄音片段跨集合；長錄音不要永遠只取開頭。",
                  "- Tiny CNN 基準可先沿用 1.5 秒窗口：訓練從不同位置取樣，驗證用固定窗口並按原錄音彙整。",
                  "- 短錄音的補齊方式及靜音、峰值正規化、高通、MFCC 順序需要固定並寫測試。",
                  "- 取樣率、窗口長度、MFCC 維度與類別順序需隨模型一起保存；若改用預訓練模型，依其輸入規格另訂。",
                  "- 有錯誤或跨集合重複時應先查明原因。本報告未自動修復、排除或重新分組任何資料。",
                  "- 舊模型可能見過這些來源；本次切分不是新舊模型公平比較的獨立實機測試集。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode manifest WAVs and audit channels, lengths and exact duplicates without training.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/datasets/emotiontalk_raw_stream/Audio.tar")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/reports/emotiontalk_four_class_20261014_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "data/reports" / datetime.now().strftime("emotiontalk_audio_audit_%Y%m%d_%H%M%S_%f")
    if output.exists():
        raise SystemExit(f"Output already exists; nothing overwritten: {output}")
    records, manifest_hashes = read_manifests(args.manifests)
    rows, errors = audit_archive(args.input, records)
    summary = summarize(rows, errors, manifest_hashes)
    summary.update({"archive_path": str(args.input.resolve()), "expected": len(records)})
    output.mkdir(parents=True, exist_ok=False)
    if rows:
        with (output / "audio_metrics.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(report_markdown(summary))
    print(f"[audit] {summary['status']}: {len(rows)}/{len(records)} decoded; report: {output / 'report.md'}")
    if summary["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
