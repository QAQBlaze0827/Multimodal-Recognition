from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tarfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_emotiontalk_audio import inspect_wav
from scripts.train_audio_emotiontalk_manifest import (
    EMOTIONS, aggregate_recordings, validate_source, write_csv, write_json,
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def indexed(rows):
    result = {}
    for row in rows:
        key = row["source_wav"]
        if key in result:
            raise ValueError(f"Duplicate source: {key}")
        result[key] = row
    return result


def join_validation(predictions, manifest, audit):
    if not manifest or any(r["split"] != "val" for r in manifest):
        raise ValueError("Only a nonempty validation manifest is accepted")
    sources = indexed(manifest)
    predicted = indexed(predictions)
    audited = indexed(r for r in audit if r["split"] == "val")
    if set(predicted) != set(sources) or not set(sources) <= set(audited):
        raise ValueError("Validation sources do not match predictions/audit")
    rows, probabilities, owners = [], [], []
    for index, (path, meta) in enumerate(sources.items()):
        p, a = predicted[path], audited[path]
        if (meta["label"] not in EMOTIONS or p["label"] != meta["label"]
                or a["label"] != meta["label"] or a["speaker_id"] != meta["speaker_id"]
                or a["pcm_sha256"] != p["pcm_sha256"]):
            raise ValueError(f"Mismatched label, speaker or PCM: {path}")
        active, inactive = p["active_windows"], p["inactive_windows"]
        if any(type(n) is not int or n < 0 for n in (active, inactive)) or active + inactive == 0:
            raise ValueError("Invalid window counts")
        scores = p["scores"]
        if active == 0:
            if p["prediction"] is not None or scores is not None:
                raise ValueError("Inactive recording must abstain")
            confidence, outcome = None, "abstain"
        else:
            scores = np.asarray(scores, dtype=float)
            if (scores.shape != (4,) or not np.isfinite(scores).all() or np.any(scores < 0)
                    or not np.isclose(scores.sum(), 1, atol=1e-4)
                    or p["prediction"] != EMOTIONS[int(np.argmax(scores))]):
                raise ValueError("Invalid prediction probabilities or label")
            probabilities.append(scores)
            owners.append(index)
            confidence = float(scores.max())
            outcome = "correct" if p["prediction"] == p["label"] else "incorrect"
        duration, rms = float(a["duration_seconds"]), float(a["mono_rms"])
        if not np.isfinite([duration, rms]).all() or duration <= 0 or rms < 0:
            raise ValueError("Invalid duration or RMS")
        if (int(a["sample_rate"]) != int(meta["sample_rate"])
                or int(a["channels"]) != int(meta["channels"])
                or abs(duration - float(meta["duration_seconds"])) > 1 / int(a["sample_rate"])):
            raise ValueError("Audit format/duration differs from manifest")
        rows.append({"source_wav": path, "label": p["label"],
                     "prediction": p["prediction"] or "abstain", "outcome": outcome,
                     "confidence": confidence, "duration_seconds": duration, "mono_rms": rms,
                     "active_windows": active, "inactive_windows": inactive,
                     "inactive_fraction": inactive / (active + inactive),
                     "speaker_id": meta["speaker_id"], "group_id": meta["group_id"],
                     "matching_votes": int(meta["matching_votes"]),
                     "review_recommended": meta["review_recommended"] == "True",
                     "pcm_sha256": p["pcm_sha256"]})
    metrics, _ = aggregate_recordings(np.asarray(probabilities).reshape(-1, 4), owners,
                                      [r["label"] for r in rows])
    return rows, metrics


def describe(rows):
    counts = Counter(r["outcome"] for r in rows)
    result = {"count": len(rows), **{k: counts[k] for k in ("correct", "incorrect", "abstain")},
              "accuracy": counts["correct"] / len(rows) if rows else None,
              "labels": dict(Counter(r["label"] for r in rows))}
    for key in ("duration_seconds", "mono_rms", "active_windows", "inactive_fraction"):
        result[f"median_{key}"] = float(np.median([r[key] for r in rows])) if rows else None
    return result


def grouped_summary(rows):
    groups = {"outcome": {}, "class_outcome": {}, "duration": {}, "rms": {}, "speaker": {}}
    for outcome in ("correct", "incorrect", "abstain"):
        groups["outcome"][outcome] = describe([r for r in rows if r["outcome"] == outcome])
        for label in EMOTIONS:
            groups["class_outcome"][f"{label}/{outcome}"] = describe(
                [r for r in rows if r["outcome"] == outcome and r["label"] == label])
    for name, low, high in (("<1.5s", 0, 1.5), ("1.5-3s", 1.5, 3),
                            ("3-6s", 3, 6), (">=6s", 6, float("inf"))):
        groups["duration"][name] = describe([r for r in rows if low <= r["duration_seconds"] < high])
    for low_rms in (True, False):
        groups["rms"]["<0.01" if low_rms else ">=0.01"] = describe(
            [r for r in rows if (r["mono_rms"] < 0.01) == low_rms])
    for speaker in sorted({r["speaker_id"] for r in rows}):
        groups["speaker"][speaker] = describe([r for r in rows if r["speaker_id"] == speaker])
    return groups


def choose_examples(rows):
    chosen = {}

    def add(candidates, reason, median=False):
        ordered = sorted(candidates, key=lambda r: (r["confidence"] or 0, r["source_wav"]))
        if ordered:
            row = ordered[len(ordered) // 2] if median else ordered[-1]
            chosen.setdefault(row["source_wav"], {**row, "selection_reason": []})["selection_reason"].append(reason)

    for label in EMOTIONS:
        add([r for r in rows if r["label"] == label and r["outcome"] == "correct"],
            f"{label}: median-confidence correct case", median=True)
        add([r for r in rows if r["label"] == label and r["outcome"] == "incorrect"],
            f"{label}: highest-confidence error")
    for label in ("happy", "sad"):
        add([r for r in rows if r["prediction"] == label and r["outcome"] == "incorrect"],
            f"false {label}: highest-confidence case")
    add([r for r in rows if r["duration_seconds"] < 1.5 and r["outcome"] == "incorrect"],
        "short recording: highest-confidence error")
    add([r for r in rows if r["outcome"] == "abstain"], "abstention example")
    return [{**r, "selection_reason": "; ".join(r["selection_reason"])} for r in chosen.values()]


def review_audio(tar, case, source):
    if source["split"] != "val" or source["source_wav"] != case["source_wav"]:
        raise ValueError("Review audio must belong to validation")
    validate_source(tar, source)
    with tar.extractfile(case["source_wav"]) as stream:
        raw = stream.read()
    if inspect_wav(raw)["pcm_sha256"] != case["pcm_sha256"]:
        raise ValueError("Review PCM differs from saved prediction")
    return raw


def require_new_outputs(report_dir, audio_dir):
    for path in (report_dir, audio_dir):
        if path.exists():
            raise FileExistsError(f"Output already exists; nothing overwritten: {path}")


def markdown(summary, report_dir, audio_dir):
    metrics = summary["metrics"]
    outcomes = summary["groups"]["outcome"]
    common_errors = "、".join(f"{p['label']} 判成 {p['prediction']}（{p['count']} 筆）"
                            for p in summary["top_confusions"][:3]) or "無分類誤判"
    lines = ["# EmotionTalk 驗證集誤判診斷", "",
             "只分析既有驗證紀錄，沒有重訓、重新推論、修改標籤或讀取 test 音訊。", "",
             f"- 最佳輪次：{summary['best_epoch']}；錄音 {metrics['recordings']} 筆。",
             f"- Accuracy：{metrics['accuracy']:.2%}；Macro-F1：{metrics['macro_f1']:.4f}；"
             f"拒判 {metrics['rejected']} 筆，仍計入分母。",
             f"- 全猜中性 accuracy 基準：{summary['majority_baseline_accuracy']:.2%}，不代表能辨識其他類別。", "",
             "## 優先看什麼", "",
             f"- 已有有效窗口但判錯 {outcomes['incorrect']['count']} 筆；完全拒判 {metrics['rejected']} 筆。"
             "錯誤不能全部歸因於 VAD 沒收到聲音，也不能因通過 VAD 就認定片段有清楚的人聲情緒。",
             "- 正確與誤判的 RMS 中位數見下表；它是描述統計，不足以證明調音量就能修好模型。",
             f"- 最多的混淆：{common_errors}；這些是人工聽音的優先方向。",
             "- 聽音清單包含高分誤判；分數是原錄音的有效窗口平均機率，沒有做機率校準，不能視為真實正確率。", "",
             "## 四類結果", "", "| 原始標籤 | Precision | Recall | F1 | 筆數 |",
             "|---|---:|---:|---:|---:|"]
    for label, m in metrics["per_class"].items():
        lines.append(f"| {label} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.4f} | {m['support']} |")
    lines += ["", "## 混淆矩陣", "", "列是原始標籤，欄是模型預測；每筆原錄音只計一次。", "",
              "| 原始／預測 | neutral | happy | sad | anger | 拒判 |", "|---|---:|---:|---:|---:|---:|"]
    for label, counts in zip(EMOTIONS, metrics["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "## 最常見混淆", ""]
    for pair in summary["top_confusions"]:
        lines.append(f"- {pair['label']} 被判為 {pair['prediction']}：{pair['count']} 筆，"
                     f"占原始 {pair['label']} 的 {pair['fraction_of_true_class']:.2%}。")
    lines += ["", "## 長度、音量與有效窗口", "",
              "以下為中位數；incorrect 不含拒判，abstain 另外列出。音量是原始整段左右平均後的 RMS，"
              "不是重取樣、補零與切片後的窗口 RMS。低 RMS 不能直接證明沒有人聲。", ""]
    for kind, title in (("outcome", "整體正確／誤判／拒判"), ("class_outcome", "各真實類別")):
        lines += [f"### {title}", "", "| 分組 | 筆數 | 秒數中位 | RMS 中位 | 有效窗口中位 | 無效窗口比例中位 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, g in summary["groups"][kind].items():
            if g["count"]:
                lines.append(f"| {name} | {g['count']} | {g['median_duration_seconds']:.3f} | "
                             f"{g['median_mono_rms']:.5f} | {g['median_active_windows']:.1f} | "
                             f"{g['median_inactive_fraction']:.2%} |")
    lines += ["", "### 分桶與說話者", "", "分桶僅供描述，不是挑選新門檻；各組情緒比例不同，不能直接推論因果。", "",
              "| 分組 | 筆數 | Accuracy（含拒判） | 拒判 | neutral/happy/sad/anger 筆數 |",
              "|---|---:|---:|---:|---|"]
    for kind in ("duration", "rms", "speaker"):
        for name, g in summary["groups"][kind].items():
            if g["count"]:
                counts = "/".join(str(g["labels"].get(e, 0)) for e in EMOTIONS)
                lines.append(f"| {kind}: {name} | {g['count']} | {g['accuracy']:.2%} | {g['abstain']} | {counts} |")
    lines += ["", "## 人工聽音清單", "",
              "案例為刻意挑選的診斷樣本，不是隨機抽樣，不能用其正確率代表整體。"
              "每類選一筆中位信心度正確案例與最高信心度誤判，另補 happy/sad 誤報、短錄音誤判及拒判，重複來源合併。", "",
              "音檔是原始完整 WAV 的副本，沒有增益、轉檔或裁切；預測則是有效窗口平均結果，不是每個瞬間的情緒。"
              "副本留在已忽略的 data/datasets，不隨報告提交；仍受原資料集授權限制。", "",
              "請先聽音再看答案，填寫 listening_cases.csv 的 heard_label、label_agreement、audio_issue、notes。"
              "可以填不確定，不要因模型判錯就直接改標籤。助理尚未聽音，以下不代表人工標註結論。", "",
              "| 音檔 | 原標籤 | 預測 | 分數 | 秒數 | 挑選理由 |", "|---|---|---|---:|---:|---|"]
    for case in summary["examples"]:
        target = Path(os.path.relpath(audio_dir / case["audio_file"], report_dir)).as_posix()
        confidence = "--" if case["confidence"] is None else f"{case['confidence']:.2%}"
        lines.append(f"| [{case['audio_file']}]({target}) | {case['label']} | {case['prediction']} | "
                     f"{confidence} | {case['duration_seconds']:.2f} | {case['selection_reason']} |")
    lines += ["", "## 結論邊界", "",
              "- 這是同一模型的驗證集診斷，不是獨立 test 成績，也不是新舊模型公平比較。",
              "- 可從混淆與分組找線索，但不能據此斷定是麥克風、標籤、MFCC、模型容量或切片造成。",
              "- 檔案正常可解碼、分數高或 VAD 通過，都不代表情緒正確；目前不建議自動替換正式模型。",
              "- 下一步先由組員聽音核對，保留不確定案例；再決定是否需要進一步窗口診斷或新舊模型耳機比較。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Diagnose saved validation predictions; never trains or evaluates test.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/reports/emotiontalk_four_class_20261014_v1/val.csv")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "data/reports/emotiontalk_audio_audit_20261014_v1")
    parser.add_argument("--input", type=Path, default=ROOT / "data/datasets/emotiontalk_raw_stream/Audio.tar")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir or ROOT / "data/reports" / f"emotiontalk_validation_diagnostics_{stamp}"
    audio_dir = args.audio_dir or ROOT / "data/datasets" / f"emotiontalk_review_{stamp}"
    audio_dir = audio_dir.resolve()
    if not audio_dir.is_relative_to((ROOT / "data/datasets").resolve()):
        raise ValueError("Listening copies must stay under ignored data/datasets")
    require_new_outputs(output, audio_dir)
    result_path = args.run_dir / "result.json"
    config_path = args.run_dir / "run_config.json"
    pred_path = args.run_dir / "val_predictions.json"
    audit_path = args.audit_dir / "audio_metrics.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit_summary_path = args.audit_dir / "summary.json"
    audit_summary = json.loads(audit_summary_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if (result["status"] != "complete" or result["smoke_only"] or config["smoke_only"]
            or result["test_audio_used"] or config["test_audio_used"]
            or digest != config["manifest_sha256"]["val.csv"]
            or digest != audit_summary["manifest_sha256"]["val.csv"]):
        raise ValueError("Expected completed full run and matching validation manifest/audit")
    manifest = read_csv(args.manifest)
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    rows, metrics = join_validation(predictions, manifest, read_csv(audit_path))
    if (metrics["confusion_matrix"] != result["val"]["confusion_matrix"]
            or not np.isclose(metrics["macro_f1"], result["val"]["macro_f1"])
            or not np.isclose(metrics["accuracy"], result["val"]["accuracy"])):
        raise ValueError("Recomputed metrics differ from saved result")
    confusions = Counter((r["label"], r["prediction"]) for r in rows if r["outcome"] == "incorrect")
    supports = Counter(r["label"] for r in rows)
    summary = {"status": "complete", "best_epoch": result["best_epoch"], "metrics": metrics,
               "majority_baseline_accuracy": max(supports.values()) / len(rows),
               "groups": grouped_summary(rows),
               "top_confusions": [{"label": a, "prediction": b, "count": n,
                                   "fraction_of_true_class": n / supports[a]}
                                  for (a, b), n in confusions.most_common(5)],
               "examples": choose_examples(rows), "test_audio_used": False,
               "listening_performed": False, "run_directory": str(args.run_dir.resolve()),
               "audio_directory": str(audio_dir),
               "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (result_path, config_path, pred_path, args.manifest, audit_path, audit_summary_path)}}
    # Validate all selected bytes before creating outputs; never extract tar paths to disk.
    sources = indexed(manifest)
    copies = []
    with tarfile.open(args.input, "r:*") as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member paths")
        for i, case in enumerate(summary["examples"], 1):
            raw = review_audio(tar, case, sources[case["source_wav"]])
            case["audio_file"] = f"{i:02}_{case['label']}_to_{case['prediction']}.wav"
            case["wav_sha256"] = hashlib.sha256(raw).hexdigest()
            copies.append(raw)
    output.mkdir(parents=True, exist_ok=False)
    audio_dir.mkdir(parents=True, exist_ok=False)
    for case, raw in zip(summary["examples"], copies):
        with (audio_dir / case["audio_file"]).open("xb") as stream:
            stream.write(raw)
    write_csv(output / "validation_diagnostics.csv", rows)
    write_csv(output / "listening_cases.csv", [{**r, "heard_label": "", "label_agreement": "",
                                               "audio_issue": "", "notes": ""} for r in summary["examples"]])
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown(summary, output, audio_dir))
    write_json(output / "summary.json", summary)
    print(f"[diagnosis] Complete: {output}; {len(rows)} val recordings; {len(copies)} listening copies in {audio_dir}")


if __name__ == "__main__":
    main()
