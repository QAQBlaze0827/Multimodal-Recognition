"""Compare two MFCC models on identical local recordings without changing them."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_audio_emotiontalk_manifest import aggregate_recordings, write_csv, write_json
from src.audio.features import extract_light_mfcc_like, highpass_filter
from src.audio.model import AudioEmotionModel
from src.audio.preprocessing import evaluation_windows, preprocessing_spec
from src.config import load_config
from src.shared_types import EMOTIONS, make_result


MODES = ("raw", "calibrated", "smoothed")
MODEL_PATHS = {
    "old": ROOT / "models/tiny_cnn_audio_emotiontalk_fp32.onnx",
    "new": ROOT / "models/emotiontalk_manifest_train_20261015_full_v1/model.onnx",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_samples(session: Path):
    session = session.resolve()
    summary = read_json(session / "summary.json")
    spec = read_json(session / "session.json")
    records = summary["recordings"]
    if summary.get("status") != "complete" or not records:
        raise ValueError("A completed recording session is required")
    names = [row["file"] for row in records]
    if len(set(names)) != len(names) or set(names) != {p.name for p in session.glob("*.wav")}:
        raise ValueError("Duplicate, missing or unlisted WAV files")
    protected = {str(session / name): digest(session / name) for name in ("summary.json", "session.json")}
    samples = []
    for row in records:
        path = (session / row["file"]).resolve()
        if path.parent != session or path.suffix != ".wav" or row["expected_label"] not in EMOTIONS:
            raise ValueError("Invalid recording path or expected label")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != row["wav_sha256"] or read_json(path.with_suffix(".json")) != row:
            raise ValueError(f"Recording hash or sidecar differs: {path.name}")
        info = sf.info(io.BytesIO(raw))
        audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        if (rate != 16000 or rate != spec["sample_rate"] or rate != row["sample_rate"]
                or row["channels"] != 1 or spec["channels"] != 1 or info.subtype != "FLOAT"
                or audio.shape != (row["frames"], 1) or not len(audio) or not np.isfinite(audio).all()
                or not np.isclose(len(audio) / rate, row["duration_seconds"], rtol=0, atol=1 / rate)):
            raise ValueError("This comparison requires intact mono 16 kHz FLOAT WAV recordings")
        protected[str(path)] = sha
        protected[str(path.with_suffix(".json"))] = digest(path.with_suffix(".json"))
        samples.append((row, audio[:, 0]))
    return samples, protected


def check_config(audio: dict) -> None:
    expected = {"sample_rate": 16000, "window_seconds": 1.5, "hop_seconds": 0.5,
                "mfcc": {"n_mfcc": 13, "n_fft": 512, "hop_length": 256, "include_delta": False}}
    if any(audio.get(key) != value for key, value in expected.items()):
        raise ValueError("Config differs from the fixed 16 kHz, 1.5s/0.5s, 13x93 comparison contract")
    alpha = float(audio["temporal_smoothing"]["alpha"])
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Invalid smoothing alpha")
    for key in ("vad_threshold", "highpass_cutoff"):
        if not np.isfinite(audio[key]) or audio[key] < 0:
            raise ValueError(f"Invalid {key}")
    cal = audio["model"]["calibration"]
    scales = [float(cal.get(key, 1)) for key in ("neutral_logit_scale", "sad_logit_scale", "anger_logit_scale")]
    floor = float(cal.get("confidence_floor", 0))
    if not all(np.isfinite(v) and v > 0 for v in scales) or not np.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError("Invalid calibration settings")


def live_features(window: np.ndarray, config: dict):
    # Use the production filter, not the training module's faster recurrence.
    samples = window.copy()
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms < config["vad_threshold"]:
        return None, rms
    peak = float(np.max(np.abs(samples)))
    if peak > 1e-8:
        samples = samples / peak
    samples = highpass_filter(samples, sample_rate=16000, cutoff_hz=config["highpass_cutoff"])
    features = extract_light_mfcc_like(samples, sample_rate=16000, **config["mfcc"])
    features = np.pad(features, ((0, 0), (0, max(0, 93 - features.shape[1]))))[:, :93]
    return features.astype(np.float32), rms


def smooth_results(results, alpha: float):
    previous = None
    output = []
    for result in results:
        if result is None:
            previous = None
            output.append(None)
            continue
        if alpha <= 0:
            output.append(result)
            continue
        if previous is None:
            previous = dict(result.scores)
        else:
            previous = {label: alpha * previous[label] + (1 - alpha) * result.scores[label]
                        for label in EMOTIONS}
        output.append(make_result("audio", previous))
    return output


def valid_scores(result):
    if result is None or result.confidence <= 0:
        return None
    values = np.array([result.scores[label] for label in EMOTIONS])
    if not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1, atol=1e-5):
        raise ValueError("Invalid model probabilities")
    return values


def summarize(traces, labels):
    probabilities, owners = [], []
    for owner, trace in enumerate(traces):
        for result in trace:
            scores = valid_scores(result)
            if scores is not None:
                probabilities.append(scores)
                owners.append(owner)
    metrics, predictions = aggregate_recordings(np.array(probabilities).reshape(-1, 4), owners, labels)
    metrics["intended_label_agreement"] = metrics.pop("accuracy")
    return metrics, predictions


def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if not path.is_file():
            raise ValueError(f"Missing model: {path}")
        model = AudioEmotionModel(str(path))
        if (model.session is None or model.backend != "onnx"
                or model.session.get_inputs()[0].shape[1:] != [13, 93]
                or model.session.get_outputs()[0].shape[1:] != [4]):
            raise ValueError(f"Expected actual four-class ONNX, not heuristic fallback: {path}")
        models[name] = model
    return models


def report_text(results: dict, recordings: list[dict]) -> str:
    lines = ["# 本機錄音新舊模型比較", "",
             "這是使用者預期情緒的診斷性對照，不是獨立標註的正式測試集。助理未人工聽音或判斷標籤正確性。", "",
             "old：既有 EmotionTalk MFCC 模型；new：完整 30 輪 manifest 模型，不是兩輪試跑模型。",
             "raw：未加權未平滑；calibrated：目前類別加權與門檻；smoothed：再加目前時間平滑。", "",
             "| 模型／模式 | 與預期標籤一致 | Macro-F1 | 覆蓋率 | 整段拒判 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for key, metrics in results.items():
        lines.append(f"| {key} | {metrics['intended_label_agreement']:.2%} | {metrics['macro_f1']:.4f} | {metrics['coverage']:.2%} | {metrics['rejected']} |")
    lines.extend(["", "## 逐段結果", "", "以下是有效窗口機率的整段平均分類，不是串流最後一次分類。", "",
                  "| 錄音 | 預期 | old/raw | old/calibrated | old/smoothed | new/raw | new/calibrated | new/smoothed |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    files = list(dict.fromkeys(row["file"] for row in recordings))
    for file in files:
        rows = [r for r in recordings if r["file"] == file]
        by_key = {r["model"] + "/" + r["mode"]: r["prediction"] or "拒判" for r in rows}
        cells = [by_key[name + "/" + mode] for name in MODEL_PATHS for mode in MODES]
        lines.append("| " + " | ".join([file, rows[0]["expected_label"], *cells]) + " |")
    lines.extend(["", "## 方法與限制", "",
                  "- 同一錄音以固定 1.5 秒窗口、0.5 秒步長及尾窗處理；兩模型共享完全相同的特徵。",
                  "- 使用目前正式 VAD、peak normalization、高通與 MFCC。訓練模組的快速高通有細小數值差異，本次沒有切換它。",
                  "- 每段錄音獨立重設平滑，遇到 VAD 拒判也重設；這是確定性的離線重播，不重現即時執行緒排程或跨錄音歷史。",
                  "- 整段無有效預測保留為拒判，仍計入分母；詳見 windows.csv 的逐窗口分數與 recordings.csv 的最後窗口結果。",
                  "- confidence-floor fallback 另外標記；smoothed 忠實沿用目前程式，可能把 confidence=0 的 neutral fallback 再平滑成有效分數，不能將它當真實中性證據。",
                  "- 個人錄音未複製到報告、不開啟麥克風、不重訓、不改設定或模型；沒有用 EmotionTalk test 音訊。",
                  "- 同一位使用者少量錄音不足以估計泛化準確率，模型分數也不等於正確率；不得只依此調參後仍把同一批當獨立測試。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "data/reports" / datetime.now().strftime("local_audio_comparison_%Y%m%d_%H%M%S_%f")
    if output.exists():
        raise SystemExit(f"Output already exists; nothing overwritten: {output}")
    if not args.config.is_file():
        raise ValueError("Missing config; do not silently use defaults")
    config = load_config(args.config)["audio"]
    check_config(config)
    samples, protected = load_samples(args.session)
    models = load_models()
    protected.update({str(p): digest(p) for p in [args.config, *MODEL_PATHS.values()]})
    spec_path = MODEL_PATHS["new"].parent / "preprocessing.json"
    if read_json(spec_path) != preprocessing_spec():
        raise ValueError("New model preprocessing contract has changed")
    protected[str(spec_path)] = digest(spec_path)
    calibrated = {name: AudioEmotionModel(str(MODEL_PATHS[name]), config["model"]["calibration"]) for name in models}
    if any(model.session is None for model in calibrated.values()):
        raise ValueError("Calibrated model did not load")
    output.mkdir(parents=True, exist_ok=False)
    smoothing = config["temporal_smoothing"]
    alpha = float(smoothing["alpha"]) if smoothing["enabled"] else 0.0
    traces = {f"{name}/{mode}": [] for name in models for mode in MODES}
    windows, recording_rows = [], []
    for source, audio in samples:
        prepared = [(start, *live_features(window, config)) for start, window in evaluation_windows(audio)]
        for name, model in models.items():
            raw = [model.predict(f) if f is not None else None for _, f, _ in prepared]
            weighted = [calibrated[name].predict(f) if f is not None else None for _, f, _ in prepared]
            modes = {"raw": raw, "calibrated": weighted, "smoothed": smooth_results(weighted, alpha)}
            for mode, trace in modes.items():
                traces[f"{name}/{mode}"].append(trace)
                for index, ((start, feature, rms), result) in enumerate(zip(prepared, trace)):
                    scores = valid_scores(result)
                    windows.append({"file": source["file"], "expected_label": source["expected_label"],
                                    "model": name, "mode": mode, "start_seconds": start / 16000, "rms": rms,
                                    "feature_sha256": hashlib.sha256(feature.tobytes()).hexdigest() if feature is not None else "",
                                    "vad_rejected": feature is None,
                                    "model_fallback": weighted[index] is not None and weighted[index].confidence <= 0 if mode != "raw" else False,
                                    "prediction": result.label if scores is not None else "",
                                    "confidence": result.confidence if result is not None else "",
                                    **{label: float(scores[i]) if scores is not None else "" for i, label in enumerate(EMOTIONS)}})
        print(f"Compared {source['file']}: {sum(f is not None for _, f, _ in prepared)}/{len(prepared)} active windows", flush=True)
    metrics = {}
    for key, values in traces.items():
        metrics[key], predictions = summarize(values, [row["expected_label"] for row, _ in samples])
        name, mode = key.split("/")
        for (source, _), trace, prediction in zip(samples, values, predictions):
            recording_rows.append({"file": source["file"], "expected_label": source["expected_label"],
                                   "wav_sha256": source["wav_sha256"], "model": name, "mode": mode,
                                   "prediction": prediction["prediction"] or "", "active_windows": sum(valid_scores(r) is not None for r in trace),
                                   "total_windows": len(trace),
                                   "last_window_prediction": trace[-1].label if valid_scores(trace[-1]) is not None else "",
                                   **{label: prediction["scores"][i] if prediction["scores"] is not None else "" for i, label in enumerate(EMOTIONS)}})
    if any(digest(Path(path)) != sha for path, sha in protected.items()):
        raise ValueError("Input changed during comparison; no completion report written")
    write_csv(output / "windows.csv", windows)
    write_csv(output / "recordings.csv", recording_rows)
    with (output / "report.md").open("x", encoding="utf-8") as file:
        file.write(report_text(metrics, recording_rows))
    write_json(output / "summary.json", {
        "status": "complete", "session": args.session.name, "recordings": len(samples),
        "labels": list(EMOTIONS), "audio_config": config, "metrics": metrics,
        "models": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for name, path in MODEL_PATHS.items()},
        "source_files_sha256": {Path(path).name: sha for path, sha in protected.items() if Path(path).parent == args.session.resolve()},
        "config_sha256": digest(args.config), "new_preprocessing_sha256": digest(spec_path),
        "source_unchanged": True, "formal_test_accuracy": False, "human_listening_performed_by_assistant": False,
        "aggregation": "mean of valid window probabilities per recording; independent smoothing per recording",
        "feature_pipeline": "production highpass_filter and MFCC; fixed evaluation_windows schedule",
    })
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
