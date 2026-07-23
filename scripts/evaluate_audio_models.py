from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.features import extract_light_mfcc_like


EMOTIONS = ("neutral", "happy", "sad", "anger")
SAMPLE_RATE = 16000
MEL_N_MELS = 64
MEL_N_FFT = 1024
MEL_HOP_LENGTH = 256
MEL_WINDOW_SECONDS = 2.0
MEL_TARGET_SAMPLES = int(SAMPLE_RATE * MEL_WINDOW_SECONDS)
MEL_TARGET_FRAMES = 1 + (MEL_TARGET_SAMPLES - MEL_N_FFT) // MEL_HOP_LENGTH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate audio emotion models and write diagnostics.")
    parser.add_argument("--data-dir", default="data/datasets/emotiontalk_clean")
    parser.add_argument("--output-dir", default="data/reports")
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("NAME", "KIND", "PATH"),
        help="Model entry. KIND must be mfcc or mel. Can be passed multiple times.",
    )
    return parser


def default_models() -> list[tuple[str, str, Path]]:
    return [
        ("current_mfcc", "mfcc", Path("models/tiny_cnn_audio_fp32.onnx")),
        ("emotiontalk_mfcc", "mfcc", Path("models/tiny_cnn_audio_emotiontalk_fp32.onnx")),
        ("emotiontalk_mel_cnn", "mel", Path("models/audio_emotiontalk_mel_cnn_fp32.onnx")),
    ]


def mel_filterbank(n_fft: int, sample_rate: int, n_mels: int, fmin: float = 60.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sample_rate / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = int(bin_indices[m - 1])
        center = int(bin_indices[m])
        right = int(bin_indices[m + 1])
        if center <= left or right <= center:
            continue
        fbank[m - 1, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        fbank[m - 1, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    return fbank


MEL_BANK = mel_filterbank(MEL_N_FFT, SAMPLE_RATE, MEL_N_MELS)


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != SAMPLE_RATE:
        from scipy import signal

        new_len = int(len(audio) * SAMPLE_RATE / int(sample_rate))
        audio = signal.resample(audio, new_len)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-8:
        audio = audio / peak
    return audio.astype(np.float32)


def mfcc_features(audio: np.ndarray) -> np.ndarray:
    features = extract_light_mfcc_like(
        audio,
        sample_rate=SAMPLE_RATE,
        n_mfcc=13,
        n_fft=512,
        hop_length=256,
        include_delta=False,
    )
    if features.shape[1] < 93:
        features = np.pad(features, ((0, 0), (0, 93 - features.shape[1])), mode="constant")
    elif features.shape[1] > 93:
        features = features[:, :93]
    return features.astype(np.float32)[None, :, :]


def mel_features(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= MEL_TARGET_SAMPLES:
        audio = audio[:MEL_TARGET_SAMPLES]
    else:
        audio = np.pad(audio, (0, MEL_TARGET_SAMPLES - audio.size), mode="constant")

    audio = np.append(audio[0:1], audio[1:] - 0.97 * audio[:-1])
    frames = []
    for start in range(0, audio.size - MEL_N_FFT + 1, MEL_HOP_LENGTH):
        frame = audio[start:start + MEL_N_FFT] * np.hanning(MEL_N_FFT)
        power = np.abs(np.fft.rfft(frame)) ** 2
        mel = np.dot(MEL_BANK, power)
        frames.append(np.log(np.maximum(mel, 1e-10)))

    spec = np.asarray(frames, dtype=np.float32).T
    if spec.shape[1] < MEL_TARGET_FRAMES:
        spec = np.pad(spec, ((0, 0), (0, MEL_TARGET_FRAMES - spec.shape[1])), mode="constant")
    elif spec.shape[1] > MEL_TARGET_FRAMES:
        spec = spec[:, :MEL_TARGET_FRAMES]
    spec = (spec - float(np.mean(spec))) / max(float(np.std(spec)), 1e-6)
    return spec[None, :, :, None].astype(np.float32)


def collect_samples(data_dir: Path, limit_per_class: int, seed: int) -> list[tuple[Path, int]]:
    rng = np.random.default_rng(seed)
    samples: list[tuple[Path, int]] = []
    for label_idx, emotion in enumerate(EMOTIONS):
        paths = sorted((data_dir / emotion).glob("*.wav"))
        if limit_per_class and len(paths) > limit_per_class:
            indices = sorted(rng.choice(len(paths), size=limit_per_class, replace=False).tolist())
            paths = [paths[i] for i in indices]
        samples.extend((path, label_idx) for path in paths)
    return samples


def evaluate_model(model_name: str, kind: str, model_path: Path, samples: list[tuple[Path, int]]) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    confusion = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=np.int64)
    pred_counts: Counter[str] = Counter()
    examples: list[dict] = []

    for wav_path, true_idx in samples:
        audio = read_audio(wav_path)
        if kind == "mfcc":
            features = mfcc_features(audio)
        elif kind == "mel":
            features = mel_features(audio)
        else:
            raise ValueError(f"Unsupported model kind: {kind}")

        probs = np.asarray(session.run(None, {input_name: features})[0]).reshape(-1)[: len(EMOTIONS)]
        pred_idx = int(np.argmax(probs))
        confusion[true_idx, pred_idx] += 1
        pred_counts[EMOTIONS[pred_idx]] += 1
        if pred_idx != true_idx and len(examples) < 20:
            examples.append(
                {
                    "file": str(wav_path),
                    "true": EMOTIONS[true_idx],
                    "pred": EMOTIONS[pred_idx],
                    "confidence": float(probs[pred_idx]),
                }
            )

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_class = {}
    for idx, emotion in enumerate(EMOTIONS):
        tp = int(confusion[idx, idx])
        support = int(confusion[idx, :].sum())
        predicted = int(confusion[:, idx].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[emotion] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = float(np.mean([item["f1"] for item in per_class.values()]))
    return {
        "name": model_name,
        "kind": kind,
        "path": str(model_path),
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "total": total,
        "confusion": confusion.tolist(),
        "prediction_counts": dict(pred_counts),
        "per_class": per_class,
        "mistake_examples": examples,
    }


def markdown_report(results: list[dict], samples: list[tuple[Path, int]], data_dir: Path) -> str:
    lines = [
        "# Audio Model Diagnostics",
        "",
        f"- data_dir: `{data_dir}`",
        f"- total_samples: {len(samples)}",
        "",
        "## Summary",
        "",
        "| Model | Kind | Accuracy | Macro F1 | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['kind']} | {result['accuracy']:.4f} | "
            f"{result['macro_f1']:.4f} | {result['total']} |"
        )

    for result in results:
        lines.extend(
            [
                "",
                f"## {result['name']}",
                "",
                f"- path: `{result['path']}`",
                f"- accuracy: {result['accuracy']:.4f}",
                f"- macro_f1: {result['macro_f1']:.4f}",
                "",
                "### Confusion Matrix",
                "",
                "| True \\ Pred | neutral | happy | sad | anger |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for emotion, row in zip(EMOTIONS, result["confusion"]):
            lines.append(f"| {emotion} | " + " | ".join(str(value) for value in row) + " |")

        lines.extend(
            [
                "",
                "### Per Class",
                "",
                "| Emotion | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for emotion in EMOTIONS:
            item = result["per_class"][emotion]
            lines.append(
                f"| {emotion} | {item['precision']:.4f} | {item['recall']:.4f} | "
                f"{item['f1']:.4f} | {item['support']} |"
            )

        lines.extend(["", "### Prediction Counts", "", "```json"])
        lines.append(json.dumps(result["prediction_counts"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.extend(["", "### First Mistakes", "", "```json"])
        lines.append(json.dumps(result["mistake_examples"], ensure_ascii=False, indent=2))
        lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_entries = (
        [(name, kind, Path(path)) for name, kind, path in args.model]
        if args.model
        else default_models()
    )
    samples = collect_samples(data_dir, limit_per_class=args.limit_per_class, seed=args.seed)
    if not samples:
        raise SystemExit(f"No wav files found in {data_dir}")

    results = []
    for name, kind, path in model_entries:
        if not path.exists():
            print(f"[eval] Warning: model not found, skipping: {path}")
            continue
        print(f"[eval] Evaluating {name}: {path}")
        results.append(evaluate_model(name, kind, path, samples))

    if not results:
        raise SystemExit("No models evaluated.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"audio_model_diagnostics_{stamp}.json"
    md_path = output_dir / f"audio_model_diagnostics_{stamp}.md"
    payload = {
        "data_dir": str(data_dir),
        "total_samples": len(samples),
        "models": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(results, samples, data_dir), encoding="utf-8")
    print(f"[eval] wrote {json_path}")
    print(f"[eval] wrote {md_path}")


if __name__ == "__main__":
    main()
