from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_emotiontalk_audio import inspect_wav, read_manifests
from scripts.prepare_emotiontalk_manifests import EMOTIONS, GROUP_SPLIT, LABEL_MAP
from scripts.train_audio_tiny_cnn import build_model
from src.audio.preprocessing import (
    evaluation_windows, preprocessing_spec, to_mono_16k, training_window, window_features,
)


def select_records(records: list[dict], split: str, limit: int, seed: int) -> list[dict]:
    if split not in ("train", "val") or limit < 0:
        raise ValueError("Training entry point only accepts train/val and nonnegative limits")
    rng = np.random.default_rng(seed)
    selected = []
    for label in EMOTIONS:
        candidates = sorted((r for r in records if r["split"] == split and r["label"] == label),
                            key=lambda r: r["source_wav"])
        if not candidates:
            raise ValueError(f"Missing class {label} in {split}")
        if limit and len(candidates) > limit:
            indices = rng.choice(len(candidates), size=limit, replace=False)
            candidates = [candidates[int(i)] for i in indices]
        selected.extend(candidates)
    return sorted(selected, key=lambda r: r["source_wav"])


def validate_source(tar, row: dict) -> None:
    if GROUP_SPLIT.get(row["group_id"]) != row["split"]:
        raise ValueError("Manifest changed the fixed group split")
    with tar.extractfile(row["source_json"]) as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != row["metadata_sha256"]:
        raise ValueError(f"Changed metadata: {row['source_json']}")
    payload = json.loads(raw)
    if (payload["emotion_result"] != row["source_label"]
            or LABEL_MAP.get(row["source_label"]) != row["label"]
            or payload["speaker_id"] != row["speaker_id"]
            or f"Audio/wav/{payload['file_path']}" != row["source_wav"]
            or payload["file_path"].split("/")[0] != row["group_id"]):
        raise ValueError(f"Manifest differs from original annotation: {row['source_wav']}")


def load_recording(tar, row: dict) -> tuple[np.ndarray, str]:
    import soundfile as sf

    with tar.extractfile(row["source_wav"]) as stream:
        raw = stream.read()
    metrics = inspect_wav(raw)
    if (metrics["sample_rate"] != int(row["sample_rate"])
            or metrics["channels"] != int(row["channels"])
            or abs(metrics["duration_seconds"] - float(row["duration_seconds"])) > 1 / metrics["sample_rate"]):
        raise ValueError(f"WAV differs from manifest: {row['source_wav']}")
    audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    return to_mono_16k(audio, rate), metrics["pcm_sha256"]


def training_arrays(tar, records: list[dict], seed: int):
    rng = np.random.default_rng(seed)
    features, labels, crops = [], [], []
    for row in records:
        if row["split"] != "train":
            raise ValueError("Only train recordings may enter optimization")
        samples, digest = load_recording(tar, row)
        start, window = training_window(samples, rng)
        feature, rms = window_features(window)
        crops.append({"source_wav": row["source_wav"], "label": row["label"],
                      "pcm_sha256": digest, "start_sample": start, "rms": rms,
                      "accepted": feature is not None})
        if feature is not None:
            features.append(feature)
            labels.append(EMOTIONS.index(row["label"]))
    if set(labels) != set(range(len(EMOTIONS))):
        raise ValueError("A training class has no active crops; inspect VAD/crop coverage before training")
    return np.stack(features), np.asarray(labels), crops


def validation_arrays(tar, records: list[dict]):
    features, owners, details = [], [], []
    for index, row in enumerate(records):
        if row["split"] != "val":
            raise ValueError("Only val recordings may enter model selection")
        samples, digest = load_recording(tar, row)
        active = inactive = 0
        for _, window in evaluation_windows(samples):
            feature, _ = window_features(window)
            if feature is None:
                inactive += 1
            else:
                active += 1
                features.append(feature)
                owners.append(index)
        details.append({"source_wav": row["source_wav"], "label": row["label"],
                        "pcm_sha256": digest, "active_windows": active, "inactive_windows": inactive})
    array = np.stack(features) if features else np.empty((0, 13, 93), dtype=np.float32)
    return array, np.asarray(owners, dtype=np.int64), details


def aggregate_recordings(probabilities, owners, labels):
    probabilities = np.asarray(probabilities)
    owners = np.asarray(owners, dtype=np.int64)
    if (probabilities.shape != (len(owners), len(EMOTIONS)) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)
            or np.any(owners < 0) or np.any(owners >= len(labels))):
        raise ValueError("Invalid window probabilities or recording indices")
    # Column 4 is abstention: it stays in the accuracy denominator and recall FN.
    matrix = np.zeros((4, 5), dtype=np.int64)
    predictions = []
    for index, label in enumerate(labels):
        windows = probabilities[owners == index]
        scores = windows.mean(axis=0) if len(windows) else None
        predicted = int(np.argmax(scores)) if scores is not None else 4
        matrix[EMOTIONS.index(label), predicted] += 1
        predictions.append({"prediction": EMOTIONS[predicted] if predicted < 4 else None,
                            "scores": scores.tolist() if scores is not None else None})
    per_class = {}
    for i, label in enumerate(EMOTIONS):
        tp, support, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "support": support,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    total, rejected, correct = len(labels), int(matrix[:, 4].sum()), int(np.trace(matrix[:, :4]))
    return {
        "recordings": total, "rejected": rejected,
        "coverage": (total - rejected) / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "per_class": per_class, "confusion_matrix": matrix.tolist(),
        "confusion_columns": [*EMOTIONS, "abstain"],
    }, predictions


def class_weights(labels) -> dict:
    counts = np.bincount(labels, minlength=4)
    if np.any(counts == 0):
        raise ValueError("All four training classes must have active samples")
    return {i: len(labels) / (4 * int(count)) for i, count in enumerate(counts)}


def predict_windows(model, features, batch_size):
    batches = [model(features[i:i + batch_size], training=False).numpy()
               for i in range(0, len(features), batch_size)]
    return np.concatenate(batches) if batches else np.empty((0, 4), dtype=np.float32)


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_onnx(tf, model, output: Path, features: np.ndarray) -> dict:
    import onnxruntime as ort
    import tf2onnx

    if output.exists():
        raise FileExistsError(output)
    signature = [tf.TensorSpec([None, 13, 93], tf.float32, name="input")]

    @tf.function(input_signature=signature)
    def forward(inputs):
        return model(inputs, training=False)

    tf2onnx.convert.from_function(forward, input_signature=signature, opset=13, output_path=str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    reference = model(features, training=False).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: features})[0]
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
    return {"samples": len(features), "max_abs_error": float(np.max(np.abs(reference - actual))),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train manifest-based Tiny CNN; defaults to a small smoke run, never evaluates test audio.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/datasets/emotiontalk_raw_stream/Audio.tar")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/reports/emotiontalk_four_class_20261014_v1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit-per-class", type=int, default=8, help="Per split; 0 uses all train/val recordings. Test remains untouched.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.limit_per_class < 0 or args.seed < 0:
        parser.error("epochs/batch-size must be positive; limit and seed must be nonnegative")
    smoke = args.limit_per_class > 0
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir or ROOT / "models" / f"emotiontalk_manifest_{'smoke' if smoke else 'train'}_{stamp}"
    if output.exists():
        raise SystemExit(f"Output already exists; nothing overwritten: {output}")
    records, manifest_hashes = read_manifests(args.manifests)
    train = select_records(records, "train", args.limit_per_class, args.seed)
    val = select_records(records, "val", args.limit_per_class, args.seed + 1)
    selected = train + val
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "preprocessing.json", preprocessing_spec())
    write_csv(output / "selected_records.csv", selected)
    write_json(output / "run_config.json", {
        "smoke_only": smoke, "epochs": args.epochs, "seed": args.seed, "batch_size": args.batch_size,
        "limit_per_class": args.limit_per_class, "learning_rate": 0.001,
        "train_records": len(train), "val_records": len(val),
        "archive": str(args.input.resolve()), "manifest_sha256": manifest_hashes,
        "test_audio_used": False, "selection_metric": "val_recording_macro_f1_including_abstention",
        "train_class_weights": "inverse active crop counts, recomputed each epoch",
        "augmentation": "random crop only", "calibration_and_temporal_smoothing": False,
        "architecture": "existing train_audio_tiny_cnn.build_model; input [batch,13,93] unchanged",
    })
    print(f"[train] {'SMOKE ONLY' if smoke else 'FULL TRAIN'}: train={len(train)}, val={len(val)}; no test audio", flush=True)

    # Keras' optional JAX import conflicts with this environment's ml_dtypes.
    # Restore module state before SciPy's array-backend detection runs.
    missing = object()
    previous_jax = sys.modules.get("jax", missing)
    try:
        sys.modules["jax"] = None
        import tensorflow as tf

        tf.keras.utils.set_random_seed(args.seed)
        model = build_model(tf, (13, 93), learning_rate=1e-3)
    finally:
        if previous_jax is missing:
            sys.modules.pop("jax", None)
        else:
            sys.modules["jax"] = previous_jax
    history, best_score, best_weights, best_epoch = [], -1.0, None, 0
    with tarfile.open(args.input, "r:*") as tar:
        members = tar.getmembers()
        names = [member.name for member in members if member.isfile()]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member paths")
        for row in selected:
            validate_source(tar, row)
        val_features, owners, val_details = validation_arrays(tar, val)
        if not len(val_features):
            raise ValueError("No active validation windows; review input before training")
        write_csv(output / "val_audio.csv", val_details)
        print(f"[train] Validation prepared: {len(val_features)} active windows", flush=True)
        for epoch in range(1, args.epochs + 1):
            x, y, crops = training_arrays(tar, train, args.seed + epoch)
            write_csv(output / f"train_crops_epoch_{epoch:03}.csv", crops)
            fit = model.fit(x, tf.keras.utils.to_categorical(y, 4), batch_size=args.batch_size,
                            epochs=1, shuffle=True, class_weight=class_weights(y), verbose=2)
            metrics, _ = aggregate_recordings(predict_windows(model, val_features, args.batch_size),
                                              owners, [r["label"] for r in val])
            history.append({"epoch": epoch, "train_active": len(y), "train_rejected": len(train) - len(y),
                            "train_class_counts": dict(Counter(EMOTIONS[i] for i in y)),
                            "loss": float(fit.history["loss"][-1]), "val": metrics})
            write_json(output / f"epoch_{epoch:03}.json", history[-1])
            print(f"[train] Epoch {epoch}: val recording macro-F1={metrics['macro_f1']:.4f}, coverage={metrics['coverage']:.4f}", flush=True)
            if metrics["macro_f1"] > best_score:
                best_score, best_epoch, best_weights = metrics["macro_f1"], epoch, model.get_weights()

    model.set_weights(best_weights)
    metrics, predictions = aggregate_recordings(predict_windows(model, val_features, args.batch_size),
                                                owners, [r["label"] for r in val])
    write_json(output / "val_predictions.json", [{**detail, **prediction} for detail, prediction in zip(val_details, predictions)])
    model_path = output / ("smoke_only.onnx" if smoke else "model.onnx")
    export_check = export_onnx(tf, model, model_path, val_features[:min(8, len(val_features))])
    write_json(output / "result.json", {
        "status": "complete", "smoke_only": smoke, "best_epoch": best_epoch, "history": history,
        "val": metrics, "onnx_check": export_check, "test_audio_used": False,
        "limitations": ["Smoke scores are not formal accuracy" if smoke else "Validation used for selection; test pending",
                        "No microphone, camera or Raspberry Pi runtime test", "Live inference and config unchanged"],
    })
    print(f"[train] Complete: {model_path}; best epoch={best_epoch}; test audio not used", flush=True)


if __name__ == "__main__":
    main()
