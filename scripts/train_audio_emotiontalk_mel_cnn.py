from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np


EMOTIONS = ("neutral", "happy", "sad", "anger")
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
WINDOW_SECONDS = 2.0
TARGET_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
TARGET_FRAMES = 1 + (TARGET_SAMPLES - N_FFT) // HOP_LENGTH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an EmotionTalk-focused log-mel 2D CNN and export ONNX.")
    parser.add_argument("--data-dir", default="data/datasets/emotiontalk_clean")
    parser.add_argument("--output", default="models/audio_emotiontalk_mel_cnn_fp32.onnx")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


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


MEL_BANK = mel_filterbank(N_FFT, SAMPLE_RATE, N_MELS)


def fit_or_crop(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= TARGET_SAMPLES:
        return audio[:TARGET_SAMPLES]
    return np.pad(audio, (0, TARGET_SAMPLES - audio.size), mode="constant")


def augment_audio(audio: np.ndarray) -> np.ndarray:
    audio = audio.copy()
    if random.random() < 0.8:
        audio *= random.uniform(0.75, 1.25)
    if random.random() < 0.5:
        shift = random.randint(-SAMPLE_RATE // 8, SAMPLE_RATE // 8)
        audio = np.roll(audio, shift)
    if random.random() < 0.5:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-8:
            audio += np.random.normal(0.0, random.uniform(0.001, 0.008) * peak, size=audio.shape)
    return audio


def extract_log_mel(audio: np.ndarray, augment: bool = False) -> np.ndarray:
    audio = fit_or_crop(audio)
    if augment:
        audio = augment_audio(audio)

    audio = np.append(audio[0:1], audio[1:] - 0.97 * audio[:-1])
    frames = []
    for start in range(0, audio.size - N_FFT + 1, HOP_LENGTH):
        frame = audio[start:start + N_FFT] * np.hanning(N_FFT)
        power = np.abs(np.fft.rfft(frame)) ** 2
        mel = np.dot(MEL_BANK, power)
        frames.append(np.log(np.maximum(mel, 1e-10)))

    spec = np.asarray(frames, dtype=np.float32).T
    if spec.shape[1] < TARGET_FRAMES:
        spec = np.pad(spec, ((0, 0), (0, TARGET_FRAMES - spec.shape[1])), mode="constant")
    elif spec.shape[1] > TARGET_FRAMES:
        spec = spec[:, :TARGET_FRAMES]

    spec = (spec - float(np.mean(spec))) / max(float(np.std(spec)), 1e-6)
    if augment:
        spec = spec_augment(spec)
    return spec[..., None].astype(np.float32)


def spec_augment(spec: np.ndarray) -> np.ndarray:
    spec = spec.copy()
    _, n_time = spec.shape
    if random.random() < 0.5 and n_time > 20:
        width = random.randint(4, min(18, n_time // 4))
        start = random.randint(0, n_time - width)
        spec[:, start:start + width] = 0.0
    if random.random() < 0.4 and N_MELS > 8:
        width = random.randint(2, 8)
        start = random.randint(0, N_MELS - width)
        spec[start:start + width, :] = 0.0
    return spec


def load_audio_files(data_dir: Path) -> list[tuple[np.ndarray, int]]:
    import soundfile as sf

    samples: list[tuple[np.ndarray, int]] = []
    for label_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = data_dir / emotion
        if not emotion_dir.exists():
            print(f"[train-mel] Warning: missing emotion dir: {emotion_dir}")
            continue
        for wav_path in sorted(emotion_dir.glob("*.wav")):
            try:
                audio, sr = sf.read(wav_path)
            except Exception as exc:
                print(f"[train-mel] Warning: skipped {wav_path}: {exc}")
                continue
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                from scipy import signal

                new_len = int(len(audio) * SAMPLE_RATE / sr)
                audio = signal.resample(audio, new_len)
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > 1e-8:
                audio = audio / peak
            samples.append((audio.astype(np.float32), label_idx))
    return samples


def make_dataset(tf, samples: list[tuple[np.ndarray, int]], batch_size: int, shuffle: bool, augment: bool):
    import tensorflow as tf_module

    X, y = [], []
    for audio, label_idx in samples:
        X.append(extract_log_mel(audio, augment=augment))
        y.append(label_idx)
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = tf_module.keras.utils.to_categorical(y, num_classes=len(EMOTIONS))
    ds = tf.data.Dataset.from_tensor_slices((X_arr, y_arr))
    if shuffle:
        ds = ds.shuffle(len(X_arr), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(tf, learning_rate: float):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(N_MELS, TARGET_FRAMES, 1), name="input")

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(EMOTIONS), activation="softmax", name="emotion")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    output = Path(args.output)

    sys.modules["jax"] = None

    import tensorflow as tf
    import tf2onnx

    samples = load_audio_files(data_dir)
    if len(samples) < 100:
        raise SystemExit(f"Only {len(samples)} audio files found in {data_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - args.val_split))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    print(f"[train-mel] Total: {len(samples)}, Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = make_dataset(tf, train_samples, args.batch_size, shuffle=True, augment=True)
    val_ds = make_dataset(tf, val_samples, args.batch_size, shuffle=False, augment=False)

    model = build_model(tf, args.learning_rate)
    model.summary()
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    spec = (tf.TensorSpec((None, N_MELS, TARGET_FRAMES, 1), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(output))
    print(f"[train-mel] exported FP32 ONNX: {output}")


if __name__ == "__main__":
    main()
