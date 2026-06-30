from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

EMOTIONS = ("neutral", "happy", "sad", "anger", "fear", "surprise", "disgust")
SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256
WINDOW_SECONDS = 1.5
TARGET_FRAMES = int(SAMPLE_RATE * WINDOW_SECONDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Tiny 1D-CNN audio emotion model and export ONNX.")
    parser.add_argument("--data-dir", default="data/datasets/audio", nargs="+",
                        help="One or more directories containing emotion subfolders (neutral/, happy/, ...)")
    parser.add_argument("--output", default="models/tiny_cnn_audio_int8.onnx")
    parser.add_argument("--fp32-output", default="models/tiny_cnn_audio_fp32.onnx")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _mel_filterbank(n_fft: int, sample_rate: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
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
        for k in range(left, center):
            fbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            fbank[m - 1, k] = (right - k) / (right - center)
    return fbank


def _dct(x: np.ndarray, n: int) -> np.ndarray:
    N = len(x)
    k = np.arange(n, dtype=np.float32)[:, None]
    i = np.arange(N, dtype=np.float32)[None, :]
    return np.sum(x[None, :] * np.cos(np.pi * k * (2 * i + 1) / (2.0 * N)), axis=1) * np.sqrt(2.0 / N)


def extract_light_mfcc_like(samples, n_mfcc=13, n_fft=512, hop_length=256, n_mels=40) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))

    samples = np.append(samples[0:1], samples[1:] - 0.97 * samples[:-1])

    fbank = _mel_filterbank(n_fft, SAMPLE_RATE, n_mels)

    frames = []
    for start in range(0, samples.size - n_fft + 1, hop_length):
        frame = samples[start: start + n_fft] * np.hanning(n_fft)
        power = np.abs(np.fft.rfft(frame)) ** 2
        mel_energy = np.dot(fbank, power)
        mel_energy = np.maximum(mel_energy, 1e-10)
        frames.append(np.log(mel_energy))

    if not frames:
        frames = [np.zeros(n_mels, dtype=np.float32)]

    log_mel = np.asarray(frames, dtype=np.float32)
    mfcc = np.apply_along_axis(lambda x: _dct(x, n_mfcc), 1, log_mel)
    return mfcc.T


def load_audio_files(data_dirs: list[Path]) -> list[tuple[np.ndarray, int]]:
    import soundfile as sf

    samples_list: list[tuple[np.ndarray, int]] = []
    seen_names: set[str] = set()

    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"[train] Warning: data dir not found, skipping: {data_dir}")
            continue

        for label_idx, emotion in enumerate(EMOTIONS):
            emotion_dir = data_dir / emotion
            if not emotion_dir.exists():
                continue
            for wav_path in sorted(emotion_dir.glob("*.wav")):
                if wav_path.name in seen_names:
                    continue
                seen_names.add(wav_path.name)
                try:
                    audio, sr = sf.read(wav_path)
                except Exception:
                    continue
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)
                if sr != SAMPLE_RATE:
                    from scipy import signal
                    new_len = int(len(audio) * SAMPLE_RATE / sr)
                    audio = signal.resample(audio, new_len)
                samples_list.append((audio.astype(np.float32), label_idx))

    return samples_list


def make_dataset(
    tf, samples_list: list[tuple[np.ndarray, int]], batch_size: int, shuffle: bool
):
    import tensorflow as tf_module

    X, y = [], []
    for audio, label_idx in samples_list:
        mfcc = extract_light_mfcc_like(audio, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        if mfcc.shape[1] < 10:
            continue
        if mfcc.shape[1] < TARGET_FRAMES // HOP_LENGTH:
            pad_width = (TARGET_FRAMES // HOP_LENGTH) - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, : TARGET_FRAMES // HOP_LENGTH]
        X.append(mfcc[None, :, :])
        y.append(label_idx)

    X = np.concatenate(X, axis=0).astype(np.float32)
    y = tf_module.keras.utils.to_categorical(y, num_classes=len(EMOTIONS))

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(tf, input_shape: tuple[int, int], learning_rate: float):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=input_shape, name="input")

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(64, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(EMOTIONS), activation="softmax", name="emotion")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def quantize_onnx(fp32_output: Path, int8_output: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(fp32_output),
        model_output=str(int8_output),
        weight_type=QuantType.QInt8,
    )


def main() -> None:
    args = build_parser().parse_args()
    data_dirs = [Path(d) for d in args.data_dir]

    import tensorflow as tf
    import tf2onnx

    print(f"[train] Scanning {len(data_dirs)} data dir(s)...")
    all_samples = load_audio_files(data_dirs)
    if len(all_samples) < 10:
        raise SystemExit(
            f"Only {len(all_samples)} audio files found. "
            "Please download audio datasets first with: "
            "python scripts/download_audio_datasets.py"
        )

    random.seed(args.seed)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * (1 - args.val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"[train] Total: {len(all_samples)}, Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = make_dataset(tf, train_samples, args.batch_size, shuffle=True)
    val_ds = make_dataset(tf, val_samples, args.batch_size, shuffle=False)

    input_shape = (N_MFCC, TARGET_FRAMES // HOP_LENGTH)
    model = build_model(tf, input_shape, learning_rate=args.learning_rate)
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    fp32_output = Path(args.fp32_output)
    fp32_output.parent.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    spec = (tf.TensorSpec((None, N_MFCC, TARGET_FRAMES // HOP_LENGTH), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(fp32_output))
    quantize_onnx(fp32_output, output)
    print(f"[train] exported FP32 ONNX: {fp32_output}")
    print(f"[train] exported int8 ONNX: {output}")


if __name__ == "__main__":
    main()
