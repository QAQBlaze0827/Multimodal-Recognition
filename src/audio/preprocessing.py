from __future__ import annotations

from math import gcd

import numpy as np

from src.audio.features import extract_light_mfcc_like, highpass_filter


SAMPLE_RATE = 16000
WINDOW_FRAMES = 24000
HOP_FRAMES = 8000
VAD_THRESHOLD = 0.01
FEATURE_FRAMES = 93


def preprocessing_spec() -> dict:
    """Candidate Tiny CNN input contract; not yet connected to live inference."""
    return {
        "version": "emotiontalk-mfcc-v1",
        "labels": ["neutral", "happy", "sad", "anger"],
        "sample_rate": SAMPLE_RATE,
        "channel_policy": "arithmetic_mean",
        "resample": "scipy.signal.resample_poly; window=('kaiser', 5.0); padtype=constant",
        "window_frames": WINDOW_FRAMES,
        "hop_frames": HOP_FRAMES,
        "short_clip": "right_zero_pad_before_vad",
        "train_crop": "uniform_random_start_per_recording",
        "eval_crop": "fixed_hop_plus_final_tail; aggregate_per_recording",
        "vad_threshold": VAD_THRESHOLD,
        "order": ["mono", "resample", "crop_or_pad", "rms_vad", "peak_normalize", "highpass", "mfcc", "feature_pad"],
        "highpass_cutoff": 80,
        "mfcc": {"n_mfcc": 13, "n_fft": 512, "hop_length": 256, "n_mels": 40, "include_delta": False},
        "feature_shape": [13, FEATURE_FRAMES],
    }


def _float_audio(samples) -> np.ndarray:
    samples = np.asarray(samples)
    if not np.issubdtype(samples.dtype, np.floating):
        raise ValueError("Expected normalized floating-point audio, not integer PCM")
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio must be non-empty and finite")
    samples = samples.astype(np.float32, copy=True)
    if not np.isfinite(samples).all():
        raise ValueError("Audio must fit float32")
    return samples


def to_mono_16k(samples, input_sample_rate: int) -> np.ndarray:
    """Accept float audio in (frames,) or (frames, 1/2) format."""
    if not isinstance(input_sample_rate, (int, np.integer)) or input_sample_rate <= 0:
        raise ValueError("input_sample_rate must be a positive integer")
    samples = _float_audio(samples)
    if samples.ndim == 2 and samples.shape[1] in (1, 2):
        samples = samples.mean(axis=1)
    elif samples.ndim != 1:
        raise ValueError("Expected mono or stereo audio, with channels last")
    if input_sample_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        factor = gcd(int(input_sample_rate), SAMPLE_RATE)
        samples = resample_poly(
            samples, SAMPLE_RATE // factor, int(input_sample_rate) // factor,
            window=("kaiser", 5.0), padtype="constant",
        )
    return samples.astype(np.float32, copy=False)


def _mono(samples) -> np.ndarray:
    samples = _float_audio(samples)
    if samples.ndim != 1:
        raise ValueError("Expected mono 16 kHz audio")
    return samples


def _window(samples: np.ndarray, start: int) -> np.ndarray:
    window = samples[start:start + WINDOW_FRAMES].copy()
    return np.pad(window, (0, WINDOW_FRAMES - len(window)))


def training_window(samples, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Sample one crop; caller must pass only training recordings and own the seed."""
    samples = _mono(samples)
    last_start = max(0, len(samples) - WINDOW_FRAMES)
    start = int(rng.integers(last_start + 1)) if last_start else 0
    return start, _window(samples, start)


def evaluation_windows(samples):
    """Yield deterministic (start_sample, window) pairs, covering the final tail."""
    samples = _mono(samples)
    last_start = max(0, len(samples) - WINDOW_FRAMES)
    for start in range(0, last_start + 1, HOP_FRAMES):
        yield start, _window(samples, start)
    if last_start % HOP_FRAMES:
        yield last_start, _window(samples, last_start)


def window_features(window) -> tuple[np.ndarray | None, float]:
    """Return (MFCC or None for inactive audio, raw RMS); never invent neutral."""
    samples = _mono(window)
    if len(samples) != WINDOW_FRAMES:
        raise ValueError(f"Expected exactly {WINDOW_FRAMES} samples at 16 kHz")
    rms = float(np.sqrt(np.mean(np.square(samples))))
    if rms < VAD_THRESHOLD:
        return None, rms
    peak = float(np.max(np.abs(samples)))
    if peak > 1e-8:
        samples = samples / peak
    samples = highpass_filter(samples, sample_rate=SAMPLE_RATE, cutoff_hz=80)
    features = extract_light_mfcc_like(
        samples, sample_rate=SAMPLE_RATE, n_mfcc=13, n_fft=512,
        hop_length=256, n_mels=40, include_delta=False,
    )
    # Match AudioEmotionModel's existing right-zero padding from 92 to 93 frames.
    features = np.pad(features, ((0, 0), (0, max(0, FEATURE_FRAMES - features.shape[1]))))
    return features[:, :FEATURE_FRAMES].astype(np.float32), rms
