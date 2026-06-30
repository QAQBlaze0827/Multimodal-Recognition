from __future__ import annotations

import numpy as np


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


def compute_delta(features: np.ndarray, window: int = 2) -> np.ndarray:
    d = np.zeros_like(features)
    denominator = 2 * sum(i * i for i in range(1, window + 1))
    for i in range(1, window + 1):
        forward = np.roll(features, -i, axis=1)
        backward = np.roll(features, i, axis=1)
        d += i * (forward - backward) / denominator
    return d


def extract_light_mfcc_like(
    samples,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 40,
    include_delta: bool = False,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))

    samples = np.append(samples[0:1], samples[1:] - 0.97 * samples[:-1])

    fbank = _mel_filterbank(n_fft, sample_rate, n_mels)

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
    mfcc = mfcc.T

    if include_delta:
        delta = compute_delta(mfcc)
        mfcc = np.concatenate([mfcc, delta], axis=0)

    return mfcc
