from __future__ import annotations

import numpy as np


def extract_light_mfcc_like(
    samples,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.size))

    frames = []
    for start in range(0, samples.size - n_fft + 1, hop_length):
        frame = samples[start : start + n_fft] * np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(frame)) + 1e-6
        log_spectrum = np.log(spectrum)
        bins = np.array_split(log_spectrum, n_mfcc)
        frames.append([float(chunk.mean()) for chunk in bins])

    if not frames:
        frames = [[0.0] * n_mfcc]
    return np.asarray(frames, dtype=np.float32).T
