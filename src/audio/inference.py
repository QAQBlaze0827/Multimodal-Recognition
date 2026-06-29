from __future__ import annotations

import threading

import numpy as np

from src.audio.features import extract_light_mfcc_like
from src.audio.model import AudioEmotionModel


class AudioEmotionThread(threading.Thread):
    def __init__(self, state, config: dict, model: AudioEmotionModel) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.config = config
        self.model = model

    def run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            print(f"[audio] disabled, sounddevice missing: {exc}")
            return

        sample_rate = int(self.config["sample_rate"])
        window_seconds = float(self.config["window_seconds"])
        frames = int(sample_rate * window_seconds)

        while not self.state.stop:
            audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
            sd.wait()
            samples = audio.reshape(-1)
            rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            if rms < 0.01:
                continue
            mfcc_cfg = self.config["mfcc"]
            features = extract_light_mfcc_like(
                samples,
                sample_rate=sample_rate,
                n_mfcc=int(mfcc_cfg["n_mfcc"]),
                n_fft=int(mfcc_cfg["n_fft"]),
                hop_length=int(mfcc_cfg["hop_length"]),
            )
            result = self.model.predict(features)
            with self.state.lock:
                self.state.audio = result
