from __future__ import annotations

import threading

import numpy as np

from src.audio.features import extract_light_mfcc_like, highpass_filter
from src.audio.model import AudioEmotionModel
from src.shared_types import make_result


class AudioEmotionThread(threading.Thread):
    def __init__(self, state, config: dict, model: AudioEmotionModel) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.config = config
        self.model = model
        smoothing = config.get("temporal_smoothing", {})
        self.smooth_alpha = float(smoothing.get("alpha", 0.7)) if smoothing.get("enabled", True) else 0.0
        self._smoothed_scores: dict[str, float] | None = None
        self._vad_threshold = float(config.get("vad_threshold", 0.02))
        self._hop_seconds = float(config.get("hop_seconds", 0.5))
        self._ring_buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()

    def run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            print(f"[audio] disabled, sounddevice missing: {exc}")
            return

        sample_rate = int(self.config["sample_rate"])
        input_sample_rate = int(self.config.get("input_sample_rate", sample_rate))
        window_seconds = float(self.config["window_seconds"])
        window_frames = int(input_sample_rate * window_seconds)
        hp_cutoff = float(self.config.get("highpass_cutoff", 80))
        device_id = int(self.config.get("device_id", -1))
        channels = int(self.config.get("channels", 1))
        channel_select = str(self.config.get("channel_select", "mono")).lower()
        device = None if device_id < 0 else device_id

        def callback(indata, frames, time_info, status):
            if indata.ndim == 2 and indata.shape[1] > 1:
                if channel_select == "right":
                    chunk = indata[:, 1]
                elif channel_select == "mix":
                    chunk = np.mean(indata, axis=1)
                else:
                    chunk = indata[:, 0]
            else:
                chunk = indata.reshape(-1)

            with self._buffer_lock:
                self._ring_buffer = np.concatenate([self._ring_buffer, chunk.astype(np.float32).reshape(-1)])
                if len(self._ring_buffer) > window_frames:
                    self._ring_buffer = self._ring_buffer[-window_frames:]

        stream = sd.InputStream(
            samplerate=input_sample_rate,
            channels=channels,
            device=device,
            dtype="float32",
            callback=callback,
        )
        stream.start()

        while not self.state.stop:
            sd.sleep(int(self._hop_seconds * 1000))

            with self._buffer_lock:
                if len(self._ring_buffer) < window_frames:
                    continue
                samples = self._ring_buffer.copy()

            if input_sample_rate != sample_rate:
                ratio = input_sample_rate // sample_rate
                samples = samples[::ratio]

            peak = float(np.max(np.abs(samples)))
            if peak > 1e-8:
                samples = samples / peak

            rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            if rms < self._vad_threshold:
                with self.state.lock:
                    self.state.audio = None
                self._smoothed_scores = None
                continue

            samples = highpass_filter(samples, sample_rate=sample_rate, cutoff_hz=hp_cutoff)

            mfcc_cfg = self.config["mfcc"]
            features = extract_light_mfcc_like(
                samples,
                sample_rate=sample_rate,
                n_mfcc=int(mfcc_cfg["n_mfcc"]),
                n_fft=int(mfcc_cfg["n_fft"]),
                hop_length=int(mfcc_cfg["hop_length"]),
                include_delta=bool(mfcc_cfg.get("include_delta", False)),
            )
            result = self.model.predict(features)

            if self.smooth_alpha > 0:
                if self._smoothed_scores is None:
                    self._smoothed_scores = dict(result.scores)
                else:
                    alpha = self.smooth_alpha
                    for emotion in result.scores:
                        self._smoothed_scores[emotion] = (
                            alpha * self._smoothed_scores[emotion]
                            + (1 - alpha) * result.scores[emotion]
                        )
                result = make_result("audio", self._smoothed_scores)

            with self.state.lock:
                self.state.audio = result

        stream.stop()