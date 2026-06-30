from __future__ import annotations

from pathlib import Path

import numpy as np

from src.shared_types import EMOTIONS, create_ort_session, make_result


class AudioEmotionModel:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = ""
        self.backend = "heuristic"
        self._time_frames = 93
        self._log_count = 0
        self._try_load_onnx()

    def predict(self, features: np.ndarray):
        if self.session is not None:
            tensor = features.astype(np.float32)[None, :, :]
            curr_frames = tensor.shape[2]
            target = self._time_frames
            if curr_frames < target:
                pad = target - curr_frames
                tensor = np.pad(tensor, ((0, 0), (0, 0), (0, pad)), mode="constant")
            elif curr_frames > target:
                tensor = tensor[:, :, :target]
            output = self.session.run(None, {self.input_name: tensor})[0]
            logits = np.asarray(output).reshape(-1)[: len(EMOTIONS)]
            exp = np.exp(logits - np.max(logits))
            probs = exp / max(float(exp.sum()), 1e-8)
            result = make_result("audio", dict(zip(EMOTIONS, probs.tolist())))
            self._log_count += 1
            if self._log_count % 10 == 0:
                scores_str = ", ".join(f"{k}:{v:.2f}" for k, v in result.scores.items())
                print(f"[audio] pred: {result.label} ({result.confidence:.2f}) {scores_str}")
            return result

        energy = float(np.mean(np.abs(features)))
        scores = {emotion: 0.02 for emotion in EMOTIONS}
        scores["neutral"] = 0.75
        if energy > 3.0:
            scores["surprise"] = 0.35
            scores["neutral"] = 0.45
        return make_result("audio", scores)

    def _try_load_onnx(self) -> None:
        if not self.model_path.exists():
            return
        session, input_name = create_ort_session(str(self.model_path))
        if session is not None:
            self.session = session
            self.input_name = input_name
            self.backend = "onnx"
            inp_shape = session.get_inputs()[0].shape
            if len(inp_shape) >= 3 and isinstance(inp_shape[2], int):
                self._time_frames = inp_shape[2]
        else:
            print(f"[audio] ONNX model load failed, fallback to heuristic")
