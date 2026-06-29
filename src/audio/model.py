from __future__ import annotations

from pathlib import Path

import numpy as np

from src.shared_types import EMOTIONS, make_result


class AudioEmotionModel:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = ""
        self.backend = "heuristic"
        self._try_load_onnx()

    def predict(self, features: np.ndarray):
        if self.session is not None:
            tensor = features.astype(np.float32)[None, :, :]
            output = self.session.run(None, {self.input_name: tensor})[0]
            logits = np.asarray(output).reshape(-1)[: len(EMOTIONS)]
            exp = np.exp(logits - np.max(logits))
            probs = exp / max(float(exp.sum()), 1e-8)
            return make_result("audio", dict(zip(EMOTIONS, probs.tolist())))

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
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.backend = "onnx"
        except Exception as exc:
            print(f"[audio] ONNX model load failed, fallback to heuristic: {exc}")
