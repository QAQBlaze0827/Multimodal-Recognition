from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.shared_types import EMOTIONS, EmotionResult, make_result


class VideoEmotionModel:
    def __init__(self, cv2, model_path: str, input_size: int = 48) -> None:
        self.cv2 = cv2
        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.session = None
        self.input_name = ""
        self.backend = "heuristic"
        self._smile = self._load_smile_cascade()
        self._try_load_onnx()

    def predict(self, face_bgr) -> EmotionResult:
        face_gray = self.cv2.cvtColor(face_bgr, self.cv2.COLOR_BGR2GRAY)
        resized = self.cv2.resize(face_gray, (self.input_size, self.input_size))
        if self.session is not None:
            return self._predict_onnx(resized)
        return self._predict_heuristic(resized)

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
            print(f"[video] ONNX model load failed, fallback to heuristic: {exc}")
            self.session = None
            self.backend = "heuristic"

    def _predict_onnx(self, face_gray) -> EmotionResult:
        tensor = face_gray.astype(np.float32) / 255.0
        tensor = tensor.reshape(1, 1, self.input_size, self.input_size)
        output = self.session.run(None, {self.input_name: tensor})[0]
        values = np.asarray(output).reshape(-1)[: len(EMOTIONS)].astype(np.float32)
        if np.all(values >= 0.0) and abs(float(values.sum()) - 1.0) < 0.05:
            probs = values / max(float(values.sum()), 1e-8)
        else:
            exp = np.exp(values - np.max(values))
            probs = exp / max(float(exp.sum()), 1e-8)
        return make_result("video", dict(zip(EMOTIONS, probs.tolist())))

    def _predict_heuristic(self, face_gray) -> EmotionResult:
        scores = {emotion: 0.02 for emotion in EMOTIONS}
        scores["neutral"] = 0.65

        if self._smile is not None:
            smiles = self._smile.detectMultiScale(
                face_gray,
                scaleFactor=1.7,
                minNeighbors=18,
                minSize=(12, 12),
            )
            if len(smiles) > 0:
                scores["happy"] = 0.85
                scores["neutral"] = 0.15

        contrast = float(face_gray.std()) / 255.0
        if contrast < 0.08:
            scores["neutral"] += 0.15
        return make_result("video", scores)

    def _load_smile_cascade(self):
        source = Path(self.cv2.data.haarcascades) / "haarcascade_smile.xml"
        if not source.exists():
            return None
        cache_dir = Path(tempfile.gettempdir()) / "multimodal_emotion_cv2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / source.name
        if not path.exists() or path.stat().st_size != source.stat().st_size:
            shutil.copy2(source, path)
        cascade = self.cv2.CascadeClassifier(str(path))
        return None if cascade.empty() else cascade
