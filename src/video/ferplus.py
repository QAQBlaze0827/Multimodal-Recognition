"""Pinned FERPlus candidate, shared by live inference and diagnostic tools."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.shared_types import EMOTIONS, EmotionResult, create_ort_session, make_result


FER_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
FER_PATH = Path(__file__).resolve().parents[2] / ".cache/ferplus/emotion-ferplus-8.onnx"
# Official happiness/sadness names are canonicalized, not merged with other classes.
FER_LABELS = ("neutral", "happy", "surprise", "sad", "anger", "disgust", "fear", "contempt")


def fer_tensor(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = Image.fromarray(gray).resize((64, 64), Image.Resampling.LANCZOS)
    # FERPlus expects raw 0..255 grayscale, NOT the legacy model's /255 input.
    return np.asarray(resized, dtype=np.float32).reshape(1, 1, 64, 64)


def fer_scores(logits):
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != 8 or not np.isfinite(values).all():
        raise ValueError("Expected eight finite FERPlus logits")
    exp = np.exp(values - values.max())
    probs = exp / exp.sum()
    selected = probs[[0, 1, 3, 4]]
    mass = float(selected.sum())
    return {
        "full_scores": dict(zip(FER_LABELS, probs.tolist())),
        "full_label": FER_LABELS[int(probs.argmax())],
        "four_scores": dict(zip(EMOTIONS, (selected / mass).tolist())) if mass > 0 else None,
        "four_label": EMOTIONS[int(selected.argmax())] if mass > 0 else None,
        "discarded_mass": float(probs[[2, 5, 6, 7]].sum()),
    }


class FerPlusEmotionModel:
    backend = "onnx-ferplus"

    def __init__(self) -> None:
        self.model_path = FER_PATH
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"FERPlus model missing: {self.model_path}. "
                "No automatic download or fallback; omit --video-model ferplus to use the configured model."
            )
        if hashlib.sha256(self.model_path.read_bytes()).hexdigest() != FER_SHA256:
            raise ValueError(f"FERPlus SHA-256 mismatch: {self.model_path}")
        self.session, self.input_name = create_ort_session(str(self.model_path))
        if self.session is None:
            raise RuntimeError("FERPlus ONNX load failed; refusing heuristic fallback")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].type != "tensor(float)"
                or inputs[0].shape != [1, 1, 64, 64]
                or len(outputs) != 1 or outputs[0].type != "tensor(float)"
                or outputs[0].shape != [1, 8]):
            raise ValueError("Unexpected FERPlus input/output contract")

    def predict(self, face_bgr) -> EmotionResult:
        tensor = fer_tensor(face_bgr)
        logits = self.session.run(None, {self.input_name: tensor})[0]
        scores = fer_scores(logits)["four_scores"]
        if scores is None:
            return EmotionResult.neutral("video")
        return make_result("video", scores)
