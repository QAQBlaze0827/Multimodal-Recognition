from __future__ import annotations

import platform
from dataclasses import dataclass, field
from time import monotonic
from typing import Any


EMOTIONS = ("neutral", "happy", "sad", "anger")


@dataclass(frozen=True)
class EmotionResult:
    source: str
    label: str
    scores: dict[str, float]
    confidence: float
    timestamp: float = field(default_factory=monotonic)

    @classmethod
    def neutral(cls, source: str) -> "EmotionResult":
        return cls(
            source=source,
            label="neutral",
            scores={emotion: 1.0 if emotion == "neutral" else 0.0 for emotion in EMOTIONS},
            confidence=0.0,
        )


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    backend: str


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    cleaned = {emotion: max(0.0, float(scores.get(emotion, 0.0))) for emotion in EMOTIONS}
    total = sum(cleaned.values())
    if total <= 0:
        return {emotion: 1.0 if emotion == "neutral" else 0.0 for emotion in EMOTIONS}
    return {emotion: value / total for emotion, value in cleaned.items()}


def create_ort_session(model_path: str) -> tuple[Any, str] | tuple[None, None]:
    try:
        import onnxruntime as ort
    except ImportError:
        return None, None

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    is_arm = platform.machine() in ("aarch64", "armv7l", "armv8l")
    if is_arm:
        providers = [
            ("XNNPACKExecutionProvider", {"intra_op_num_threads": 2}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]

    try:
        session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
        return session, session.get_inputs()[0].name
    except Exception:
        return None, None


def make_result(source: str, scores: dict[str, float]) -> EmotionResult:
    normalized = normalize_scores(scores)
    label, confidence = max(normalized.items(), key=lambda item: item[1])
    return EmotionResult(source=source, label=label, scores=normalized, confidence=confidence)
