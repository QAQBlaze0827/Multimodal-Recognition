from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


EMOTIONS = ("neutral", "happy", "sad", "anger", "fear", "surprise", "disgust")


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


def make_result(source: str, scores: dict[str, float]) -> EmotionResult:
    normalized = normalize_scores(scores)
    label, confidence = max(normalized.items(), key=lambda item: item[1])
    return EmotionResult(source=source, label=label, scores=normalized, confidence=confidence)
