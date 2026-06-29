from __future__ import annotations

from src.shared_types import EMOTIONS, EmotionResult, make_result


def confidence_weighted_fusion(
    video: EmotionResult | None,
    audio: EmotionResult | None,
    min_confidence: float = 0.3,
) -> EmotionResult:
    available = [item for item in (video, audio) if item is not None]
    if not available:
        return EmotionResult.neutral("fused")

    total_conf = sum(max(0.0, item.confidence) for item in available)
    if total_conf < min_confidence:
        return EmotionResult.neutral("fused")

    scores = {emotion: 0.0 for emotion in EMOTIONS}
    for item in available:
        weight = max(0.0, item.confidence) / max(total_conf, 1e-8)
        for emotion in EMOTIONS:
            scores[emotion] += item.scores.get(emotion, 0.0) * weight
    return make_result("fused", scores)
