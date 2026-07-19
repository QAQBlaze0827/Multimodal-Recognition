from __future__ import annotations

from src.shared_types import EMOTIONS, EmotionResult, make_result


def confidence_weighted_fusion(
    video: EmotionResult | None,
    audio: EmotionResult | None,
    min_confidence: float = 0.3,
    audio_weight: float = 0.5,
    neutral_penalty: float = 0.3,
) -> EmotionResult:
    available: list[EmotionResult] = []
    effective_confs: list[float] = []
    for item in (video, audio):
        if item is None:
            continue
        conf = max(0.0, item.confidence)
        if item.source == "audio":
            conf *= audio_weight
            if item.label == "neutral" and conf >= 0.8:
                conf *= neutral_penalty
        available.append(item)
        effective_confs.append(conf)

    if not available:
        return EmotionResult.neutral("fused")

    total_conf = sum(effective_confs)
    if total_conf < min_confidence:
        return EmotionResult.neutral("fused")

    scores = {emotion: 0.0 for emotion in EMOTIONS}
    for item, conf in zip(available, effective_confs):
        weight = conf / max(total_conf, 1e-8)
        for emotion in EMOTIONS:
            scores[emotion] += item.scores.get(emotion, 0.0) * weight
    return make_result("fused", scores)
