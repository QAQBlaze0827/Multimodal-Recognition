from __future__ import annotations

import numpy as np


PREVIEW_SIZE = (960, 700)
IMAGE_HEIGHT = 540


def draw_overlay(cv2, frame, video_result, audio_result, fused_result, fps: float, temp: float | None, audio_gated: bool = False):
    # Resize only the presentation copy; inference keeps the original camera frame.
    width, height = PREVIEW_SIZE
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, IMAGE_HEIGHT / source_height)
    image_width = max(1, round(source_width * scale))
    image_height = max(1, round(source_height * scale))
    left = (width - image_width) // 2
    top = (IMAGE_HEIGHT - image_height) // 2
    image = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    if video_result.face is not None:
        x, y, w, h = video_result.face.bbox
        cv2.rectangle(image, (round(x * scale), round(y * scale)),
                      (round((x + w) * scale), round((y + h) * scale)), (0, 220, 0), 2)
    preview[top:top + image_height, left:left + image_width] = image
    preview[IMAGE_HEIGHT:] = (35, 35, 35)

    def emotion_text(result):
        if result is None or result.confidence <= 0:
            return "--"
        return f"{result.label}  {result.confidence:.0%}"

    face_status = "Detected" if video_result.face is not None else "Not detected"
    video_text = emotion_text(video_result.emotion) if video_result.face is not None else "--"
    audio_text = "Gated" if audio_gated else emotion_text(audio_result)
    has_signal = video_text != "--" or (not audio_gated and audio_text != "--")
    fused_text = emotion_text(fused_result) if has_signal else "--"
    temperature = f"{temp:.1f} C" if temp is not None else "n/a"
    rows = [
        (f"Face: {face_status}", f"Video: {video_text}"),
        (f"Audio: {audio_text}", f"Fused: {fused_text}"),
        (f"FPS: {fps:.1f}   Temp: {temperature}", "Relative scores, not accuracy | Q: quit"),
    ]
    for row_index, columns in enumerate(rows):
        for column_index, text in enumerate(columns):
            font_scale = 0.65 if row_index < 2 else 0.5
            cv2.putText(preview, text, (24 + column_index * 480, IMAGE_HEIGHT + 40 + row_index * 48),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), 1, cv2.LINE_AA)
    return preview
