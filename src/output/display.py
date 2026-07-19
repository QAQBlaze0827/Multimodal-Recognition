from __future__ import annotations


def draw_overlay(cv2, frame, video_result, audio_result, fused_result, fps: float, temp: float | None, audio_gated: bool = False) -> None:
    face_status = "NO"
    face_backend = "none"
    face_bbox = "-"
    if video_result.face is not None:
        x, y, w, h = video_result.face.bbox
        face_status = "YES"
        face_backend = video_result.face.backend
        face_bbox = f"{x},{y},{w},{h}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
        label = f"{video_result.emotion.label} {video_result.emotion.confidence:.2f}"
        cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

    audio_display = "(gated)" if audio_gated else (
        f"{audio_result.label} {audio_result.confidence:.2f}" if audio_result else "none"
    )
    rows = [
        f"Face: {face_status} [{face_backend}] bbox={face_bbox}",
        f"Video: {video_result.emotion.label} {video_result.emotion.confidence:.2f} [{video_result.model_backend}]",
        f"Audio: {audio_display}",
        f"Fused: {fused_result.label} {fused_result.confidence:.2f}",
        f"FPS: {fps:.1f} | Temp: {temp:.1f}C" if temp is not None else f"FPS: {fps:.1f} | Temp: n/a",
        "press q to quit",
    ]

    y = frame.shape[0] - 110
    for row in rows:
        cv2.putText(frame, row, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, row, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 18
