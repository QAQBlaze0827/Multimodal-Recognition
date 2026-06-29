from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class CsvLogger:
    def __init__(self, path_template: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self.file = None
        self.writer = None
        if not enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(path_template.format(timestamp=timestamp))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "timestamp",
                "face_detected",
                "face_backend",
                "face_bbox",
                "video_emotion",
                "video_conf",
                "audio_emotion",
                "audio_conf",
                "fused_emotion",
                "fused_conf",
                "fps",
                "cpu_temp",
            ],
        )
        self.writer.writeheader()

    def write(self, video_result, audio_result, fused_result, fps: float, temp: float | None) -> None:
        if not self.enabled or self.writer is None:
            return
        self.writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "face_detected": video_result.face is not None,
                "face_backend": video_result.face.backend if video_result.face is not None else "",
                "face_bbox": (
                    ",".join(str(value) for value in video_result.face.bbox)
                    if video_result.face is not None
                    else ""
                ),
                "video_emotion": video_result.emotion.label,
                "video_conf": f"{video_result.emotion.confidence:.3f}",
                "audio_emotion": audio_result.label if audio_result else "",
                "audio_conf": f"{audio_result.confidence:.3f}" if audio_result else "",
                "fused_emotion": fused_result.label,
                "fused_conf": f"{fused_result.confidence:.3f}",
                "fps": f"{fps:.1f}",
                "cpu_temp": f"{temp:.1f}" if temp is not None else "",
            }
        )
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
