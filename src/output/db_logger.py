from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.database import (
    cleanup_old_sessions,
    create_session,
    end_session,
    init_db,
    insert_log,
)


class DbLogger:
    def __init__(
        self,
        db_path: str | Path,
        session_id: str,
        log_interval: float = 1.0,
        retention_days: int = 30,
        notes: str = "",
    ) -> None:
        self.db_path = str(db_path)
        self.session_id = session_id
        self.log_interval = log_interval
        self.retention_days = retention_days
        self.last_log_time = 0.0
        self.total_frames = 0
        self._enabled = True

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        init_db(self.db_path)
        create_session(self.db_path, session_id, notes)

    def write(
        self,
        video_result: Any,
        audio_result: Any,
        fused_result: Any,
        fps: float,
        temp: float | None,
    ) -> None:
        if not self._enabled:
            return
        now = time.time()
        if now - self.last_log_time < self.log_interval:
            self.total_frames += 1
            return
        self.last_log_time = now
        self.total_frames += 1

        face_result = getattr(video_result, "face", None)
        video_emotion = getattr(video_result, "emotion", None)
        audio_scores = {}
        video_scores = {}
        if video_emotion is not None:
            video_scores = getattr(video_emotion, "scores", {})
        if audio_result is not None:
            audio_scores = getattr(audio_result, "scores", {})

        row = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "face_detected": face_result is not None,
            "face_backend": getattr(face_result, "backend", "") if face_result else "",
            "video_emotion": getattr(video_emotion, "label", ""),
            "video_conf": getattr(video_emotion, "confidence", 0.0),
            "audio_emotion": getattr(audio_result, "label", ""),
            "audio_conf": getattr(audio_result, "confidence", 0.0),
            "fused_emotion": getattr(fused_result, "label", "neutral"),
            "fused_conf": getattr(fused_result, "confidence", 0.0),
            "fps": fps,
            "cpu_temp": temp,
            "video_scores": video_scores,
            "audio_scores": audio_scores,
        }
        insert_log(self.db_path, row)

    def close(self) -> None:
        self._enabled = False
        end_session(self.db_path, self.session_id, self.total_frames)
        cleanup_old_sessions(self.db_path, self.retention_days)
