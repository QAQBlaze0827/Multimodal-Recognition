from __future__ import annotations

from dataclasses import dataclass

from src.shared_types import EmotionResult, FaceDetection
from src.video.face_detection import FaceDetector
from src.video.model import VideoEmotionModel


@dataclass
class VideoInferenceResult:
    face: FaceDetection | None
    emotion: EmotionResult
    model_backend: str


class VideoEmotionPipeline:
    def __init__(self, cv2, detector: FaceDetector, model: VideoEmotionModel, frame_skip: int) -> None:
        self.cv2 = cv2
        self.detector = detector
        self.model = model
        self.frame_skip = max(1, int(frame_skip))
        self.counter = 0
        self._log_counter = 0
        self.cached = EmotionResult.neutral("video")

    def process(self, frame) -> VideoInferenceResult:
        face = self.detector.detect(frame)
        if face is None:
            return VideoInferenceResult(None, EmotionResult.neutral("video"), self.model.backend)

        self.counter += 1
        if self.counter >= self.frame_skip:
            self.counter = 0
            x, y, w, h = face.bbox
            crop = frame[y : y + h, x : x + w]
            if crop.size:
                self.cached = self.model.predict(crop)
                self._log_counter += 1
                if self._log_counter % 30 == 0:
                    scores_str = ", ".join(f"{k}:{v:.2f}" for k, v in self.cached.scores.items())
                    print(f"[video] pred: {self.cached.label} ({self.cached.confidence:.2f}) [{self.cached.source}] {scores_str}")

        return VideoInferenceResult(face, self.cached, self.model.backend)
