from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.shared_types import FaceDetection


@dataclass
class FaceDetectorConfig:
    backend: str = "auto"
    min_detection_confidence: float = 0.7
    model_selection: int = 0


class FaceDetector:
    def __init__(self, cv2, config: FaceDetectorConfig) -> None:
        self.cv2 = cv2
        self.config = config
        self.backend = "none"
        self._mp_detector = None
        self._haar = None
        self._warning: str | None = None
        self._init_backend()

    @property
    def warning(self) -> str | None:
        return self._warning

    def close(self) -> None:
        if self._mp_detector is not None:
            self._mp_detector.close()

    def detect(self, frame) -> FaceDetection | None:
        if self.backend == "mediapipe":
            detection = self._detect_mediapipe(frame)
            if detection is not None:
                return detection
        if self._haar is not None:
            return self._detect_haar(frame)
        return None

    def _init_backend(self) -> None:
        requested = self.config.backend.lower()
        if requested in ("auto", "mediapipe"):
            try:
                self._mp_detector = self._create_mediapipe_detector()
                self.backend = "mediapipe"
                return
            except Exception as exc:
                self._warning = (
                    "MediaPipe face detection failed to initialize; "
                    f"fallback to OpenCV Haar. Reason: {exc}"
                )
                if requested == "mediapipe":
                    raise

        haar_path = self._ascii_cache_file(
            Path(self.cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        )
        haar = self.cv2.CascadeClassifier(haar_path)
        if haar.empty():
            raise RuntimeError(f"Could not load Haar cascade: {haar_path}")
        self._haar = haar
        self.backend = "haar"

    def _ascii_cache_file(self, source: Path) -> str:
        if not source.exists():
            raise RuntimeError(f"OpenCV resource does not exist: {source}")
        cache_dir = Path(tempfile.gettempdir()) / "multimodal_emotion_cv2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / source.name
        if not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        return str(target)

    def _create_mediapipe_detector(self):
        import os
        from pathlib import Path

        cache_dir = Path.cwd() / ".cache" / "matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

        try:
            import mediapipe as mp

            face_detection = mp.solutions.face_detection
        except AttributeError:
            from mediapipe.python.solutions import face_detection

        return face_detection.FaceDetection(
            model_selection=int(self.config.model_selection),
            min_detection_confidence=float(self.config.min_detection_confidence),
        )

    def _detect_mediapipe(self, frame) -> FaceDetection | None:
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        results = self._mp_detector.process(rgb)
        if not results.detections:
            return None

        height, width = frame.shape[:2]
        best = max(results.detections, key=lambda item: item.score[0])
        rel = best.location_data.relative_bounding_box
        x = max(0, int(rel.xmin * width))
        y = max(0, int(rel.ymin * height))
        w = min(width - x, int(rel.width * width))
        h = min(height - y, int(rel.height * height))
        if w <= 0 or h <= 0:
            return None
        return FaceDetection((x, y, w, h), float(best.score[0]), "mediapipe")

    def _detect_haar(self, frame) -> FaceDetection | None:
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return FaceDetection((int(x), int(y), int(w), int(h)), 0.65, "haar")
