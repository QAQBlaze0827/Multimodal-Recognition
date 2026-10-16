"""Display camera, face crop and model grayscale input locally; save no images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.video.capture import open_camera
from src.video.face_detection import FaceDetector, FaceDetectorConfig


WINDOW = "Face input inspection"
PANEL_WIDTH = 320
IMAGE_HEIGHT = 240


def inspect_frame(frame, detector, input_size: int):
    face = detector.detect(frame)
    if face is None:
        return None, None, None
    x, y, w, h = face.bbox
    height, width = frame.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("Detector returned an invalid face rectangle")
    crop = frame[y:y + h, x:x + w].copy()
    # Match VideoEmotionPipeline.process() and VideoEmotionModel.predict().
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (input_size, input_size))
    return face, crop, gray


def image_panel(image, nearest=False):
    panel = np.zeros((IMAGE_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    if image is None:
        return panel
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    scale = min(PANEL_WIDTH / width, IMAGE_HEIGHT / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    enlarged = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    x, y = (PANEL_WIDTH - size[0]) // 2, (IMAGE_HEIGHT - size[1]) // 2
    panel[y:y + size[1], x:x + size[0]] = enlarged
    return panel


def render(frame, face, crop, gray, backend, input_size):
    canvas = np.full((320, PANEL_WIDTH * 3, 3), 25, dtype=np.uint8)
    camera = frame.copy()
    if face is not None:
        x, y, w, h = face.bbox
        cv2.rectangle(camera, (x, y), (x + w - 1, y + h - 1), (70, 210, 100), 1)
    titles = ("CAMERA", "FACE CROP", f"MODEL GRAY {input_size}x{input_size}")
    for i, (title, image) in enumerate(zip(titles, (camera, crop, gray))):
        x = i * PANEL_WIDTH
        canvas[40:40 + IMAGE_HEIGHT, x:x + PANEL_WIDTH] = image_panel(image, nearest=i == 2)
        cv2.putText(canvas, title, (x + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    details = [f"{frame.shape[1]}x{frame.shape[0]} | {backend}", "No face", "No face"]
    if face is not None:
        details[1] = f"x={face.bbox[0]} y={face.bbox[1]} | {crop.shape[1]}x{crop.shape[0]}"
        details[2] = f"min={int(gray.min())} max={int(gray.max())} mean={float(gray.mean()):.1f}"
    for i, text in enumerate(details):
        cv2.putText(canvas, text, (i * PANEL_WIDTH + 10, 305), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def run(video, camera_id=None, max_frames=0):
    cap = detector = None
    try:
        # Fail before opening the camera when OpenCV has no desktop GUI support.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        detector = FaceDetector(cv2, FaceDetectorConfig(**video["face_detection"]))
        if detector.warning:
            print(detector.warning, flush=True)
        camera_id = video["camera_id"] if camera_id is None else camera_id
        print(f"Camera index: {camera_id}; face detector: {detector.backend}", flush=True)
        print("Local preview only: no images, video, database or emotion predictions are saved.", flush=True)
        print("Focus the preview and press Q or Esc to exit; closing the window also exits.", flush=True)
        cap = open_camera(cv2, camera_id, video["width"], video["height"], video["fps_target"])
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError("Camera opened but did not return a valid frame")
            size = int(video["model"]["input_size"])
            face, crop, gray = inspect_frame(frame, detector, size)
            cv2.imshow(WINDOW, render(frame, face, crop, gray, detector.backend, size))
            key = cv2.waitKey(1) & 0xFF
            count += 1
            if key in (27, ord("q"), ord("Q")) or (max_frames and count >= max_frames):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if cap is not None:
            cap.release()
        if detector is not None:
            detector.close()
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--camera-id", type=int, help="Override the camera index for this run only.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 runs until closed.")
    args = parser.parse_args()
    if not args.config.is_file() or args.max_frames < 0 or (args.camera_id is not None and args.camera_id < 0):
        parser.error("Use an existing config and nonnegative camera-id/max-frames")
    video = load_config(args.config)["video"]
    if not 1 <= int(video["model"]["input_size"]) <= 512:
        parser.error("input_size must be between 1 and 512")
    try:
        run(video, args.camera_id, args.max_frames)
    except KeyboardInterrupt:
        print("Preview stopped.")
    except (cv2.error, RuntimeError) as exc:
        raise SystemExit(f"Preview failed: {exc}\nRequires a desktop display and GUI-enabled OpenCV; close other camera apps first.") from exc


if __name__ == "__main__":
    main()
