"""Local FERPlus webcam preview; no downloads, image saving or production changes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_face_models import FER_LABELS, FER_PATH, FER_SHA256, fer_scores, fer_tensor, open_session
from scripts.inspect_face_input import inspect_frame, render as render_input
from src.config import load_config
from src.video.capture import open_camera
from src.video.face_detection import FaceDetector, FaceDetectorConfig


WINDOW = "FERPlus live preview"


def process_frame(frame, detector, session, name):
    face, crop, _ = inspect_frame(frame, detector, 64)
    if face is None:
        return None, None, None, None, None
    # Use exactly the same tensor as the offline comparison, not the viewer's linear resize.
    tensor = fer_tensor(crop)
    start = perf_counter()
    logits = session.run(None, {name: tensor})[0]
    elapsed_ms = (perf_counter() - start) * 1000
    return face, crop, tensor[0, 0].astype(np.uint8), fer_scores(logits), elapsed_ms


def render(frame, face, crop, gray, backend, scores, elapsed_ms):
    canvas = np.full((560, 960, 3), 25, dtype=np.uint8)
    canvas[:320] = render_input(frame, face, crop, gray, backend, 64)

    def text(value, x, y, color=(230, 230, 230), scale=0.52):
        cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    if scores is None:
        text("FULL (8): No face", 10, 344)
        text("FOUR (relative): --", 490, 344)
        text("Excluded mass: --", 10, 370)
        text("Model inference: --", 490, 370)
    else:
        label, four = scores["full_label"], scores["four_label"]
        text(f"FULL (8): {label} {scores['full_scores'][label]:.1%}", 10, 344)
        four_value = f"{four} {scores['four_scores'][four]:.1%}" if four else "Abstain"
        text(f"FOUR (relative): {four_value}", 490, 344)
        text(f"Excluded mass: {scores['discarded_mass']:.1%}", 10, 370)
        text(f"Model inference: {elapsed_ms:.1f} ms", 490, 370)
    for i, label in enumerate(FER_LABELS):
        x, y = (i // 4) * 480 + 10, 406 + (i % 4) * 34
        text(label, x, y)
        cv2.rectangle(canvas, (x + 120, y - 13), (x + 320, y + 1), (55, 55, 55), -1)
        value = scores["full_scores"][label] if scores is not None else None
        if value is not None:
            width = round(200 * value)
            if width:
                cv2.rectangle(canvas, (x + 120, y - 13), (x + 119 + width, y + 1), (120, 190, 90), -1)
        text(f"{value:.1%}" if value is not None else "--", x + 340, y)
    text("FERPlus | CPU | raw per-frame scores", 10, 548, color=(170, 170, 170), scale=0.45)
    return canvas


def run(video, camera_id=None, max_frames=0):
    if not FER_PATH.is_file():
        raise FileNotFoundError("FERPlus cache missing; use compare_face_models.py --download-model with your photo session first")
    session, name = open_session(FER_PATH, FER_SHA256, 64, 8)
    cap = detector = None
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        detector = FaceDetector(cv2, FaceDetectorConfig(**video["face_detection"]))
        if detector.warning:
            print(detector.warning, flush=True)
        index = video["camera_id"] if camera_id is None else camera_id
        cap = open_camera(cv2, index, video["width"], video["height"], video["fps_target"])
        print(f"Camera: {index}; detector: {detector.backend}; model: FERPlus (CPU)", flush=True)
        print("Local preview only. No images, predictions or database records are saved.", flush=True)
        print("Eight-class probabilities and four-class relative scores are different; no smoothing or calibration.", flush=True)
        print("Focus the window; Q/ESC or closing it exits. Ctrl+C also stops.", flush=True)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError("Camera did not return a valid frame")
            face, crop, gray, scores, elapsed_ms = process_frame(frame, detector, session, name)
            cv2.imshow(WINDOW, render(frame, face, crop, gray, detector.backend, scores, elapsed_ms))
            key = cv2.waitKey(1) & 0xFF
            count += 1
            if key in (27, ord("q"), ord("Q")) or (max_frames and count >= max_frames):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        try:
            if cap is not None:
                cap.release()
        finally:
            try:
                if detector is not None:
                    detector.close()
            finally:
                try:
                    cv2.destroyWindow(WINDOW)
                except cv2.error:
                    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if not args.config.is_file() or args.max_frames < 0 or (args.camera_id is not None and args.camera_id < 0):
        parser.error("Use an existing config and nonnegative camera-id/max-frames")
    try:
        run(load_config(args.config)["video"], args.camera_id, args.max_frames)
    except KeyboardInterrupt:
        print("Preview stopped.")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(f"Preview failed: {exc}\nRequires desktop GUI OpenCV; close other camera apps first. No automatic download or model fallback.") from exc


if __name__ == "__main__":
    main()
