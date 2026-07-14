from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

from src.audio.inference import AudioEmotionThread
from src.audio.model import AudioEmotionModel
from src.config import load_config
from src.fusion.late_fusion import confidence_weighted_fusion
from src.output.db_logger import DbLogger
from src.output.display import draw_overlay
from src.output.logger import CsvLogger
from src.shared_types import EmotionResult
from src.video.capture import open_camera
from src.video.face_detection import FaceDetector, FaceDetectorConfig
from src.video.inference import VideoEmotionPipeline
from src.video.model import VideoEmotionModel


@dataclass
class RuntimeState:
    audio: EmotionResult | None = None
    stop: bool = False

    def __post_init__(self) -> None:
        self.lock = threading.Lock()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Architecture-based multimodal emotion prototype.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--vision-only", action="store_true")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; useful for smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.audio_only:
        run_audio_only(config)
        return
    run_video_loop(config, args)


def run_video_loop(config: dict, args) -> None:
    import cv2

    state = RuntimeState()
    video_cfg = config["video"]
    output_cfg = config["output"]

    camera_id = video_cfg["camera_id"] if args.camera_index is None else args.camera_index
    cap = open_camera(
        cv2,
        int(camera_id),
        int(video_cfg["width"]),
        int(video_cfg["height"]),
        int(video_cfg["fps_target"]),
    )

    detector_cfg = FaceDetectorConfig(**video_cfg["face_detection"])
    detector = FaceDetector(cv2, detector_cfg)
    if detector.warning:
        print(f"[video] {detector.warning}")

    model_cfg = video_cfg["model"]
    video_model = VideoEmotionModel(cv2, model_cfg["path"], int(model_cfg["input_size"]))
    video_pipeline = VideoEmotionPipeline(cv2, detector, video_model, int(video_cfg["frame_skip"]))

    audio_thread = None
    if not args.vision_only and config["audio"].get("enabled", True):
        audio_model = AudioEmotionModel(config["audio"]["model"]["path"], config["audio"]["model"].get("calibration"))
        audio_thread = AudioEmotionThread(state, config["audio"], audio_model)
        audio_thread.start()

    logger_cfg = output_cfg["csv_log"]
    logger = CsvLogger(logger_cfg["path"], bool(logger_cfg["enabled"]))
    backend_cfg = config.get("backend", {})
    db_logger = None
    if backend_cfg.get("enabled", False):
        from datetime import datetime
        session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        db_logger = DbLogger(
            db_path=backend_cfg["db_path"],
            session_id=session_id,
            log_interval=float(backend_cfg.get("log_interval", 1.0)),
            retention_days=int(backend_cfg.get("retention_days", 30)),
        )
        print(f"[backend] logging to {backend_cfg['db_path']} (session={session_id})")

    fps = 0.0
    frame_count = 0
    total_frames = 0
    last_fps_at = time.monotonic()
    display = output_cfg["display"]["show_video"] and not args.no_display

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            result = video_pipeline.process(frame)
            with state.lock:
                audio = state.audio
            fused = confidence_weighted_fusion(
                result.emotion,
                audio,
                min_confidence=float(config["fusion"]["min_confidence"]),
            )

            frame_count += 1
            total_frames += 1
            now = time.monotonic()
            elapsed = now - last_fps_at
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_at = now

            temp = read_cpu_temp()
            logger.write(result, audio, fused, fps, temp)
            if db_logger is not None:
                db_logger.write(result, audio, fused, fps, temp)

            if display:
                draw_overlay(cv2, frame, result, audio, fused, fps, temp)
                cv2.imshow("Multimodal Emotion Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames and total_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        state.stop = True
        if audio_thread is not None:
            audio_thread.join(timeout=2.0)
        logger.close()
        if db_logger is not None:
            db_logger.close()
        detector.close()
        cap.release()
        if display:
            cv2.destroyAllWindows()


def run_audio_only(config: dict) -> None:
    state = RuntimeState()
    audio_model = AudioEmotionModel(config["audio"]["model"]["path"], config["audio"]["model"].get("calibration"))
    thread = AudioEmotionThread(state, config["audio"], audio_model)
    thread.start()
    print("[audio] audio-only mode. Press Ctrl+C to stop.")
    try:
        while True:
            with state.lock:
                audio = state.audio
            if audio is not None:
                print(f"[audio] {audio.label} {audio.confidence:.2f}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        state.stop = True
        thread.join(timeout=2.0)


def read_cpu_temp() -> float | None:
    try:
        from pathlib import Path

        thermal = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal.exists():
            return float(thermal.read_text(encoding="utf-8").strip()) / 1000.0
    except Exception:
        return None
    return None


if __name__ == "__main__":
    main()
