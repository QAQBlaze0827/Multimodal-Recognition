"""Manually capture 12 local face samples and compare seven/four-class scores."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inspect_face_input import inspect_frame, render
from src.config import load_config
from src.shared_types import create_ort_session
from src.video.capture import open_camera
from src.video.face_detection import FaceDetector, FaceDetectorConfig


LABELS = ("neutral", "happy", "sad", "anger", "fear", "surprise", "disgust")
TARGETS = tuple(label for label in LABELS[:4] for _ in range(3))
MODEL_SHA256 = "cc3d71c2b35a08dbc1eef4cf2fb32e699ac9dea5f77e021367a41e9d8c887d23"
WINDOW = "Face sample capture"


def load_model(video):
    path = ROOT / video["model"]["path"]
    if int(video["model"]["input_size"]) != 48:
        raise ValueError("This diagnostic requires the verified 48x48 model")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Unknown model: cannot assume its seven-class label order")
    session, name = create_ort_session(str(path))
    if session is None:
        raise RuntimeError("ONNX model failed to load; heuristic fallback is not allowed")
    return session, name


def predict_scores(session, name, gray):
    if gray.shape != (48, 48) or gray.dtype != np.uint8:
        raise ValueError("Expected uint8 48x48 grayscale model input")
    tensor = gray.astype(np.float32).reshape(1, 1, 48, 48) / 255.0
    values = np.asarray(session.run(None, {name: tensor})[0]).reshape(-1).astype(np.float32)
    if (values.size != 7 or not np.isfinite(values).all() or (values < 0).any()
            or abs(float(values.sum()) - 1.0) >= 0.05):
        raise ValueError("Expected seven-class Softmax probabilities")
    mass = float(values[:4].sum())
    retained = values[:4] / mass if mass > 0 else None
    return {
        "seven_scores": dict(zip(LABELS, values.tolist())),
        "seven_label": LABELS[int(values.argmax())],
        "four_scores": dict(zip(LABELS[:4], retained.tolist())) if retained is not None else None,
        "four_label": LABELS[int(retained.argmax())] if retained is not None else None,
        "retained_mass": mass,
        "discarded_mass": float(values[4:].sum()),
    }


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_sample(output, index, expected, frame, face, crop, gray, backend, scores):
    stem = f"{index:02d}_{expected}"
    record = {
        "index": index, "expected_label": expected,
        "captured_at": datetime.now().astimezone().isoformat(),
        "face_backend": backend, "bbox": [int(v) for v in face.bbox],
        "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
        **scores,
    }
    for kind, image in (("crop", crop), ("gray", gray)):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"Could not encode {kind}")
        data = encoded.tobytes()
        filename = f"{stem}_{kind}.png"
        with (output / filename).open("xb") as handle:
            handle.write(data)
        record[f"{kind}_file"] = filename
        record[f"{kind}_sha256"] = hashlib.sha256(data).hexdigest()
    # The per-photo JSON is written last; partial PNGs are preserved on failure.
    write_json(output / f"{stem}.json", record)
    return record


def finish_report(output, records, error=None):
    complete = len(records) == len(TARGETS) and error is None
    summary = {
        "status": "complete" if complete else "incomplete", "error": error,
        "model_sha256": MODEL_SHA256, "label_order": LABELS,
        "expected_count": len(TARGETS), "saved_count": len(records), "samples": records,
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# 人臉表情診斷", "",
        f"狀態：{'完成' if complete else '未完成'}；已存 {len(records)} / {len(TARGETS)} 張。", "",
        "預期表情由使用者指定，不是獨立標註或內在情緒的證明；12 張不能作為正式準確率。",
        "四類分數是排除 fear/surprise/disgust 後重新正規化的相對分數，不等於原始七類機率。",
        "排除機率高時，即使四類信心度高，也不能視為模型非常確定。", "",
        "| 編號 | 預期 | 七類最高／原始機率 | 四類最高／相對分數 | 排除機率 | 影像 |",
        "|---|---|---|---|---|---|",
    ]
    for item in records:
        seven, four = item["seven_label"], item["four_label"]
        four_text = f"{four} {item['four_scores'][four]:.1%}" if four else "拒判（四類總和為零）"
        lines.append(
            f"| {item['index']:02d} | {item['expected_label']} | "
            f"{seven} {item['seven_scores'][seven]:.1%} | {four_text} | "
            f"{item['discarded_mass']:.1%} | [裁切]({item['crop_file']}) / [48x48]({item['gray_file']}) |"
        )
    lines.extend(["", "完整七類及四類分數、偵測框與圖片雜湊見 summary.json。",
                  "只儲存裁切與灰階輸入，未儲存全畫面；所有資料留在本機。"])
    if error:
        lines.extend(["", "本次中斷，已完成的照片與個別 JSON 保留；原因見 summary.json。"])
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run(video, output, camera_id=None):
    session, name = load_model(video)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "session.json", {
        "started_at": datetime.now().astimezone().isoformat(),
        "model_sha256": MODEL_SHA256, "video_config": video,
        "camera_id": video["camera_id"] if camera_id is None else camera_id,
        "targets": TARGETS,
    })
    cap = detector = None
    records = []
    error = None
    ready = True
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        detector = FaceDetector(cv2, FaceDetectorConfig(**video["face_detection"]))
        if detector.warning:
            print(detector.warning, flush=True)
        cap = open_camera(cv2, video["camera_id"] if camera_id is None else camera_id,
                          video["width"], video["height"], video["fps_target"])
        print(f"Local output: {output}", flush=True)
        print("Fixed light/distance; neutral, happy, sad, anger: 3 photos each.", flush=True)
        print("SPACE: save current face. ENTER: arm next photo. Q/ESC: stop.", flush=True)
        while len(records) < len(TARGETS):
            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                raise RuntimeError("Camera did not return a valid frame")
            face, crop, gray = inspect_frame(frame, detector, 48)
            target = TARGETS[len(records)]
            preview = render(frame, face, crop, gray, detector.backend, 48)
            canvas = np.full((370, preview.shape[1], 3), 25, dtype=np.uint8)
            canvas[:320] = preview
            status = "READY" if ready else "PAUSED"
            cv2.putText(canvas, f"{len(records) + 1}/12 | {target} | {status}", (10, 352),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (10, 13):
                ready = True
            elif key == 32 and ready:
                if face is None:
                    print("No face: nothing saved.", flush=True)
                    continue
                scores = predict_scores(session, name, gray)
                record = save_sample(output, len(records) + 1, target, frame, face, crop,
                                     gray, detector.backend, scores)
                records.append(record)
                ready = False
                print(f"Saved {len(records)}/12: {target}. ENTER to continue.", flush=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
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
                finish_report(output, records, error)
                print(f"Report: {output / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/config.yaml")
    parser.add_argument("--camera-id", type=int)
    args = parser.parse_args()
    if not args.config.is_file() or (args.camera_id is not None and args.camera_id < 0):
        parser.error("Use an existing config and nonnegative camera-id")
    output = ROOT / "data/datasets/local_faces" / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    try:
        run(load_config(args.config)["video"], output, args.camera_id)
    except KeyboardInterrupt:
        print("Stopped; existing captures retained.")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise SystemExit(f"Capture failed: {exc}\nUse a desktop display and GUI-enabled OpenCV.") from exc


if __name__ == "__main__":
    main()
