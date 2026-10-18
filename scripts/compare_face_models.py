"""Offline comparison of captured faces; --download-model explicitly fetches FERPlus."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.capture_face_samples import LABELS, MODEL_SHA256, TARGETS, predict_scores, write_json

REVISION = "c5cae0f1942c8d7727372c7b1f6b485aa39e4421"
UPSTREAM = "validated/vision/body_analysis/emotion_ferplus"
RAW = f"https://raw.githubusercontent.com/onnx/models/{REVISION}"
MODEL_URL = f"https://media.githubusercontent.com/media/onnx/models/{REVISION}/{UPSTREAM}/model/emotion-ferplus-8.onnx"
FER_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
FER_BYTES = 35040571
FER_PATH = ROOT / ".cache/ferplus/emotion-ferplus-8.onnx"
# Official happiness/sadness names are canonicalized, not merged with other classes.
FER_LABELS = ("neutral", "happy", "surprise", "sad", "anger", "disgust", "fear", "contempt")
FOUR = LABELS[:4]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require_hash(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return data


def download_model():
    FER_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Preserve upstream documentation; the model README says MIT, the repo license Apache-2.0.
    for filename, url in (("UPSTREAM_README.md", f"{RAW}/{UPSTREAM}/README.md"),
                          ("UPSTREAM_REPO_LICENSE.txt", f"{RAW}/LICENSE")):
        target = FER_PATH.parent / filename
        if not target.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            with target.open("xb") as handle:
                handle.write(data)
    if FER_PATH.exists():
        require_hash(FER_PATH, FER_SHA256)
        return
    partial = FER_PATH.with_name(f"{FER_PATH.name}.{datetime.now():%Y%m%d_%H%M%S_%f}.download")
    size = 0
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open("xb") as handle:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > FER_BYTES:
                raise ValueError("Unexpected model size; partial download preserved")
            handle.write(chunk)
    if size != FER_BYTES:
        raise ValueError("Incomplete model; partial download preserved")
    require_hash(partial, FER_SHA256)
    if FER_PATH.exists():
        raise FileExistsError(FER_PATH)
    partial.rename(FER_PATH)
    write_json(FER_PATH.parent / "source.json", {
        "url": MODEL_URL, "revision": REVISION, "sha256": FER_SHA256, "bytes": size,
        "readme_url": f"{RAW}/{UPSTREAM}/README.md",
        "license_note": "Model README: MIT; repository LICENSE: Apache-2.0. Preserve both; review before redistribution.",
    })
    print(f"Downloaded and verified FERPlus: {size} bytes", flush=True)


def read_image(folder, item, kind):
    filename = item[f"{kind}_file"]
    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("Expected a local image basename")
    path = folder / filename
    if path.resolve().parent != folder.resolve():
        raise ValueError("Image escapes session directory")
    raw = require_hash(path, item[f"{kind}_sha256"])
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError(f"Invalid PNG: {filename}")
    return image


def load_samples(folder):
    raw = (folder / "summary.json").read_bytes()
    summary = json.loads(raw)
    samples = summary["samples"]
    if (summary["status"] != "complete" or summary.get("error") is not None
            or summary["saved_count"] != 12 or summary["expected_count"] != 12
            or summary["model_sha256"] != MODEL_SHA256
            or tuple(summary["label_order"]) != LABELS
            or tuple(item["expected_label"] for item in samples) != TARGETS
            or [item["index"] for item in samples] != list(range(1, 13))):
        raise ValueError("Expected a complete, verified 12-photo capture session")
    result, used = [], set()
    for item in samples:
        sidecar = folder / f"{item['index']:02d}_{item['expected_label']}.json"
        if json.loads(sidecar.read_text(encoding="utf-8")) != item:
            raise ValueError("Photo sidecar differs from session summary")
        for kind in ("crop", "gray"):
            filename = item[f"{kind}_file"]
            if filename in used:
                raise ValueError("Duplicate image path")
            used.add(filename)
        crop, gray = read_image(folder, item, "crop"), read_image(folder, item, "gray")
        if crop.ndim != 3 or crop.shape[2] != 3 or gray.shape != (48, 48):
            raise ValueError("Unexpected crop/grayscale dimensions")
        expected = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (48, 48))
        if not np.array_equal(gray, expected):
            raise ValueError("Saved grayscale differs from original crop preprocessing")
        result.append((item, crop, gray))
    return result, digest(raw)


def fer_tensor(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = Image.fromarray(gray).resize((64, 64), Image.Resampling.LANCZOS)
    # FERPlus expects raw 0..255 grayscale, NOT the legacy model's /255 input.
    return np.asarray(resized, dtype=np.float32).reshape(1, 1, 64, 64)


def fer_scores(logits):
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != 8 or not np.isfinite(values).all():
        raise ValueError("Expected eight finite FERPlus logits")
    exp = np.exp(values - values.max())
    probs = exp / exp.sum()
    selected = probs[[0, 1, 3, 4]]
    mass = float(selected.sum())
    return {
        "full_scores": dict(zip(FER_LABELS, probs.tolist())),
        "full_label": FER_LABELS[int(probs.argmax())],
        "four_scores": dict(zip(FOUR, (selected / mass).tolist())) if mass > 0 else None,
        "four_label": FOUR[int(selected.argmax())] if mass > 0 else None,
        "discarded_mass": float(probs[[2, 5, 6, 7]].sum()),
    }


def legacy_scores(session, name, gray, saved):
    result = predict_scores(session, name, gray)
    if result["seven_label"] != saved["seven_label"] or result["four_label"] != saved["four_label"]:
        raise ValueError("Legacy prediction no longer matches captured result")
    for key in ("seven_scores", "four_scores"):
        if result[key] is None or saved[key] is None:
            if result[key] != saved[key]:
                raise ValueError("Legacy abstention mismatch")
        elif set(result[key]) != set(saved[key]) or not np.allclose(
                list(result[key].values()), [saved[key][label] for label in result[key]], rtol=1e-4, atol=1e-6):
            raise ValueError("Legacy probabilities differ from captured result")
    return {"full_scores": result["seven_scores"], "full_label": result["seven_label"],
            "four_scores": result["four_scores"], "four_label": result["four_label"],
            "discarded_mass": result["discarded_mass"]}


def open_session(path, sha, size, classes):
    require_hash(path, sha)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape[1:] != [1, size, size]
            or len(outputs) != 1 or outputs[0].shape[-1] != classes):
        raise ValueError("Unexpected model input/output contract")
    return session, inputs[0].name


def metrics(rows, model):
    return {mode: {label: {
        "matched": sum(row["expected_label"] == label and row[model][mode + "_label"] == label for row in rows),
        "total": sum(row["expected_label"] == label for row in rows),
    } for label in FOUR} for mode in ("full", "four")}


def report_text(result):
    lines = ["# 同照片影像模型比較", "", "這是單人 12 張照片與預期表情的一致率，不是正式測試準確率。",
             "兩模型使用相同原始裁切，各自套用規定的前處理；沒有重拍、重新偵測、調參或上傳照片。",
             "四類只選出中性／開心／難過／生氣後正規化；其他類別不硬合併，完整分類結果另列。", "",
             "| 預期表情 | 舊模型完整／四類符合 | FERPlus 完整／四類符合 |", "|---|---|---|"]
    for label in FOUR:
        values = []
        for model in ("legacy", "ferplus"):
            m = result["metrics"][model]
            values.append(f"{m['full'][label]['matched']}/{m['full'][label]['total']} / {m['four'][label]['matched']}/{m['four'][label]['total']}")
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines.extend(["", "| 編號 | 預期 | 舊完整 | 舊四類 | 舊排除機率 | FER+完整 | FER+四類 | FER+排除機率 |",
                  "|---|---|---|---|---|---|---|---|"])
    for row in result["samples"]:
        a, b = row["legacy"], row["ferplus"]
        lines.append(f"| {row['index']:02d} | {row['expected_label']} | {a['full_label']} | {a['four_label']} | "
                     f"{a['discarded_mass']:.1%} | {b['full_label']} | {b['four_label']} | {b['discarded_mass']:.1%} |")
    lines.extend(["", "完整機率、裁切雜湊、模型來源與計時見 summary.json；本報告不複製個人照片。",
                  "計時只含 CPU session.run（預熱後），不含讀圖、前處理、相機或人臉偵測，不代表樹莓派效能。",
                  "本批曾用於診斷，不能作為最終獨立測試；需新樣本、其他使用者與樹莓派實測再決定部署。",
                  "FERPlus 上游模型 README 標 MIT，倉庫 LICENSE 標 Apache-2.0；已保留原始聲明，重新散布前應釐清。"])
    return "\n".join(lines) + "\n"


def compare(folder, output):
    if output.exists():
        raise FileExistsError(output)
    samples, summary_hash = load_samples(folder)
    old, old_name = open_session(ROOT / "models/mini_xception_fp32.onnx", MODEL_SHA256, 48, 7)
    new, new_name = open_session(FER_PATH, FER_SHA256, 64, 8)
    # Warm up outside recorded per-photo timings.
    for _ in range(3):
        old.run(None, {old_name: samples[0][2].astype(np.float32)[None, None] / 255.0})
        new.run(None, {new_name: fer_tensor(samples[0][1])})
    rows = []
    for item, crop, gray in samples:
        legacy = legacy_scores(old, old_name, gray, item)
        tensor_old = gray.astype(np.float32)[None, None] / 255.0
        start = perf_counter()
        old.run(None, {old_name: tensor_old})
        old_ms = (perf_counter() - start) * 1000
        tensor_new = fer_tensor(crop)
        start = perf_counter()
        logits = new.run(None, {new_name: tensor_new})[0]
        new_ms = (perf_counter() - start) * 1000
        rows.append({"index": item["index"], "expected_label": item["expected_label"],
                     "crop_sha256": item["crop_sha256"], "gray_sha256": item["gray_sha256"],
                     "legacy": legacy, "ferplus": fer_scores(logits),
                     "legacy_ms": old_ms, "ferplus_ms": new_ms})
    result = {
        "status": "complete", "session": folder.name, "source_summary_sha256": summary_hash,
        "created_at": datetime.now().astimezone().isoformat(),
        "models": {"legacy": MODEL_SHA256, "ferplus": FER_SHA256},
        "ferplus_source": {"url": MODEL_URL, "revision": REVISION, "readme": f"{RAW}/{UPSTREAM}/README.md"},
        "preprocessing": {"legacy": "BGR->OpenCV gray->48x48 linear->float32 /255 NCHW; Softmax probabilities",
                          "ferplus": "BGR->OpenCV gray->Pillow 64x64 LANCZOS->float32 0..255 NCHW; logits->Softmax"},
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "onnxruntime": ort.__version__, "opencv": cv2.__version__, "pillow": PILLOW_VERSION,
                        "provider": "CPUExecutionProvider", "intra_threads": 2, "inter_threads": 1},
        "median_inference_ms": {model: float(np.median([row[model + "_ms"] for row in rows])) for model in ("legacy", "ferplus")},
        "metrics": {model: metrics(rows, model) for model in ("legacy", "ferplus")}, "samples": rows,
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(report_text(result))
    write_json(output / "summary.json", result)
    print(f"Report: {output / 'report.md'}", flush=True)
    for model in ("legacy", "ferplus"):
        correct = sum(m["matched"] for m in result["metrics"][model]["four"].values())
        print(f"{model}: four-class agreement {correct}/12; median inference {result['median_inference_ms'][model]:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--download-model", action="store_true", help="Fetch the pinned 35 MB model if missing; no photo uploads.")
    args = parser.parse_args()
    output = args.output or ROOT / "data/reports" / datetime.now().strftime("face_model_comparison_%Y%m%d_%H%M%S_%f")
    if output.exists():
        parser.error("Output already exists; choose a new directory, do not delete old reports")
    if args.download_model:
        download_model()
    compare(args.session, output)


if __name__ == "__main__":
    main()
