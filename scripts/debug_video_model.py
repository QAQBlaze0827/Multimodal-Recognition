from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EMOTIONS = ("neutral", "happy", "sad", "anger")
MODEL_PATH = Path("models/mini_xception_fp32.onnx")


def check_onnxruntime() -> bool:
    try:
        import onnxruntime
        print(f"[check] onnxruntime {onnxruntime.__version__} loaded OK")
        return True
    except ImportError:
        print("[check] onnxruntime NOT available (not installed or wrong venv)")
        return False


def check_model_file() -> bool:
    if MODEL_PATH.exists():
        print(f"[check] model exists: {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")
        return True
    print(f"[check] model NOT found: {MODEL_PATH}")
    return False


def test_onnx_model():
    import onnxruntime as ort

    sess = ort.InferenceSession(str(MODEL_PATH))
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"\n[onnx] input:  name={inp.name}, shape={inp.shape}, type={inp.type}")
    print(f"[onnx] output: name={out.name}, shape={out.shape}, type={out.type}")

    tests = {
        "zeros (black)": np.zeros((1, 1, 48, 48), dtype=np.float32),
        "ones  (white)": np.ones((1, 1, 48, 48), dtype=np.float32),
        "half  (gray)":  np.full((1, 1, 48, 48), 0.5, dtype=np.float32),
        "random-1":      np.random.rand(1, 1, 48, 48).astype(np.float32),
        "random-2":      np.random.rand(1, 1, 48, 48).astype(np.float32),
    }

    print("\n--- ONNX model predictions ---")
    prev = None
    for label, data in tests.items():
        out_val = sess.run(None, {inp.name: data})[0]
        probs = out_val.reshape(-1)
        top_idx = int(probs.argmax())
        print(f"  {label:20s} → {EMOTIONS[top_idx]:10s} {probs[top_idx]:.4f}  {probs}")
        if prev is not None and not np.allclose(prev, probs):
            print(f"  {'':20s}   ^ different from previous input ✓")
        prev = probs

    all_out = {k: sess.run(None, {inp.name: v})[0].reshape(-1) for k, v in tests.items()}
    ref = list(all_out.values())[0]
    all_same = all(np.allclose(ref, v) for v in all_out.values())
    print(f"\n  All outputs identical? {'YES (BUG!)' if all_same else 'NO (model varies correctly)'}")

    print("\n--- Heuristic fallback simulation (for comparison) ---")
    test_heuristic()


def test_heuristic():
    scores = {e: 0.02 for e in EMOTIONS}
    scores["neutral"] = 0.65

    def normalize(s):
        total = sum(s.values())
        return {k: v / total for k, v in s.items()}

    for label, contrast_trigger, smile_trigger in [
        ("no smile, contrast >= 0.08", False, False),
        ("no smile, contrast < 0.08 ", True,  False),
        ("smile detected",             False, True),
    ]:
        s = dict(scores)
        if smile_trigger:
            s["happy"] = 0.85
            s["neutral"] = 0.15
        elif contrast_trigger:
            s["neutral"] += 0.15
        norm = normalize(s)
        top_e, top_c = max(norm.items(), key=lambda item: item[1])
        print(f"  {label:30s} → {top_e:10s} {top_c:.4f}")


if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f".venv : {sys.prefix}")
    print()

    has_ort = check_onnxruntime()
    has_model = check_model_file()

    if has_ort and has_model:
        test_onnx_model()
    elif not has_model:
        print("\nModel file missing. Cannot proceed.")
    else:
        print("\nonnxruntime not available. Please run with .venv:")
        print("  source .venv/bin/activate")
        print("  python scripts/debug_video_model.py")
