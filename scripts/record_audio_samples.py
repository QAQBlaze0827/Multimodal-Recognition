"""Record local headset samples for later model comparison; no inference."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("neutral", "happy", "sad", "anger")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--list-devices", action="store_true", help="List devices without recording.")
    result.add_argument("--device", type=int, help="Input device index; defaults to the system input.")
    result.add_argument("--label", choices=LABELS, help="Record one label only; default: all four.")
    result.add_argument("--takes", type=int, default=3, help="Recordings per label (1-20).")
    result.add_argument("--seconds", type=float, default=5.0, help="Seconds per recording (1-30).")
    result.add_argument("--sample-rate", type=int, default=16000, help="Capture rate; no software resampling.")
    return result


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def capture(sd, frames: int, sample_rate: int, device: int) -> np.ndarray:
    try:
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, device=device,
                       dtype="float32", blocking=False)
        status = sd.wait()
        if status:
            raise RuntimeError(f"Capture reported {status}; this take was not saved. Retry in a new session.")
        if audio.shape != (frames, 1) or not np.isfinite(audio).all():
            raise ValueError("Capture returned invalid or incomplete samples")
        return audio
    finally:
        sd.stop()


def wav_payload(sf, audio: np.ndarray, sample_rate: int) -> tuple[bytes, dict]:
    if audio.ndim != 2 or audio.shape[1] != 1 or not len(audio):
        raise ValueError("Expected nonempty mono frames x 1 audio")
    if audio.dtype != np.float32 or not np.isfinite(audio).all() or sample_rate <= 0:
        raise ValueError("Expected finite float32 audio and a positive sample rate")
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="FLOAT")
    payload = buffer.getvalue()
    values = audio.astype(np.float64)
    return payload, {
        "frames": len(audio), "sample_rate": sample_rate, "channels": 1,
        "duration_seconds": len(audio) / sample_rate, "wav_subtype": "FLOAT",
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "peak": float(np.max(np.abs(values))),
        "full_scale_fraction": float(np.mean(np.abs(values) >= 1.0)),
        "all_zero": bool(np.all(values == 0)),
        "wav_sha256": hashlib.sha256(payload).hexdigest(),
    }


def save_take(output: Path, name: str, sf, audio: np.ndarray, sample_rate: int,
              label: str, started_at: str) -> dict:
    payload, metrics = wav_payload(sf, audio, sample_rate)
    record = {
        "file": name + ".wav", "expected_label": label,
        "label_source": "user_intended_emotion_not_independently_verified",
        "recording_started_at": started_at, "heard_label": "", "notes": "", **metrics,
    }
    with (output / record["file"]).open("xb") as file:
        file.write(payload)
    write_json(output / (name + ".json"), record)
    return record


def record_session(args, sd, sf, prompt=input) -> Path:
    device = args.device if args.device is not None else int(sd.default.device[0])
    if device < 0:
        raise ValueError("No default input device; select one with --device")
    info = sd.query_devices(device, "input")
    sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=args.sample_rate)
    labels = [args.label] if args.label else list(LABELS)
    frames = round(args.seconds * args.sample_rate)
    output = ROOT / "data/datasets/local_audio" / datetime.now().astimezone().strftime("session_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "session.json", {
        "created_at": datetime.now().astimezone().isoformat(), "device_index": device,
        "device_name": info["name"], "host_api": sd.query_hostapis(info["hostapi"])["name"],
        "sample_rate": args.sample_rate, "channels": 1, "seconds_requested": args.seconds,
        "frames_per_take": frames, "labels": labels, "takes_per_label": args.takes,
        "processing": "none; float32 samples from audio API; OS/device processing may still apply",
        "purpose": "local diagnostic recordings, not training data or a formal test set",
    })
    print(f"Input: [{device}] {info['name']}; mono {args.sample_rate} Hz", flush=True)
    print(f"Output: {output}", flush=True)
    print("Each take starts after Enter. Ctrl+C stops; completed files are kept.", flush=True)
    records = []
    for label in labels:
        for take in range(1, args.takes + 1):
            prompt(f"[{label} {take}/{args.takes}] Ready? Press Enter to record {args.seconds:g}s: ")
            started_at = datetime.now().astimezone().isoformat()
            print("RECORDING...", flush=True)
            audio = capture(sd, frames, args.sample_rate, device)
            record = save_take(output, f"{label}_{take:02}", sf, audio, args.sample_rate, label, started_at)
            records.append(record)
            print(f"Saved {record['file']}; RMS={record['rms']:.4f}, peak={record['peak']:.4f}", flush=True)
            if record["all_zero"] or record["full_scale_fraction"] > 0:
                print("Review audio: all-zero or full-scale samples detected; original take retained.", flush=True)
    write_json(output / "summary.json", {"status": "complete", "recordings": records})
    print("Complete. Listen to the WAV files before comparing models.", flush=True)
    return output


def main(argv=None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    if not 1 <= args.takes <= 20 or not np.isfinite(args.seconds) or not 1 <= args.seconds <= 30:
        cli.error("takes must be 1-20 and seconds must be finite and 1-30")
    if not 8000 <= args.sample_rate <= 192000 or (args.device is not None and args.device < 0):
        cli.error("sample-rate must be 8000-192000; device must be nonnegative")
    try:
        import sounddevice as sd
        if args.list_devices:
            print(sd.query_devices())
            return
        import soundfile as sf
        record_session(args, sd, sf)
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("Stopped. Existing files were kept. Check files if interrupted during saving; no cleanup performed.")
    except Exception as exc:
        raise SystemExit(f"Recording failed: {exc}. Existing files were kept; no cleanup performed.") from exc


if __name__ == "__main__":
    main()
