"""Local microphone STT; optional loopback text delivery, no audio files or uploads."""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

from dog_reply import choose_reply

SAMPLE_RATE = 16000
BLOCK_SIZE = 1600


class PhraseBuffer:
    """Energy-based endpointing; Whisper's VAD checks the completed phrase too."""

    def __init__(self, threshold=0.002):
        self.threshold = threshold
        self.pre_roll = deque(maxlen=3)
        self.reset()

    def reset(self):
        self.parts = []
        self.total = 0
        self.voiced = 0
        self.quiet = 0
        self.pre_roll.clear()

    def feed(self, chunk):
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size != BLOCK_SIZE or not np.isfinite(chunk).all():
            raise ValueError("Expected a finite 100 ms mono audio block")
        loud = float(np.sqrt(np.mean(chunk * chunk))) >= self.threshold
        if not self.parts:
            if not loud:
                self.pre_roll.append(chunk.copy())
                return None
            self.parts = list(self.pre_roll)
            self.total = sum(part.size for part in self.parts)
            self.pre_roll.clear()
        self.parts.append(chunk.copy())
        self.total += chunk.size
        self.voiced += chunk.size if loud else 0
        self.quiet = 0 if loud else self.quiet + chunk.size
        # Bound continuous speech/noise, and reject very short clicks.
        if self.quiet >= int(0.8 * SAMPLE_RATE) or self.total >= 12 * SAMPLE_RATE:
            result = np.concatenate(self.parts) if self.voiced >= int(0.3 * SAMPLE_RATE) else None
            self.reset()
            return result
        return None


def load_model(model_dir=None):
    if model_dir is None:
        cache_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
        model_dir = cache_root / "emotion-stt-test" / "models"
    model_dir = Path(model_dir).expanduser().resolve()
    print(f"Model cache: {model_dir}", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    from faster_whisper import WhisperModel

    return WhisperModel(
        "base", device="cpu", compute_type="int8",
        download_root=str(model_dir),
        local_files_only=True,
    )


def transcribe(model, audio):
    started = time.perf_counter()
    segments, _ = model.transcribe(
        audio, language="zh", beam_size=5, vad_filter=True,
        condition_on_previous_text=False,
    )
    text = "".join(segment.text for segment in segments).strip()
    return text, time.perf_counter() - started


def transcription_worker(model, phrases, stop, errors, publisher=None):
    while not stop.is_set():
        try:
            number, audio, queued_at = phrases.get(timeout=0.2)
        except queue.Empty:
            continue
        if stop.is_set():
            return
        try:
            wait = time.perf_counter() - queued_at
            text, elapsed = transcribe(model, audio)
            print(f"\n[{number:03d}] {text or '(No speech recognized)'}", flush=True)
            print(f"  audio={audio.size / SAMPLE_RATE:.2f}s | "
                  f"queue={wait:.2f}s | STT={elapsed:.2f}s", flush=True)
            reply = choose_reply(text)
            if reply is not None:
                intent, message = reply
                print(f"  Dog [{intent}]: {message}", flush=True)
                if publisher is not None and not stop.is_set():
                    publisher.publish(text, intent, message)
        except Exception as exc:
            errors.put(exc)
            stop.set()
            return


def run(sd, model, device, threshold, publisher=None):
    blocks = queue.Queue(maxsize=50)
    phrases = queue.Queue(maxsize=3)
    errors = queue.SimpleQueue()
    stop = threading.Event()
    capture_failed = threading.Event()
    segmenter = PhraseBuffer(threshold)
    worker = threading.Thread(
        target=transcription_worker, args=(model, phrases, stop, errors, publisher),
        name="local-stt",
    )

    def callback(indata, frames, time_info, status):
        if status or frames != BLOCK_SIZE:
            capture_failed.set()
            return
        try:
            blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            capture_failed.set()

    worker.start()
    number = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1,
            device=device, dtype="float32", callback=callback,
        ) as stream:
            if publisher is not None:
                publisher.start()
            print("Listening. Speak a short sentence, then pause about 0.8 seconds.", flush=True)
            print("Ctrl+C stops recording. Audio stays in memory; no files are saved.", flush=True)
            last_block = time.monotonic()
            while not stop.is_set():
                if capture_failed.is_set():
                    raise RuntimeError("Microphone overflow/status error. Stopped to avoid transcribing broken audio.")
                if not stream.active:
                    raise RuntimeError("Microphone stream stopped unexpectedly")
                try:
                    chunk = blocks.get(timeout=0.2)
                except queue.Empty:
                    if time.monotonic() - last_block > 3:
                        raise RuntimeError("No microphone samples received for 3 seconds")
                    continue
                last_block = time.monotonic()
                was_idle = not segmenter.parts
                audio = segmenter.feed(chunk)
                if was_idle and segmenter.parts:
                    print("[capture] Sound detected...", flush=True)
                if audio is not None:
                    number += 1
                    try:
                        phrases.put_nowait((number, audio, time.perf_counter()))
                        print(f"[capture] Phrase {number:03d} queued; microphone remains active.", flush=True)
                    except queue.Full:
                        print(f"[busy] Phrase {number:03d} skipped: transcription queue is full.", flush=True)
            if not errors.empty():
                raise RuntimeError(f"Transcription failed: {errors.get()}")
    except KeyboardInterrupt:
        print("\nStopping microphone; waiting for any active transcription...", flush=True)
    finally:
        stop.set()
        if publisher is not None:
            publisher.close()
        worker.join()
        print("Stopped. Pending/incomplete audio was not saved.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="List devices without recording or loading the model")
    parser.add_argument("--device", type=int, help="Input device index; default is the OS default microphone")
    parser.add_argument("--threshold", type=float, default=0.002, help="Raw RMS endpointing threshold (default: 0.002)")
    parser.add_argument("--web-url", help="Optional local dashboard, e.g. http://127.0.0.1:8000; sends text only")
    parser.add_argument("--model-dir", type=Path,
                        help="Existing base model cache root; default: LOCALAPPDATA/emotion-stt-test/models "
                             "(or ~/.cache/emotion-stt-test/models). Offline only.")
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    publisher = None
    if args.web_url:
        from web_reply import WebReplyPublisher
        try:
            publisher = WebReplyPublisher(args.web_url)
        except ValueError as exc:
            parser.error(str(exc))
    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return 0
    try:
        info = sd.query_devices(args.device, "input")
        sd.check_input_settings(device=args.device, channels=1, dtype="float32", samplerate=SAMPLE_RATE)
        print(f"Input: {info['name']} | 16000 Hz mono | threshold={args.threshold}", flush=True)
        print("Loading cached base model (CPU INT8, offline)...", flush=True)
        model = load_model(args.model_dir)
        run(sd, model, args.device, args.threshold, publisher)
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.", flush=True)
        return 130
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        print("Check cached model and microphone. Use --list-devices / --device N if needed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
