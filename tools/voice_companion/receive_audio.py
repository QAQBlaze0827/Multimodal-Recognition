"""Receive Pi PCM over a restricted private link; transcribe to the terminal only."""

from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import queue
import sys
import threading
import time
from uuid import UUID

import av
import numpy as np

# This standalone laptop tool needs only the transport helper, not the emotion models.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.audio.remote import RATES, checked_address, read_token
from dog_reply import choose_reply
from transcribe_mic import BLOCK_SIZE, SAMPLE_RATE, PhraseBuffer, load_model, transcribe

SOURCE_TTL = 3.0
PHRASE_TTL = 5.0


class RemoteAudio:
    def __init__(self, threshold=0.002):
        self.lock = threading.Lock()
        self.phrases = queue.Queue(maxsize=3)
        self.segmenter = PhraseBuffer(threshold)
        self.session_id = None
        self.sequence = 0
        self.rate = None
        self.last_seen = None
        self.generation = 0
        self.number = 0
        self._reset()

    def _reset(self):
        self.generation += 1
        self.segmenter.reset()
        self.resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        self.remainder = np.zeros(0, dtype=np.float32)
        while True:
            try:
                self.phrases.get_nowait()
            except queue.Empty:
                break

    def _expire(self, now):
        if self.last_seen is not None and now - self.last_seen >= SOURCE_TTL:
            self.last_seen = None
            self._reset()
            print("[remote] Source timed out; partial/queued speech discarded.", flush=True)

    def expire(self):
        with self.lock:
            self._expire(time.monotonic())

    def accept(self, session_id, sequence, rate, payload, stopped=False):
        now = time.monotonic()
        with self.lock:
            self._expire(now)
            if session_id == self.session_id and sequence <= self.sequence:
                return 409
            if session_id != self.session_id:
                if self.last_seen is not None or stopped:
                    return 409
                self._reset()
                self.session_id, self.sequence, self.rate = session_id, 0, rate
                print(f"[remote] Source connected: {rate} Hz mono; resampling to 16000 Hz on laptop.", flush=True)
            if sequence != self.sequence + 1 or rate != self.rate:
                self._reset()
            self.sequence, self.rate = sequence, rate
            if stopped:
                self.last_seen = None
                self._reset()
                print("[remote] Source stopped; partial/queued speech discarded.", flush=True)
                return 204
            self.last_seen = now
            samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768
            frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
            frame.sample_rate = rate
            for converted in self.resampler.resample(frame):
                self.remainder = np.concatenate((self.remainder, converted.to_ndarray().reshape(-1)))
                while self.remainder.size >= BLOCK_SIZE:
                    block, self.remainder = self.remainder[:BLOCK_SIZE], self.remainder[BLOCK_SIZE:]
                    audio = self.segmenter.feed(block)
                    if audio is not None:
                        self.number += 1
                        try:
                            self.phrases.put_nowait((self.generation, self.number, audio, now))
                            print(f"[remote] Phrase {self.number:03d} queued.", flush=True)
                        except queue.Full:
                            print("[remote] Transcription busy; new phrase skipped.", flush=True)
            return 204

    def process_one(self, model):
        try:
            generation, number, audio, queued_at = self.phrases.get(timeout=0.2)
        except queue.Empty:
            return
        with self.lock:
            self._expire(time.monotonic())
            if generation != self.generation or time.monotonic() - queued_at > PHRASE_TTL:
                return
        text, elapsed = transcribe(model, audio)
        with self.lock:
            self._expire(time.monotonic())
            if generation != self.generation or time.monotonic() - queued_at > PHRASE_TTL:
                print("[remote] Stale transcription discarded.", flush=True)
                return
            print(f"\n[{number:03d}] {text or '(No speech recognized)'}", flush=True)
            print(f"  audio={audio.size / SAMPLE_RATE:.2f}s | STT={elapsed:.2f}s", flush=True)
            reply = choose_reply(text)
            if reply is not None:
                print(f"  Dog [{reply[0]}]: {reply[1]}", flush=True)

    def close(self):
        with self.lock:
            self.last_seen = None
            self._reset()


class AudioHandler(BaseHTTPRequestHandler):
    server_version = "PrivateAudioReceiver"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(2)

    def log_message(self, *args):
        pass  # Never echo headers, tokens, raw audio or arbitrary request paths.

    def respond(self, status):
        # Bounded drain avoids a TCP reset hiding rejection status on Windows.
        if status >= 400 and not getattr(self, "body_consumed", False):
            try:
                length = min(48000, max(0, int(self.headers.get("Content-Length", "0"))))
                self.connection.settimeout(0.1)
                self.rfile.read(length)
            except (ValueError, OSError):
                pass
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def do_POST(self):
        try:
            if self.client_address[0] != self.server.allowed_peer or "Origin" in self.headers:
                return self.respond(403)
            fields = ("Host", "Authorization", "Content-Length", "Content-Type",
                      "X-Audio-Session", "X-Audio-Sequence", "X-Audio-Rate")
            if any(len(self.headers.get_all(field, [])) != 1 for field in fields):
                return self.respond(400)
            host, port = self.server.server_address
            if self.headers["Host"] != f"{host}:{port}":
                return self.respond(403)
            if not hmac.compare_digest(self.headers["Authorization"].encode("utf-8"),
                                       f"Bearer {self.server.token}".encode("ascii")):
                return self.respond(401)
            if self.path not in ("/audio", "/stop"):
                return self.respond(404)
            if ("Transfer-Encoding" in self.headers or
                    self.headers["Content-Type"] != "application/octet-stream"):
                return self.respond(415)
            length = int(self.headers["Content-Length"])
            rate = int(self.headers["X-Audio-Rate"])
            sequence = int(self.headers["X-Audio-Sequence"])
            session_id = str(UUID(self.headers["X-Audio-Session"]))
            stopped = self.path == "/stop"
            # Exactly half a second of mono signed 16-bit little-endian PCM.
            if rate not in RATES or not 1 <= sequence < 2**63:
                return self.respond(400)
            if length != (0 if stopped else rate):
                return self.respond(413)
            payload = self.rfile.read(length)
            self.body_consumed = True
            if len(payload) != length:
                return self.respond(400)
            self.respond(self.server.audio.accept(session_id, sequence, rate, payload, stopped))
        except (ValueError, OverflowError):
            self.respond(400)
        except (OSError, TimeoutError):
            self.close_connection = True
        except Exception as exc:
            self.server.audio.close()
            print(f"[remote] Invalid stream discarded ({type(exc).__name__}).", flush=True)
            self.respond(400)


class AudioServer(HTTPServer):
    def __init__(self, host, port, peer, token, audio):
        host, peer = checked_address(host), checked_address(peer)
        if (host == "127.0.0.1") != (peer == "127.0.0.1"):
            raise ValueError("Use loopback on both sides for tests, or Tailscale addresses on both sides")
        self.allowed_peer, self.token, self.audio = peer, token, audio
        super().__init__((host, port), AudioHandler)

    def service_actions(self):
        self.audio.expire()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Laptop Tailscale IPv4; loopback by default")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-peer", required=True, help="Only this Pi Tailscale IPv4 may send audio")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.002)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 0 < args.threshold < 1:
        parser.error("Invalid port or RMS threshold")
    try:
        host, peer, token = checked_address(args.host), checked_address(args.allow_peer), read_token()
        audio = RemoteAudio(args.threshold)
        with AudioServer(host, args.port, peer, token, audio) as server:
            model = load_model(args.model_dir)
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
            thread.start()
            print(f"[remote] Listening on {host}:{args.port}; allowed source: {peer}.", flush=True)
            print("[remote] Terminal replies only. No laptop microphone, recording files or web delivery.", flush=True)
            try:
                while thread.is_alive():
                    audio.process_one(model)
            finally:
                audio.close()
                server.shutdown()
                thread.join(timeout=3)
        return 0
    except KeyboardInterrupt:
        print("\n[remote] Stopped. Pending audio was not saved.", flush=True)
        return 0
    except Exception as exc:
        print(f"[remote] Cannot run: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
