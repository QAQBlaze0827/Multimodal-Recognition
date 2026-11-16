"""Opt-in PCM transport to a trusted laptop over Tailscale (or local tests)."""

from __future__ import annotations

import http.client
import ipaddress
import os
import queue
import re
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

TOKEN_ENV = "COMPANION_AUDIO_TOKEN"
RATES = (16000, 48000)
TAILNET = ipaddress.ip_network("100.64.0.0/10")


def checked_address(value):
    address = ipaddress.ip_address(value)
    if address.version != 4 or (str(address) != "127.0.0.1" and address not in TAILNET):
        raise ValueError("Use an explicit Tailscale IPv4 address or 127.0.0.1; not a LAN/public/wildcard address")
    return str(address)


def read_token():
    token = os.environ.get(TOKEN_ENV, "")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", token):
        raise ValueError(f"Set {TOKEN_ENV} to the same random 64-character hex token on both devices")
    return token


class AudioSender:
    def __init__(self, url, token, sample_rate):
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.username is not None or parsed.password is not None
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                or parsed.port is None or not 1 <= parsed.port <= 65535):
            raise ValueError("--stt-url must be http://TAILSCALE_IP:PORT without a path or credentials")
        self.host, self.port = checked_address(parsed.hostname or ""), parsed.port
        if not re.fullmatch(r"[0-9a-fA-F]{64}", token):
            raise ValueError("Invalid audio transport token")
        if sample_rate not in RATES:
            raise ValueError("Remote audio currently supports 16000 or 48000 Hz input only")
        self.token, self.sample_rate = token, sample_rate
        self.block_size = sample_rate // 10
        self.pending = queue.Queue(maxsize=10)
        self.stop = threading.Event()
        self.thread = None
        self.session_id = str(uuid4())
        self.sequence = 0
        self.capture_sequence = 0
        self.online = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="pi-audio-send", daemon=True)
        self.thread.start()
        print(f"[stt-send] Sending selected microphone channel to {self.host}:{self.port}; no recordings saved.", flush=True)

    def offer(self, samples, status=False):
        # Called by the existing microphone callback: never do network I/O here.
        if self.stop.is_set():
            return
        self.capture_sequence += 1
        if status or samples.size != self.block_size:
            return
        try:
            self.pending.put_nowait((time.monotonic(), self.capture_sequence, samples.copy()))
        except queue.Full:
            pass  # The skipped sequence resets partial speech at the receiver.

    def _send(self, payload=b"", stopped=False):
        self.sequence += 1
        connection = http.client.HTTPConnection(self.host, self.port, timeout=0.5)
        try:
            connection.request("POST", "/stop" if stopped else "/audio", body=payload, headers={
                "Authorization": f"Bearer {self.token}", "Content-Type": "application/octet-stream",
                "X-Audio-Session": self.session_id, "X-Audio-Sequence": str(self.sequence),
                "X-Audio-Rate": str(self.sample_rate),
            })
            response = connection.getresponse()
            if response.status != 204:
                if response.status in (401, 403):
                    self.stop.set()
                raise RuntimeError(f"HTTP {response.status}")
            if not stopped and self.online is not True:
                print("[stt-send] Laptop audio receiver connected.", flush=True)
            self.online = True
            return True
        except Exception as exc:
            if not stopped and self.online is not False:
                detail = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                print(f"[stt-send] Unavailable ({detail}); emotion recognition continues. Old audio is not retried.", flush=True)
            self.online = False
            return False
        finally:
            connection.close()

    def _run(self):
        parts, previous = [], 0
        previous_at = None
        try:
            while not self.stop.is_set():
                try:
                    captured_at, number, samples = self.pending.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self.stop.is_set():
                    break
                if number != previous + 1 or (previous_at is not None and captured_at - previous_at > 0.5):
                    parts.clear()
                    self.sequence += 1
                    print("[stt-send] Capture gap; partial audio discarded.", flush=True)
                previous = number
                previous_at = captured_at
                if time.monotonic() - captured_at > 1 or not np.isfinite(samples).all():
                    parts.clear()
                    self.sequence += 1
                    continue
                parts.append(samples)
                if len(parts) == 5:
                    mono = np.concatenate(parts)
                    parts.clear()
                    pcm = (np.clip(mono, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()
                    if not self._send(pcm):
                        self.sequence += 1
                        self.stop.wait(1)
        except Exception as exc:
            print(f"[stt-send] Transport stopped ({type(exc).__name__}); emotion recognition continues.", flush=True)
            self.stop.set()
        finally:
            if self.sequence:
                self._send(stopped=True)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        while True:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break
