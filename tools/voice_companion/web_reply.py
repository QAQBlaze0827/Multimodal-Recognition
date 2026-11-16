"""Optional loopback-only text sender; no audio, uploads or third-party packages."""

import json
import queue
import threading
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4


def validate_web_url(value):
    url = urlsplit(value)
    if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
            or url.username is not None or url.password is not None
            or url.path not in {"", "/"} or url.query or url.fragment
            or url.port is None or not 1 <= url.port <= 65535):
        raise ValueError("--web-url must be a local URL such as http://127.0.0.1:8000")
    return value.rstrip("/")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class WebReplyPublisher:
    def __init__(self, url):
        self.url = validate_web_url(url) + "/api/companion/events"
        self.session_id = str(uuid4())
        self.sequence = 0
        self.pending = queue.Queue(maxsize=3)
        self.stop = threading.Event()
        self.thread = None
        self.online = None
        # Do not route private text through environment proxies or redirects.
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def start(self):
        self.thread = threading.Thread(target=self._run, name="local-web-reply")
        self.thread.start()

    def publish(self, text, intent, reply):
        if self.stop.is_set():
            return
        if len(text) > 2000 or len(reply) > 500:
            print("[web] Text too long; kept in terminal only.", flush=True)
            return
        try:
            self.pending.put_nowait((time.monotonic(), text, intent, reply))
        except queue.Full:
            print("[web] Busy: web reply skipped; terminal output is unchanged.", flush=True)

    def _send(self, state="listening", message=None):
        self.sequence += 1
        data = {"session_id": self.session_id, "sequence": self.sequence, "state": state}
        if message:
            data.update(zip(("text", "intent", "reply"), message))
        request = Request(self.url, data=json.dumps(data).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=1) as response:
                result = json.loads(response.read(1024))
                if response.status != 200 or result.get("accepted") is not True:
                    raise ValueError("Backend did not accept event")
            if self.online is not True and state != "stopped":
                print("[web] Connected to local dashboard.", flush=True)
            self.online = True
        except Exception as exc:
            if self.online is not False:
                print(f"[web] Delivery unavailable ({type(exc).__name__}); terminal continues. "
                      "Check local backend and --companion-local.", flush=True)
            if message:
                print("[web] This reply was not confirmed; it will not be retried.", flush=True)
            self.online = False

    def _run(self):
        try:
            self._send()
            while not self.stop.is_set():
                try:
                    queued_at, text, intent, reply = self.pending.get(timeout=2)
                except queue.Empty:
                    if not self.stop.is_set():
                        self._send()
                    continue
                if self.stop.is_set():
                    break
                if time.monotonic() - queued_at > 5:
                    print("[web] Expired reply skipped.", flush=True)
                    continue
                self._send(message=(text, intent, reply))
        finally:
            self._send(state="stopped")

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        while True:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                break
