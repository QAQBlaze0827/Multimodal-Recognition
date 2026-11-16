import contextlib
import http.client
import io
import threading
import time
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np

from receive_audio import AudioServer, RemoteAudio
from src.audio.remote import AudioSender

TOKEN = "b" * 64


def pcm(rate=16000, loud=False, frequency=440):
    wave = .1 * np.sin(2 * np.pi * frequency * np.arange(rate // 2) / rate) if loud else np.zeros(rate // 2)
    return (wave * 32768).astype("<i2").tobytes()


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.audio = RemoteAudio()
        self.session = str(uuid4())
        self.sequence = 0

    def feed(self, loud=False, rate=16000):
        self.sequence += 1
        return self.audio.accept(self.session, self.sequence, rate, pcm(rate, loud))

    def phrase(self, rate=16000):
        self.feed(True, rate)
        for _ in range(3):
            self.feed(False, rate)

    def test_silence_produces_no_phrase(self):
        for _ in range(12):
            self.feed()
        self.assertTrue(self.audio.phrases.empty())

    def test_both_rates_resample_and_segment(self):
        for rate in (16000, 48000):
            with self.subTest(rate=rate):
                self.audio = RemoteAudio()
                self.phrase(rate)
                self.assertEqual(self.audio.phrases.qsize(), 1)
                _, _, audio, _ = self.audio.phrases.get_nowait()
                self.assertEqual(audio.dtype, np.float32)
                self.assertGreaterEqual(audio.size, 16000)
                self.assertLessEqual(audio.size, 24000)

    def test_resampling_filters_aliasing(self):
        def energy(frequency):
            audio = RemoteAudio()
            collected = []
            audio.segmenter.feed = lambda block: collected.append(block.copy())
            for n in range(1, 3):
                audio.accept(self.session, n, 48000, pcm(48000, True, frequency))
            return np.sqrt(np.mean(np.concatenate(collected)[1600:] ** 2))
        self.assertGreater(energy(1000), .05)
        self.assertLess(energy(12000), .001)

    def test_gap_drops_partial_and_queued_audio(self):
        self.phrase()
        self.feed(True)
        old_generation = self.audio.generation
        self.sequence += 1
        self.feed()
        self.assertGreater(self.audio.generation, old_generation)
        self.assertTrue(self.audio.phrases.empty())
        self.assertEqual(self.audio.segmenter.voiced, 0)

    def test_rate_change_resets_partial(self):
        self.feed(True)
        old = self.audio.generation
        self.feed(False, 48000)
        self.assertGreater(self.audio.generation, old)
        self.assertEqual(self.audio.segmenter.voiced, 0)

    def test_duplicate_and_simultaneous_source_rejected(self):
        self.feed(True)
        self.assertEqual(self.audio.accept(self.session, 1, 16000, pcm()), 409)
        self.assertEqual(self.audio.accept(str(uuid4()), 1, 16000, pcm()), 409)

    def test_timeout_resets_and_allows_new_source(self):
        self.phrase()
        self.audio.last_seen = time.monotonic() - 4
        self.audio.expire()
        self.assertTrue(self.audio.phrases.empty())
        self.assertIsNone(self.audio.last_seen)
        self.assertEqual(self.audio.accept(str(uuid4()), 1, 16000, pcm()), 204)

    def test_stop_clears_pending_and_inflight_results(self):
        self.phrase()
        def transcribe(*args):
            self.audio.accept(self.session, self.sequence + 1, 16000, b"", stopped=True)
            return "我想玩", .1
        with patch("receive_audio.transcribe", side_effect=transcribe), patch("receive_audio.choose_reply") as reply, contextlib.redirect_stdout(io.StringIO()) as output:
            self.audio.process_one(Mock())
        reply.assert_not_called()
        self.assertNotIn("我想玩", output.getvalue())
        self.assertTrue(self.audio.phrases.empty())

    def test_phrase_queue_bounded_and_old_phrases_not_transcribed(self):
        for _ in range(6):
            self.phrase()
        self.assertEqual(self.audio.phrases.qsize(), 3)
        self.audio.phrases.queue[0] = (self.audio.generation, 1, np.zeros(1600), time.monotonic() - 6)
        with patch("receive_audio.transcribe") as transcribe:
            self.audio.process_one(Mock())
            transcribe.assert_not_called()

    def test_terminal_uses_existing_reply_rules(self):
        self.phrase()
        with patch("receive_audio.transcribe", return_value=("我想玩", .1)), contextlib.redirect_stdout(io.StringIO()) as output:
            self.audio.process_one(Mock())
        self.assertIn("Dog [play]", output.getvalue())
        self.assertIn("我想玩", output.getvalue())


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.audio = RemoteAudio()
        self.server = AudioServer("127.0.0.1", 0, "127.0.0.1", TOKEN, self.audio)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02})
        self.thread.start()
        self.port = self.server.server_address[1]
        self.session = str(uuid4())

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(3)
        self.server.server_close()
        self.assertFalse(self.thread.is_alive())

    def post(self, changes=None, body=None, path="/audio"):
        headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/octet-stream",
                   "X-Audio-Session": self.session, "X-Audio-Sequence": "1", "X-Audio-Rate": "16000"}
        headers.update(changes or {})
        with contextlib.closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)) as connection:
            connection.request("POST", path, body=pcm() if body is None else body, headers=headers)
            response = connection.getresponse()
            response.read()
            return response.status

    def test_auth_origin_host_peer_restrictions(self):
        self.assertEqual(self.post({"Authorization": "Bearer wrong"}), 401)
        self.assertEqual(self.post({"Origin": "http://127.0.0.1"}), 403)
        self.assertEqual(self.post({"Host": "other:8765"}), 403)
        self.server.allowed_peer = "100.64.1.1"
        self.assertEqual(self.post(), 403)
        self.assertIsNone(self.audio.last_seen)

    def test_request_format_and_size_validation(self):
        for changes, expected in (
            ({"Content-Length": "999999999"}, 413),
            ({"Content-Length": "-1"}, 413),
            ({"X-Audio-Rate": "44100"}, 400),
            ({"X-Audio-Sequence": "0"}, 400),
            ({"X-Audio-Session": "invalid"}, 400),
            ({"Content-Type": "application/json"}, 415),
            ({"Transfer-Encoding": "chunked", "Content-Length": "16000"}, 415),
        ):
            with self.subTest(changes=changes):
                self.assertEqual(self.post(changes), expected)
        self.assertEqual(self.post(path="/api/companion/events"), 404)
        self.assertIsNone(self.audio.last_seen)

    def test_real_sender_http_and_stop(self):
        sender = AudioSender(f"http://127.0.0.1:{self.port}", TOKEN, 48000)
        self.assertTrue(sender._send(pcm(48000, True)))
        for _ in range(3):
            self.assertTrue(sender._send(pcm(48000)))
        self.assertEqual(self.audio.phrases.qsize(), 1)
        with patch("receive_audio.transcribe", return_value=("我想休息一下", .1)), contextlib.redirect_stdout(io.StringIO()) as output:
            self.audio.process_one(Mock())
        self.assertIn("Dog [rest]", output.getvalue())
        self.assertTrue(sender._send(stopped=True))
        self.assertIsNone(self.audio.last_seen)

    def test_duplicate_request_rejected(self):
        self.assertEqual(self.post(), 204)
        self.assertEqual(self.post(), 409)

    def test_real_sender_bad_token_stops(self):
        sender = AudioSender(f"http://127.0.0.1:{self.port}", "c" * 64, 16000)
        self.assertFalse(sender._send(pcm()))
        self.assertTrue(sender.stop.is_set())
        self.assertIsNone(self.audio.last_seen)

    def test_mixed_loopback_and_remote_binding_rejected(self):
        with self.assertRaises(ValueError):
            AudioServer("127.0.0.1", 0, "100.64.1.1", TOKEN, RemoteAudio())


if __name__ == "__main__":
    unittest.main()
