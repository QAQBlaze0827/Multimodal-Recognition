import contextlib
import io
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.audio.inference import AudioEmotionThread
from src.audio.remote import AudioSender, checked_address, read_token
from src.main import build_parser, main

TOKEN = "a" * 64


class SenderTests(unittest.TestCase):
    def make_sender(self, rate=16000):
        return AudioSender("http://127.0.0.1:8765", TOKEN, rate)

    def test_addresses_and_urls_are_restricted(self):
        self.assertEqual(checked_address("100.74.11.81"), "100.74.11.81")
        for address in ("0.0.0.0", "192.168.1.5", "8.8.8.8", "::1", "localhost"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                checked_address(address)
        for url in ("https://127.0.0.1:8765", "http://127.0.0.1", "http://user@127.0.0.1:8765",
                    "http://127.0.0.1:8765/audio", "http://127.0.0.1:8765?token=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                AudioSender(url, TOKEN, 16000)
        with self.assertRaises(ValueError):
            self.make_sender(44100)

    def test_token_is_required_and_not_echoed(self):
        with patch.dict(os.environ, {"COMPANION_AUDIO_TOKEN": "private-secret"}):
            with self.assertRaises(ValueError) as error:
                read_token()
            self.assertNotIn("private-secret", str(error.exception))
        with patch.dict(os.environ, {"COMPANION_AUDIO_TOKEN": TOKEN}):
            self.assertEqual(read_token(), TOKEN)

    def test_offer_is_bounded_copies_and_skips_bad_blocks(self):
        sender = self.make_sender()
        block = np.ones(1600, dtype=np.float32)
        sender.offer(block)
        block[:] = 0
        self.assertEqual(sender.pending.queue[0][2][0], 1)
        sender.offer(block, status=True)
        sender.offer(block[:100])
        for _ in range(20):
            sender.offer(block)
        self.assertEqual(sender.pending.qsize(), 10)
        self.assertEqual(sender.capture_sequence, 23)
        sender.close()
        self.assertTrue(sender.pending.empty())

    def run_packets(self, entries):
        sender = self.make_sender()
        for timestamp, number, value in entries:
            sender.pending.put((timestamp, number, np.full(1600, value, dtype=np.float32)))
        packets = []

        def send(payload=b"", stopped=False):
            if not stopped:
                packets.append((sender.sequence, np.frombuffer(payload, dtype="<i2")))
                sender.stop.set()
            return True

        sender._send = send
        with patch("src.audio.remote.time.monotonic", return_value=10):
            sender._run()
        return packets

    def test_pcm_clipping_and_half_second_packets(self):
        packets = self.run_packets([(10, n, 2 if n < 3 else -2) for n in range(1, 6)])
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0][1].size, 8000)
        self.assertEqual(packets[0][1][0], 32767)
        self.assertEqual(packets[0][1][-1], -32768)

    def test_gaps_and_stale_audio_discard_partial_packets(self):
        for entries in (
            [(10, 1, .9)] + [(10, n, .1) for n in range(3, 8)],
            [(9, 1, .9)] + [(10, n, .1) for n in range(2, 7)],
            [(8, 1, .9)] + [(10, n, .1) for n in range(2, 7)],
            [(10, 1, float("nan"))] + [(10, n, .1) for n in range(2, 7)],
        ):
            with self.subTest(entries=entries):
                packets = self.run_packets(entries)
                self.assertGreater(packets[0][0], 0)
                self.assertTrue(np.all(packets[0][1] == 3276))

    def test_http_auth_failure_stops_sender_without_raising(self):
        sender = self.make_sender()
        connection = Mock()
        connection.getresponse.return_value.status = 401
        with patch("src.audio.remote.http.client.HTTPConnection", return_value=connection), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(sender._send(bytes(16000)))
        self.assertTrue(sender.stop.is_set())
        self.assertNotIn(TOKEN, output.getvalue())
        connection.close.assert_called_once()

    def test_network_failure_is_not_retried_and_thread_closes(self):
        sender = self.make_sender()
        with patch("src.audio.remote.http.client.HTTPConnection") as factory:
            factory.return_value.request.side_effect = OSError("test only")
            self.assertFalse(sender._send(bytes(16000)))
            self.assertFalse(sender.stop.is_set())
            factory.return_value.request.assert_called_once()
        sender.start()
        sender.close()
        self.assertFalse(sender.thread.is_alive())

    def test_cli_default_and_invalid_combination_before_hardware(self):
        self.assertIsNone(build_parser().parse_args([]).stt_url)
        with patch("sys.argv", ["app.py", "--vision-only", "--stt-url", "http://127.0.0.1:8765"]), patch("src.main.load_config", return_value={"audio": {"enabled": True}}), patch("src.main.open_camera") as camera, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()
            camera.assert_not_called()


class CaptureTests(unittest.TestCase):
    def run_capture(self, sender, fail=False):
        state = SimpleNamespace(stop=True, audio=None, lock=threading.Lock())
        config = {"sample_rate": 16000, "input_sample_rate": 48000,
                  "window_seconds": 1.5, "channels": 2, "channel_select": "left", "device_id": 0}
        sd = Mock()
        stream = sd.InputStream.return_value
        if fail:
            stream.start.side_effect = RuntimeError("microphone failed")
        else:
            def start():
                callback = sd.InputStream.call_args.kwargs["callback"]
                callback(np.column_stack((np.full(4800, .125), np.full(4800, -.99))), 4800, None, False)
            stream.start.side_effect = start
        worker = AudioEmotionThread(state, config, Mock(), sender=sender)
        with patch.dict("sys.modules", {"sounddevice": sd}):
            if fail:
                with self.assertRaises(RuntimeError):
                    worker.run()
            else:
                worker.run()
        stream.close.assert_called_once()
        sd.InputStream.assert_called_once()
        return sd, worker

    def test_single_reader_only_left_channel_raw_audio(self):
        sender = Mock(block_size=4800)
        sd, worker = self.run_capture(sender)
        self.assertEqual(sd.InputStream.call_args.kwargs["blocksize"], 4800)
        self.assertEqual(sd.InputStream.call_args.kwargs["device"], 0)
        np.testing.assert_array_equal(sender.offer.call_args.args[0], np.full(4800, .125))
        self.assertEqual(worker._ring_buffer.size, 4800)
        sender.start.assert_called_once()
        sender.close.assert_called_once()

    def test_disabled_preserves_default_stream_blocksize(self):
        sd, _ = self.run_capture(None)
        self.assertNotIn("blocksize", sd.InputStream.call_args.kwargs)

    def test_failed_microphone_never_starts_network(self):
        sender = Mock(block_size=4800)
        self.run_capture(sender, fail=True)
        sender.start.assert_not_called()
        sender.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
