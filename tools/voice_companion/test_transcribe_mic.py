"""Synthetic tests only: no real microphone, model download, or audio files."""

import os
from pathlib import Path
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import transcribe_mic as stt


def block(value=0):
    return np.full(stt.BLOCK_SIZE, value, dtype=np.float32)


class PhraseTests(unittest.TestCase):
    def test_silence_is_bounded_and_never_queued(self):
        buf = stt.PhraseBuffer()
        for _ in range(1000):
            self.assertIsNone(buf.feed(block()))
        self.assertEqual(len(buf.pre_roll), 3)
        self.assertEqual(buf.parts, [])

    def test_pause_preroll_and_no_duplicate(self):
        buf = stt.PhraseBuffer()
        for _ in range(3):
            buf.feed(block())
        for _ in range(5):
            self.assertIsNone(buf.feed(block(0.1)))
        for _ in range(7):
            self.assertIsNone(buf.feed(block()))
        result = buf.feed(block())
        self.assertEqual(result.size, 16 * stt.BLOCK_SIZE)
        self.assertEqual(result.dtype, np.float32)
        self.assertIsNone(buf.feed(block()))

    def test_click_rejected(self):
        buf = stt.PhraseBuffer()
        buf.feed(block(0.1))
        for _ in range(8):
            self.assertIsNone(buf.feed(block()))
        self.assertFalse(buf.parts)

    def test_quiet_voice_above_threshold_is_not_lost(self):
        buf = stt.PhraseBuffer()
        for _ in range(5):
            buf.feed(block(0.003))
        for _ in range(7):
            self.assertIsNone(buf.feed(block(0.0001)))
        self.assertIsNotNone(buf.feed(block(0.0001)))

    def test_continuous_sound_capped(self):
        buf = stt.PhraseBuffer()
        for _ in range(119):
            self.assertIsNone(buf.feed(block(0.1)))
        self.assertEqual(buf.feed(block(0.1)).size, 12 * stt.SAMPLE_RATE)

    def test_two_phrases_and_input_copy(self):
        buf = stt.PhraseBuffer()
        for _ in range(2):
            for _ in range(3):
                source = block(0.1)
                buf.feed(source)
                source[:] = 0
            for _ in range(7):
                buf.feed(block())
            result = buf.feed(block())
            self.assertGreater(float(result.max()), 0.09)

    def test_invalid_block(self):
        for value in (np.zeros(3), block(float("nan"))):
            with self.assertRaises(ValueError):
                stt.PhraseBuffer().feed(value)

    def test_transcription_options_and_iterator(self):
        model = Mock()
        model.transcribe.return_value = (iter([SimpleNamespace(text=" hello"), SimpleNamespace(text=" world")]), None)
        text, seconds = stt.transcribe(model, block())
        self.assertEqual(text, "hello world")
        self.assertGreaterEqual(seconds, 0)
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertFalse(model.transcribe.call_args.kwargs["condition_on_previous_text"])

    def test_worker_exception_stops(self):
        phrases, errors, stop = queue.Queue(), queue.SimpleQueue(), threading.Event()
        phrases.put((1, block(), time.perf_counter()))
        model = Mock()
        model.transcribe.side_effect = ValueError("bad model")
        stt.transcription_worker(model, phrases, stop, errors)
        self.assertTrue(stop.is_set())
        self.assertIsInstance(errors.get_nowait(), ValueError)

    def test_stream_closed_on_ctrl_c(self):
        stream = Mock(active=True)
        manager = Mock()
        manager.__enter__ = Mock(return_value=stream)
        manager.__exit__ = Mock(return_value=False)
        sd = Mock()
        sd.InputStream.return_value = manager
        with patch.object(stt.PhraseBuffer, "feed", side_effect=KeyboardInterrupt):
            def enter():
                sd.InputStream.call_args.kwargs["callback"](block()[:, None], stt.BLOCK_SIZE, None, False)
                return stream
            manager.__enter__.side_effect = enter
            stt.run(sd, Mock(), None, 0.01)
        manager.__exit__.assert_called_once()
        self.assertFalse(any(t.name == "local-stt" for t in threading.enumerate()))

    def test_device_listing_does_not_load_model_or_record(self):
        fake_sd = Mock()
        with patch.dict("sys.modules", {"sounddevice": fake_sd}), patch.object(stt, "load_model") as load:
            self.assertEqual(stt.main(["--list-devices"]), 0)
        load.assert_not_called()
        fake_sd.InputStream.assert_not_called()

    def test_capture_can_continue_while_transcription_waits(self):
        phrases, errors, stop = queue.Queue(maxsize=3), queue.SimpleQueue(), threading.Event()
        entered, release = threading.Event(), threading.Event()
        model = Mock()

        def slow_transcribe(*args, **kwargs):
            entered.set()
            release.wait(timeout=3)
            return iter([]), None

        model.transcribe.side_effect = slow_transcribe
        worker = threading.Thread(target=stt.transcription_worker, args=(model, phrases, stop, errors))
        phrases.put((1, block(), time.perf_counter()))
        worker.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            for number in range(2, 5):
                phrases.put_nowait((number, block(), time.perf_counter()))
            with self.assertRaises(queue.Full):
                phrases.put_nowait((5, block(), time.perf_counter()))
        finally:
            stop.set()
            release.set()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertTrue(errors.empty())

    def test_stream_open_failure_releases_worker(self):
        sd = Mock()
        sd.InputStream.side_effect = RuntimeError("device unavailable")
        with self.assertRaisesRegex(RuntimeError, "device unavailable"):
            stt.run(sd, Mock(), None, 0.01)
        self.assertFalse(any(t.name == "local-stt" for t in threading.enumerate()))


class ModelCacheTests(unittest.TestCase):
    def test_default_reuses_external_cache_and_stays_offline(self):
        whisper = Mock()
        root = Path.home() / "test-local-appdata"
        with patch.dict(os.environ, {"LOCALAPPDATA": str(root)}), \
                patch.dict("sys.modules", {"faster_whisper": whisper}):
            stt.load_model()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        whisper.WhisperModel.assert_called_once_with(
            "base", device="cpu", compute_type="int8",
            download_root=str((root / "emotion-stt-test" / "models").resolve()),
            local_files_only=True,
        )

    def test_explicit_cache_overrides_default(self):
        whisper = Mock()
        cache = Path.home() / "cache with spaces" / "models"
        with patch.dict(os.environ), patch.dict("sys.modules", {"faster_whisper": whisper}):
            stt.load_model(cache)
        self.assertEqual(whisper.WhisperModel.call_args.kwargs["download_root"], str(cache.resolve()))
        self.assertTrue(whisper.WhisperModel.call_args.kwargs["local_files_only"])

    def test_without_localappdata_uses_home_cache_not_script_directory(self):
        whisper = Mock()
        with patch.dict(os.environ), patch.dict("sys.modules", {"faster_whisper": whisper}):
            os.environ.pop("LOCALAPPDATA", None)
            stt.load_model()
        expected = Path.home() / ".cache" / "emotion-stt-test" / "models"
        self.assertEqual(whisper.WhisperModel.call_args.kwargs["download_root"], str(expected.resolve()))

    def test_missing_cache_does_not_retry_online(self):
        whisper = Mock()
        whisper.WhisperModel.side_effect = FileNotFoundError("Missing offline model")
        with patch.dict(os.environ), patch.dict("sys.modules", {"faster_whisper": whisper}):
            with self.assertRaisesRegex(FileNotFoundError, "Missing offline model"):
                stt.load_model(Path.home() / "missing-test-cache")
        whisper.WhisperModel.assert_called_once()
        self.assertTrue(whisper.WhisperModel.call_args.kwargs["local_files_only"])

    def test_cli_passes_cache_without_changing_capture_defaults(self):
        sd = Mock()
        sd.query_devices.return_value = {"name": "mock microphone"}
        cache = Path.home() / "cache with spaces"
        with patch.dict("sys.modules", {"sounddevice": sd}), \
                patch.object(stt, "load_model") as load, patch.object(stt, "run") as run:
            self.assertEqual(stt.main(["--model-dir", str(cache)]), 0)
        load.assert_called_once_with(cache)
        run.assert_called_once_with(sd, load.return_value, None, 0.002, None)
        sd.InputStream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
