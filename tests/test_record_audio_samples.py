import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from scripts import record_audio_samples as recorder


class RecordingTests(unittest.TestCase):
    def test_defaults_are_twelve_five_second_mono_takes(self):
        args = recorder.parser().parse_args([])
        self.assertEqual((args.takes, args.seconds, args.sample_rate, args.label), (3, 5, 16000, None))
        self.assertEqual(recorder.LABELS, ("neutral", "happy", "sad", "anger"))

    def test_float_wav_round_trip_preserves_samples_without_normalization(self):
        audio = np.array([0, 0.0001, -0.02, 0.5, 1.2, -1.1], dtype=np.float32)[:, None]
        before = audio.copy()
        payload, metrics = recorder.wav_payload(sf, audio, 48000)
        decoded, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        np.testing.assert_array_equal(decoded, before)
        np.testing.assert_array_equal(audio, before)
        self.assertEqual(rate, 48000)
        self.assertEqual(metrics["wav_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertAlmostEqual(metrics["full_scale_fraction"], 2 / 6)
        self.assertAlmostEqual(metrics["duration_seconds"], 6 / 48000)

    def test_silence_is_preserved_not_relabelled_or_rejected(self):
        payload, metrics = recorder.wav_payload(sf, np.zeros((160, 1), np.float32), 16000)
        self.assertTrue(payload)
        self.assertTrue(metrics["all_zero"])
        self.assertEqual(metrics["rms"], 0)

    def test_invalid_audio_is_rejected(self):
        for audio in (np.zeros((0, 1), np.float32), np.zeros((2, 2), np.float32),
                      np.zeros(2, np.float32), np.zeros((2, 1), np.int16),
                      np.array([[np.nan]], np.float32), np.array([[np.inf]], np.float32)):
            with self.subTest(shape=audio.shape), self.assertRaises(ValueError):
                recorder.wav_payload(sf, audio, 16000)

    def test_capture_uses_selected_device_and_waits_before_returning(self):
        audio = np.zeros((20, 1), np.float32)
        sd = Mock()
        sd.rec.return_value = audio
        sd.wait.return_value = None
        self.assertIs(recorder.capture(sd, 20, 48000, 4), audio)
        sd.rec.assert_called_once_with(20, samplerate=48000, channels=1, device=4,
                                       dtype="float32", blocking=False)
        sd.wait.assert_called_once()
        sd.stop.assert_called_once()

    def test_overflow_or_interrupt_stops_capture(self):
        for interrupted in (False, True):
            sd = Mock()
            sd.wait.return_value = "input overflow"
            if interrupted:
                sd.wait.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt if interrupted else RuntimeError):
                recorder.capture(sd, 20, 16000, 0)
            sd.stop.assert_called_once()

    def test_wrong_capture_length_is_rejected(self):
        sd = Mock()
        sd.rec.return_value = np.zeros((10, 1), np.float32)
        sd.wait.return_value = None
        with self.assertRaises(ValueError):
            recorder.capture(sd, 20, 16000, 0)

    def test_existing_wav_is_never_overwritten(self):
        with patch.object(Path, "open", side_effect=FileExistsError) as opening, \
                patch.object(recorder, "write_json") as metadata:
            with self.assertRaises(FileExistsError):
                recorder.save_take(Path("unused"), "neutral_01", sf,
                                   np.zeros((20, 1), np.float32), 16000, "neutral", "now")
        opening.assert_called_once_with("xb")
        metadata.assert_not_called()

    def test_session_prompts_each_take_and_writes_completion_last(self):
        args = recorder.parser().parse_args(["--label", "sad", "--takes", "2", "--device", "3"])
        sd = Mock()
        sd.query_devices.return_value = {"name": "Fake headset", "hostapi": 0}
        sd.query_hostapis.return_value = {"name": "Fake host"}
        prompt = Mock()
        record = {"file": "sad.wav", "rms": 0.1, "peak": 0.2, "all_zero": False, "full_scale_fraction": 0}
        with patch.object(Path, "mkdir") as mkdir, patch.object(recorder, "write_json") as writing, \
                patch.object(recorder, "capture") as capture, \
                patch.object(recorder, "save_take", return_value=record) as save, patch("builtins.print"):
            output = recorder.record_session(args, sd, sf, prompt)
        mkdir.assert_called_once_with(parents=True, exist_ok=False)
        self.assertTrue(output.is_relative_to(recorder.ROOT / "data/datasets/local_audio"))
        self.assertEqual(prompt.call_count, 2)
        self.assertEqual(capture.call_count, 2)
        self.assertEqual([call.args[1] for call in save.call_args_list], ["sad_01", "sad_02"])
        self.assertEqual(writing.call_args_list[-1].args[0].name, "summary.json")
        self.assertEqual(writing.call_args_list[-1].args[1]["status"], "complete")
        sd.check_input_settings.assert_called_once_with(device=3, channels=1, dtype="float32", samplerate=16000)

    def test_interrupt_keeps_completed_take_without_success_summary(self):
        args = recorder.parser().parse_args(["--label", "neutral", "--takes", "2", "--device", "0"])
        sd = Mock()
        sd.query_devices.return_value = {"name": "Fake", "hostapi": 0}
        sd.query_hostapis.return_value = {"name": "Fake"}
        record = {"file": "neutral_01.wav", "rms": 0, "peak": 0, "all_zero": True, "full_scale_fraction": 0}
        with patch.object(Path, "mkdir"), patch.object(recorder, "write_json") as writing, \
                patch.object(recorder, "capture"), patch.object(recorder, "save_take", return_value=record) as save, \
                patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                recorder.record_session(args, sd, sf, Mock(side_effect=["", KeyboardInterrupt]))
        self.assertEqual(save.call_count, 1)
        self.assertEqual([call.args[0].name for call in writing.call_args_list], ["session.json"])

    def test_unsupported_device_fails_before_creating_output(self):
        args = recorder.parser().parse_args(["--device", "0"])
        sd = Mock()
        sd.check_input_settings.side_effect = ValueError("Invalid sample rate")
        with patch.object(Path, "mkdir") as mkdir, self.assertRaises(ValueError):
            recorder.record_session(args, sd, sf)
        mkdir.assert_not_called()
        sd.rec.assert_not_called()

    def test_invalid_cli_arguments_do_not_start_recording(self):
        for arguments in (["--seconds", "nan"], ["--takes", "0"], ["--device", "-1"],
                          ["--seconds", "31"], ["--sample-rate", "0"]):
            with patch.object(recorder, "record_session") as session, patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    recorder.main(arguments)
            session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
