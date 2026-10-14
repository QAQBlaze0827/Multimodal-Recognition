import io
import unittest
import wave

import numpy as np

from scripts.audit_emotiontalk_audio import duplicate_groups, inspect_wav, summarize


def wav_bytes(values, rate=16000):
    values = np.asarray(values, dtype="<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(values.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(values.tobytes())
    return buffer.getvalue()


class AudioAuditTests(unittest.TestCase):
    def test_mono_full_decode_and_peak(self):
        result = inspect_wav(wav_bytes([[0], [32767], [-32768]]))
        self.assertEqual(result["channels"], 1)
        self.assertAlmostEqual(result["duration_seconds"], 3 / 16000)
        self.assertEqual(result["peak"], 1.0)
        self.assertAlmostEqual(result["full_scale_fraction"], 2 / 3)
        self.assertFalse(result["exact_silence"])

    def test_identical_stereo_preserves_rms(self):
        result = inspect_wav(wav_bytes([[1000, 1000], [-1000, -1000]]))
        self.assertTrue(result["identical_stereo_channels"])
        self.assertFalse(result["mono_cancellation"])
        self.assertAlmostEqual(result["mono_rms"], result["left_rms"])

    def test_opposite_phase_flags_cancellation(self):
        result = inspect_wav(wav_bytes([[1000, -1000], [-1000, 1000]]))
        self.assertTrue(result["mono_cancellation"])
        self.assertAlmostEqual(result["channel_correlation"], -1)
        self.assertFalse(result["exact_silence"])

    def test_one_silent_channel_flags_imbalance(self):
        result = inspect_wav(wav_bytes([[1000, 0], [-1000, 0]]))
        self.assertTrue(result["channel_imbalance"])
        self.assertIsNone(result["channel_correlation"])

    def test_silence_has_no_nan_correlation(self):
        result = inspect_wav(wav_bytes([[0, 0], [0, 0]]))
        self.assertTrue(result["exact_silence"])
        self.assertTrue(result["low_whole_clip_rms"])
        self.assertIsNone(result["channel_correlation"])

    def test_truncated_payload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Truncated"):
            inspect_wav(wav_bytes([[1], [2], [3]])[:-2])

    def test_hash_includes_sampling_rate(self):
        a = inspect_wav(wav_bytes([[1], [2]], rate=16000))
        b = inspect_wav(wav_bytes([[1], [2]], rate=44100))
        self.assertNotEqual(a["pcm_sha256"], b["pcm_sha256"])

    def test_cross_split_identical_pcm_blocks_summary(self):
        metrics = inspect_wav(wav_bytes([[1], [2]]))
        rows = [{**metrics, "split": split, "label": "sad", "source_wav": split + ".wav"}
                for split in ("train", "test")]
        self.assertTrue(duplicate_groups(rows)[0]["cross_split"])
        self.assertEqual(summarize(rows, [], {})["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
