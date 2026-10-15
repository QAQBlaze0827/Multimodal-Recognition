import copy
import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from scripts import compare_local_audio_models as comparison
from src.audio.features import extract_light_mfcc_like, highpass_filter
from src.config import load_config
from src.shared_types import EmotionResult, make_result


def result(values):
    return make_result("audio", dict(zip(comparison.EMOTIONS, values)))


class LocalComparisonTests(unittest.TestCase):
    def test_smoothing_matches_production_recurrence_and_resets_on_silence(self):
        a, b = result([1, 0, 0, 0]), result([0, 1, 0, 0])
        trace = comparison.smooth_results([a, b, None, b], 0.7)
        self.assertEqual(trace[0].scores, a.scores)
        np.testing.assert_allclose(list(trace[1].scores.values()), [0.7, 0.3, 0, 0])
        self.assertIsNone(trace[2])
        self.assertEqual(trace[3].scores, b.scores)
        self.assertEqual(comparison.smooth_results([a, None, b], 0), [a, None, b])
        self.assertEqual(comparison.smooth_results([b], 0.7)[0].scores, b.scores)

    def test_existing_zero_confidence_fallback_behavior_is_not_silently_fixed(self):
        fallback = EmotionResult.neutral("audio")
        self.assertIsNone(comparison.valid_scores(fallback))
        smoothed = comparison.smooth_results([fallback], 0.7)[0]
        self.assertEqual(smoothed.label, "neutral")
        self.assertEqual(smoothed.confidence, 1.0)

    def test_aggregation_gives_recordings_equal_weight_and_keeps_abstention(self):
        traces = [[result([1, 0, 0, 0])] * 8, [result([0, 1, 0, 0])], [None, None]]
        metrics, predictions = comparison.summarize(traces, ["neutral", "happy", "sad"])
        self.assertEqual(metrics["intended_label_agreement"], 2 / 3)
        self.assertEqual(metrics["coverage"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"][2][4], 1)
        self.assertIsNone(predictions[-1]["prediction"])
        self.assertNotIn("accuracy", metrics)

    def test_feature_pipeline_matches_live_operations_and_does_not_change_input(self):
        cfg = load_config()["audio"]
        x = (0.1 * np.sin(np.arange(24000) * 0.1)).astype(np.float32)
        before = x.copy()
        features, rms = comparison.live_features(x, cfg)
        y = highpass_filter(x / np.max(np.abs(x)), 16000, cfg["highpass_cutoff"])
        expected = extract_light_mfcc_like(y, sample_rate=16000, **cfg["mfcc"])
        expected = np.pad(expected, ((0, 0), (0, 93 - expected.shape[1])))
        np.testing.assert_array_equal(features, expected)
        np.testing.assert_array_equal(x, before)
        self.assertGreater(rms, cfg["vad_threshold"])
        self.assertIsNone(comparison.live_features(np.zeros(24000, np.float32), cfg)[0])

    def test_config_mismatch_and_invalid_weights_are_rejected(self):
        cfg = load_config()["audio"]
        comparison.check_config(cfg)
        for key, value in (("sample_rate", 48000), ("window_seconds", 2), ("hop_seconds", 1)):
            wrong = copy.deepcopy(cfg)
            wrong[key] = value
            with self.assertRaises(ValueError):
                comparison.check_config(wrong)
        wrong = copy.deepcopy(cfg)
        wrong["model"]["calibration"]["anger_logit_scale"] = float("nan")
        with self.assertRaises(ValueError):
            comparison.check_config(wrong)

    def fixture(self):
        root = Path("unused_local_audio_fixture").resolve()
        stream = io.BytesIO()
        sf.write(stream, np.ones((160, 1), np.float32) * 0.1, 16000, format="WAV", subtype="FLOAT")
        raw = stream.getvalue()
        row = {"file": "neutral_01.wav", "expected_label": "neutral", "frames": 160,
               "sample_rate": 16000, "channels": 1, "duration_seconds": 0.01,
               "wav_sha256": hashlib.sha256(raw).hexdigest()}
        metadata = {"summary.json": {"status": "complete", "recordings": [row]},
                    "session.json": {"sample_rate": 16000, "channels": 1}, "neutral_01.json": copy.deepcopy(row)}
        return root, raw, row, metadata

    def test_load_checks_hash_sidecar_and_format_without_writing(self):
        root, raw, row, metadata = self.fixture()
        with patch.object(comparison, "read_json", side_effect=lambda p: metadata[p.name]), \
                patch.object(Path, "read_bytes", return_value=raw), \
                patch.object(Path, "glob", return_value=[root / row["file"]]):
            samples, hashes = comparison.load_samples(root)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][1].shape, (160,))
        self.assertEqual(len(hashes), 4)

    def test_changed_hash_sidecar_or_unsafe_path_is_rejected(self):
        for issue in ("hash", "sidecar", "path", "duplicate", "incomplete"):
            root, raw, row, metadata = self.fixture()
            if issue == "hash":
                row["wav_sha256"] = "changed"
            elif issue == "sidecar":
                metadata["neutral_01.json"]["expected_label"] = "sad"
            elif issue == "path":
                row["file"] = "../neutral_01.wav"
            elif issue == "duplicate":
                metadata["summary.json"]["recordings"].append(row)
            else:
                metadata["summary.json"]["status"] = "incomplete"
            with patch.object(comparison, "read_json", side_effect=lambda p: metadata[p.name]), \
                    patch.object(Path, "read_bytes", return_value=raw), \
                    patch.object(Path, "glob", return_value=[root / row["file"]]), \
                    self.subTest(issue=issue), self.assertRaises(ValueError):
                comparison.load_samples(root)

    def test_model_load_failure_never_becomes_heuristic_comparison(self):
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(comparison, "AudioEmotionModel", return_value=Mock(session=None)), \
                self.assertRaisesRegex(ValueError, "heuristic"):
            comparison.load_models()


if __name__ == "__main__":
    unittest.main()
