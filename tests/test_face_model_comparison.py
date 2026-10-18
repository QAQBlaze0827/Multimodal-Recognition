import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from scripts import compare_face_models as compare


class FaceModelComparisonTests(unittest.TestCase):
    def test_preprocessing_uses_raw_gray_lanczos_and_does_not_mutate(self):
        crop = np.random.default_rng(42).integers(0, 256, (90, 100, 3), dtype=np.uint8)
        before = crop.copy()
        expected = np.asarray(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)).resize(
            (64, 64), Image.Resampling.LANCZOS), dtype=np.float32)[None, None]
        tensor = compare.fer_tensor(crop)
        np.testing.assert_array_equal(tensor, expected)
        np.testing.assert_array_equal(crop, before)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertGreater(tensor.max(), 1)

    def test_label_order_selects_anger_index_four_not_surprise(self):
        logits = np.array([0, 1, 2, 3, 5, 0, 0, 0], dtype=np.float32)
        before = logits.copy()
        result = compare.fer_scores(logits)
        self.assertEqual(result["full_label"], "anger")
        self.assertEqual(result["four_label"], "anger")
        self.assertEqual(tuple(result["four_scores"]), compare.FOUR)
        np.testing.assert_array_equal(logits, before)

    def test_excluded_emotion_is_preserved_not_merged(self):
        result = compare.fer_scores([0, 1, 10, 0, 0, 0, 0, 0])
        self.assertEqual(result["full_label"], "surprise")
        self.assertEqual(result["four_label"], "happy")
        self.assertGreater(result["discarded_mass"], 0.99)
        self.assertLess(result["four_scores"]["happy"], 0.5)

    def test_stable_softmax_and_zero_retained_abstention(self):
        a = compare.fer_scores([10000, 10001, 10000, 10000, 10000, 10000, 10000, 10000])
        b = compare.fer_scores([0, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(list(a["full_scores"].values()), list(b["full_scores"].values()))
        zero = compare.fer_scores([-10000, -10000, 10000, -10000, -10000, 0, 0, 0])
        self.assertIsNone(zero["four_label"])
        self.assertIsNone(zero["four_scores"])

    def test_invalid_logits_rejected(self):
        for values in ([0] * 7, [float("nan")] * 8, [float("inf")] * 8):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compare.fer_scores(values)

    def test_hash_mismatch_rejected(self):
        path = Mock()
        path.read_bytes.return_value = b"modified"
        with self.assertRaises(ValueError):
            compare.require_hash(path, compare.FER_SHA256)

    def test_unsafe_paths_rejected_before_reading(self):
        for name in ("../other.png", "..\\other.png", "C:/other.png", "/other.png"):
            with self.subTest(name=name), patch.object(compare, "require_hash") as reader:
                with self.assertRaises(ValueError):
                    compare.read_image(Path("session"), {"crop_file": name}, "crop")
                reader.assert_not_called()

    def fixture(self):
        crop = np.full((100, 100, 3), 100, np.uint8)
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (48, 48))
        records = [{"index": i, "expected_label": label, "crop_file": f"{i}_crop.png", "gray_file": f"{i}_gray.png"}
                   for i, label in enumerate(compare.TARGETS, 1)]
        summary = {"status": "complete", "error": None, "saved_count": 12, "expected_count": 12,
                   "model_sha256": compare.MODEL_SHA256, "label_order": compare.LABELS, "samples": records}
        return summary, crop, gray

    def load_fixture(self, summary, crop, gray, sidecar=None):
        with patch.object(Path, "read_bytes", return_value=json.dumps(summary).encode()), \
                patch.object(Path, "read_text", side_effect=[json.dumps(sidecar or r) for r in summary["samples"]]), \
                patch.object(compare, "read_image", side_effect=lambda folder, item, kind: crop if kind == "crop" else gray):
            return compare.load_samples(Path("session"))

    def test_complete_session_loaded_with_summary_hash(self):
        summary, crop, gray = self.fixture()
        records, sha = self.load_fixture(summary, crop, gray)
        self.assertEqual(len(records), 12)
        self.assertEqual(sha, compare.digest(json.dumps(summary).encode()))

    def test_incomplete_or_wrong_order_session_rejected(self):
        summary, crop, gray = self.fixture()
        summary["status"] = "incomplete"
        with self.assertRaises(ValueError):
            self.load_fixture(summary, crop, gray)
        summary["status"] = "complete"
        summary["samples"][0]["expected_label"] = "anger"
        with self.assertRaises(ValueError):
            self.load_fixture(summary, crop, gray)

    def test_sidecar_and_grayscale_mismatch_rejected(self):
        summary, crop, gray = self.fixture()
        with self.assertRaises(ValueError):
            self.load_fixture(summary, crop, gray, sidecar={"wrong": True})
        with self.assertRaises(ValueError):
            self.load_fixture(summary, crop, np.zeros_like(gray))

    def test_duplicate_photo_rejected(self):
        summary, crop, gray = self.fixture()
        summary["samples"][1]["crop_file"] = summary["samples"][0]["crop_file"]
        with self.assertRaises(ValueError):
            self.load_fixture(summary, crop, gray)

    def test_existing_output_rejected_before_model_or_photo_reads(self):
        with patch.object(Path, "exists", return_value=True), patch.object(compare, "load_samples") as reader:
            with self.assertRaises(FileExistsError):
                compare.compare(Path("session"), Path("old_report"))
            reader.assert_not_called()

    def test_metrics_count_full_errors_and_abstentions_in_denominator(self):
        rows = [{"expected_label": "happy", "ferplus": {"full_label": "surprise", "four_label": "happy"}},
                {"expected_label": "happy", "ferplus": {"full_label": "fear", "four_label": None}}]
        result = compare.metrics(rows, "ferplus")
        self.assertEqual(result["full"]["happy"], {"matched": 0, "total": 2})
        self.assertEqual(result["four"]["happy"], {"matched": 1, "total": 2})

    def test_legacy_result_must_reproduce_capture(self):
        session = Mock()
        session.run.return_value = [np.array([[0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]], np.float32)]
        gray = np.zeros((48, 48), np.uint8)
        saved = compare.predict_scores(session, "input", gray)
        result = compare.legacy_scores(session, "input", gray, saved)
        self.assertEqual(result["four_label"], "neutral")
        changed = copy.deepcopy(saved)
        changed["seven_scores"]["neutral"] = 0.9
        with self.assertRaises(ValueError):
            compare.legacy_scores(session, "input", gray, changed)


if __name__ == "__main__":
    unittest.main()
