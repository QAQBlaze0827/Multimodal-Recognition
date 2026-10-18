import copy
from contextlib import contextmanager, ExitStack
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from scripts import capture_face_samples as capture
from src.config import DEFAULT_CONFIG
from src.shared_types import FaceDetection
from src.video.model import VideoEmotionModel


class MemoryPath:
    """Exclusive in-memory output; tests never create or delete photo files."""

    def __init__(self, files=None, name="session"):
        self.files = {} if files is None else files
        self.name = name

    def __truediv__(self, name):
        return MemoryPath(self.files, self.name + "/" + name)

    def __str__(self):
        return self.name

    def mkdir(self, **kwargs):
        if self.name in self.files:
            raise FileExistsError(self.name)
        self.files[self.name] = None

    @contextmanager
    def open(self, mode, **kwargs):
        assert mode in ("x", "xb")
        if self.name in self.files:
            raise FileExistsError(self.name)
        stream = io.BytesIO() if mode == "xb" else io.StringIO()
        self.files[self.name] = None
        try:
            yield stream
        finally:
            self.files[self.name] = stream.getvalue()
            stream.close()


class FaceSampleCaptureTests(unittest.TestCase):
    def setUp(self):
        self.output = MemoryPath()
        self.frame = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        self.face = FaceDetection((40, 20, 60, 80), 0.9, "fake")
        self.detector = Mock(backend="fake", warning=None)
        self.detector.detect.return_value = self.face
        _, self.crop, self.gray = capture.inspect_frame(self.frame, self.detector, 48)
        self.values = np.array([[0.01, 0.01, 0.07, 0.01, 0.80, 0.05, 0.05]], np.float32)
        self.session = Mock()
        self.session.run.return_value = [self.values]

    def scores(self):
        return capture.predict_scores(self.session, "input", self.gray)

    def test_seven_and_four_scores_match_production_and_preserve_input(self):
        original = self.values.copy()
        scores = self.scores()
        self.assertEqual(scores["seven_label"], "fear")
        self.assertEqual(scores["four_label"], "sad")
        self.assertAlmostEqual(scores["discarded_mass"], 0.9, places=6)
        self.assertAlmostEqual(scores["four_scores"]["sad"], 0.7, places=6)
        model = VideoEmotionModel.__new__(VideoEmotionModel)
        model.input_size, model.session, model.input_name = 48, self.session, "input"
        result = model._predict_onnx(self.gray)
        np.testing.assert_allclose(list(result.scores.values()), list(scores["four_scores"].values()))
        np.testing.assert_array_equal(self.values, original)
        tensor = self.session.run.call_args.args[1]["input"]
        np.testing.assert_array_equal(tensor, self.gray.astype(np.float32)[None, None] / 255.0)

    def test_zero_retained_mass_is_not_neutral_certainty(self):
        self.session.run.return_value = [np.array([[0, 0, 0, 0, 1, 0, 0]], np.float32)]
        scores = self.scores()
        self.assertIsNone(scores["four_label"])
        self.assertIsNone(scores["four_scores"])

    def test_invalid_output_and_input_are_rejected(self):
        for values in ([0.25] * 4, [0] * 7, [float("nan")] * 7, [-1, 2, 0, 0, 0, 0, 0]):
            self.session.run.return_value = [np.array([values])]
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.scores()
        with self.assertRaises(ValueError):
            capture.predict_scores(self.session, "input", self.gray.astype(np.float32))

    def test_unknown_model_rejected_before_session_creation(self):
        with patch.object(Path, "read_bytes", return_value=b"unknown"), patch.object(capture, "create_ort_session") as create:
            with self.assertRaises(ValueError):
                capture.load_model(copy.deepcopy(DEFAULT_CONFIG["video"]))
            create.assert_not_called()

    def test_save_roundtrip_hashes_exclusivity_and_no_full_frame(self):
        before = self.crop.copy()
        record = capture.save_sample(self.output, 1, "neutral", self.frame, self.face,
                                     self.crop, self.gray, "fake", self.scores())
        for kind, expected in (("crop", self.crop), ("gray", self.gray)):
            raw = self.output.files[f"session/{record[kind + '_file']}"]
            actual = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record[kind + "_sha256"])
        self.assertEqual(len(self.output.files), 3)
        np.testing.assert_array_equal(self.crop, before)
        with self.assertRaises(FileExistsError):
            capture.save_sample(self.output, 1, "neutral", self.frame, self.face,
                                self.crop, self.gray, "fake", self.scores())

    def mocked_run(self, keys, *, faces=None, read_ok=True, headless=False, visible=1):
        cap = Mock()
        cap.read.return_value = (read_ok, self.frame if read_ok else None)
        if faces is not None:
            self.detector.detect.side_effect = faces
        with ExitStack() as stack:
            stack.enter_context(patch.object(capture, "load_model", return_value=(self.session, "input")))
            opening = stack.enter_context(patch.object(capture, "open_camera", return_value=cap))
            stack.enter_context(patch.object(capture, "FaceDetector", return_value=self.detector))
            stack.enter_context(patch.object(cv2, "namedWindow", side_effect=cv2.error("headless") if headless else None))
            stack.enter_context(patch.object(cv2, "imshow"))
            stack.enter_context(patch.object(cv2, "waitKey", side_effect=keys))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=visible))
            stack.enter_context(patch.object(cv2, "destroyWindow"))
            stack.enter_context(patch("builtins.print"))
            if headless or not read_ok:
                with self.assertRaises((cv2.error, RuntimeError)):
                    capture.run(copy.deepcopy(DEFAULT_CONFIG["video"]), self.output)
            else:
                capture.run(copy.deepcopy(DEFAULT_CONFIG["video"]), self.output, camera_id=2)
                self.assertEqual(opening.call_args.args[1], 2)
            if headless:
                opening.assert_not_called()
            else:
                cap.release.assert_called_once()
                self.detector.close.assert_called_once()
        return json.loads(self.output.files["session/summary.json"])

    def test_no_key_capture_or_window_close_saves_no_photos(self):
        summary = self.mocked_run([-1], visible=0)
        self.assertEqual(summary["saved_count"], 0)
        self.assertEqual(summary["status"], "incomplete")
        self.assertFalse(any(name.endswith(".png") for name in self.output.files))
        self.session.run.assert_not_called()

    def test_held_space_does_not_repeat_capture(self):
        summary = self.mocked_run([32, 32, 32, ord("q")])
        self.assertEqual(summary["saved_count"], 1)

    def test_no_face_does_not_save_or_advance(self):
        summary = self.mocked_run([32, 32, ord("q")], faces=[None, self.face, self.face])
        self.assertEqual(summary["saved_count"], 1)
        self.assertEqual(summary["samples"][0]["index"], 1)

    def test_complete_session_has_twelve_photos_in_expected_order(self):
        summary = self.mocked_run([32, 13] * 12)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(tuple(item["expected_label"] for item in summary["samples"]), capture.TARGETS)
        self.assertEqual(sum(name.endswith(".png") for name in self.output.files), 24)
        report = self.output.files["session/report.md"]
        self.assertIn("fear", report)
        self.assertIn("48x48", report)

    def test_camera_error_records_incomplete_report(self):
        summary = self.mocked_run([], read_ok=False)
        self.assertEqual(summary["status"], "incomplete")
        self.assertIn("RuntimeError", summary["error"])

    def test_headless_does_not_open_camera(self):
        summary = self.mocked_run([], headless=True)
        self.assertEqual(summary["saved_count"], 0)

    def test_interrupt_preserves_completed_sample_and_reports_partial(self):
        with self.assertRaises(KeyboardInterrupt):
            self.mocked_run([32, KeyboardInterrupt()])
        summary = json.loads(self.output.files["session/summary.json"])
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["saved_count"], 1)
        self.assertIn("session/01_neutral.json", self.output.files)
        self.assertIn("KeyboardInterrupt", summary["error"])

    def test_encoding_failure_keeps_partial_image_without_completion_json(self):
        ok, encoded = cv2.imencode(".png", self.crop)
        with patch.object(cv2, "imencode", side_effect=[(ok, encoded), (False, None)]):
            with self.assertRaises(RuntimeError):
                capture.save_sample(self.output, 1, "neutral", self.frame, self.face,
                                    self.crop, self.gray, "fake", self.scores())
        self.assertEqual(list(self.output.files), ["session/01_neutral_crop.png"])

    def test_report_refuses_overwrite(self):
        capture.finish_report(self.output, [])
        original = dict(self.output.files)
        with self.assertRaises(FileExistsError):
            capture.finish_report(self.output, [])
        self.assertEqual(self.output.files, original)


if __name__ == "__main__":
    unittest.main()
