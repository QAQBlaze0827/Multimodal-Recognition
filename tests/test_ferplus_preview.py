import copy
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from scripts import preview_ferplus as preview
from scripts.compare_face_models import fer_scores, fer_tensor
from src.config import DEFAULT_CONFIG
from src.shared_types import FaceDetection


class FERPlusPreviewTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        self.face = FaceDetection((40, 20, 60, 80), 0.9, "fake")
        self.detector = Mock(backend="fake", warning=None)
        self.detector.detect.return_value = self.face
        self.session = Mock()
        self.session.run.return_value = [np.array([[0, 1, 2, 3, 5, 0, 0, 0]], np.float32)]

    def test_live_tensor_and_scores_match_offline_comparison(self):
        original = self.frame.copy()
        face, crop, gray, scores, elapsed = preview.process_frame(self.frame, self.detector, self.session, "input")
        expected = fer_tensor(self.frame[20:100, 40:100])
        np.testing.assert_array_equal(self.session.run.call_args.args[1]["input"], expected)
        np.testing.assert_array_equal(gray, expected[0, 0].astype(np.uint8))
        np.testing.assert_array_equal(self.frame, original)
        self.assertIs(face, self.face)
        self.assertEqual(scores, fer_scores(self.session.run.return_value[0]))
        self.assertGreaterEqual(elapsed, 0)

    def test_no_face_clears_previous_result_without_inference(self):
        preview.process_frame(self.frame, self.detector, self.session, "input")
        self.detector.detect.return_value = None
        self.session.reset_mock()
        self.assertEqual(preview.process_frame(self.frame, self.detector, self.session, "input"), (None,) * 5)
        self.session.run.assert_not_called()

    def test_render_has_fixed_dimensions_and_does_not_mutate_inputs(self):
        face, crop, gray, scores, elapsed = preview.process_frame(self.frame, self.detector, self.session, "input")
        originals = [a.copy() for a in (self.frame, crop, gray)]
        with patch.object(cv2, "putText", wraps=cv2.putText) as write:
            canvas = preview.render(self.frame, face, crop, gray, "mediapipe", scores, elapsed)
        self.assertEqual(canvas.shape, (560, 960, 3))
        for a, b in zip((self.frame, crop, gray), originals):
            np.testing.assert_array_equal(a, b)
        for call in write.call_args_list:
            _, text, (x, y), font, scale, _, thickness, *_ = call.args
            (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
            self.assertGreaterEqual(y - height, 0)
            self.assertLess(x + width, 960)
            self.assertLess(y + baseline, 560)

    def test_no_face_render_has_empty_crops_and_no_stale_scores(self):
        with patch.object(cv2, "putText", wraps=cv2.putText) as write:
            canvas = preview.render(self.frame, None, None, None, "haar", None, None)
        self.assertFalse(canvas[40:280, 320:].any())
        texts = [call.args[1] for call in write.call_args_list]
        self.assertIn("FULL (8): No face", texts)
        self.assertIn("FOUR (relative): --", texts)
        self.assertFalse(any("%" in text for text in texts))

    def test_renderer_keeps_excluded_class_and_four_class_abstention(self):
        scores = fer_scores([-10000, -10000, -10000, -10000, -10000, -10000, -10000, 10000])
        with patch.object(cv2, "putText", wraps=cv2.putText) as write:
            preview.render(self.frame, self.face, self.frame[20:100, 40:100], np.zeros((64, 64), np.uint8), "haar", scores, 12.3)
        texts = [call.args[1] for call in write.call_args_list]
        self.assertIn("FULL (8): contempt 100.0%", texts)
        self.assertIn("FOUR (relative): Abstain", texts)

    def mocked_run(self, *, keys=(ord("q"),), visible=1, max_frames=0, read_ok=True,
                   headless=False, error=None):
        cap = Mock()
        cap.read.return_value = (read_ok, self.frame if read_ok else None)
        if error:
            self.session.run.side_effect = error
        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "is_file", return_value=True))
            stack.enter_context(patch.object(preview, "open_session", return_value=(self.session, "input")))
            opening = stack.enter_context(patch.object(preview, "open_camera", return_value=cap))
            stack.enter_context(patch.object(preview, "FaceDetector", return_value=self.detector))
            stack.enter_context(patch.object(cv2, "namedWindow", side_effect=cv2.error("headless") if headless else None))
            stack.enter_context(patch.object(cv2, "imshow"))
            stack.enter_context(patch.object(cv2, "waitKey", side_effect=keys))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=visible))
            destroy = stack.enter_context(patch.object(cv2, "destroyWindow"))
            stack.enter_context(patch("builtins.print"))
            if headless or not read_ok or error:
                with self.assertRaises((cv2.error, RuntimeError, ValueError)):
                    preview.run(copy.deepcopy(DEFAULT_CONFIG["video"]), camera_id=2)
            else:
                preview.run(copy.deepcopy(DEFAULT_CONFIG["video"]), camera_id=2, max_frames=max_frames)
            if headless:
                opening.assert_not_called()
                self.detector.close.assert_not_called()
            else:
                self.assertEqual(opening.call_args.args[1], 2)
                cap.release.assert_called_once()
                self.detector.close.assert_called_once()
            destroy.assert_called_once()

    def test_q_exit_releases_camera(self):
        self.mocked_run()

    def test_frame_limit_releases_camera(self):
        self.mocked_run(keys=(-1,), max_frames=1)

    def test_window_close_releases_camera(self):
        self.mocked_run(keys=(-1,), visible=0)

    def test_failed_capture_releases_camera(self):
        self.mocked_run(read_ok=False)

    def test_failed_inference_releases_camera(self):
        self.mocked_run(error=ValueError("invalid output"))

    def test_headless_does_not_open_camera(self):
        self.mocked_run(headless=True)

    def test_missing_or_invalid_model_stops_before_camera(self):
        with patch.object(preview, "open_camera") as opening, patch.object(cv2, "namedWindow") as window:
            with patch.object(Path, "is_file", return_value=False), self.assertRaises(FileNotFoundError):
                preview.run(DEFAULT_CONFIG["video"])
            with patch.object(Path, "is_file", return_value=True), \
                    patch.object(preview, "open_session", side_effect=ValueError("hash")), self.assertRaises(ValueError):
                preview.run(DEFAULT_CONFIG["video"])
            opening.assert_not_called()
            window.assert_not_called()


if __name__ == "__main__":
    unittest.main()
