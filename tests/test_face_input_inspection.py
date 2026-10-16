import copy
from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from scripts import inspect_face_input as inspector
from src.config import DEFAULT_CONFIG
from src.shared_types import FaceDetection
from src.video.model import VideoEmotionModel


class FaceInputInspectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(42).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        self.face = FaceDetection((40, 20, 60, 80), 0.9, "fake")
        self.detector = Mock(backend="fake", warning=None)
        self.detector.detect.return_value = self.face

    def test_crop_and_grayscale_match_production_without_modifying_frame(self):
        before = self.frame.copy()
        face, crop, gray = inspector.inspect_frame(self.frame, self.detector, 48)
        self.assertIs(face, self.face)
        np.testing.assert_array_equal(crop, before[20:100, 40:100])
        expected = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (48, 48))
        np.testing.assert_array_equal(gray, expected)
        np.testing.assert_array_equal(self.frame, before)
        self.assertFalse(np.shares_memory(self.frame, crop))

    def test_displayed_gray_matches_actual_model_input_before_normalization(self):
        _, crop, gray = inspector.inspect_frame(self.frame, self.detector, 48)
        model = VideoEmotionModel.__new__(VideoEmotionModel)
        model.cv2, model.input_size, model.input_name = cv2, 48, "input"
        model.session = Mock()
        model.session.run.return_value = [np.array([[0.1, 0.2, 0.3, 0.4]], np.float32)]
        model.predict(crop)
        tensor = model.session.run.call_args.args[1]["input"]
        np.testing.assert_array_equal(tensor, gray.astype(np.float32)[None, None] / 255.0)

    def test_missing_face_clears_both_panels(self):
        self.detector.detect.side_effect = [self.face, None]
        inspector.inspect_frame(self.frame, self.detector, 48)
        face, crop, gray = inspector.inspect_frame(self.frame, self.detector, 48)
        self.assertEqual((face, crop, gray), (None, None, None))
        canvas = inspector.render(self.frame, face, crop, gray, "fake", 48)
        self.assertEqual(canvas.shape, (320, 960, 3))
        self.assertFalse(canvas[40:280, 320:].any())

    def test_render_preserves_source_arrays_and_has_fixed_size(self):
        face, crop, gray = inspector.inspect_frame(self.frame, self.detector, 48)
        before = [array.copy() for array in (self.frame, crop, gray)]
        canvas = inspector.render(self.frame, face, crop, gray, "mediapipe", 48)
        self.assertEqual(canvas.shape, (320, 960, 3))
        for actual, expected in zip((self.frame, crop, gray), before):
            np.testing.assert_array_equal(actual, expected)
        for x in (0, 320, 640):
            self.assertGreater(float(canvas[40:280, x:x+320].std()), 0)

    def test_gray_preview_uses_nearest_neighbor_and_preserves_square_aspect(self):
        gray = np.arange(48 * 48, dtype=np.uint8).reshape(48, 48)
        panel = inspector.image_panel(gray, nearest=True)
        expected = cv2.cvtColor(cv2.resize(gray, (240, 240), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        np.testing.assert_array_equal(panel[:, 40:280], expected)
        self.assertFalse(panel[:, :40].any())
        self.assertFalse(panel[:, 280:].any())

    def test_invalid_rectangle_is_rejected_not_silently_recropped(self):
        for box in ((-1, 0, 20, 20), (0, 0, 0, 20), (140, 0, 30, 20), (0, 110, 20, 20)):
            self.detector.detect.return_value = FaceDetection(box, 1, "fake")
            with self.subTest(box=box), self.assertRaises(ValueError):
                inspector.inspect_frame(self.frame, self.detector, 48)

    def mocked_run(self, *, key=-1, visible=1, max_frames=0, read_ok=True, headless=False):
        cap = Mock()
        cap.read.return_value = (read_ok, self.frame if read_ok else None)
        with ExitStack() as stack:
            opening = stack.enter_context(patch.object(inspector, "open_camera", return_value=cap))
            stack.enter_context(patch.object(inspector, "FaceDetector", return_value=self.detector))
            stack.enter_context(patch.object(cv2, "namedWindow", side_effect=cv2.error("headless") if headless else None))
            stack.enter_context(patch.object(cv2, "imshow"))
            stack.enter_context(patch.object(cv2, "waitKey", return_value=key))
            stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=visible))
            destroy = stack.enter_context(patch.object(cv2, "destroyWindow"))
            stack.enter_context(patch("builtins.print"))
            if headless:
                with self.assertRaises(cv2.error):
                    inspector.run(copy.deepcopy(DEFAULT_CONFIG["video"]))
                opening.assert_not_called()
            elif not read_ok:
                with self.assertRaises(RuntimeError):
                    inspector.run(copy.deepcopy(DEFAULT_CONFIG["video"]))
            else:
                inspector.run(copy.deepcopy(DEFAULT_CONFIG["video"]), camera_id=2, max_frames=max_frames)
                self.assertEqual(opening.call_args.args[1], 2)
            if not headless:
                cap.release.assert_called_once()
                self.detector.close.assert_called_once()
            destroy.assert_called_once()

    def test_q_exit_releases_resources(self):
        self.mocked_run(key=ord("q"))

    def test_window_close_releases_resources(self):
        self.mocked_run(visible=0)

    def test_frame_limit_releases_resources(self):
        self.mocked_run(max_frames=1)

    def test_failed_read_releases_resources(self):
        self.mocked_run(read_ok=False)

    def test_headless_error_does_not_open_camera(self):
        self.mocked_run(headless=True)


if __name__ == "__main__":
    unittest.main()
