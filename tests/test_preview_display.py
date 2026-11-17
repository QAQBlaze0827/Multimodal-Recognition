import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src import main as app
from src.config import load_config
from src.output.display import IMAGE_HEIGHT, PREVIEW_SIZE, draw_overlay
from src.shared_types import EmotionResult, FaceDetection, make_result
from src.video.inference import VideoInferenceResult


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((240, 320, 3), 110, dtype=np.uint8)
        self.emotion = make_result("video", {"happy": 0.8, "neutral": 0.2})
        self.video = VideoInferenceResult(None, EmotionResult.neutral("video"), "onnx-ferplus")

    def render(self, audio=None, gated=False):
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            preview = draw_overlay(cv2, self.frame, self.video, audio, self.emotion, 19.2, 75.5, gated)
        return preview, put_text.call_args_list

    def test_no_face_or_audio_does_not_show_fallback_emotion(self):
        preview, calls = self.render()
        texts = [call.args[1] for call in calls]
        self.assertIn("Face: Not detected", texts)
        for text in ("Video: --", "Audio: --", "Fused: --"):
            self.assertIn(text, texts)
        self.assertEqual(preview.shape, (PREVIEW_SIZE[1], PREVIEW_SIZE[0], 3))

    def test_scaled_box_and_original_frame_unchanged(self):
        self.video.face = FaceDetection((40, 30, 80, 100), 0.9, "mediapipe")
        self.video.emotion = self.emotion
        before = self.frame.copy()
        preview, calls = self.render()
        # 320x240 becomes 720x540, with 120px of padding on each side.
        self.assertTrue(np.array_equal(preview[68, 210], (0, 220, 0)))
        self.assertTrue(np.array_equal(before, self.frame))
        self.assertIn("Video: happy  80%", [call.args[1] for call in calls])

    def test_all_text_fits_in_separate_panel(self):
        _, calls = self.render()
        for call in calls:
            _, text, (x, y), font, scale, _, thickness, _ = call.args
            (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
            self.assertGreaterEqual(y - height, IMAGE_HEIGHT)
            self.assertLess(y + baseline, PREVIEW_SIZE[1])
            self.assertLess(x + width, 480 if x < 480 else PREVIEW_SIZE[0])

    def test_landscape_and_portrait_preserve_aspect_ratio(self):
        for height, width in ((240, 320), (360, 640), (640, 360)):
            self.frame = np.full((height, width, 3), 110, dtype=np.uint8)
            preview, _ = self.render()
            ys, xs = np.where(np.any(preview[:IMAGE_HEIGHT] != 0, axis=2))
            actual_width = xs.max() - xs.min() + 1
            actual_height = ys.max() - ys.min() + 1
            self.assertAlmostEqual(actual_width / actual_height, width / height, delta=0.005)

    def test_audio_only_signal_and_gated_signal(self):
        _, calls = self.render(self.emotion)
        self.assertIn("Fused: happy  80%", [call.args[1] for call in calls])
        _, calls = self.render(self.emotion, gated=True)
        self.assertIn("Audio: Gated", [call.args[1] for call in calls])
        self.assertIn("Fused: --", [call.args[1] for call in calls])

    def test_zero_confidence_face_and_missing_temperature(self):
        self.video.face = FaceDetection((0, 0, 100, 100), 0.9, "haar")
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            draw_overlay(cv2, self.frame, self.video, None, self.emotion, 0, None)
        texts = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Video: --", texts)
        self.assertIn("FPS: 0.0   Temp: n/a", texts)

    def test_video_loop_display_and_headless(self):
        for headless in (False, True):
            with self.subTest(headless=headless), ExitStack() as stack:
                args = app.build_parser().parse_args(["--vision-only", "--max-frames", "1"])
                args.no_display = headless
                config = load_config(args.config)
                config["backend"]["enabled"] = False
                config["output"]["display"]["show_video"] = True
                camera = Mock()
                camera.read.return_value = (True, self.frame)
                stack.enter_context(patch.object(app, "open_camera", return_value=camera))
                stack.enter_context(patch.object(app, "VideoEmotionModel"))
                detector = stack.enter_context(patch.object(app, "FaceDetector"))
                detector.return_value.warning = None
                pipeline = stack.enter_context(patch.object(app, "VideoEmotionPipeline"))
                pipeline.return_value.process.return_value = self.video
                logger = stack.enter_context(patch.object(app, "CsvLogger"))
                database = stack.enter_context(patch.object(app, "DbLogger"))
                windows = {name: stack.enter_context(patch.object(cv2, name))
                           for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows")}
                stack.enter_context(patch.object(cv2, "waitKey", return_value=-1))
                app.run_video_loop(config, args)
                database.assert_not_called()
                camera.release.assert_called_once()
                logger.return_value.close.assert_called_once()
                if headless:
                    for mocked in windows.values():
                        mocked.assert_not_called()
                else:
                    windows["namedWindow"].assert_called_once_with(
                        "Multimodal Emotion Recognition",
                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL,
                    )
                    windows["resizeWindow"].assert_called_once_with("Multimodal Emotion Recognition", *PREVIEW_SIZE)
                    self.assertEqual(windows["imshow"].call_args.args[1].shape, (700, 960, 3))
                    windows["destroyAllWindows"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
