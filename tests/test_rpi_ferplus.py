import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src import main as app
from src.config import load_config
from src.video import ferplus


class FerPlusSelectionTests(unittest.TestCase):
    def test_default_selects_configured_legacy_before_camera(self):
        args = app.build_parser().parse_args([])
        config = load_config(args.config)
        with patch.object(app, "VideoEmotionModel") as legacy, patch.object(
            app, "open_camera", side_effect=RuntimeError("stop before capture")
        ), self.assertRaisesRegex(RuntimeError, "stop before capture"):
            app.run_video_loop(config, args)
        cfg = config["video"]["model"]
        legacy.assert_called_once_with(cv2, cfg["path"], cfg["input_size"])

    def test_missing_ferplus_never_opens_camera_or_uses_legacy(self):
        args = app.build_parser().parse_args(["--video-model", "ferplus"])
        with patch.object(ferplus, "FER_PATH") as path, patch.object(
            app, "open_camera"
        ) as camera, patch.object(app, "VideoEmotionModel") as legacy:
            path.is_file.return_value = False
            with self.assertRaises(FileNotFoundError):
                app.run_video_loop(load_config(args.config), args)
        camera.assert_not_called()
        legacy.assert_not_called()

    def test_four_class_mapping(self):
        for index, label in ((0, "neutral"), (1, "happy"), (3, "sad"), (4, "anger")):
            logits = np.zeros(8)
            logits[index] = 5
            scores = ferplus.fer_scores(logits)["four_scores"]
            self.assertEqual(max(scores, key=scores.get), label)
            self.assertAlmostEqual(sum(scores.values()), 1)


if __name__ == "__main__":
    unittest.main()
