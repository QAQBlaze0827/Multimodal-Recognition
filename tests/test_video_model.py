import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.shared_types import EMOTIONS
from src.video.model import VideoEmotionModel


class VideoEmotionModelTests(unittest.TestCase):
    def predict(self, values):
        with patch.object(VideoEmotionModel, "_try_load_onnx"), \
                patch.object(VideoEmotionModel, "_load_smile_cascade", return_value=None):
            model = VideoEmotionModel(Mock(), "unused.onnx")
        output = np.array([values], dtype=np.float32)
        before = output.copy()
        model.session = Mock()
        model.session.run.return_value = [output]
        model.input_name = "input"
        gray = np.arange(48 * 48, dtype=np.uint8).reshape(48, 48)
        result = model._predict_onnx(gray)
        np.testing.assert_array_equal(output, before)
        tensor = model.session.run.call_args.args[1]["input"]
        self.assertEqual(tensor.shape, (1, 1, 48, 48))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor[0, 0], gray.astype(np.float32) / 255.0)
        return result

    def test_four_class_probabilities_are_preserved(self):
        result = self.predict([0.1, 0.2, 0.6, 0.1])
        np.testing.assert_allclose(list(result.scores.values()), [0.1, 0.2, 0.6, 0.1])
        self.assertEqual(result.label, "sad")

    def test_seven_class_probabilities_are_renormalized_not_softmaxed(self):
        result = self.predict([0.01, 0.01, 0.07, 0.01, 0.5, 0.2, 0.2])
        np.testing.assert_allclose(list(result.scores.values()), [0.1, 0.1, 0.7, 0.1])
        self.assertEqual(result.label, "sad")
        self.assertAlmostEqual(result.confidence, 0.7)

    def test_legacy_first_four_label_positions(self):
        for index, label in enumerate(EMOTIONS):
            values = np.zeros(7, np.float32)
            values[index] = 0.6
            values[4] = 0.4
            self.assertEqual(self.predict(values).label, label)

    def test_seven_classes_with_nearly_all_mass_retained(self):
        result = self.predict([0.1, 0.2, 0.3, 0.38, 0.01, 0.005, 0.005])
        expected = np.array([0.1, 0.2, 0.3, 0.38]) / 0.98
        np.testing.assert_allclose(list(result.scores.values()), expected)

    def test_logits_keep_stable_softmax_behavior(self):
        for values in ([1000, 1001, 999, 998], [-2, -1, 0, 2, 10, 8, 5]):
            retained = np.array(values[:4], np.float32)
            expected = np.exp(retained - retained.max())
            expected /= expected.sum()
            np.testing.assert_allclose(list(self.predict(values).scores.values()), expected, rtol=1e-6)

    def test_zero_retained_probability_has_zero_confidence(self):
        result = self.predict([0, 0, 0, 0, 1, 0, 0])
        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.label, "neutral")

    def test_tiny_retained_probability_still_preserves_ratios(self):
        result = self.predict([1e-12, 2e-12, 6e-12, 1e-12, 1, 0, 0])
        np.testing.assert_allclose(list(result.scores.values()), [0.1, 0.2, 0.6, 0.1])

    def test_invalid_output_is_rejected(self):
        for values in ([0.1] * 3, [0.2] * 5, [0.1] * 8,
                       [float("nan"), 0, 0, 1], [0, float("inf"), 0, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.predict(values)


if __name__ == "__main__":
    unittest.main()
