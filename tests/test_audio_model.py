import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.audio.model import AudioEmotionModel
from src.shared_types import EMOTIONS


class AudioEmotionModelTests(unittest.TestCase):
    def predict(self, probabilities, calibration=None):
        with patch.object(AudioEmotionModel, "_try_load_onnx"):
            model = AudioEmotionModel("unused.onnx", calibration=calibration)
        output = np.array([probabilities], dtype=np.float32)
        model.session = Mock()
        model.session.run.return_value = [output]
        result = model.predict(np.zeros((13, 93), dtype=np.float32))
        np.testing.assert_array_equal(output, np.array([probabilities], dtype=np.float32))
        return result

    def test_preserves_probabilities_without_calibration(self):
        result = self.predict([0.1, 0.1, 0.7, 0.1])
        self.assertEqual(result.label, "sad")
        self.assertAlmostEqual(result.confidence, 0.7)
        np.testing.assert_allclose(list(result.scores.values()), [0.1, 0.1, 0.7, 0.1])

    def test_applies_class_weights_then_normalizes(self):
        result = self.predict(
            [0.1, 0.2, 0.3, 0.4],
            {"neutral_logit_scale": 1.2, "sad_logit_scale": 1.5, "anger_logit_scale": 0.8},
        )
        expected = np.array([0.12, 0.2, 0.45, 0.32])
        expected /= expected.sum()
        np.testing.assert_allclose([result.scores[e] for e in EMOTIONS], expected)
        self.assertEqual(result.label, "sad")
        self.assertAlmostEqual(sum(result.scores.values()), 1.0)

    def test_confident_prediction_is_not_rejected_by_floor(self):
        result = self.predict([0.1, 0.1, 0.7, 0.1], {"confidence_floor": 0.6})
        self.assertEqual(result.label, "sad")
        self.assertAlmostEqual(result.confidence, 0.7)

    def test_low_confidence_keeps_existing_fallback(self):
        result = self.predict([0.25, 0.25, 0.25, 0.25], {"confidence_floor": 0.35})
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.confidence, 0.0)

    def test_legacy_seven_class_output_renormalizes_retained_classes(self):
        result = self.predict([0.1, 0.1, 0.35, 0.15, 0.1, 0.1, 0.1])
        self.assertEqual(result.label, "sad")
        self.assertAlmostEqual(result.confidence, 0.5)
        self.assertEqual(set(result.scores), set(EMOTIONS))


if __name__ == "__main__":
    unittest.main()
