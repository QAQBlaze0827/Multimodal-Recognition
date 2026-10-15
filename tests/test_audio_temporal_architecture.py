import importlib.util
import sys
import unittest

import numpy as np

from scripts.train_audio_emotiontalk_manifest import model_structure
from scripts.train_audio_tiny_cnn import build_model


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Training dependency TensorFlow is not installed")
class TemporalArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = object()
        previous = sys.modules.get("jax", missing)
        try:
            sys.modules["jax"] = None
            import tensorflow as tf

            cls.tf = tf
            tf.keras.utils.set_random_seed(42)
            cls.legacy = build_model(tf, (13, 93), 0.001)
            tf.keras.utils.set_random_seed(42)
            cls.temporal = build_model(tf, (13, 93), 0.001, temporal_axis=True)
        finally:
            if previous is missing:
                sys.modules.pop("jax", None)
            else:
                sys.modules["jax"] = previous

    def test_default_retains_original_layers_and_seeded_weights(self):
        self.tf.keras.utils.set_random_seed(42)
        explicit = build_model(self.tf, (13, 93), 0.001, temporal_axis=False)
        self.assertEqual([layer.__class__.__name__ for layer in self.legacy.layers],
                         ["InputLayer", "Conv1D", "BatchNormalization", "ReLU", "Conv1D",
                          "BatchNormalization", "ReLU", "GlobalAveragePooling1D", "Dense", "Dropout", "Dense"])
        for default, supplied in zip(self.legacy.get_weights(), explicit.get_weights()):
            np.testing.assert_array_equal(default, supplied)

    def test_temporal_permute_preserves_values_and_external_shape(self):
        probe = self.tf.keras.Model(self.temporal.inputs, self.temporal.get_layer("mfcc_to_time").output)
        x = np.arange(2 * 13 * 93, dtype=np.float32).reshape(2, 13, 93)
        np.testing.assert_array_equal(probe(x).numpy(), x.transpose(0, 2, 1))
        self.assertEqual(self.temporal.input_shape, (None, 13, 93))
        self.assertEqual(self.temporal.output_shape, (None, 4))

    def test_conv_steps_channels_kernels_and_parameter_difference(self):
        legacy = model_structure(self.legacy, "legacy")
        temporal = model_structure(self.temporal, "temporal")
        self.assertEqual(legacy["conv_kernel_shapes"], [[5, 93, 32], [3, 32, 64]])
        self.assertEqual(temporal["conv_kernel_shapes"], [[5, 13, 32], [3, 32, 64]])
        self.assertEqual(legacy["conv_output_shapes"], [[None, 7, 32], [None, 4, 64]])
        self.assertEqual(temporal["conv_output_shapes"], [[None, 47, 32], [None, 24, 64]])
        self.assertEqual(legacy["parameter_count"] - temporal["parameter_count"], 12800)

    def test_both_models_output_finite_normalized_four_class_probabilities(self):
        for model in (self.legacy, self.temporal):
            for batch in (1, 3):
                x = np.random.default_rng(42).normal(size=(batch, 13, 93)).astype(np.float32)
                result = model(x, training=False).numpy()
                self.assertEqual(result.shape, (batch, 4))
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue((result >= 0).all())
                np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-6)

    def test_structure_rejects_wrong_or_unknown_axis(self):
        for model, name in ((self.legacy, "temporal"), (self.temporal, "legacy"),
                            (self.temporal, "unknown")):
            with self.assertRaises(ValueError):
                model_structure(model, name)


if __name__ == "__main__":
    unittest.main()
