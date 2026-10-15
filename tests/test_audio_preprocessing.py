import unittest

import numpy as np

from src.audio.features import extract_light_mfcc_like, highpass_filter
from src.audio.preprocessing import (
    WINDOW_FRAMES, _highpass_filter_fast, evaluation_windows, preprocessing_spec,
    to_mono_16k, training_window, window_features,
)


class AudioPreprocessingTests(unittest.TestCase):
    def test_stereo_mean_and_input_unchanged(self):
        source = np.array([[0.2, 0.4], [-0.4, 0.2]], dtype=np.float32)
        saved = source.copy()
        result = to_mono_16k(source, 16000)
        np.testing.assert_allclose(result, [0.3, -0.1], atol=1e-7)
        np.testing.assert_array_equal(source, saved)
        self.assertEqual(result.dtype, np.float32)

    def test_mono_passthrough_returns_independent_copy(self):
        source = np.array([0.1, -0.2], dtype=np.float32)
        result = to_mono_16k(source, 16000)
        np.testing.assert_array_equal(source, result)
        self.assertFalse(np.shares_memory(source, result))

    def test_resampling_duration_for_both_hardware_rates(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                source = np.sin(2 * np.pi * 1000 * np.arange(rate) / rate).astype(np.float32)
                result = to_mono_16k(source, rate)
                self.assertEqual(len(result), 16000)
                self.assertTrue(np.isfinite(result).all())
                self.assertGreater(np.sqrt(np.mean(result[100:-100] ** 2)), 0.65)

    def test_downsampling_rejects_out_of_band_tone(self):
        time = np.arange(48000) / 48000
        signal = np.sin(2 * np.pi * 12000 * time).astype(np.float32)
        result = to_mono_16k(signal, 48000)
        self.assertLess(np.sqrt(np.mean(result[100:-100] ** 2)), 0.01)

    def test_invalid_audio_is_rejected(self):
        for source in (np.array([], dtype=float), np.array([np.nan]),
                       np.array([np.inf]), np.array([1], dtype=np.int16),
                       np.zeros((4, 3)), np.zeros((1, 2, 3))):
            with self.subTest(shape=source.shape):
                with self.assertRaises(ValueError):
                    to_mono_16k(source, 16000)
        for rate in (0, -1, 16000.5):
            with self.assertRaises(ValueError):
                to_mono_16k(np.ones(10, dtype=np.float32), rate)

    def test_short_clip_is_right_padded(self):
        source = np.full(8000, 0.1, dtype=np.float32)
        windows = list(evaluation_windows(source))
        self.assertEqual(len(windows), 1)
        start, window = windows[0]
        self.assertEqual(start, 0)
        self.assertEqual(len(window), WINDOW_FRAMES)
        np.testing.assert_array_equal(window[:8000], source)
        self.assertTrue(np.all(window[8000:] == 0))

    def test_eval_windows_include_tail_without_duplicates(self):
        for length, expected in ((24000, [0]), (40000, [0, 8000, 16000]),
                                 (41000, [0, 8000, 16000, 17000])):
            source = np.arange(length, dtype=np.float32) / length
            windows = list(evaluation_windows(source))
            self.assertEqual([s for s, _ in windows], expected)
            self.assertTrue(all(len(w) == WINDOW_FRAMES for _, w in windows))
            self.assertEqual(windows[-1][1][-1], source[-1])

    def test_training_crop_is_seeded_and_not_always_start(self):
        source = np.linspace(-0.1, 0.1, 80000, dtype=np.float32)
        rng1, rng2 = np.random.default_rng(42), np.random.default_rng(42)
        starts = []
        for _ in range(4):
            start, window = training_window(source, rng1)
            other_start, other = training_window(source, rng2)
            self.assertEqual(start, other_start)
            np.testing.assert_array_equal(window, other)
            np.testing.assert_array_equal(window, source[start:start + WINDOW_FRAMES])
            starts.append(start)
        self.assertGreater(len(set(starts)), 1)
        self.assertTrue(any(s > 0 for s in starts))

    def test_training_short_clip_matches_eval_padding(self):
        source = np.ones(200, dtype=np.float32) * 0.1
        start, window = training_window(source, np.random.default_rng(42))
        self.assertEqual(start, 0)
        np.testing.assert_array_equal(window, next(evaluation_windows(source))[1])

    def test_quiet_audio_is_gated_before_normalization(self):
        for amplitude in (0.0, 0.001):
            features, rms = window_features(np.full(WINDOW_FRAMES, amplitude, dtype=np.float32))
            self.assertIsNone(features)
            self.assertLess(rms, 0.01)

    def test_features_match_current_inference_and_model_padding(self):
        time = np.arange(WINDOW_FRAMES) / 16000
        window = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
        saved = window.copy()
        features, rms = window_features(window)
        expected = highpass_filter(window / np.max(np.abs(window)), 16000, 80)
        expected = extract_light_mfcc_like(expected, 16000, 13, 512, 256, include_delta=False)
        expected = np.pad(expected, ((0, 0), (0, 93 - expected.shape[1])))
        # Tiny filter rounding changes are magnified in the near-silent tone bins.
        np.testing.assert_allclose(features, expected, rtol=0, atol=0.005)
        np.testing.assert_array_equal(window, saved)
        self.assertEqual(features.shape, (13, 93))
        self.assertEqual(features.dtype, np.float32)
        self.assertGreater(rms, 0.01)

    def test_wrong_window_size_is_rejected(self):
        with self.assertRaises(ValueError):
            window_features(np.zeros(100, dtype=np.float32))

    def test_fast_highpass_preserves_initial_state_and_waveform(self):
        rng = np.random.default_rng(42)
        time = np.arange(WINDOW_FRAMES) / 16000
        signals = [np.zeros(WINDOW_FRAMES), np.ones(WINDOW_FRAMES),
                   -np.ones(WINDOW_FRAMES), np.eye(1, WINDOW_FRAMES).ravel(),
                   np.sin(2 * np.pi * 220 * time), rng.uniform(-1, 1, WINDOW_FRAMES),
                   np.array([0.4])]
        for index, signal in enumerate(signals):
            with self.subTest(signal=index):
                source = signal.astype(np.float32)
                saved = source.copy()
                actual = _highpass_filter_fast(source, 16000, 80)
                expected = highpass_filter(source, 16000, 80)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(source, saved)
                self.assertEqual(actual.dtype, np.float32)
                self.assertEqual(actual[0], source[0])

    def test_fast_features_match_reference_on_broadband_audio(self):
        source = np.random.default_rng(42).uniform(-0.2, 0.2, WINDOW_FRAMES).astype(np.float32)
        actual, _ = window_features(source)
        filtered = highpass_filter(source / np.max(np.abs(source)), 16000, 80)
        expected = extract_light_mfcc_like(filtered, 16000, 13, 512, 256, include_delta=False)
        expected = np.pad(expected, ((0, 0), (0, 93 - expected.shape[1])))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

    def test_spec_does_not_share_mutable_state(self):
        spec = preprocessing_spec()
        spec['labels'].clear()
        self.assertEqual(preprocessing_spec()['labels'], ['neutral', 'happy', 'sad', 'anger'])


if __name__ == '__main__':
    unittest.main()
