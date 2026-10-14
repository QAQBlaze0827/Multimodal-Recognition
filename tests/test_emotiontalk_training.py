import hashlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from scripts.train_audio_emotiontalk_manifest import (
    aggregate_recordings, class_weights, select_records, training_arrays,
    validate_source, validation_arrays,
)


def records():
    return [{"split": split, "label": label, "source_wav": f"{split}/{label}/{i}.wav"}
            for split in ("train", "val", "test")
            for label in ("neutral", "happy", "sad", "anger") for i in range(5)]


def annotation():
    payload = {"file_path": "G00002/clip.wav", "speaker_id": "07", "emotion_result": "angry"}
    raw = json.dumps(payload).encode()
    row = {"group_id": "G00002", "split": "train", "source_json": "annotation.json",
           "source_wav": "Audio/wav/G00002/clip.wav", "metadata_sha256": hashlib.sha256(raw).hexdigest(),
           "source_label": "angry", "label": "anger", "speaker_id": "07"}
    return raw, row


class FakeTar:
    def __init__(self, raw):
        self.raw = raw

    def extractfile(self, name):
        return io.BytesIO(self.raw)


class EmotionTalkTrainingTests(unittest.TestCase):
    def test_selection_is_seeded_and_keeps_split(self):
        source = records()
        chosen = select_records(source, "train", 2, 42)
        self.assertEqual(chosen, select_records(list(reversed(source)), "train", 2, 42))
        self.assertEqual(len(chosen), 8)
        self.assertEqual({r["split"] for r in chosen}, {"train"})
        for label in ("neutral", "happy", "sad", "anger"):
            self.assertEqual(sum(r["label"] == label for r in chosen), 2)
        self.assertEqual(source, records())

    def test_full_selection_keeps_natural_distribution(self):
        source = records() + [{"split": "train", "label": "neutral", "source_wav": "extra.wav"}]
        chosen = select_records(source, "train", 0, 42)
        self.assertEqual(len(chosen), 21)
        self.assertEqual(sum(r["label"] == "neutral" for r in chosen), 6)

    def test_selection_rejects_test_negative_limit_and_missing_class(self):
        for split, limit in (("test", 1), ("train", -1)):
            with self.assertRaises(ValueError):
                select_records(records(), split, limit, 42)
        with self.assertRaisesRegex(ValueError, "Missing class"):
            select_records([r for r in records() if r["label"] != "sad"], "train", 1, 42)

    def test_annotation_is_checked_against_manifest(self):
        raw, row = annotation()
        validate_source(FakeTar(raw), row)
        for key, value in (("metadata_sha256", "changed"), ("label", "sad"),
                           ("speaker_id", "other"), ("source_label", "neutral"),
                           ("source_wav", "wrong.wav"), ("split", "val")):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_source(FakeTar(raw), {**row, key: value})

    def test_training_rejects_val_or_test_before_decoding(self):
        for split in ("val", "test"):
            with self.assertRaisesRegex(ValueError, "Only train"):
                training_arrays(None, [{"split": split}], 42)

    def test_validation_rejects_test_before_decoding(self):
        with self.assertRaisesRegex(ValueError, "Only val"):
            validation_arrays(None, [{"split": "test"}])

    def test_silent_validation_recording_is_preserved(self):
        row = {"split": "val", "label": "sad", "source_wav": "silent.wav"}
        with patch("scripts.train_audio_emotiontalk_manifest.load_recording",
                   return_value=(np.zeros(24000, dtype=np.float32), "hash")):
            features, owners, details = validation_arrays(None, [row])
        self.assertEqual(features.shape, (0, 13, 93))
        self.assertEqual(len(owners), 0)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["inactive_windows"], 1)

    def test_no_active_training_class_stops_instead_of_inventing_neutral(self):
        row = {"split": "train", "label": "sad", "source_wav": "silent.wav"}
        with patch("scripts.train_audio_emotiontalk_manifest.load_recording",
                   return_value=(np.zeros(24000, dtype=np.float32), "hash")):
            with self.assertRaisesRegex(ValueError, "no active crops"):
                training_arrays(None, [row], 42)

    def test_metrics_are_per_recording_and_include_abstention(self):
        probabilities = np.array([[1, 0, 0, 0]] * 10 + [[1, 0, 0, 0]], dtype=np.float32)
        metrics, predictions = aggregate_recordings(probabilities, [0] * 10 + [1], ["neutral", "happy", "sad"])
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)
        self.assertEqual(metrics["rejected"], 1)
        self.assertIsNone(predictions[2]["prediction"])
        self.assertEqual(metrics["confusion_matrix"][2][4], 1)
        self.assertEqual(metrics["per_class"]["sad"]["support"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3) / 4)

    def test_probabilities_are_averaged_before_argmax(self):
        metrics, predictions = aggregate_recordings(
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.8, 0.05, 0.05]], [0, 0], ["happy"])
        self.assertEqual(predictions[0]["prediction"], "happy")
        self.assertEqual(metrics["accuracy"], 1)

    def test_all_rejected_records_have_zero_coverage_and_accuracy(self):
        metrics, _ = aggregate_recordings(np.empty((0, 4)), [], ["neutral", "happy", "sad", "anger"])
        self.assertEqual(metrics["coverage"], 0)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["macro_f1"], 0)
        self.assertEqual(metrics["rejected"], 4)

    def test_invalid_probabilities_and_owner_are_rejected(self):
        for probs, owners in (([[np.nan, 0, 0, 1]], [0]), ([[1, 0, 0, 0]], [1]),
                              ([[0, 0, 0, 0]], [0]), ([[2, -1, 0, 0]], [0])):
            with self.assertRaises(ValueError):
                aggregate_recordings(probs, owners, ["neutral"])

    def test_class_weights_come_from_active_training_labels(self):
        weights = class_weights(np.array([0, 0, 1, 2, 3]))
        self.assertAlmostEqual(weights[0], 5 / 8)
        self.assertAlmostEqual(weights[3], 5 / 4)
        with self.assertRaises(ValueError):
            class_weights(np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
