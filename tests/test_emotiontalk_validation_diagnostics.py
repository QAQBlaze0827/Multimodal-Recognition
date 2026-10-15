import copy
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.diagnose_emotiontalk_validation import (
    choose_examples, describe, grouped_summary, join_validation,
    require_new_outputs, review_audio,
)


def fixture():
    manifest, predictions, audit = [], [], []
    for i, label in enumerate(("neutral", "happy", "sad", "anger")):
        path = f"Audio/wav/G00001/{i}.wav"
        manifest.append({"source_wav": path, "split": "val", "label": label,
                         "speaker_id": "01", "group_id": "G00001", "matching_votes": "4",
                         "review_recommended": "False", "duration_seconds": "2",
                         "sample_rate": "16000", "channels": "1"})
        scores = [0.1] * 4
        scores[i] = 0.7
        predictions.append({"source_wav": path, "label": label, "pcm_sha256": str(i),
                            "scores": scores, "prediction": label,
                            "active_windows": 2, "inactive_windows": 0})
        audit.append({**manifest[-1], "pcm_sha256": str(i), "mono_rms": "0.05"})
    predictions[1].update(prediction="sad", scores=[0.1, 0.1, 0.7, 0.1])
    predictions[2].update(prediction=None, scores=None, active_windows=0, inactive_windows=2)
    return predictions, manifest, audit


class ValidationDiagnosisTests(unittest.TestCase):
    def test_join_recomputes_metrics_and_keeps_abstention(self):
        inputs = fixture()
        before = copy.deepcopy(inputs)
        rows, metrics = join_validation(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual([r["outcome"] for r in rows], ["correct", "incorrect", "abstain", "correct"])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertEqual(metrics["confusion_matrix"][2], [0, 0, 0, 0, 1])
        self.assertEqual(metrics["per_class"]["sad"]["recall"], 0)

    def test_test_manifest_is_rejected_and_other_audit_splits_are_not_joined(self):
        p, m, a = fixture()
        a.append({"split": "test", "source_wav": m[0]["source_wav"]})
        self.assertEqual(len(join_validation(p, m, a)[0]), 4)
        m[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation manifest"):
            join_validation(p, m, a)

    def test_duplicate_missing_or_foreign_sources_fail(self):
        for position in range(3):
            values = list(fixture())
            values[position].append(values[position][0])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                join_validation(*values)
        p, m, a = fixture()
        with self.assertRaisesRegex(ValueError, "sources"):
            join_validation(p[:-1], m, a)
        p[0]["source_wav"] = "test.wav"
        with self.assertRaisesRegex(ValueError, "sources"):
            join_validation(p, m, a)

    def test_inconsistent_hash_label_speaker_or_format_fails(self):
        for field, value in (("pcm_sha256", "changed"), ("label", "sad"),
                             ("speaker_id", "other"), ("duration_seconds", "3"),
                             ("sample_rate", "48000"), ("mono_rms", "nan")):
            p, m, a = fixture()
            a[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                join_validation(p, m, a)

    def test_invalid_probabilities_or_window_counts_fail(self):
        changes = ({"scores": [float("nan"), 0, 0, 0]}, {"scores": [1, 1, 1, 1]},
                   {"scores": [1.1, -0.1, 0, 0]}, {"prediction": "sad"},
                   {"active_windows": -1}, {"inactive_windows": 0.5},
                   {"active_windows": 0, "inactive_windows": 1})
        for change in changes:
            p, m, a = fixture()
            p[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                join_validation(p, m, a)

    def test_all_abstention_and_empty_subgroup_are_supported(self):
        p, m, a = fixture()
        for item in p:
            item.update(prediction=None, scores=None, active_windows=0, inactive_windows=1)
        rows, metrics = join_validation(p, m, a)
        self.assertEqual(metrics["rejected"], 4)
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(grouped_summary(rows)["outcome"]["correct"]["count"], 0)
        self.assertIsNone(describe([])["accuracy"])

    def test_groups_include_denominators_and_bucket_boundaries(self):
        rows, _ = join_validation(*fixture())
        for row, duration in zip(rows, (1.49, 1.5, 3, 6)):
            row["duration_seconds"] = duration
        rows[0]["mono_rms"] = 0.01
        rows[1]["mono_rms"] = 0.009
        groups = grouped_summary(rows)
        self.assertEqual([g["count"] for g in groups["duration"].values()], [1, 1, 1, 1])
        self.assertEqual(groups["rms"]["<0.01"]["count"], 1)
        self.assertEqual(groups["speaker"]["01"]["accuracy"], 0.5)

    def test_examples_are_deterministic_deduplicated_and_not_modified(self):
        rows, _ = join_validation(*fixture())
        before = copy.deepcopy(rows)
        examples = choose_examples(rows)
        self.assertEqual(examples, choose_examples(list(reversed(rows))))
        self.assertEqual(len({r["source_wav"] for r in examples}), len(examples))
        self.assertEqual(rows, before)
        wrong = next(r for r in examples if r["outcome"] == "incorrect")
        self.assertIn("false sad", wrong["selection_reason"])
        self.assertTrue(any(r["outcome"] == "abstain" for r in examples))

    def test_existing_outputs_are_refused_before_writing(self):
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaises(FileExistsError):
                require_new_outputs(Path("report"), Path("audio"))
        with patch.object(Path, "exists", side_effect=[False, True]):
            with self.assertRaises(FileExistsError):
                require_new_outputs(Path("report"), Path("audio"))

    def test_review_only_accepts_val_and_checks_original_annotation_and_pcm(self):
        raw_json = json.dumps({"emotion_result": "neutral", "speaker_id": "01",
                               "file_path": "G00001/clip.wav"}).encode()
        source = {"split": "val", "group_id": "G00001", "source_label": "neutral",
                  "label": "neutral", "speaker_id": "01", "source_json": "annotation.json",
                  "source_wav": "Audio/wav/G00001/clip.wav",
                  "metadata_sha256": hashlib.sha256(raw_json).hexdigest()}
        case = {"source_wav": source["source_wav"], "pcm_sha256": "expected"}

        class FakeTar:
            def extractfile(self, name):
                return io.BytesIO(raw_json if name == "annotation.json" else b"wav")

        with patch("scripts.diagnose_emotiontalk_validation.inspect_wav", return_value={"pcm_sha256": "expected"}):
            self.assertEqual(review_audio(FakeTar(), case, source), b"wav")
        with patch("scripts.diagnose_emotiontalk_validation.inspect_wav", return_value={"pcm_sha256": "changed"}):
            with self.assertRaisesRegex(ValueError, "PCM"):
                review_audio(FakeTar(), case, source)
        for field, value in (("split", "test"), ("metadata_sha256", "changed")):
            with self.assertRaises(ValueError):
                review_audio(FakeTar(), case, {**source, field: value})


if __name__ == "__main__":
    unittest.main()
