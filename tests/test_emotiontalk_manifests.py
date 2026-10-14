import io
import json
import tarfile
import unittest
import wave

from scripts.prepare_emotiontalk_manifests import scan_archive


def fixture_members():
    members = []
    for group, speaker in (("G00002", "07"), ("G00001", "01"), ("G00013", "16")):
        for index, label in enumerate(("neutral", "happy", "sad", "angry", "fearful")):
            relative = f"{group}/{group}_01/{group}_01_01/{group}_01_01_{index:03}.wav"
            payload = {
                "file_path": relative,
                "speaker_id": speaker,
                "emotion_result": label,
                "data": {str(i): {"emotion": label if i < 3 else "surprised"} for i in range(5)},
                "paragraphs": {"startTime": 0, "endTime": 0.01, "duration": 0.01},
            }
            members.append((f"Audio/json/{relative[:-4]}.json", json.dumps(payload).encode()))
            audio = io.BytesIO()
            with wave.open(audio, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * 160)
            members.append((f"Audio/wav/{relative}", audio.getvalue()))
    return members


def scan(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in members:
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            tar.addfile(entry, io.BytesIO(content))
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode="r:") as tar:
        return scan_archive(tar)


class EmotionTalkManifestTests(unittest.TestCase):
    def test_keeps_only_original_four_classes_with_provenance(self):
        rows, audit = scan(fixture_members())
        self.assertEqual(len(rows), 12)
        self.assertEqual(audit["excluded_label_counts"]["fearful"], 3)
        self.assertEqual({r["split"] for r in rows}, {"train", "val", "test"})
        self.assertEqual({r["label"] for r in rows}, {"neutral", "happy", "sad", "anger"})
        for row in rows:
            self.assertEqual(row["sample_rate"], 16000)
            self.assertAlmostEqual(row["duration_seconds"], 0.01)
            self.assertEqual(row["matching_votes"], 3)
            self.assertTrue(row["review_recommended"])
            self.assertEqual(len(row["metadata_sha256"]), 64)
            if row["label"] == "anger":
                self.assertEqual(row["source_label"], "angry")

    def test_output_order_is_independent_of_archive_order(self):
        members = fixture_members()
        self.assertEqual(scan(members), scan(list(reversed(members))))

    def test_missing_audio_stops_audit(self):
        members = fixture_members()
        del members[1]
        with self.assertRaisesRegex(ValueError, "counts differ|Missing WAV"):
            scan(members)

    def test_duplicate_archive_path_stops_audit(self):
        members = fixture_members()
        members.append(members[0])
        with self.assertRaisesRegex(ValueError, "Duplicate archive path"):
            scan(members)

    def test_speaker_overlap_in_excluded_emotion_still_stops_audit(self):
        members = fixture_members()
        for index, (name, content) in enumerate(members):
            if name.endswith(".json") and "G00001" in name:
                payload = json.loads(content)
                if payload["emotion_result"] == "fearful":
                    payload["speaker_id"] = "07"
                    members[index] = (name, json.dumps(payload).encode())
        with self.assertRaisesRegex(ValueError, "Speaker overlap"):
            scan(members)

    def test_wrong_metadata_path_stops_audit(self):
        members = fixture_members()
        name, content = members[0]
        payload = json.loads(content)
        payload["file_path"] = "../outside.wav"
        members[0] = (name, json.dumps(payload).encode())
        with self.assertRaisesRegex(ValueError, "Invalid source path"):
            scan(members)

    def test_unknown_emotion_is_not_silently_merged(self):
        members = fixture_members()
        name, content = members[0]
        payload = json.loads(content)
        payload["emotion_result"] = "unknown"
        members[0] = (name, json.dumps(payload).encode())
        with self.assertRaisesRegex(ValueError, "Unknown emotion label"):
            scan(members)


if __name__ == "__main__":
    unittest.main()
