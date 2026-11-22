import sqlite3
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import database as db
from backend.app import app


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.addCleanup(self.conn.close)
        patched = patch.object(db, "_get_conn", return_value=self.conn)
        patched.start()
        self.addCleanup(patched.stop)
        db.init_db(":memory:")
        self.now = datetime.now() - timedelta(seconds=2)

    def add(self, **fields):
        row = dict(session_id="test", timestamp=self.now.isoformat(),
                   face_detected=True, video_emotion="happy", video_conf=0.8,
                   fused_emotion="happy", fused_conf=0.8)
        row.update(fields)
        db.insert_log(":memory:", row)

    def summary(self, days=30):
        changes = self.conn.total_changes
        result = db.get_analytics_summary(":memory:", days)
        self.assertEqual(self.conn.total_changes, changes)
        return result

    def test_empty(self):
        data = self.summary()
        self.assertEqual(data["total_logs"], 0)
        self.assertEqual(data["valid_logs"], 0)
        self.assertEqual(data["distribution"], {})
        self.assertIsNone(data["first_record"])
        self.assertEqual(data["session_count"], 0)

    def test_no_source_fallback_is_not_neutral_distribution(self):
        self.add(face_detected=False, video_conf=0, fused_emotion="neutral", fused_conf=0)
        self.add(face_detected=False, video_conf=0, fused_emotion="neutral", fused_conf=1,
                 audio_emotion="neutral", audio_conf=0)
        data = self.summary()
        self.assertEqual(data["no_source_logs"], 2)
        self.assertEqual(data["excluded_logs"], 2)
        self.assertEqual(data["distribution"], {})

    def test_valid_denominator_includes_low_scores(self):
        self.add(fused_emotion="happy", fused_conf=0.49)
        self.add(fused_emotion="neutral", fused_conf=0.5)
        self.add(face_detected=False, video_conf=0, fused_conf=0)
        self.add(fused_conf=0)
        data = self.summary()
        self.assertEqual((data["total_logs"], data["valid_logs"], data["low_score_logs"]), (4, 2, 1))
        self.assertEqual((data["no_source_logs"], data["unresolved_logs"]), (1, 1))
        self.assertEqual(data["distribution"]["happy"], dict(count=1, pct=50.0, avg_conf=0.49))
        self.assertEqual(data["distribution"]["neutral"]["pct"], 50.0)

    def test_audio_source_does_not_require_face(self):
        self.add(face_detected=False, video_conf=0, audio_emotion="sad", audio_conf=0.7,
                 fused_emotion="sad", fused_conf=0.7)
        data = self.summary()
        self.assertEqual(data["valid_logs"], 1)
        self.assertEqual(data["distribution"]["sad"]["count"], 1)

    def test_invalid_scores_labels_and_missing_fields(self):
        for score in (0, -0.1, 1.1, float("inf"), None, "bad"):
            self.add(video_conf=score)
        self.add(video_emotion="surprise")
        self.add(session_id="missing-face")
        self.conn.execute("UPDATE emotion_logs SET face_detected = NULL WHERE session_id = 'missing-face'")
        # Old/imported rows without source metadata cannot prove a usable source.
        db.insert_log(":memory:", dict(session_id="legacy", timestamp=self.now.isoformat(),
                                     fused_emotion="neutral", fused_conf=0.9))
        self.add(fused_emotion="surprise")
        self.add(fused_conf=1.2)
        data = self.summary()
        self.assertEqual(data["valid_logs"], 0)
        self.assertEqual(data["no_source_logs"], 9)
        self.assertEqual(data["unresolved_logs"], 2)

    def test_period_actual_range_and_sessions_with_records(self):
        old = (self.now - timedelta(days=100)).isoformat()
        self.conn.execute("INSERT INTO sessions(id, start_time) VALUES (?,?)", ("old-session", old))
        self.conn.execute("INSERT INTO sessions(id, start_time) VALUES (?,?)", ("empty-session", self.now.isoformat()))
        self.add(session_id="old-session")
        self.add(session_id="old-session", timestamp=(self.now - timedelta(days=1)).isoformat())
        self.add(session_id="outside", timestamp=(self.now - timedelta(days=8)).isoformat())
        self.add(session_id="future", timestamp=(self.now + timedelta(days=1)).isoformat())
        data = self.summary(7)
        self.assertEqual(data["total_logs"], 2)
        self.assertEqual(data["session_count"], 1)
        self.assertEqual(data["first_record"], (self.now - timedelta(days=1)).isoformat())
        self.assertEqual(data["last_record"], self.now.isoformat())

    def test_api_contract_and_days_validation(self):
        self.add()
        with TestClient(app) as client:
            data = client.get("/api/analytics?days=7").json()
            self.assertEqual(data["valid_logs"], 1)
            self.assertEqual(data["low_score_threshold"], 0.5)
            self.assertEqual(client.get("/api/analytics?days=0").status_code, 422)
            self.assertEqual(client.get("/api/analytics?days=366").status_code, 422)


if __name__ == "__main__":
    unittest.main()
