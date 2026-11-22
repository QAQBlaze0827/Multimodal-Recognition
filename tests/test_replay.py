import sqlite3
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import database as db
from backend.app import app


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.addCleanup(self.conn.close)
        patched = patch.object(db, '_get_conn', return_value=self.conn)
        patched.start()
        self.addCleanup(patched.stop)
        db.init_db(':memory:')
        self.now = datetime.now() - timedelta(seconds=5)

    def add(self, **fields):
        row = dict(session_id='sample', timestamp=self.now.isoformat(), face_detected=True,
                   video_emotion='happy', video_conf=.8, fused_emotion='happy', fused_conf=.8)
        row.update(fields)
        db.insert_log(':memory:', row)

    def test_same_validity_as_history_and_analytics(self):
        self.add(fused_conf=.4)
        self.add(face_detected=False, fused_emotion='neutral', fused_conf=1)
        self.add(fused_conf=0)
        self.add(face_detected=False, video_conf=0, audio_emotion='sad', audio_conf=.6,
                 fused_emotion='sad', fused_conf=.6)
        changes = self.conn.total_changes
        rows = db.get_emotion_timeline(':memory:', 'sample')
        self.assertEqual([r['quality'] for r in rows], ['valid', 'no_source', 'unresolved', 'valid'])
        self.assertIsNone(rows[1]['fused_emotion'])
        self.assertIsNone(rows[1]['fused_conf'])
        self.assertIsNone(rows[1]['video_emotion'])
        self.assertEqual(rows[2]['video_emotion'], 'happy')
        self.assertEqual(rows[3]['audio_emotion'], 'sad')
        self.assertEqual(sum(r['quality'] == 'valid' for r in rows), db.get_analytics_summary(':memory:')['valid_logs'])
        self.assertEqual([r['quality'] for r in reversed(rows)], [r['quality'] for r in db.get_history_records(':memory:', 'sample')['items']])
        self.assertEqual(self.conn.total_changes, changes)

    def test_order_ties_future_and_session_isolation(self):
        self.add(timestamp=(self.now + timedelta(days=1)).isoformat())
        self.add()
        self.add()
        self.add(session_id='other')
        rows = db.get_emotion_timeline(':memory:', 'sample')
        self.assertEqual([r['id'] for r in rows], [2, 3])
        self.assertEqual(db.get_emotion_timeline(':memory:', 'missing'), [])

    def test_invalid_values_do_not_break_json_or_become_sources(self):
        self.add(fused_conf=float('inf'), video_conf=float('inf'), audio_emotion='unknown', audio_conf=.8)
        self.add(fused_conf=-1)
        with TestClient(app) as client:
            response = client.get('/api/sessions/sample/timeline')
            self.assertEqual(response.status_code, 200)
            first, second = response.json()
            self.assertEqual(first['quality'], 'no_source')
            for field in ('fused_conf', 'video_conf', 'audio_conf'):
                self.assertIsNone(first[field])
            self.assertEqual(second['quality'], 'unresolved')
            self.assertIsNone(second['fused_conf'])
