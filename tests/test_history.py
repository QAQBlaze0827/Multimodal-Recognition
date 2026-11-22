import sqlite3
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import database as db
from backend.app import app


class HistoryTests(unittest.TestCase):
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

    def test_counts_match_analytics_without_writes(self):
        self.add()
        self.add(fused_emotion='sad', fused_conf=.49)
        self.add(fused_conf=.5)
        self.add(face_detected=False, video_conf=0, fused_emotion='neutral', fused_conf=1)
        self.add(fused_conf=0)
        self.add(session_id='audio', face_detected=False, video_conf=0,
                 audio_emotion='anger', audio_conf=.7, fused_emotion='anger')
        changes = self.conn.total_changes
        summary = db.get_history_sessions(':memory:')
        analytics = db.get_analytics_summary(':memory:')
        self.assertEqual(summary['total'], 2)
        for field in ('total_logs', 'valid_logs', 'low_score_logs', 'no_source_logs', 'unresolved_logs'):
            self.assertEqual(sum(item[field] for item in summary['items']), analytics[field])
        details = db.get_history_records(':memory:', 'sample')
        self.assertEqual([r['quality'] for r in details['items']], ['unresolved', 'no_source', 'valid', 'valid', 'valid'])
        self.assertEqual(self.conn.total_changes, changes)

    def test_pagination_deterministic_and_missing_session(self):
        for index in range(25):
            self.add()
        first = db.get_history_records(':memory:', 'sample', limit=20)
        second = db.get_history_records(':memory:', 'sample', limit=20, offset=20)
        self.assertEqual(first['total'], 25)
        self.assertEqual(len(second['items']), 5)
        self.assertFalse({r['id'] for r in first['items']} & {r['id'] for r in second['items']})
        self.assertEqual(db.get_history_records(':memory:', 'missing'), {'total': 0, 'items': []})
        self.add(session_id='another')
        pages = [db.get_history_sessions(':memory:', limit=1, offset=n) for n in (0, 1)]
        self.assertNotEqual(pages[0]['items'][0]['session_id'], pages[1]['items'][0]['session_id'])

    def test_period_applies_to_both_summary_and_details(self):
        self.add(timestamp=(self.now - timedelta(days=40)).isoformat())
        self.add(timestamp=(self.now + timedelta(days=1)).isoformat())
        self.add()
        summary = db.get_history_sessions(':memory:')['items'][0]
        self.assertEqual(summary['total_logs'], 1)
        self.assertEqual(summary['first_record'], self.now.isoformat())
        self.assertEqual(db.get_history_records(':memory:', 'sample')['total'], 1)

    def test_empty_and_all_excluded(self):
        self.assertEqual(db.get_history_sessions(':memory:'), {'total': 0, 'items': []})
        self.add(face_detected=False, video_conf=0, fused_conf=1)
        summary = db.get_history_sessions(':memory:')['items'][0]
        self.assertEqual(summary['valid_logs'], 0)
        self.assertEqual(sum(summary['distribution'].values()), 0)

    def test_api_validation_and_session_identifier(self):
        session = 'special/<>&?'
        self.add(session_id=session)
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/history/sessions').json()['total'], 1)
            self.assertEqual(client.get('/api/history/records', params={'session_id': session}).json()['total'], 1)
            for url in ('/api/history/sessions?days=0', '/api/history/sessions?limit=101',
                        '/api/history/records', '/api/history/records?session_id=x&offset=-1'):
                self.assertEqual(client.get(url).status_code, 422)

    def test_invalid_nonfinite_scores_are_safe_for_json(self):
        self.add(fused_conf=float('inf'))
        with TestClient(app) as client:
            response = client.get('/api/history/records', params={'session_id': 'sample'})
            self.assertEqual(response.status_code, 200)
            record = response.json()['items'][0]
            self.assertEqual(record['quality'], 'unresolved')
            self.assertIsNone(record['fused_conf'])
            self.assertIsNone(record['fused_emotion'])
