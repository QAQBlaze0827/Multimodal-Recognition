from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


_local = threading.local()


def _get_conn(db_path: str) -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(db_path, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


def init_db(db_path: str | Path) -> None:
    db_path = str(db_path)
    conn = _get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT,
            total_frames INTEGER DEFAULT 0,
            notes TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS emotion_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            timestamp TEXT NOT NULL,
            face_detected INTEGER DEFAULT 0,
            face_backend TEXT DEFAULT '',
            video_emotion TEXT DEFAULT '',
            video_conf REAL DEFAULT 0.0,
            audio_emotion TEXT DEFAULT '',
            audio_conf REAL DEFAULT 0.0,
            fused_emotion TEXT NOT NULL,
            fused_conf REAL NOT NULL,
            fps REAL DEFAULT 0.0,
            cpu_temp REAL,
            video_scores_json TEXT DEFAULT '{}',
            audio_scores_json TEXT DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_logs_session ON emotion_logs(session_id);
        CREATE INDEX IF NOT EXISTS idx_logs_ts ON emotion_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_logs_emotion ON emotion_logs(fused_emotion);

        CREATE TABLE IF NOT EXISTS emotion_summary (
            session_id TEXT PRIMARY KEY REFERENCES sessions(id),
            emotion_counts TEXT NOT NULL DEFAULT '{}',
            avg_confidence REAL DEFAULT 0.0,
            total_duration_sec REAL DEFAULT 0.0
        );
    """)
    conn.commit()


def create_session(db_path: str, session_id: str, notes: str = "") -> None:
    conn = _get_conn(db_path)
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, start_time, notes) VALUES (?, ?, ?)",
        (session_id, datetime.now().isoformat(timespec="milliseconds"), notes),
    )
    conn.commit()


def end_session(db_path: str, session_id: str, total_frames: int = 0) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        "UPDATE sessions SET end_time=?, total_frames=? WHERE id=?",
        (datetime.now().isoformat(timespec="milliseconds"), total_frames, session_id),
    )
    conn.commit()


def insert_log(db_path: str, row: dict[str, Any]) -> None:
    conn = _get_conn(db_path)
    conn.execute(
        """INSERT INTO emotion_logs (
            session_id, timestamp, face_detected, face_backend,
            video_emotion, video_conf, audio_emotion, audio_conf,
            fused_emotion, fused_conf, fps, cpu_temp,
            video_scores_json, audio_scores_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row["session_id"],
            row.get("timestamp", datetime.now().isoformat(timespec="milliseconds")),
            int(row.get("face_detected", False)),
            row.get("face_backend", ""),
            row.get("video_emotion", ""),
            row.get("video_conf", 0.0),
            row.get("audio_emotion", ""),
            row.get("audio_conf", 0.0),
            row["fused_emotion"],
            row["fused_conf"],
            row.get("fps", 0.0),
            row.get("cpu_temp"),
            json.dumps(row.get("video_scores", {})),
            json.dumps(row.get("audio_scores", {})),
        ),
    )
    conn.commit()


def query_logs(
    db_path: str,
    session_id: str | None = None,
    emotion: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    parts = ["SELECT * FROM emotion_logs"]
    params: list[Any] = []
    conditions = []
    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    if emotion:
        conditions.append("fused_emotion = ?")
        params.append(emotion)
    if conditions:
        parts.append("WHERE " + " AND ".join(conditions))
    parts.append("ORDER BY timestamp DESC LIMIT ? OFFSET ?")
    params.extend([limit, offset])
    rows = conn.execute(" ".join(parts), params).fetchall()
    return [dict(r) for r in rows]


def get_recent_logs(db_path: str, limit: int = 100) -> list[dict[str, Any]]:
    return query_logs(db_path, limit=limit)


def get_latest_log_id(db_path: str) -> int:
    conn = _get_conn(db_path)
    row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM emotion_logs").fetchone()
    return int(row[0] or 0)


def get_logs_after_id(db_path: str, last_id: int, limit: int = 100) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM emotion_logs WHERE id > ? ORDER BY id ASC LIMIT ?",
        (last_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_session_logs_for_replay(db_path: str, session_id: str) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT * FROM emotion_logs WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_sessions(db_path: str) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        """SELECT s.*, COUNT(l.id) as log_count,
                  COALESCE(SUM(CASE WHEN l.face_detected THEN 1 ELSE 0 END), 0) as face_count
           FROM sessions s
           LEFT JOIN emotion_logs l ON s.id = l.session_id
           GROUP BY s.id
           ORDER BY s.start_time DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def get_session_stats(db_path: str, session_id: str) -> dict[str, Any]:
    conn = _get_conn(db_path)
    row = conn.execute(
        """SELECT fused_emotion, COUNT(*) as cnt,
                  ROUND(AVG(fused_conf), 3) as avg_conf,
                  ROUND(AVG(fps), 1) as avg_fps
           FROM emotion_logs
           WHERE session_id = ?
           GROUP BY fused_emotion
           ORDER BY cnt DESC""",
        (session_id,),
    ).fetchall()
    total = sum(r["cnt"] for r in row)
    emotion_dist = {}
    for r in row:
        emotion_dist[r["fused_emotion"]] = {
            "count": r["cnt"],
            "pct": round(r["cnt"] / total * 100, 1) if total else 0,
            "avg_conf": r["avg_conf"],
        }
    return {
        "total_logs": total,
        "emotion_distribution": emotion_dist,
        "avg_fps": row[0]["avg_fps"] if row else 0,
    }


def get_emotion_timeline(db_path: str, session_id: str) -> list[dict[str, Any]]:
    conn = _get_conn(db_path)
    rows = conn.execute(
        """SELECT timestamp, fused_emotion, fused_conf, video_scores_json, audio_scores_json
           FROM emotion_logs
           WHERE session_id = ?
           ORDER BY timestamp ASC""",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_analytics_summary(db_path: str, days: int = 30) -> dict[str, Any]:
    conn = _get_conn(db_path)
    since = (datetime.now() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        """SELECT fused_emotion, COUNT(*) as cnt, AVG(fused_conf) as avg_conf
           FROM emotion_logs
           WHERE timestamp >= ?
           GROUP BY fused_emotion
           ORDER BY cnt DESC""",
        (since,),
    ).fetchall()
    total = sum(r["cnt"] for r in rows)
    distribution = {}
    for r in rows:
        distribution[r["fused_emotion"]] = {
            "count": r["cnt"],
            "pct": round(r["cnt"] / total * 100, 1) if total else 0,
            "avg_conf": round(r["avg_conf"], 3),
        }
    session_count = conn.execute(
        "SELECT COUNT(*) FROM sessions WHERE start_time >= ?", (since,)
    ).fetchone()[0]
    return {
        "total_logs": total,
        "session_count": session_count,
        "distribution": distribution,
    }


def cleanup_old_sessions(db_path: str, retain_days: int = 30) -> int:
    conn = _get_conn(db_path)
    cutoff = (datetime.now() - timedelta(days=retain_days)).isoformat()
    deleted = conn.execute(
        "DELETE FROM emotion_logs WHERE timestamp < ?", (cutoff,)
    ).rowcount
    conn.execute("DELETE FROM sessions WHERE start_time < ?", (cutoff,))
    conn.execute(
        "DELETE FROM emotion_summary WHERE session_id NOT IN (SELECT id FROM sessions)"
    )
    conn.commit()
    return deleted
