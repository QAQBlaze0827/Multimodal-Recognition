from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


_local = threading.local()

_QUALITY_SQL = """CASE
    WHEN NOT (
        COALESCE(face_detected = 1 AND video_emotion IN ('neutral','happy','sad','anger')
            AND typeof(video_conf) IN ('real','integer') AND video_conf > 0 AND video_conf <= 1, 0)
        OR COALESCE(audio_emotion IN ('neutral','happy','sad','anger')
            AND typeof(audio_conf) IN ('real','integer') AND audio_conf > 0 AND audio_conf <= 1, 0)
    ) THEN 'no_source'
    WHEN fused_emotion IN ('neutral','happy','sad','anger')
        AND typeof(fused_conf) IN ('real','integer') AND fused_conf > 0 AND fused_conf <= 1
        THEN 'valid'
    ELSE 'unresolved'
END"""


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
    now = datetime.now()
    since = (now - timedelta(days=days)).isoformat()
    # This is a display grouping, not an inference threshold or an accuracy claim.
    low_score_threshold = 0.5
    rows = conn.execute(
        f"""WITH classified AS (
            SELECT *, {_QUALITY_SQL} AS quality
            FROM emotion_logs WHERE timestamp >= ? AND timestamp <= ?
        )
        SELECT quality, CASE WHEN quality = 'valid' THEN fused_emotion END AS emotion,
            COUNT(*) AS cnt,
            AVG(CASE WHEN quality = 'valid' THEN fused_conf END) AS avg_conf,
            SUM(CASE WHEN quality = 'valid' AND fused_conf < ? THEN 1 ELSE 0 END) AS low_count,
            MIN(timestamp) AS first_record, MAX(timestamp) AS last_record,
            (SELECT COUNT(DISTINCT session_id) FROM classified) AS session_count
        FROM classified GROUP BY quality, emotion ORDER BY cnt DESC, emotion""",
        (since, now.isoformat(), low_score_threshold),
    ).fetchall()
    total = sum(r["cnt"] for r in rows)
    valid = sum(r["cnt"] for r in rows if r["quality"] == "valid")
    distribution = {}
    for r in rows:
        if r["quality"] != "valid":
            continue
        distribution[r["emotion"]] = {
            "count": r["cnt"],
            "pct": round(r["cnt"] / valid * 100, 1),
            "avg_conf": round(r["avg_conf"], 3),
        }
    return {
        "total_logs": total,
        "valid_logs": valid,
        "excluded_logs": total - valid,
        "no_source_logs": sum(r["cnt"] for r in rows if r["quality"] == "no_source"),
        "unresolved_logs": sum(r["cnt"] for r in rows if r["quality"] == "unresolved"),
        "low_score_logs": sum(r["low_count"] for r in rows),
        "low_score_threshold": low_score_threshold,
        "session_count": rows[0]["session_count"] if rows else 0,
        "first_record": min((r["first_record"] for r in rows), default=None),
        "last_record": max((r["last_record"] for r in rows), default=None),
        "distribution": distribution,
    }


def get_history_sessions(db_path: str, days: int = 30, limit: int = 10,
                         offset: int = 0) -> dict[str, Any]:
    conn = _get_conn(db_path)
    now = datetime.now()
    params = ((now - timedelta(days=days)).isoformat(), now.isoformat())
    cte = f"""WITH classified AS (
        SELECT *, {_QUALITY_SQL} AS quality FROM emotion_logs
        WHERE timestamp >= ? AND timestamp <= ?
    ) """
    total = conn.execute(cte + "SELECT COUNT(DISTINCT session_id) FROM classified", params).fetchone()[0]
    counts = ", ".join(
        f"SUM(CASE WHEN quality = 'valid' AND fused_emotion = '{emotion}' THEN 1 ELSE 0 END) AS {emotion}"
        for emotion in ('neutral', 'happy', 'sad', 'anger')
    )
    rows = conn.execute(cte + f"""SELECT session_id,
        MIN(timestamp) AS first_record, MAX(timestamp) AS last_record,
        COUNT(*) AS total_logs, SUM(quality = 'valid') AS valid_logs,
        SUM(quality = 'no_source') AS no_source_logs,
        SUM(quality = 'unresolved') AS unresolved_logs,
        SUM(quality = 'valid' AND fused_conf < 0.5) AS low_score_logs, {counts}
        FROM classified GROUP BY session_id
        ORDER BY last_record DESC, session_id LIMIT ? OFFSET ?""", (*params, limit, offset)).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item['distribution'] = {key: item.pop(key) for key in ('neutral', 'happy', 'sad', 'anger')}
        items.append(item)
    return {'total': total, 'items': items}


def get_history_records(db_path: str, session_id: str, days: int = 30,
                        limit: int = 20, offset: int = 0) -> dict[str, Any]:
    conn = _get_conn(db_path)
    now = datetime.now()
    params = (session_id, (now - timedelta(days=days)).isoformat(), now.isoformat())
    where = "WHERE session_id = ? AND timestamp >= ? AND timestamp <= ?"
    total = conn.execute(f"SELECT COUNT(*) FROM emotion_logs {where}", params).fetchone()[0]
    rows = conn.execute(f"""WITH classified AS (
        SELECT *, {_QUALITY_SQL} AS quality FROM emotion_logs {where}
        ) SELECT id, timestamp, quality,
        CASE WHEN quality = 'valid' THEN fused_emotion END AS fused_emotion,
        CASE WHEN quality = 'valid' THEN fused_conf END AS fused_conf FROM classified
        ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?""", (*params, limit, offset)).fetchall()
    return {'total': total, 'items': [dict(row) for row in rows]}


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
