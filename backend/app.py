from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as script: python backend/app.py
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.companion import CompanionBridge, LOCAL_HOSTS, router as companion_router
from backend.database import (
    cleanup_old_sessions,
    create_session,
    get_all_sessions,
    get_analytics_summary,
    get_emotion_timeline,
    get_history_sessions,
    get_history_records,
    get_latest_log_id,
    get_logs_after_id,
    get_recent_logs,
    get_session_logs_for_replay,
    get_session_stats,
    init_db,
    insert_log,
    query_logs,
)

DB_PATH: str = ""
CONNECTIONS: set[WebSocket] = set()
POLL_TASK: asyncio.Task | None = None


class EmotionLogIn(BaseModel):
    session_id: str
    timestamp: str | None = None
    face_detected: bool = False
    face_backend: str = ""
    video_emotion: str = ""
    video_conf: float = 0.0
    audio_emotion: str = ""
    audio_conf: float = 0.0
    fused_emotion: str = "neutral"
    fused_conf: float = 0.0
    fps: float = 0.0
    cpu_temp: float | None = None
    video_scores: dict[str, float] = {}
    audio_scores: dict[str, float] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global POLL_TASK
    init_db(DB_PATH)
    POLL_TASK = asyncio.create_task(_poll_database_logs())
    try:
        yield
    finally:
        POLL_TASK.cancel()
        try:
            await POLL_TASK
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Multimodal Emotion Recognition API", version="1.0.0", lifespan=lifespan)
app.state.companion_enabled = False
app.state.companion_bridge = CompanionBridge()
app.include_router(companion_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "db_path": DB_PATH}


@app.post("/api/emotions")
async def post_emotion(data: EmotionLogIn) -> dict[str, str]:
    row = data.model_dump()
    if row["timestamp"] is None:
        row["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
    create_session(DB_PATH, row["session_id"], "api")
    insert_log(DB_PATH, row)
    return {"status": "ok"}


@app.get("/api/emotions")
async def get_emotions(
    session_id: str | None = Query(None),
    emotion: str | None = Query(None),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    return query_logs(DB_PATH, session_id, emotion, limit, offset)


@app.get("/api/emotions/recent")
async def recent_emotions(n: int = Query(100, ge=1, le=10000)) -> list[dict[str, Any]]:
    return get_recent_logs(DB_PATH, n)


@app.get("/api/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return get_all_sessions(DB_PATH)


@app.get("/api/sessions/{session_id}")
async def session_detail(session_id: str) -> dict[str, Any]:
    stats = get_session_stats(DB_PATH, session_id)
    if stats["total_logs"] == 0:
        raise HTTPException(404, "Session not found")
    return stats


@app.get("/api/sessions/{session_id}/logs")
async def session_logs(session_id: str) -> list[dict[str, Any]]:
    return get_session_logs_for_replay(DB_PATH, session_id)


@app.get("/api/sessions/{session_id}/timeline")
async def session_timeline(session_id: str) -> list[dict[str, Any]]:
    return get_emotion_timeline(DB_PATH, session_id)


@app.get("/api/analytics")
async def analytics(days: int = Query(30, ge=1, le=365)) -> dict[str, Any]:
    return get_analytics_summary(DB_PATH, days)


@app.post("/api/cleanup")
async def cleanup(days: int = Query(30, ge=1)) -> dict[str, Any]:
    deleted = cleanup_old_sessions(DB_PATH, days)
    return {"deleted": deleted}


@app.get("/api/history/sessions")
async def history_sessions(days: int = Query(30, ge=1, le=365),
                           limit: int = Query(10, ge=1, le=100),
                           offset: int = Query(0, ge=0)) -> dict[str, Any]:
    return get_history_sessions(DB_PATH, days, limit, offset)


@app.get("/api/history/records")
async def history_records(session_id: str = Query(..., min_length=1),
                          days: int = Query(30, ge=1, le=365),
                          limit: int = Query(20, ge=1, le=100),
                          offset: int = Query(0, ge=0)) -> dict[str, Any]:
    return get_history_records(DB_PATH, session_id, days, limit, offset)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    CONNECTIONS.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        CONNECTIONS.discard(ws)


async def _broadcast(data: dict[str, Any]) -> None:
    if not CONNECTIONS:
        return
    message = json.dumps(data, default=str)
    dead: set[WebSocket] = set()
    for ws in list(CONNECTIONS):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    CONNECTIONS.difference_update(dead)


async def _poll_database_logs(interval_seconds: float = 0.25) -> None:
    last_id = get_latest_log_id(DB_PATH)
    while True:
        try:
            rows = get_logs_after_id(DB_PATH, last_id, 100)
            for row in rows:
                last_id = max(last_id, int(row["id"]))
                await _broadcast(row)
        except Exception as exc:
            print(f"[backend] database poll failed: {exc}")
        await asyncio.sleep(interval_seconds)


def serve(host: str = "0.0.0.0", port: int = 8000, db_path: str = "data/emotion.db",
          companion_local: bool = False) -> None:
    global DB_PATH
    if companion_local and host not in LOCAL_HOSTS:
        raise ValueError("--companion-local requires --host 127.0.0.1, localhost or ::1")
    DB_PATH = db_path
    app.state.companion_enabled = companion_local
    app.state.companion_bridge = CompanionBridge()
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    uvicorn.run(app, host=host, port=port, log_level="info", proxy_headers=not companion_local)


def main() -> None:
    parser = argparse.ArgumentParser(description="Emotion recognition backend server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default="data/emotion.db")
    parser.add_argument("--companion-local", action="store_true",
                        help="Enable the laptop-only text bridge (requires a loopback host)")
    args = parser.parse_args()
    if args.companion_local and args.host not in LOCAL_HOSTS:
        parser.error("--companion-local requires --host 127.0.0.1, localhost or ::1")
    serve(args.host, args.port, args.db, args.companion_local)


if __name__ == "__main__":
    main()
