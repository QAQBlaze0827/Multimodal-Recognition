"""Opt-in loopback-only text bridge. No audio or database persistence."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

router = APIRouter()
SOURCE_TTL = 8.0
REPLY_TTL = 60.0
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


class CompanionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: UUID
    sequence: int = Field(strict=True, ge=1)
    state: Literal["listening", "stopped"]
    text: str = Field(default="", max_length=2000)
    reply: str = Field(default="", max_length=500)
    intent: Literal["", "company", "play", "rest", "quiet", "clarify"] = ""

    @model_validator(mode="after")
    def check_reply(self):
        values = (bool(self.text.strip()), bool(self.reply.strip()), bool(self.intent))
        if any(values) and (not all(values) or self.state != "listening"):
            raise ValueError("Replies require text, intent, reply and listening state")
        return self


class CompanionBridge:
    def __init__(self):
        self.session_id = None
        self.sequence = 0
        self.last_seen = 0.0
        self.state = "waiting"
        self.reply = None
        self.reply_at = 0.0

    def accept(self, event: CompanionEvent, now: float):
        if event.session_id != self.session_id:
            if self.state == "listening" and now - self.last_seen < SOURCE_TTL:
                raise HTTPException(409, "Another transcription process is active")
            self.session_id = event.session_id
            self.sequence = 0
            self.reply = None
        if event.sequence <= self.sequence:
            return False
        self.sequence = event.sequence
        self.last_seen = now
        self.state = event.state
        if event.reply:
            self.reply = {"id": f"{event.session_id}:{event.sequence}",
                          "text": event.text, "reply": event.reply, "intent": event.intent,
                          "received_at": datetime.now(timezone.utc).isoformat()}
            self.reply_at = now
        if event.state == "stopped":
            self.reply = None
        return True

    def snapshot(self, now: float):
        remaining = max(0.0, SOURCE_TTL - (now - self.last_seen))
        state = self.state
        if state == "listening" and not remaining:
            state = "stale"
        if state != "listening" or now - self.reply_at >= REPLY_TTL:
            self.reply = None
        return {"type": "companion", "state": state,
                "expires_in": remaining if state == "listening" else 0,
                "message": self.reply}


def local_client(connection):
    return (connection.client is not None
            and connection.client.host in {"127.0.0.1", "::1"}
            and connection.url.hostname in LOCAL_HOSTS)


@router.post("/api/companion/events")
async def receive_event(request: Request):
    if not getattr(request.app.state, "companion_enabled", False):
        raise HTTPException(404, "Local companion bridge is disabled")
    # This endpoint is for the local Python sender, never browser-originated writes.
    if not local_client(request) or request.headers.get("origin") is not None:
        raise HTTPException(403, "Local Python sender only")
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, "Expected application/json")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 16384:
            raise HTTPException(413, "Event too large")
    try:
        event = CompanionEvent.model_validate_json(bytes(body))
    except ValidationError:
        raise HTTPException(422, "Invalid companion event") from None
    accepted = request.app.state.companion_bridge.accept(event, time.monotonic())
    return {"accepted": accepted}


@router.websocket("/ws/companion")
async def companion_socket(ws: WebSocket):
    if not getattr(ws.app.state, "companion_enabled", False):
        await ws.accept()
        await ws.close(code=4404)
        return
    try:
        origin = urlsplit(ws.headers.get("origin", ""))
    except ValueError:
        await ws.close(code=1008)
        return
    if (not local_client(ws) or origin.scheme not in {"http", "https"}
            or origin.netloc != ws.headers.get("host")):
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        while True:
            snapshot = ws.app.state.companion_bridge.snapshot(time.monotonic())
            await asyncio.wait_for(ws.send_json(snapshot), timeout=2)
            # A one-second snapshot also expires dead senders without a database poll.
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=1)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
        pass
