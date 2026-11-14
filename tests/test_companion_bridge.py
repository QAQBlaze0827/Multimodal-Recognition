import unittest
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.companion import CompanionBridge, CompanionEvent, router


def payload(**changes):
    data = {"session_id": str(uuid4()), "sequence": 1, "state": "listening",
            "text": "我好難過可不可以陪我一下", "intent": "company",
            "reply": "我在這裡，陪你待一下。"}
    data.update(changes)
    return data


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.state.companion_enabled = True
        self.bridge = self.app.state.companion_bridge = CompanionBridge()
        self.app.include_router(router)
        self.client = TestClient(self.app, base_url="http://127.0.0.1:8000",
                                 client=("127.0.0.1", 12345))

    def test_reply_http_to_websocket_and_stop(self):
        data = payload()
        with self.client.websocket_connect("ws://127.0.0.1:8000/ws/companion", headers={"origin": "http://127.0.0.1:8000"}) as ws:
            self.assertEqual(ws.receive_json()["state"], "waiting")
            self.assertTrue(self.client.post("/api/companion/events", json=data).json()["accepted"])
            ws.send_text("snapshot")
            snapshot = ws.receive_json()
            self.assertEqual(snapshot["message"]["text"], data["text"])
            self.assertEqual(snapshot["message"]["reply"], data["reply"])
            stop = payload(session_id=data["session_id"], sequence=2, state="stopped", text="", reply="", intent="")
            self.client.post("/api/companion/events", json=stop)
            ws.send_text("snapshot")
            snapshot = ws.receive_json()
            self.assertEqual(snapshot["state"], "stopped")
            self.assertIsNone(snapshot["message"])

    def test_duplicate_and_reordered_events_do_not_refresh_lease(self):
        data = payload(sequence=3)
        event = CompanionEvent(**data)
        self.assertTrue(self.bridge.accept(event, 100))
        self.assertFalse(self.bridge.accept(event, 105))
        self.assertFalse(self.bridge.accept(CompanionEvent(**{**data, "sequence": 2}), 106))
        self.assertEqual(self.bridge.snapshot(108)["state"], "stale")
        self.assertIsNone(self.bridge.snapshot(108)["message"])

    def test_heartbeat_retains_latest_reply_but_expires_after_sixty_seconds(self):
        event = CompanionEvent(**payload())
        self.bridge.accept(event, 100)
        self.bridge.accept(event.model_copy(update={"sequence": 2, "text": "", "reply": "", "intent": ""}), 159)
        self.assertIsNotNone(self.bridge.snapshot(159)["message"])
        self.assertIsNone(self.bridge.snapshot(160)["message"])
        self.assertEqual(self.bridge.snapshot(160)["state"], "listening")

    def test_second_sender_rejected_until_stop_or_expiry(self):
        event = CompanionEvent(**payload())
        self.bridge.accept(event, 100)
        with self.assertRaises(HTTPException) as error:
            self.bridge.accept(CompanionEvent(**payload()), 101)
        self.assertEqual(error.exception.status_code, 409)
        self.bridge.accept(CompanionEvent(**payload(text="", reply="", intent="")), 109)
        self.assertIsNone(self.bridge.snapshot(109)["message"])

    def test_disabled_by_default_and_no_db_routes_in_bridge(self):
        self.app.state.companion_enabled = False
        self.assertEqual(self.client.post("/api/companion/events", json=payload()).status_code, 404)
        with self.client.websocket_connect("ws://127.0.0.1:8000/ws/companion", headers={"origin": "http://127.0.0.1:8000"}) as ws:
            with self.assertRaises(WebSocketDisconnect) as error:
                ws.receive_json()
        self.assertEqual(error.exception.code, 4404)
        self.assertFalse(hasattr(self.bridge, "db_path"))

    def test_browser_writes_remote_peers_and_rebinding_hosts_rejected(self):
        for origin in ["http://127.0.0.1:8000", "https://evil.example", "null"]:
            self.assertEqual(self.client.post("/api/companion/events", json=payload(),
                                              headers={"origin": origin}).status_code, 403)
        remote = TestClient(self.app, base_url="http://127.0.0.1:8000", client=("192.168.1.2", 1))
        self.assertEqual(remote.post("/api/companion/events", json=payload(),
                                     headers={"x-forwarded-for": "127.0.0.1"}).status_code, 403)
        self.assertEqual(self.client.post("/api/companion/events", json=payload(),
                                          headers={"host": "evil.example"}).status_code, 403)
        for origin in [None, "null", "https://evil.example", "http://127.0.0.1:9000", "http://[bad"]:
            headers = {} if origin is None else {"origin": origin}
            with self.assertRaises(WebSocketDisconnect):
                with self.client.websocket_connect("ws://127.0.0.1:8000/ws/companion", headers=headers):
                    pass

    def test_schema_and_body_limits(self):
        for changes in [{"sequence": True}, {"sequence": 0}, {"intent": "unknown"},
                        {"reply": ""}, {"text": "x" * 2001}, {"audio": "not allowed"},
                        {"state": "stopped"}, {"text": "   "}]:
            with self.subTest(changes=changes):
                self.assertEqual(self.client.post("/api/companion/events", json=payload(**changes)).status_code, 422)
        self.assertEqual(self.client.post("/api/companion/events", content=b"x" * 16385,
                                          headers={"content-type": "application/json"}).status_code, 413)
        self.assertEqual(self.client.post("/api/companion/events", content="{}").status_code, 415)


if __name__ == "__main__":
    unittest.main()
