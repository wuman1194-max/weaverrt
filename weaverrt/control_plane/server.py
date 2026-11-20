"""Control Plane REST API & Live Observability Console (zero-dependency ThreadingHTTPServer)."""

from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import json
import threading

from weaverrt.control_plane.registry import AgentArtifact, ControlPlaneStore
from weaverrt.runtime.store import RuntimeStore


DASHBOARD_HTML = Path(__file__).parent / "dashboard.html"


class ControlPlaneServer:
    def __init__(
        self,
        runtime_store: RuntimeStore,
        cp_store: ControlPlaneStore,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.runtime_store = runtime_store
        self.cp_store = cp_store
        self.host = host
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self.port = int(self._httpd.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def _make_handler(self):
        rt = self.runtime_store
        cp = self.cp_store

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, code: int, body: bytes | str, ctype: str = "application/json; charset=utf-8") -> None:
                raw = body.encode("utf-8") if isinstance(body, str) else body
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(raw)

            def _json(self, code: int, obj: Any) -> None:
                self._send(code, json.dumps(obj, ensure_ascii=False))

            def _read_json(self) -> dict[str, Any]:
                n = int(self.headers.get("Content-Length") or "0")
                if n <= 0:
                    return {}
                return json.loads(self.rfile.read(n).decode("utf-8"))

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    html = DASHBOARD_HTML.read_text(encoding="utf-8") if DASHBOARD_HTML.exists() else "<h1>WeaverRT</h1>"
                    return self._send(200, html, "text/html; charset=utf-8")
                if path == "/api/artifacts":
                    return self._json(200, {"artifacts": [a.to_dict() for a in cp.list_artifacts()]})
                if path == "/api/sessions":
                    return self._json(
                        200,
                        {"sessions": [dict(asdict(s), status=s.status.value) for s in rt.list_sessions()]},
                    )
                if path.startswith("/api/sessions/") and path.endswith("/events"):
                    sid = path.split("/")[3]
                    evs = [
                        {
                            "seq": e.seq,
                            "turn_index": e.turn_index,
                            "event_type": e.event_type.value,
                            "payload": e.payload,
                            "created_at_ms": e.created_at_ms,
                        }
                        for e in rt.get_events(sid)
                    ]
                    return self._json(200, {"session_id": sid, "events": evs})
                if path.startswith("/api/sessions/") and path.endswith("/traces"):
                    sid = path.split("/")[3]
                    spans = [asdict(sp) for sp in rt.get_traces(sid)]
                    return self._json(200, {"session_id": sid, "traces": spans})
                if path == "/api/approvals":
                    return self._json(200, {"approvals": rt.list_approvals()})
                if path == "/api/evals":
                    return self._json(200, {"eval_runs": cp.list_eval_runs()})
                if path == "/api/bad_cases":
                    return self._json(200, {"bad_cases": cp.list_bad_cases()})
                return self._json(404, {"error": f"Not found: {path}"})

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                body = self._read_json()
                if path == "/api/artifacts":
                    art = AgentArtifact.from_dict(body)
                    cp.register_artifact(art, actor=str(body.get("actor", "api")))
                    return self._json(201, {"artifact": art.to_dict()})
                if path == "/api/artifacts/release":
                    ok, msg = cp.promote_to_release(str(body["name"]), str(body["version"]), actor=str(body.get("actor", "api")))
                    return self._json(200 if ok else 409, {"ok": ok, "message": msg})
                if path == "/api/artifacts/rollback":
                    ok, msg = cp.rollback(str(body["name"]), str(body["version"]), actor=str(body.get("actor", "api")))
                    return self._json(200 if ok else 409, {"ok": ok, "message": msg})
                if path.startswith("/api/approvals/") and path.endswith("/resolve"):
                    aid = path.split("/")[3]
                    ok = rt.resolve_approval(aid, bool(body.get("approved", False)), str(body.get("decided_by", "console")))
                    return self._json(200 if ok else 404, {"ok": ok, "approval_id": aid})
                return self._json(404, {"error": f"Not found: {path}"})

        return Handler
