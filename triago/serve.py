"""HTTP scoring endpoint — for ticketing systems that can call a webhook (ServiceNow, Jira SM, Zammad, ...).

    python -m triago.serve                  # http://127.0.0.1:8642
    TRIAGO_API_KEY=secret TRIAGO_HOST=0.0.0.0 python -m triago.serve

    curl -s localhost:8642/dispatch -d '{"subject": "VPN drops", "body": "every 10 min since Monday"}'

Stateless: it returns the routing decision as JSON and your system applies it.
Single-threaded on purpose — one model, one request at a time (~1 s each).
"""
from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from .examples import ExampleStore
from .pipeline import Dispatch, Policy, dispatch

HOST = os.environ.get("TRIAGO_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRIAGO_PORT", "8642"))
API_KEY = os.environ.get("TRIAGO_API_KEY", "")
TRIAGE_QUEUE = "triage"


def to_json(d: Dispatch) -> dict:
    """The response contract. `assign_to` is what to do; `group` is the model's best guess either way."""
    imp, urg = d.decisions.get("impact"), d.decisions.get("urgency")
    return {
        "assign_to": d.group if d.auto else TRIAGE_QUEUE,
        "group": d.group,
        "auto": d.auto,
        "priority": d.priority,
        "impact": imp.choice if imp else None,
        "urgency": urg.choice if urg else None,
        "security_incident": d.group == "security",
        "reason": d.reason,
        "probabilities": {k: {c: round(p, 4) for c, p in v.probabilities.items()} for k, v in d.decisions.items()},
        "latency_ms": round(d.latency_ms),
    }


def ticket_text(payload: dict) -> str:
    if payload.get("text"):
        return str(payload["text"])
    text = f"{payload.get('subject', '')}\n{payload.get('body', '')}".strip()
    if not text:
        raise ValueError('send {"subject": ..., "body": ...} or {"text": ...}')
    return text


def make_handler(scorer, policy: Policy, store: ExampleStore | None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if not API_KEY:
                return True
            return hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {API_KEY}")

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "examples": len(store.rows) if store else 0})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/dispatch":
                return self._send(404, {"error": "not found"})
            if not self._authorized():
                return self._send(401, {"error": "missing or wrong Authorization: Bearer <TRIAGO_API_KEY>"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                text = ticket_text(json.loads(self.rfile.read(n) or b"{}"))
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(200, to_json(dispatch(scorer, text, policy, store)))

    return Handler


def main() -> None:
    from .scorer import Scorer
    if HOST not in ("127.0.0.1", "localhost") and not API_KEY:
        raise SystemExit("refusing to listen on a non-local address without TRIAGO_API_KEY")
    scorer, store = Scorer(), ExampleStore.from_data()
    policy = Policy.for_scorer(scorer)
    print(f"triago listening on http://{HOST}:{PORT}  (POST /dispatch, GET /health)", flush=True)
    HTTPServer((HOST, PORT), make_handler(scorer, policy, store)).serve_forever()


if __name__ == "__main__":
    main()
