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
from pathlib import Path

from .examples import ExampleStore
from .pipeline import GROUPS, IMPACT, URGENCY, Dispatch, Policy, dispatch, recalibrate

PAGE = Path(__file__).with_name("static") / "index.html"

HOST = os.environ.get("TRIAGO_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRIAGO_PORT", "8642"))
API_KEY = os.environ.get("TRIAGO_API_KEY", "")
TRIAGE_QUEUE = "triage"


def to_json(d: Dispatch, policy: Policy | None = None) -> dict:
    """The response contract. `assign_to` is what to do; `group` is the model's best guess either way.
    `security_p` is the calibrated probability the security gate compared with `security_trip`."""
    imp, urg = d.decisions.get("impact"), d.decisions.get("urgency")
    policy = policy or Policy()
    sec = d.decisions.get("security")
    sec_p = (recalibrate(sec, policy.security_temperature).probabilities["security"]
             if sec is not None and sec.logits else None)
    return {
        "assign_to": d.group if d.auto else TRIAGE_QUEUE,
        "group": d.group,
        "auto": d.auto,
        "priority": d.priority,
        "impact": imp.choice if imp else None,
        "urgency": urg.choice if urg else None,
        "security_incident": d.group == "security",
        "security_p": round(sec_p, 4) if sec_p is not None else None,
        "security_trip": policy.security_trip,
        "reason": d.reason,
        "probabilities": {k: {c: round(p, 4) for c, p in v.probabilities.items()} for k, v in d.decisions.items()},
        "latency_ms": round(d.latency_ms),
    }


def info(scorer, policy: Policy, store: ExampleStore | None) -> dict:
    """What the UI needs to explain a decision: who the teams are and what the rules are."""
    return {
        "model": scorer.model_path,
        "backend": getattr(scorer.backend, "name", "?"),
        "device": getattr(scorer.backend, "device", None),
        "adapter": scorer.adapter_path,
        "examples": len(store.rows) if store else 0,
        "auth": bool(API_KEY),
        "policy": {"min_top": policy.min_top, "min_margin": policy.min_margin,
                   "security_trip": policy.security_trip, "shots": policy.shots},
        "groups": {k: v.split(":")[0] for k, v in GROUPS.items()} | {"security": "Security Operations (SOC)",
                                                                       "other": "Not IT / no action"},
        "impact": IMPACT, "urgency": URGENCY,
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
            if self.path in ("/", "/index.html"):
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if self.path == "/health":
                return self._send(200, {"ok": True, "examples": len(store.rows) if store else 0})
            if self.path == "/info":
                return self._send(200, info(scorer, policy, store))
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
            d = dispatch(scorer, text, policy, store)
            out = to_json(d, policy)
            if d.group != "security" and store is not None and policy.shots:
                # same deterministic lookup the group decision used
                out["examples"] = [{"text": t, "group": g} for t, g in store.nearest(text, policy.shots, allowed=GROUPS)]
            self._send(200, out)

    return Handler


def main() -> None:
    from .scorer import Scorer
    if HOST not in ("127.0.0.1", "localhost") and not API_KEY:
        raise SystemExit("refusing to listen on a non-local address without TRIAGO_API_KEY")
    scorer, store = Scorer(), ExampleStore.from_data()
    policy = Policy.for_scorer(scorer)
    print(f"triago listening on http://{HOST}:{PORT}  (UI at /, POST /dispatch, GET /health)", flush=True)
    HTTPServer((HOST, PORT), make_handler(scorer, policy, store)).serve_forever()


if __name__ == "__main__":
    main()
