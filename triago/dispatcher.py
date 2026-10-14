"""Dispatcher service: polls GLPI for New tickets, scores them, writes routing back.

    python -m triago.dispatcher run        # poll loop (every 10s)
    python -m triago.dispatcher once       # process the current queue and exit
    python -m triago.dispatcher feedback   # export human reassignments as labeled data (GLPI)
    python -m triago.dispatcher run --connector mypkg.mod:Class   # any ticketing system, see connector.py
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

from . import connector as connectors
from .connector import Connector
from .examples import ExampleStore
from .glpi import GLPI, GROUP_NAMES, plain as _plain
from .pipeline import Policy, dispatch
from .scorer import Scorer

MARK = "[TRIAGO]"
OLD_MARKS = ("[JEV-DISPATCH]",)  # notes written before the rename
FEEDBACK_FILE = Path(__file__).resolve().parent.parent / "data" / "feedback.jsonl"


def _note(d) -> str:
    lines = [f"{MARK} group={d.group} priority=P{d.priority} mode={'AUTO' if d.auto else 'TRIAGE'}",
             f"why: {d.reason}"]
    for name, dec in d.decisions.items():
        probs = " ".join(f"{k}={v:.2f}" for k, v in sorted(dec.probabilities.items(), key=lambda kv: -kv[1]))
        lines.append(f"{name}: {probs}")
    lines.append(f"scored in {d.latency_ms:.0f} ms (no tokens generated)")
    return "<br>".join(html.escape(l) for l in lines)


def process_queue(conn: Connector, scorer: Scorer, policy: Policy, store: ExampleStore | None = None) -> int:
    n = 0
    for t in conn.pending():
        d = dispatch(scorer, t.text, policy, store)
        conn.apply(t, d, _note(d))
        print(f"#{t.id!s:>6} {'AUTO  ' if d.auto else 'TRIAGE'} {d.group:<16} P{d.priority} "
              f"{d.latency_ms:6.0f}ms  {t.subject[:60]}", flush=True)
        n += 1
    return n


def export_feedback(glpi: GLPI) -> None:
    """Tickets where a human moved the ticket away from (or out of triage into) a group
    are ground-truth labels — the raw material for calibration and threshold tuning."""
    id_to_key = {}
    groups = glpi.groups()
    for key, name in GROUP_NAMES.items():
        if name in groups and key != "other":
            id_to_key[groups[name]] = key
    out = []
    for t in glpi.all_tickets():
        notes = [f for f in glpi.followups(t["id"])
                 if any(m in html.unescape(f["content"]) for m in (MARK, *OLD_MARKS))]
        if not notes:
            continue
        predicted = re.search(r"group=(\w+)", html.unescape(notes[0]["content"])).group(1)
        current = [id_to_key[g] for g in glpi.assigned_groups(t["id"]) if g in id_to_key]
        if current:
            out.append({"text": f"{t['name']}\n{_plain(t['content'])}", "predicted": predicted,
                        "group": current[-1], "corrected": current[-1] != predicted})
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    FEEDBACK_FILE.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {len(out)} labeled tickets ({sum(r['corrected'] for r in out)} corrections) -> {FEEDBACK_FILE}")


def _connect(spec: str | None) -> Connector:
    if spec:
        return connectors.load(spec)
    glpi = GLPI()
    glpi.ensure_groups()
    return glpi


def main() -> None:
    args = sys.argv[1:]
    spec = None
    if "--connector" in args:
        i = args.index("--connector")
        spec = args[i + 1]
        del args[i:i + 2]
    cmd = args[0] if args else "run"
    conn = _connect(spec)
    if cmd == "feedback":
        if not isinstance(conn, GLPI):
            sys.exit("feedback export is GLPI-only; for other systems write data/feedback.jsonl yourself "
                     '(one {"text": ..., "group": ...} per line)')
        return export_feedback(conn)
    scorer, store = Scorer(), ExampleStore.from_data()
    policy = Policy.for_scorer(scorer)
    print(f"example store: {len(store.rows)} labeled tickets (incl. feedback)", flush=True)
    if cmd == "once":
        print(f"processed {process_queue(conn, scorer, policy, store)} tickets")
        return
    print("dispatcher polling every 10s — Ctrl+C to stop", flush=True)
    while True:
        try:
            process_queue(conn, scorer, policy, store)
        except Exception as e:  # keep the loop alive; sessions expire
            print(f"error: {e}; reconnecting", flush=True)
            conn = _connect(spec)
        time.sleep(10)


if __name__ == "__main__":
    main()
