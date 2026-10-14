"""Retrieved examples: the k most similar labeled tickets, shown to the model before the ticket.

This is how the dispatcher learns your team conventions without training: every human
reassignment exported from GLPI (data/feedback.jsonl) becomes a new example.

Similarity is TF-IDF cosine over character 3-5-grams: no extra dependencies, robust to
typos ("pwd", "acces"), and partly language-independent through shared terms (VPN, SAP, sudo).
"""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_CHARS = 200  # examples are truncated: they only need to show the gist, and they cost latency


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" | ", "\n").lower()).strip()


def _grams(text: str) -> collections.Counter:
    t = f" {_norm(text)} "
    return collections.Counter(t[i:i + n] for n in (3, 4, 5) for i in range(len(t) - n + 1))


class ExampleStore:
    def __init__(self, rows: list[tuple[str, str]]):
        """rows: (ticket text, correct group key). Later rows win on duplicate text (corrections)."""
        uniq = {_norm(t): (t, g) for t, g in rows}
        self.rows = list(uniq.values())
        docs = [_grams(t) for t, _ in self.rows]
        df = collections.Counter(g for d in docs for g in d)
        n = len(docs)
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1 for g, c in df.items()}
        self.vecs = [self._unit(d) for d in docs]

    def _unit(self, counts: collections.Counter) -> dict[str, float]:
        v = {g: (1 + math.log(c)) * self.idf.get(g, 0.0) for g, c in counts.items()}
        z = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / z for g, x in v.items() if x}

    def nearest(self, text: str, k: int = 3, allowed=None) -> list[tuple[str, str]]:
        """Top-k (text, group) by cosine similarity. The ticket itself is never returned
        (leave-one-out), so the store can be built from the same data it is evaluated on."""
        if k <= 0 or not self.rows:
            return []
        q, me = self._unit(_grams(text)), _norm(text)
        scored = []
        for (t, g), v in zip(self.rows, self.vecs):
            if (allowed and g not in allowed) or _norm(t) == me:
                continue
            small, big = (q, v) if len(q) < len(v) else (v, q)
            scored.append((sum(x * big.get(gr, 0.0) for gr, x in small.items()), t, g))
        scored.sort(key=lambda s: -s[0])
        return [(t[:MAX_CHARS], g) for _, t, g in scored[:k]]

    @classmethod
    def from_data(cls) -> "ExampleStore":
        """Production store: all labeled tickets + exported GLPI feedback (human corrections last)."""
        rows = []
        for f in [DATA_DIR / "tickets.jsonl", *sorted((DATA_DIR / "gen").glob("*.jsonl"))]:
            if f.exists():
                rows += [(c["t"], c["g"]) for c in map(json.loads, filter(str.strip, f.read_text().splitlines()))]
        fb = DATA_DIR / "feedback.jsonl"
        if fb.exists():
            rows += [(r["text"], r["group"]) for r in map(json.loads, filter(str.strip, fb.read_text().splitlines()))]
        return cls(rows)
