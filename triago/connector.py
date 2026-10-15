"""What triago needs from a ticketing system — two methods. GLPI (triago/glpi.py) is one implementation.

Plug in your own without forking:

    python -m triago.dispatcher run --connector mycompany.zammad:Zammad

`mycompany/zammad.py` defines a class with a no-argument constructor (read settings from env vars)
that implements `pending()` and `apply()`. See the README section "Integrate your ticketing system".
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Iterable, Protocol

from .pipeline import Dispatch


@dataclass
class Ticket:
    id: str | int
    subject: str
    body: str  # plain text; strip HTML before handing it over

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}"


class Connector(Protocol):
    def pending(self) -> Iterable[Ticket]:
        """Tickets that still need routing (new and not yet assigned to a group)."""

    def apply(self, ticket: Ticket, d: Dispatch, note: str) -> None:
        """Write the result back. If d.auto is False, send it to your human triage queue, not d.group.
        d.decisions has the probabilities; impact/urgency are absent when the security gate tripped."""


def load(spec: str) -> Connector:
    """'package.module:ClassName' -> instance."""
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"connector must look like 'package.module:ClassName', got {spec!r}")
    return getattr(importlib.import_module(module), name)()


def note_lines(d: Dispatch, mark: str = "[TRIAGO]") -> list[str]:
    """The explanation written back to the ticket (a private note), as plain-text lines.
    The first line is machine-readable: feedback export parses `group=` from it."""
    lines = [f"{mark} group={d.group} priority=P{d.priority} mode={'AUTO' if d.auto else 'TRIAGE'}",
             f"why: {d.reason}"]
    for name, dec in d.decisions.items():
        probs = " ".join(f"{k}={v:.2f}" for k, v in sorted(dec.probabilities.items(), key=lambda kv: -kv[1]))
        lines.append(f"{name}: {probs}")
    lines.append(f"scored in {d.latency_ms:.0f} ms (no tokens generated)")
    return lines
