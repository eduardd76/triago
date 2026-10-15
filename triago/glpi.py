"""Minimal GLPI legacy REST client (apirest.php) — just what dispatching needs."""
from __future__ import annotations

import os

import html
import re

import requests

from .connector import Ticket

GLPI_URL = os.environ.get("GLPI_URL", "http://localhost:8080/apirest.php")
GLPI_USER = os.environ.get("GLPI_USER", "glpi")
GLPI_PASSWORD = os.environ.get("GLPI_PASSWORD", "glpi")

TRIAGE_GROUP = "Service Desk Triage"
GROUP_NAMES = {
    "security": "Security Operations (SOC)",
    "network": "Network",
    "server_infra": "Server & Infrastructure",
    "end_user": "End-user Computing",
    "identity_access": "Identity & Access",
    "application": "Business Applications",
    "telephony": "Telephony",
    "facilities": "Facilities",
    "other": TRIAGE_GROUP,
}
# our labels -> GLPI 1..5 scale (5 = very high)
LEVEL = {"high": 4, "medium": 3, "low": 2}
# our P1..P5 -> GLPI priority (6 = major, 5 = very high ... 1 = very low)
GLPI_PRIORITY = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2}
STATUS_NEW = 1
# A ticket created by a technician is auto-assigned to them and starts as 2 = "Processing (assigned)",
# still without a group: both count as waiting for routing. (3 planned, 4 pending, 5 solved, 6 closed)
STATUS_UNROUTED = (1, 2)
GROUP_TYPE_ASSIGN = 2


def plain(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", html.unescape(text or "")).strip()


class GLPI:
    def __init__(self):
        r = requests.get(f"{GLPI_URL}/initSession", auth=(GLPI_USER, GLPI_PASSWORD), timeout=15)
        r.raise_for_status()
        self.s = requests.Session()
        self.s.headers.update({"Session-Token": r.json()["session_token"],
                               "Content-Type": "application/json"})
        self._groups: dict[str, int] | None = None

    def _req(self, method: str, path: str, **kw):
        r = self.s.request(method, f"{GLPI_URL}/{path}", timeout=30, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    # ---- setup -----------------------------------------------------------------
    def groups(self) -> dict[str, int]:
        if self._groups is None:
            rows = self._req("GET", "Group", params={"range": "0-999"}) or []
            self._groups = {g["name"]: g["id"] for g in rows}
        return self._groups

    def ensure_groups(self) -> dict[str, int]:
        existing = self.groups()
        for name in set(GROUP_NAMES.values()):
            if name not in existing:
                res = self._req("POST", "Group", json={"input": {
                    "name": name, "is_assign": 1, "is_requester": 0, "is_watcher": 1}})
                existing[name] = res["id"]
        return existing

    # ---- tickets ---------------------------------------------------------------
    def create_ticket(self, name: str, content: str) -> int:
        return self._req("POST", "Ticket", json={"input": {"name": name, "content": content}})["id"]

    def new_tickets(self) -> list[dict]:
        rows = self._req("GET", "Ticket", params={"range": "0-999", "order": "ASC"}) or []
        return [t for t in rows if t["status"] in STATUS_UNROUTED and not t.get("is_deleted")]

    def assigned_groups(self, ticket_id: int) -> list[int]:
        rows = self._req("GET", f"Ticket/{ticket_id}/Group_Ticket") or []
        return [r["groups_id"] for r in rows if r["type"] == GROUP_TYPE_ASSIGN]

    def apply_dispatch(self, ticket_id: int, group_key: str, auto: bool, priority: int,
                       impact: str, urgency: str, note: str) -> None:
        groups = self.groups()
        target = GROUP_NAMES[group_key] if auto else TRIAGE_GROUP
        self._req("PUT", f"Ticket/{ticket_id}", json={"input": {
            "impact": LEVEL[impact], "urgency": LEVEL[urgency], "priority": GLPI_PRIORITY[priority]}})
        self._req("POST", "Group_Ticket", json={"input": {
            "tickets_id": ticket_id, "groups_id": groups[target], "type": GROUP_TYPE_ASSIGN}})
        self._req("POST", "ITILFollowup", json={"input": {
            "itemtype": "Ticket", "items_id": ticket_id, "is_private": 1, "content": note}})

    # ---- Connector interface (triago/connector.py) ------------------------------
    def pending(self) -> list[Ticket]:
        return [Ticket(t["id"], t["name"], plain(t["content"]))
                for t in self.new_tickets() if not self.assigned_groups(t["id"])]

    def apply(self, ticket: Ticket, d, note: str) -> None:
        imp, urg = d.decisions.get("impact"), d.decisions.get("urgency")
        self.apply_dispatch(ticket.id, d.group, d.auto, d.priority,
                            imp.choice if imp else "high", urg.choice if urg else "high", note)

    def followups(self, ticket_id: int) -> list[dict]:
        return self._req("GET", f"Ticket/{ticket_id}/ITILFollowup") or []

    def all_tickets(self) -> list[dict]:
        return self._req("GET", "Ticket", params={"range": "0-9999"}) or []
