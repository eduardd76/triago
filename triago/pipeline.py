"""Ticket dispatch pipeline: a chain of bounded decisions, each one scored (not generated)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .scorer import Decision, Scorer

# Decision 1 — security gate. Runs first and short-circuits.
SECURITY = {
    "security": ("security incident: phishing link clicked or credentials/QR code entered, malware or ransomware, "
                 "suspicious sign-in or MFA change the user did not make, lost or stolen laptop/phone/bag with a "
                 "company device, data sent to the wrong recipient or shared publicly, leaked passwords"),
    # A narrower "normal" (listing lockouts, MFA re-setup, BitLocker...) was tested 2026-10-13: 7 fewer false
    # alarms but 3 more missed incidents over 450 held-out tickets -> rejected.
    "normal": ("normal IT issue or request with no actual threat - including ordinary outages, lockouts, "
               "expired certificates or warnings where the user only wonders whether it is a hack, and empty "
               "tickets with no content such as 'see attachment'"),
}

# Decision 2 — assignment group. Keep <= ~8 options; OTHER is the escape hatch.
GROUPS = {
    "network": ("Network: wifi/LAN/WAN for a site or many users, VPN connectivity, firewall rules and blocked "
                "connections, proxy, DNS/DHCP, IP address and VLAN requests, load-balancer pool changes, a "
                "server or app reachable from one site/branch but not another, everything slow at once"),
    "server_infra": ("Server & Infrastructure: Windows/Linux servers, VMware/Hyper-V VMs, storage/SAN/NAS capacity, "
                     "file-share creation and restores, backup, databases (SQL Server, Oracle, PostgreSQL), "
                     "cloud (Azure/AWS), TLS/SSL certificates expired on websites and portals, monitoring alerts, "
                     "Kubernetes, datacenter hardware"),
    "end_user": ("End-user Computing: one user's laptop/desktop (incl. its wifi adapter, Windows updates, BitLocker "
                 "recovery), macOS, printers, monitors, docking stations, peripherals, Office apps and add-ins "
                 "crashing on one PC (incl. Outlook), software installation on a PC, device setup"),
    "identity_access": ("Identity & Access: passwords, account lockout, MFA, AD/Entra ID accounts and groups; ANY "
                        "request to grant or fix access or permissions - shared folders, SharePoint sites, SSH, "
                        "sudo, admin rights, guest accounts, service-account expiry; joiner/mover/leaver"),
    "application": ("Business Applications: ERP/SAP, CRM/Salesforce, Exchange/Outlook mailboxes and mail flow, "
                    "Teams, SharePoint/OneDrive sync, room-booking and other internal or customer-facing web apps, "
                    "software licenses (Visio, Acrobat, Power BI), application errors and bugs"),
    "telephony": ("Telephony & Mobile: desk phones, softphones (Webex/Teams Phone) incl. registration, contact "
                  "center agent states/queues/greetings, voicemail and its PIN, company mobile phones (orders, "
                  "broken screens, SIM PIN/PUK), mobile email setup and MDM/Company Portal enrollment"),
    "facilities": ("Facilities: meeting-room audio/video and room devices, badges incl. visitor badges, door "
                   "access, building power, lights, HVAC, desks, parking and EV chargers, server-room temperature/"
                   "humidity/cooling (CRAC) and UPS alerts"),
    "other": ("NOT an IT matter or no actionable content: HR, payroll/payslips, vacation, expenses, travel/visa, "
              "benefits, canteen, non-IT purchasing; empty tickets ('see attachment'); automatic emails such as "
              "out-of-office replies, meeting invitations and mail delivery failure notices"),
}

# Decisions 3+4 — scored separately, combined by an ITIL matrix in code.
IMPACT = {
    "high": "many users, a whole site/department, or a business-critical service is affected",
    "medium": "a team or a small group of users is affected",
    "low": "a single user is affected",
}
URGENCY = {
    "high": "work is completely stopped right now, no workaround",
    "medium": "work is degraded but a workaround exists",
    "low": "a request or question that can wait",
}

# ITIL impact x urgency -> priority (1 = critical)
PRIORITY_MATRIX = {
    ("high", "high"): 1, ("high", "medium"): 2, ("high", "low"): 3,
    ("medium", "high"): 2, ("medium", "medium"): 3, ("medium", "low"): 4,
    ("low", "high"): 3, ("low", "medium"): 4, ("low", "low"): 5,
}


import math

QUESTIONS = {
    "security": ("Is this a security incident?", SECURITY),
    "group": ("Which support group should handle this ticket?", GROUPS),
    "impact": ("What is the business impact?", IMPACT),
    "urgency": ("How urgent is it?", URGENCY),
}


@dataclass
class Policy:
    min_top: float = 0.90      # top probability required to auto-dispatch (after calibration)
    min_margin: float = 0.25   # lead over runner-up required to auto-dispatch
    # Values below were fitted on the CALIB half (`python -m triago.bench`) for Qwen3-4B-Instruct-2507.
    temperature: float = 4.75  # group-decision temperature
    security_temperature: float = 5.75  # security-gate temperature
    security_trip: float = 0.05  # calibrated security prob above this -> SOC (recall over false alarms)
    shots: int = 3               # retrieved past tickets as examples: +7 pts on real Zenodo and realistic holdout

    @classmethod
    def for_scorer(cls, scorer) -> "Policy":
        """Defaults above for the base model; a fine-tuned adapter overrides them with its own fit."""
        return cls(**getattr(scorer, "calibration", {}))


def recalibrate(d: Decision, temperature: float) -> Decision:
    """Re-apply the restricted softmax to the raw label logits at a given temperature."""
    m = max(d.logits.values())
    e = {k: math.exp((v - m) / temperature) for k, v in d.logits.items()}
    z = sum(e.values())
    p = {k: v / z for k, v in e.items()}
    return Decision(max(p, key=p.get), p, d.latency_ms, d.logits)


def is_confident(d: Decision, policy: Policy) -> bool:
    """Auto-dispatch rule for the group decision.

    This is the business rule that decides how much work the model takes off
    the service desk vs. how many misroutes you tolerate. Tune it on the
    calibration report (python -m triago.bench).
    """
    return d.choice != "other" and d.probabilities[d.choice] >= policy.min_top and d.margin >= policy.min_margin


@dataclass
class Dispatch:
    group: str
    priority: int
    auto: bool                       # True -> assign directly; False -> human triage queue
    reason: str
    decisions: dict[str, Decision] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return sum(d.latency_ms for d in self.decisions.values())


def security_tripped(raw: dict[str, Decision], policy: Policy) -> tuple[bool, float]:
    p = recalibrate(raw["security"], policy.security_temperature).probabilities["security"]
    # Asymmetric cost: a missed incident is far worse than a false alarm -> trip low, route to SOC, P1.
    return p >= policy.security_trip, p


def route(raw: dict[str, Decision], policy: Policy) -> Dispatch:
    """Turn raw (T=1) decisions into a dispatch: calibrate, gate, matrix, policy.
    Pure function of the logits -> the benchmark can re-evaluate any policy offline."""
    tripped, p_sec = security_tripped(raw, policy)
    if tripped:
        return Dispatch("security", 1, True, f"security gate p={p_sec:.2f}", raw)
    grp = recalibrate(raw["group"], policy.temperature)
    prio = PRIORITY_MATRIX[(raw["impact"].choice, raw["urgency"].choice)]
    ranked = sorted(grp.probabilities.items(), key=lambda kv: -kv[1])[:2]
    reason = ", ".join(f"{k}={v:.2f}" for k, v in ranked) + f"; security p={p_sec:.2f}"
    return Dispatch(grp.choice, prio, is_confident(grp, policy), reason, {**raw, "group": grp})


def score(scorer: Scorer, ticket: str, names=tuple(QUESTIONS), store=None, shots: int = 0) -> dict[str, Decision]:
    """store: optional ExampleStore; its nearest labeled tickets become examples for the group decision."""
    ex = store.nearest(ticket, shots, allowed=GROUPS) if store is not None and "group" in names else None
    return {n: scorer.decide(ticket, *QUESTIONS[n], examples=ex if n == "group" else None) for n in names}


def dispatch(scorer: Scorer, ticket: str, policy: Policy | None = None, store=None) -> Dispatch:
    policy = policy or Policy()
    raw = score(scorer, ticket, ("security",))
    if not security_tripped(raw, policy)[0]:  # short-circuit: incidents skip the other decisions
        raw |= score(scorer, ticket, ("group", "impact", "urgency"), store, policy.shots)
    return route(raw, policy)
