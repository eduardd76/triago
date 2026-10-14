"""Real-world check on the Zenodo "Classification of IT Support Tickets" dataset (CC BY 4.0,
Benitez Pereira 2022, doi:10.5281/zenodo.7648117): 2,229 real, PII-masked MSP tickets, 7 queues.

Their own taxonomy is used as-is (no mapping to ours). Descriptions below were written from the
TRAIN split only; the TEST split is scored once. Reference: the authors' trained model reports
0.79 accuracy / 0.72 macro-F1 on the same test split.

    python -m triago.zenodo_bench            # zero-shot and with 3 retrieved TRAIN examples
"""
from __future__ import annotations

import collections
import csv
import math
import statistics
from pathlib import Path

from .examples import ExampleStore
from .pipeline import recalibrate
from .scorer import Scorer

DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "zenodo_it_tickets"
QUESTION = "Which support queue of this IT service provider should handle this ticket?"
QUEUES = {
    "Active Directory": ("Active Directory: user accounts and logins - passwords, unlock/blocked user, credentials "
                         "not working, new/leaving employee accounts, adding users, AD groups, GPOs, device "
                         "accounts"),
    "Computer-Services": ("Computer-Services: printers (offline, errors, install), scanners and scan-to-mail, local "
                          "PC/notebook hardware and setup"),
    "EOL": "EOL: automated Nexthink notice that a system must be set to End Of Life",
    "Fileservice": ("Fileservice: file shares and network drives - File Share Access requests (grant/remove), "
                    "folder permissions (e.g. access to a folder on drive J), missing folders, fileserver "
                    "migrations"),
    "O365": ("O365: Outlook and Exchange e-mail - mailbox access, mail groups/distribution lists, undeliverable "
             "mails, attachments, Skype/Teams, Office 365"),
    "Software": ("Software: installing or re-installing applications (Java, R, BarTender ...), software "
                 "install requests, application problems, automated 'Initiated by GIT' alerts about missing "
                 "update/AV connections on a system"),
    "Support general": ("Support general: everything else - network and site alarms (check_mk), servers, "
                        "firmware, planned maintenance notices, shared mailbox creation, light/external user "
                        "creation, websites, mobile phones, general questions"),
}


def load(split: str) -> list[tuple[str, str]]:
    X = {r["id"]: r["text"] for r in csv.DictReader(open(DIR / f"X_{split}.csv", encoding="utf-8"))}
    y = {r["id"]: r["category_truth"] for r in csv.DictReader(open(DIR / f"y_{split}.csv", encoding="utf-8"))}
    return [(" ".join(X[i].split())[:1500], y[i]) for i in X]


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    f1 = []
    for q in QUEUES:
        tp = sum(p == q == y for p, y in pairs)
        fp = sum(p == q != y for p, y in pairs)
        fn = sum(y == q != p for p, y in pairs)
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return statistics.mean(f1)


def fit_T(rows) -> float:
    def nll(T):
        return -sum(math.log(max(recalibrate(d, T).probabilities[y], 1e-12)) for d, y in rows) / len(rows)
    return min((0.5 + 0.25 * i for i in range(80)), key=nll)


def run(scorer: Scorer, cases, store=None, shots=0):
    out = []
    for t, y in cases:
        ex = store.nearest(t, shots) if store else None
        out.append((scorer.decide(t, QUESTION, QUEUES, examples=ex), y))
    return out


def report(name: str, res, T: float, test) -> None:
    pairs = [(d.choice, y) for d, y in res]
    n = len(pairs)
    uniq = {}
    for (t, y), (p, _) in zip(test, pairs):
        uniq[t] = (p, y)
    print(f"\n== {name} ==  (T={T:.2f})")
    print(f"accuracy        {sum(p == y for p, y in pairs) / n:.1%}   (authors' trained model: 79%)")
    print(f"macro-F1        {macro_f1(pairs):.2f}    (authors' trained model: 0.72)")
    print(f"dedup accuracy  {sum(p == y for p, y in uniq.values()) / len(uniq):.1%}   ({len(uniq)} unique texts)")
    print(f"mean latency    {statistics.mean(d.latency_ms for d, _ in res):.0f} ms")
    print(f"{'queue':<19}{'n':>5}{'recall':>8}{'precision':>11}")
    for q in QUEUES:
        tp = sum(p == q == y for p, y in pairs)
        sup = sum(y == q for _, y in pairs)
        pred = sum(p == q for p, _ in pairs)
        print(f"{q:<19}{sup:>5}{tp / max(1, sup):>8.0%}{tp / max(1, pred):>11.0%}")
    conf = [(recalibrate(d, T).probabilities[d.choice], d.choice == y) for d, y in res]
    print(f"{'threshold':>10}{'coverage':>10}{'accuracy':>10}")
    for th in (0.5, 0.7, 0.8, 0.9, 0.95):
        sel = [k for p, k in conf if p >= th]
        print(f"{th:>10.2f}{len(sel) / n:>10.0%}{(sum(sel) / len(sel) if sel else float('nan')):>10.1%}")
    confusion = collections.Counter((y, p) for p, y in pairs if p != y)
    print("top confusions (true -> predicted):", ", ".join(f"{a}->{b} {c}" for (a, b), c in confusion.most_common(6)))


def main() -> None:
    train, test = load("train"), load("test")
    print(f"train {len(train)} / test {len(test)} tickets")
    scorer = Scorer()
    scorer.decide("warmup", "warmup?", {"a": "x", "b": "y"})
    store = ExampleStore(train)
    fit = list({t: (t, y) for t, y in train}.values())[::3]  # ~300 unique TRAIN tickets for T
    for name, s, k in (("zero-shot", None, 0), ("3 retrieved TRAIN examples", store, 3)):
        T = fit_T(run(scorer, fit, s, k))
        report(f"Zenodo TEST, {name}", run(scorer, test, s, k), T, test)


if __name__ == "__main__":
    main()
