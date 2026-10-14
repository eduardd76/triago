"""Benchmark + calibration on the labeled ticket set, with an honest held-out split.

    python -m triago.bench               # fit T + security threshold on CALIB half, report on TEST half
    python -m triago.bench --eval calib  # dev loop: report on CALIB (use while tuning descriptions)
    python -m triago.bench --race        # also: scoring vs generation head-to-head (64 test tickets)
    python -m triago.bench --shots 0     # disable retrieved examples (default: Policy.shots)
    python -m triago.bench --holdout data/real_like   # fit on CALIB, report on a never-tuned set
                                      # (rows with "agree": false = blind reviewer disagreed)
    python -m triago.bench --seed     # push data/tickets.jsonl into GLPI as New tickets

Data: data/tickets.jsonl + data/gen/*.jsonl, keys t, g, i, u (+ optional hard, topic).
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from .examples import ExampleStore
from .pipeline import GROUPS, PRIORITY_MATRIX, SECURITY, Policy, recalibrate, route, score
from .scorer import Scorer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# For the head-to-head race both lanes answer the same single 9-way question.
ALL_GROUPS = {"security": SECURITY["security"], **GROUPS}


def load() -> list[dict]:
    files = [DATA_DIR / "tickets.jsonl", *sorted((DATA_DIR / "gen").glob("*.jsonl"))]
    seen, cases = set(), []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                key = c["t"].strip().lower()
                if key not in seen:
                    seen.add(key)
                    cases.append(c)
    return cases


def load_dir(path: str) -> list[dict]:
    files = sorted(Path(path).glob("*.jsonl")) if Path(path).is_dir() else [Path(path)]
    return [json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()]


def split(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic stratified 50/50: within each group, order by hash and alternate."""
    calib, test = [], []
    by_group = collections.defaultdict(list)
    for c in cases:
        by_group[c["g"]].append(c)
    for rows in by_group.values():
        rows.sort(key=lambda c: hashlib.sha1(c["t"].encode()).hexdigest())
        calib += rows[0::2]
        test += rows[1::2]
    return calib, test


# ---- calibration ---------------------------------------------------------------
def fit_temperature(rows: list[tuple[object, str]], name: str) -> float:
    """Temperature scaling: one scalar T minimizing NLL of the true label. Keeps argmax,
    fixes overconfidence. With few labels this is the only calibration you should fit."""
    def nll(T):
        return -sum(math.log(max(recalibrate(d, T).probabilities[y], 1e-12)) for d, y in rows) / len(rows)
    T = min((0.5 + 0.25 * i for i in range(80)), key=nll)
    print(f"{name:<9} T={T:.2f}  NLL {nll(1.0):.3f} -> {nll(T):.3f}  ({len(rows)} decisions)")
    return T


def fit_security_trip(results, policy: Policy) -> float:
    """Highest threshold that still catches every security ticket in CALIB (recall first)."""
    p = [(recalibrate(raw["security"], policy.security_temperature).probabilities["security"],
          c["g"] == "security") for c, raw in results]
    print(f"\n{'trip':>6}{'recall':>9}{'false alarms':>14}   (security gate on CALIB)")
    best = 0.05
    for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        rec = sum(x >= t for x, y in p if y) / max(1, sum(y for _, y in p))
        fp = sum(x >= t for x, y in p if not y)
        print(f"{t:>6.2f}{rec:>9.0%}{fp:>14}")
        if rec == 1.0:
            best = t
    return best


# ---- run + report --------------------------------------------------------------
def run_all(scorer: Scorer, cases: list[dict], store=None, shots: int = 0) -> list[tuple[dict, dict]]:
    """Score all four decisions for every ticket at T=1; calibration is applied offline by route().
    Examples come only from `store` (the CALIB half), never from the ticket itself."""
    out = []
    for i, c in enumerate(cases, 1):
        out.append((c, score(scorer, c["t"], store=store, shots=shots)))
        if i % 50 == 0:
            print(f"  scored {i}/{len(cases)}", flush=True)
    return out


def report(results: list[tuple[dict, dict]], policy: Policy, label: str) -> None:
    rows = []  # (case, predicted_group, auto, priority)
    for c, raw in results:
        d = route(raw, policy)
        rows.append((c, d.group, d.auto, d.priority))
    n = len(rows)
    ok = lambda r: r[1] == r[0]["g"]  # noqa: E731
    auto = [r for r in rows if r[2]]
    print(f"\n== {label} set: {n} tickets  {policy} ==")
    print(f"group accuracy (argmax)      {sum(map(ok, rows)) / n:.1%}")
    for flag, name in ((False, "normal"), (True, "hard")):
        sub = [r for r in rows if bool(r[0].get("hard")) == flag]
        if sub:
            print(f"  {name:<8} ({len(sub):>3})           {sum(map(ok, sub)) / len(sub):.1%}")
    print(f"auto-dispatched              {len(auto)}/{n} ({len(auto) / n:.0%})  "
          f"accuracy {sum(map(ok, auto)) / max(1, len(auto)):.1%}  "
          f"misroutes {sum(not ok(r) for r in auto)}")
    sec_true = [r for r in rows if r[0]["g"] == "security"]
    sec_fp = [r for r in rows if r[1] == "security" and r[0]["g"] != "security"]
    print(f"security recall              {sum(map(ok, sec_true))}/{len(sec_true)}   "
          f"false alarms to SOC {len(sec_fp)}")
    want = lambda c: 1 if c["g"] == "security" else PRIORITY_MATRIX[(c["i"], c["u"])]  # noqa: E731
    print(f"priority exact / ±1          {sum(r[3] == want(r[0]) for r in rows) / n:.1%} / "
          f"{sum(abs(r[3] - want(r[0])) <= 1 for r in rows) / n:.1%}")
    for k, lab in (("impact", "i"), ("urgency", "u")):
        print(f"{k + ' accuracy':<29}{sum(raw[k].choice == c[lab] for c, raw in results) / n:.1%}")
    print(f"mean latency / ticket        "
          f"{statistics.mean(sum(d.latency_ms for d in raw.values()) for _, raw in results):.0f} ms (all 4 decisions)")

    print(f"\n{'group':<17}{'n':>4}{'acc':>8}{'auto':>7}{'auto acc':>10}")
    for g in ALL_GROUPS:
        sub = [r for r in rows if r[0]["g"] == g]
        if sub:
            a = [r for r in sub if r[2]]
            aa = f"{sum(map(ok, a)) / len(a):.0%}" if a else "-"
            print(f"{g:<17}{len(sub):>4}{sum(map(ok, sub)) / len(sub):>8.0%}{len(a) / len(sub):>7.0%}{aa:>10}")

    conf = []
    for c, raw in results:
        if c["g"] != "security":
            gd = recalibrate(raw["group"], policy.temperature)
            conf.append((gd.probabilities[gd.choice], gd.choice == c["g"]))
    print(f"\n{'threshold':>10}{'coverage':>10}{'accuracy':>10}   (group decision, calibrated)")
    for t in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95):
        sel = [k for p, k in conf if p >= t]
        print(f"{t:>10.2f}{len(sel) / len(conf):>10.0%}{(sum(sel) / len(sel) if sel else float('nan')):>10.1%}")

    bad = [r for r in auto if not ok(r)] + sec_fp
    if label.startswith("CALIB"):  # dev loop: show every error, triaged or not
        bad = [r for r in rows if not ok(r)]
    if bad:
        print("\nAUTO misroutes + SOC false alarms (predicted -> expected):")
        seen = set()
        for c, g, _, _ in bad:
            if c["t"] not in seen:
                seen.add(c["t"])
                print(f"  {g:<16}-> {c['g']:<16}{c['t'][:70]}")
    print(f"\n(triaged misroutes, harmless but cost a human touch: "
          f"{sum(not ok(r) and not r[2] for r in rows)})")


def race(scorer: Scorer, cases: list[dict]) -> None:
    q = "Which support group should handle this ticket?"
    s_lat, g_lat, s_ok, g_ok, g_unparsed = [], [], 0, 0, 0
    for c in cases:
        d = scorer.decide(c["t"], q, ALL_GROUPS)
        s_lat.append(d.latency_ms)
        s_ok += d.choice == c["g"]
        parsed, _, ms = scorer.generate_choice(c["t"], q, ALL_GROUPS, max_tokens=32)
        g_lat.append(ms)
        g_ok += parsed == c["g"]
        g_unparsed += parsed is None
    n = len(cases)
    print(f"\n== Race: one 9-way routing decision, {n} tickets ==")
    print(f"{'lane':<12}{'accuracy':>10}{'p50 ms':>10}{'p95 ms':>10}{'total s':>10}")
    for name, k, lat in (("scoring", s_ok, s_lat), ("generation", g_ok, g_lat)):
        p95 = sorted(lat)[int(0.95 * (n - 1))]
        print(f"{name:<12}{k / n:>10.1%}{statistics.median(lat):>10.0f}{p95:>10.0f}{sum(lat) / 1000:>10.1f}")
    print(f"generation answers that could not be parsed: {g_unparsed}")
    print(f"speedup (total time): {sum(g_lat) / sum(s_lat):.1f}x")


def main() -> None:
    cases = load()
    if "--seed" in sys.argv:
        from .glpi import GLPI
        g = GLPI()
        seed = [json.loads(l) for l in (DATA_DIR / "tickets.jsonl").read_text().splitlines() if l.strip()]
        for c in seed:
            name, _, body = c["t"].partition(" | ")
            g.create_ticket(name, body or name)
        print(f"seeded {len(seed)} tickets into GLPI")
        return
    calib, test = split(cases)
    eval_calib = "--eval" in sys.argv and sys.argv[sys.argv.index("--eval") + 1] == "calib"
    print(f"{len(cases)} tickets -> CALIB {len(calib)} / TEST {len(test)}  "
          f"by group: {dict(collections.Counter(c['g'] for c in cases))}")
    shots = int(sys.argv[sys.argv.index("--shots") + 1]) if "--shots" in sys.argv else Policy().shots
    store = ExampleStore([(c["t"], c["g"]) for c in calib])  # examples only from CALIB
    print(f"retrieved examples per group decision: {shots}")
    scorer = Scorer()
    scorer.decide("warmup", "warmup?", {"a": "x", "b": "y"})  # compile Metal kernels
    calib_res = run_all(scorer, calib, store, shots)  # leave-one-out inside nearest()
    print("\n== Fit on CALIB ==")
    T = fit_temperature([(raw["group"], c["g"]) for c, raw in calib_res if c["g"] != "security"], "group")
    T_sec = fit_temperature([(raw["security"], "security" if c["g"] == "security" else "normal")
                             for c, raw in calib_res], "security")
    policy = Policy(temperature=T, security_temperature=T_sec, shots=shots)
    policy.security_trip = fit_security_trip(calib_res, policy)
    if "--holdout" in sys.argv:
        hold = load_dir(sys.argv[sys.argv.index("--holdout") + 1])
        res = run_all(scorer, hold, store, shots)
        report(res, policy, f"HOLDOUT {sys.argv[sys.argv.index('--holdout') + 1]} (never tuned on)")
        agreed = [r for r in res if r[0].get("agree", True)]
        if len(agreed) < len(res):
            report(agreed, policy, "HOLDOUT, blind-reviewer agreed subset")
        by_src = collections.defaultdict(list)
        for r in res:
            by_src[r[0].get("src", "?")].append(r)
        for src, rs in sorted(by_src.items()):
            print(f"  source {src:<22}{len(rs):>4} tickets  group accuracy "
                  f"{sum(route(raw, policy).group == c['g'] for c, raw in rs) / len(rs):.1%}")
    elif eval_calib:
        report(calib_res, policy, "CALIB (in-sample, dev only)")
    else:
        report(run_all(scorer, test, store, shots), policy, "TEST (held-out)")
    if "--race" in sys.argv:
        race(scorer, test[:64])


if __name__ == "__main__":
    main()
