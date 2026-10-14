"""Fine-tune the local Jev on your own tickets: LoRA on this Mac, nothing leaves the machine.

    python -m triago.finetune data/my_tickets.jsonl   # one {"t": "<subject> | <body>", "g": "<group key>"} per line
    python -m triago.finetune --zenodo                # reproducible check on real tickets (Zenodo, its own queues)

What it does, in order:
  1. scrub   e-mails, URLs, IPs, phone numbers, MACs, secrets -> placeholders (names are NOT detected)
  2. split   train / valid / test by hash of the text (test is never trained or tuned on)
  3. train   LoRA on the exact prompt the scorer reads, loss on the answer letter only
  4. fit     temperatures on valid (fine-tuning changes confidence; old temperatures would lie)
  5. judge   base vs fine-tuned, with and without retrieved examples; the choice is made on valid,
             test is only reported. Your tickets: the adapter must also not miss more security incidents.
  6. write   adapters/<name>/triago.json; activate with TRIAGO_ADAPTER=adapters/<name> only if accepted.

Rollback = unset TRIAGO_ADAPTER. The adapter has seen your tickets: keep it as private as they are.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .examples import ExampleStore
from .pipeline import GROUPS, SECURITY, Policy, recalibrate
from .scorer import Scorer

ROOT = Path(__file__).resolve().parent.parent
GROUP_Q = "Which support group should handle this ticket?"
SECURITY_Q = "Is this a security incident?"

# ---- 1. scrub ------------------------------------------------------------------------------------
_SCRUB = [
    (re.compile(r"(?i)\b(password|passwort|mot de passe|contraseña|parola|pwd|pw|pin)\b(\s*(?:is|:|=|ist|est)\s*)\S+"),
     r"\1\2<SECRET>"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<EMAIL>"),
    (re.compile(r"(?i)\b(?:https?://|www\.)\S+"), "<URL>"),
    (re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"), "<MAC>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b"), "<IP>"),
    (re.compile(r"\b[A-Za-z0-9+/_=-]{32,}\b"), "<SECRET>"),
    (re.compile(r"(?<![\w.])\+?\d[\d ()/.-]{7,}\d\b"), "<PHONE>"),
]


def scrub(text: str) -> str:
    """Remove what a LoRA could memorise and leak. Names and hostnames need your own list or an NER pass."""
    for pattern, repl in _SCRUB:
        text = pattern.sub(repl, text)
    return text


# ---- 2. data -------------------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    question: str
    choices: dict[str, str]
    train: list[tuple[str, str]]
    valid: list[tuple[str, str]]
    test: list[tuple[str, str]]
    allowed: dict | None = None  # restrict retrieved examples to these keys


def bucket(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest(), 16) % 10


def dedupe(rows):
    return list({t: (t, g) for t, g in rows}.values())


def own_tasks(path: str, do_scrub: bool) -> list[Task]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            t, g = r.get("t") or r.get("text"), r.get("g") or r.get("group")
            if g not in GROUPS and g != "security":
                sys.exit(f"unknown group {g!r}: use the keys in triago/pipeline.py GROUPS (or 'security')")
            rows.append(((scrub(t) if do_scrub else t)[:1500], g))
    rows = dedupe(rows)
    parts = {"train": [], "valid": [], "test": []}
    for t, g in rows:  # 70 / 10 / 20
        b = bucket(t)
        parts["test" if b < 2 else "valid" if b == 2 else "train"].append((t, g))
    sec = {k: [(t, "security" if g == "security" else "normal") for t, g in v] for k, v in parts.items()}
    grp = {k: [(t, g) for t, g in v if g != "security"] for k, v in parts.items()}
    return [Task("security", SECURITY_Q, SECURITY, **sec), Task("group", GROUP_Q, GROUPS, allowed=GROUPS, **grp)]


def zenodo_tasks() -> list[Task]:
    from . import zenodo_bench as z
    train = dedupe(z.load("train"))
    return [Task("group", z.QUESTION, z.QUEUES,
                 train=[r for r in train if bucket(r[0]) != 0], valid=[r for r in train if bucket(r[0]) == 0],
                 test=z.load("test"))]


# ---- 3. train ------------------------------------------------------------------------------------
class _Examples:
    """mlx_lm dataset protocol: process(item) -> (tokens, offset); loss covers tokens[offset:]."""

    def __init__(self, items: list[tuple[list[int], int]]):
        self.items = items

    def __getitem__(self, i):
        return self.items[i][0]

    def __len__(self):
        return len(self.items)

    def process(self, tokens):
        return tokens, self._offset[id(tokens)]

    @classmethod
    def build(cls, items):
        ds = cls(items)
        ds._offset = {id(tok): off for tok, off in items}
        return ds


def encode_examples(scorer: Scorer, task: Task, rows, store: ExampleStore | None, shots: int):
    """Exactly the scorer's prompt + the answer letter. (Chat-templating an assistant turn would insert
    an empty <think></think> before the letter, i.e. train a different position than the one we read.)"""
    keys = list(task.choices)
    out = []
    for t, g in rows:
        k = shots if store is not None and bucket(t) % 2 else 0  # half with retrieved examples, half without
        ex = store.nearest(t, k, allowed=task.allowed) if k else None
        prompt, labels = scorer._render(t, task.question, task.choices, ex)
        ids = scorer._encode(prompt)
        out.append((ids + [scorer._label_token_ids(prompt, labels)[keys.index(g)]], len(ids)))
    return out


def answer_loss(model, batch, lengths):
    """Cross-entropy on the answer letter only. (mlx_lm's default mask also scores one padding
    position after the sequence; with a one-token answer that padding term would dominate.)"""
    import mlx.core as mx
    import mlx.nn as nn
    logits = model(batch[:, :-1])
    steps = mx.arange(1, batch.shape[1])
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, batch[:, 1:]) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


def train(tasks: list[Task], out: Path, iters: int | None, lr: float, shots: int) -> None:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.lora import CONFIG_DEFAULTS, linear_to_lora_layers, save_config
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train as mlx_train

    scorer = Scorer(adapter_path=None, backend="mlx")  # training uses MLX; PyTorch users: see README
    tr, va = [], []
    for task in tasks:
        store = ExampleStore(task.train) if task.name == "group" else None  # the security gate runs without examples
        tr += encode_examples(scorer, task, task.train, store, shots)
        va += encode_examples(scorer, task, task.valid, store, shots)
    batch = 4
    iters = iters or max(100, math.ceil(2 * len(tr) / batch))  # ~2 epochs
    print(f"training on {len(tr)} examples ({len(va)} valid), {iters} iterations, "
          f"longest {max(len(t) for t, _ in tr)} tokens", flush=True)
    cfg = {**CONFIG_DEFAULTS, "model": scorer.model_path, "adapter_path": str(out), "iters": iters,
           "batch_size": batch, "learning_rate": lr}
    mx.random.seed(cfg["seed"])
    model = scorer.model
    model.freeze()
    linear_to_lora_layers(model, cfg["num_layers"], cfg["lora_parameters"])
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "adapter_config.json")  # what mlx_lm.load(adapter_path=...) reads back
    mlx_train(model=model, optimizer=optim.Adam(learning_rate=lr), loss=answer_loss,
              train_dataset=CacheDataset(_Examples.build(tr)), val_dataset=CacheDataset(_Examples.build(va)),
              args=TrainingArgs(batch_size=batch, iters=iters, val_batches=max(1, min(25, len(va) // batch)),
                                steps_per_report=20, steps_per_eval=100, steps_per_save=100,
                                adapter_file=out / "adapters.safetensors", max_seq_length=2048,
                                grad_checkpoint=True))
    del scorer, model
    gc.collect()


def scorer_model() -> str:
    import os
    from . import backends
    return os.environ.get("TRIAGO_MODEL") or backends.MODELS["mlx"]


# ---- 4/5. fit + judge ----------------------------------------------------------------------------
def fit_T(res) -> float:
    def nll(T):
        return -sum(math.log(max(recalibrate(d, T).probabilities[y], 1e-12)) for d, y in res) / len(res)
    return min((0.25 + 0.25 * i for i in range(80)), key=nll)


def run(scorer: Scorer, task: Task, rows, shots: int):
    store = ExampleStore(task.train) if shots else None
    return [(scorer.decide(t, task.question, task.choices,
                           examples=store.nearest(t, shots, allowed=task.allowed) if store else None), y)
            for t, y in rows]


def accuracy(res) -> float:
    return sum(d.choice == y for d, y in res) / len(res)


def security_stats(res, T: float, trip: float) -> tuple[int, int, int]:
    hit = [recalibrate(d, T).probabilities["security"] >= trip for d, _ in res]
    pos = [y == "security" for _, y in res]
    return sum(h and p for h, p in zip(hit, pos)), sum(pos), sum(h and not p for h, p in zip(hit, pos))


def judge(tasks: list[Task], out: Path, shots: int, kind: str, model: str) -> dict:
    base, tuned = Scorer(adapter_path=None), Scorer(adapter_path=str(out))
    report, policy = {}, {"shots": 0}
    seen_train = {t for task in tasks for t, _ in task.train}
    for task in tasks:
        configs = [("base", base, 0), ("tuned", tuned, 0)]
        if task.name == "group":
            configs += [("base", base, shots), ("tuned", tuned, shots)]
        rows = {}
        for who, sc, k in configs:
            t0 = time.perf_counter()
            v = run(sc, task, task.valid, k)
            T = fit_T(v)
            te = run(sc, task, task.test, k)
            unseen = [(d, y) for (d, y), (t, _) in zip(te, task.test) if t not in seen_train]
            rows[(who, k)] = {"T": T, "valid_acc": accuracy(v), "test_acc": accuracy(te),
                              "test_unseen_acc": accuracy(unseen) if unseen else None, "n_unseen": len(unseen),
                              "ms": statistics.mean(d.latency_ms for d, _ in te), "_v": v, "_te": te}
            if task.name == "security":
                rows[(who, k)]["valid_sec"] = security_stats(v, T, Policy().security_trip)
                rows[(who, k)]["test_sec"] = security_stats(te, T, Policy().security_trip)
            print(f"  {task.name:<9} {who:<6} shots={k}  valid {accuracy(v):.1%}  test {accuracy(te):.1%}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        report[task.name] = rows
    # decide on VALID only
    g = report["group"]
    best = max(g, key=lambda c: g[c]["valid_acc"])
    prod = (("base", shots) if ("base", shots) in g else ("base", 0))
    reasons = []
    accepted = best[0] == "tuned" and g[best]["valid_acc"] > g[prod]["valid_acc"]
    if not accepted:
        reasons.append(f"best on valid is {best[0]} shots={best[1]}, not better than base shots={prod[1]}")
    if "security" in report:
        s = report["security"]
        caught_b, caught_t = s[("base", 0)]["valid_sec"][0], s[("tuned", 0)]["valid_sec"][0]
        if caught_t < caught_b:
            accepted = False
            reasons.append(f"fine-tuned gate caught {caught_t} valid incidents vs {caught_b} for base")
        policy["security_temperature"] = s[("tuned", 0)]["T"]
    chosen = best if best[0] == "tuned" else ("tuned", 0)
    policy.update(temperature=g[chosen]["T"], shots=chosen[1])
    meta = {"kind": kind, "base_model": model, "created": datetime.date.today().isoformat(), "accepted": accepted,
            "reasons": reasons, "policy": policy,
            "results": {name: {f"{w} shots={k}": {m: v for m, v in r.items() if not m.startswith("_")}
                               for (w, k), r in rows.items()} for name, rows in report.items()}}
    (out / "triago.json").write_text(json.dumps(meta, indent=2))
    return meta


def print_summary(meta: dict, out: Path) -> None:
    print(f"\n== results ({meta['kind']}; choice made on valid, test reported once) ==")
    for name, rows in meta["results"].items():
        print(f"{name}:")
        for cfg, r in rows.items():
            unseen = f"  unseen-test {r['test_unseen_acc']:.1%} (n={r['n_unseen']})" if r["test_unseen_acc"] is not None else ""
            sec = ""
            if "test_sec" in r:
                c, n, fa = r["test_sec"]
                sec = f"  test incidents caught {c}/{n}, false alarms {fa}"
            print(f"  {cfg:<16} valid {r['valid_acc']:6.1%}  test {r['test_acc']:6.1%}{unseen}  "
                  f"{r['ms']:5.0f} ms  T={r['T']:.2f}{sec}")
    if meta["accepted"]:
        print(f"\nACCEPTED. Activate:  export TRIAGO_ADAPTER={out}   (rollback: unset TRIAGO_ADAPTER)")
        if meta["kind"] == "zenodo":
            print("(This adapter knows Zenodo's queues, not yours: it is a benchmark, not for production.)")
    else:
        print("\nNOT accepted: keep the base model.  " + "; ".join(meta["reasons"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickets", nargs="?", help="your labeled tickets (.jsonl)")
    ap.add_argument("--zenodo", action="store_true", help="fine-tune and judge on the Zenodo dataset")
    ap.add_argument("--out", help="adapter directory (default adapters/<name>)")
    ap.add_argument("--iters", type=int, help="training iterations (default ~2 epochs)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--no-scrub", action="store_true", help="skip PII scrubbing (only for pre-masked data)")
    ap.add_argument("--judge-only", action="store_true", help="re-run evaluation on an existing adapter")
    a = ap.parse_args()
    if not a.zenodo and not a.tickets:
        ap.error("give a tickets file or --zenodo")
    tasks = zenodo_tasks() if a.zenodo else own_tasks(a.tickets, not a.no_scrub)
    out = Path(a.out or ROOT / "adapters" / ("zenodo" if a.zenodo else Path(a.tickets).stem))
    for t in tasks:
        print(f"{t.name}: train {len(t.train)} / valid {len(t.valid)} / test {len(t.test)}")
    if sum(len(t.train) for t in tasks if t.name == "group") < 300:
        print("warning: under ~300 training tickets; retrieved examples alone are usually the better bet")
    if not a.judge_only:
        train(tasks, out, a.iters, a.lr, Policy().shots)
    print_summary(judge(tasks, out, Policy().shots, "zenodo" if a.zenodo else "triago", scorer_model()), out)


if __name__ == "__main__":
    main()
