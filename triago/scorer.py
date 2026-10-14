"""Jev-style fixed-answer scorer. Runs on MLX (Apple Silicon) or PyTorch (CUDA/CPU): see backends.py.

Equivalent of SGLang's /v1/score: run the prompt once, read the next-token
logits for the label tokens only, softmax across those labels. No decoding.
"""
from __future__ import annotations

import json
import os
import string
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import backends

DEFAULT_ADAPTER = os.environ.get("TRIAGO_ADAPTER") or None  # LoRA dir from `python -m triago.finetune`
LABELS = string.ascii_uppercase  # A, B, C, ... -> single tokens in the answer position

SYSTEM = (
    "You are an IT service desk dispatcher. Tickets may be written in any language "
    "(English, French, German, Romanian, ...); judge the meaning, not the language. "
    "Read the ticket and answer with exactly one label from the allowed list. Return only the label."
)


@dataclass
class Decision:
    choice: str
    probabilities: dict[str, float]
    latency_ms: float
    logits: dict[str, float] = None  # raw label logits, kept for calibration

    @property
    def margin(self) -> float:
        top = sorted(self.probabilities.values(), reverse=True)
        return top[0] - (top[1] if len(top) > 1 else 0.0)


class Scorer:
    def __init__(self, model_path: str | None = None, adapter_path: str | None = DEFAULT_ADAPTER,
                 backend: str | None = None):
        self.backend = backends.load(backend or backends.pick(), model_path, adapter_path)
        self.model, self.tokenizer = self.backend.model, self.backend.tokenizer
        self.model_path, self.adapter_path = self.backend.model_path, adapter_path
        # a fine-tuned adapter ships its own calibration (temperatures, shots): see Policy.for_scorer
        cal = os.path.join(adapter_path, "triago.json") if adapter_path else None
        self.calibration = json.load(open(cal))["policy"] if cal and os.path.exists(cal) else {}
        self._lock = threading.Lock()  # one forward pass at a time (shared prefix caches)
        self._label_ids_cache: dict[int, list[int]] = {}
        self._prefix: dict[tuple, tuple[list[int], list]] = {}  # (question, choices) -> (ids, kv cache)

    # ---- prompt construction -------------------------------------------------
    def messages(self, ticket: str, question: str, choices: dict[str, str],
                 examples=None) -> tuple[list[dict], list[str]]:
        """The chat the model sees. Fine-tuning trains on exactly this (see triago/finetune.py)."""
        labels = list(LABELS[: len(choices)])
        lines = "\n".join(f"{lab} = {desc}" for lab, desc in zip(labels, choices.values()))
        letter = dict(zip(choices, labels))
        shots = ""
        if examples:  # retrieved past tickets with their correct label (see triago/examples.py)
            shots = "Similar past tickets and their correct label:\n" + "\n".join(
                f"- {' '.join(t.split())} -> {letter[g]}" for t, g in examples if g in letter) + "\n\n"
        # Static part first (question + labels), ticket last: the long prefix is identical for
        # every ticket, so its KV cache is computed once per decision type and reused.
        user = (
            f"Question:\n{question}\n\nAllowed labels:\n{lines}\n\n{shots}"
            f"Ticket:\n{ticket}\n\nReturn only the label."
        )
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], labels

    def _render(self, ticket: str, question: str, choices: dict[str, str],
                examples=None) -> tuple[str, list[str]]:
        msgs, labels = self.messages(ticket, question, choices, examples)
        prompt = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # hybrid Qwen3: emit empty <think></think> so the answer slot comes next
        )
        return prompt, labels

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prefix_cache(self, question: str, choices: dict[str, str]) -> tuple[list[int], list]:
        """KV cache of the ticket-independent prompt prefix. The prefix is the longest common
        *token* prefix of renders with different tickets and with/without examples
        (robust to BPE boundary merges)."""
        key = (question, tuple(choices.items()))
        if key not in self._prefix:
            dummy = [("Zulu 42", next(iter(choices)))]
            a = self._encode(self._render("alpha", question, choices)[0])
            n = len(a)
            for t, ex in (("Zulu 42", None), ("alpha", dummy)):
                b = self._encode(self._render(t, question, choices, ex)[0])
                n = min(n, next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))))
            ids = a[: max(0, n - 1)]  # keep one token back for safety
            self._prefix[key] = (ids, self.backend.prefill(ids))
        return self._prefix[key]

    def _label_token_ids(self, prompt: str, labels: list[str]) -> list[int]:
        """Resolve label token ids *in context*: tokens(prompt+label) must equal
        tokens(prompt) + exactly one token. Guards against ' A' vs 'A' surprises."""
        n = len(labels)
        if n in self._label_ids_cache:
            return self._label_ids_cache[n]
        base = self.tokenizer.encode(prompt, add_special_tokens=False)
        ids = []
        for lab in labels:
            full = self.tokenizer.encode(prompt + lab, add_special_tokens=False)
            if full[: len(base)] != base or len(full) != len(base) + 1:
                raise ValueError(f"label {lab!r} is not a single token at the answer position")
            ids.append(full[-1])
        if len(set(ids)) != n:
            raise ValueError("label token ids collide")
        self._label_ids_cache[n] = ids
        return ids

    # ---- scoring lane ----------------------------------------------------------
    def decide(self, ticket: str, question: str, choices: dict[str, str],
               temperature: float = 1.0, examples=None) -> Decision:
        """choices: {semantic_key: description}. Returns probabilities keyed by semantic_key.
        temperature > 1 softens overconfident distributions (fit it on labeled data).
        examples: optional [(text, key)] shown before the ticket as worked examples."""
        t0 = time.perf_counter()
        prompt, labels = self._render(ticket, question, choices, examples)
        label_ids = self._label_token_ids(prompt, labels)
        ids = self._encode(prompt)
        with self._lock:
            pre_ids, cache = self._prefix_cache(question, choices)
            use_cache = ids[: len(pre_ids)] == pre_ids
            # vocab-sized vector at the "answer" slot; the backend rewinds the cache to the shared prefix
            logits = (self.backend.next_logits(ids[len(pre_ids):], cache) if use_cache
                      else self.backend.next_logits(ids))
        sel = logits[label_ids].astype(np.float64)
        e = np.exp((sel - sel.max()) / temperature)  # restricted softmax over the labels only
        probs = e / e.sum()
        p = {key: float(v) for key, v in zip(choices, probs.tolist(), strict=True)}
        raw = {key: float(v) for key, v in zip(choices, sel.tolist(), strict=True)}
        best = max(p, key=p.get)
        return Decision(best, p, (time.perf_counter() - t0) * 1000, raw)

    # ---- baseline lane: ordinary generation -----------------------------------
    def generate_choice(self, ticket: str, question: str, choices: dict[str, str],
                        max_tokens: int = 32) -> tuple[str | None, str, float]:
        """Ask the model to *write* an answer, then parse the label back. For benchmarking."""
        t0 = time.perf_counter()
        prompt, labels = self._render(ticket, question, choices)
        prompt = prompt.replace("Return only the label.<|im_end|>",
                                "Answer with the label and a short reason.<|im_end|>")
        with self._lock:
            text = self.backend.generate(prompt, max_tokens)
        head = text.strip()[:100]
        parsed = None
        for lab, key in zip(labels, choices):
            if head.startswith(lab) or f"{lab} =" in head or f"**{lab}**" in head:
                parsed = key
                break
        return parsed, text, (time.perf_counter() - t0) * 1000
