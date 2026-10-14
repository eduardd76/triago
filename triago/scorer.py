"""Jev-style fixed-answer scorer on MLX (Apple Silicon).

Equivalent of SGLang's /v1/score: run the prompt once, read the next-token
logits for the label tokens only, softmax across those labels. No decoding.
"""
from __future__ import annotations

import os
import string
import threading
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

DEFAULT_MODEL = os.environ.get("TRIAGO_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")
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
    def __init__(self, model_path: str = DEFAULT_MODEL):
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)
        self._lock = threading.Lock()  # MLX model is not safe for concurrent forward passes
        self._label_ids_cache: dict[int, list[int]] = {}
        self._prefix: dict[tuple, tuple[list[int], list]] = {}  # (question, choices) -> (ids, kv cache)

    # ---- prompt construction -------------------------------------------------
    def _render(self, ticket: str, question: str, choices: dict[str, str],
                examples=None) -> tuple[str, list[str]]:
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
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
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
            cache = make_prompt_cache(self.model)
            self.model(mx.array(ids)[None], cache=cache)
            mx.eval([c.state for c in cache])
            self._prefix[key] = (ids, cache)
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
            if use_cache:
                suffix = ids[len(pre_ids):]
                logits = self.model(mx.array(suffix)[None], cache=cache)[0, -1]
            else:
                logits = self.model(mx.array(ids)[None])[0, -1]  # vocab-sized vector at "answer" slot
            sel = logits[mx.array(label_ids)].astype(mx.float32)
            probs = mx.softmax(sel / temperature)       # restricted softmax over the labels only
            mx.eval(probs, sel)
            if use_cache:
                trim_prompt_cache(cache, len(suffix))  # rewind to the shared prefix for the next ticket
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
            text = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)
        head = text.strip()[:100]
        parsed = None
        for lab, key in zip(labels, choices):
            if head.startswith(lab) or f"{lab} =" in head or f"**{lab}**" in head:
                parsed = key
                break
        return parsed, text, (time.perf_counter() - t0) * 1000
