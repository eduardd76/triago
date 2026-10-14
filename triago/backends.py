"""Where the forward pass runs. The scorer only needs next-token logits plus a reusable prompt-prefix cache.

    TRIAGO_BACKEND=mlx     Apple Silicon (default on Macs)
    TRIAGO_BACKEND=torch   PyTorch: NVIDIA CUDA on Linux/Windows, Apple MPS, or CPU (default elsewhere)
    TRIAGO_TORCH_DEVICE=cuda|mps|cpu   force a device (default: the best one available)

Each backend picks its own default model: the same Qwen3-4B-Instruct-2507, 4-bit for MLX, bf16 for PyTorch.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import sys

import numpy as np

MODELS = {"mlx": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "torch": "Qwen/Qwen3-4B-Instruct-2507"}


def pick() -> str:
    name = os.environ.get("TRIAGO_BACKEND")
    if name:
        return name
    apple = sys.platform == "darwin" and platform.machine() == "arm64"
    return "mlx" if apple and importlib.util.find_spec("mlx") else "torch"


def load(name: str, model_path: str | None, adapter_path: str | None):
    cls = {"mlx": MLXBackend, "torch": TorchBackend}.get(name)
    if cls is None:
        raise ValueError(f"TRIAGO_BACKEND must be 'mlx' or 'torch', got {name!r}")
    return cls(model_path or os.environ.get("TRIAGO_MODEL") or MODELS[name], adapter_path)


class MLXBackend:
    name = "mlx"

    def __init__(self, model_path: str, adapter_path: str | None):
        from mlx_lm import load
        self.model_path = model_path
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)

    def prefill(self, ids: list[int]):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.model)
        self.model(mx.array(ids)[None], cache=cache)
        mx.eval([c.state for c in cache])
        return cache

    def next_logits(self, ids: list[int], cache=None) -> np.ndarray:
        """Logits at the position after `ids`. With a cache, `ids` is the suffix and the cache is
        rewound afterwards, so the same prefix serves the next ticket."""
        import mlx.core as mx
        from mlx_lm.models.cache import trim_prompt_cache
        out = self.model(mx.array(ids)[None], cache=cache)[0, -1].astype(mx.float32)
        mx.eval(out)
        if cache is not None:
            trim_prompt_cache(cache, len(ids))
        return np.array(out)

    def generate(self, prompt: str, max_tokens: int) -> str:
        from mlx_lm import generate
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)


class TorchBackend:
    name = "torch"

    def __init__(self, model_path: str, adapter_path: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        forced = os.environ.get("TRIAGO_TORCH_DEVICE")  # cuda | mps | cpu; default: best available
        if forced == "cuda" or (not forced and torch.cuda.is_available()):
            self.device, dtype = "cuda", torch.bfloat16
        elif forced == "mps" or (not forced and torch.backends.mps.is_available()):
            self.device, dtype = "mps", torch.float16
        else:
            self.device, dtype = "cpu", torch.bfloat16 if os.environ.get("TRIAGO_CPU_BF16") else torch.float32
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
        if adapter_path:
            if not os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
                raise ValueError(f"{adapter_path} is not a PEFT adapter (MLX adapters only work with TRIAGO_BACKEND=mlx)")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path).eval()

    def prefill(self, ids: list[int]):
        import torch
        from transformers import DynamicCache
        cache = DynamicCache()
        with torch.inference_mode():
            self.model(torch.tensor([ids], device=self.device), past_key_values=cache, use_cache=True)
        return cache

    def next_logits(self, ids: list[int], cache=None) -> np.ndarray:
        import torch
        with torch.inference_mode():
            out = self.model(torch.tensor([ids], device=self.device), past_key_values=cache,
                             use_cache=cache is not None, logits_to_keep=1)
        if cache is not None:
            cache.crop(-len(ids))  # drop the suffix again (negative = tokens to remove)
        return out.logits[0, -1].float().cpu().numpy()

    def generate(self, prompt: str, max_tokens: int) -> str:
        import torch
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
