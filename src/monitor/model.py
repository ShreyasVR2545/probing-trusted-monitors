"""Monitor model loading.

One place that knows how to turn `configs/hardware.yaml` into a loaded model,
so the bf16 guard and the quantization choice cannot drift between scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.hardware import Plan, assert_dtype_supported

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class LoadedMonitor:
    model: Any
    tokenizer: Any
    model_id: str
    dtype: torch.dtype
    quantization: str
    device: str
    n_layers: int
    hidden_size: int

    @property
    def layers(self):
        return self.model.model.layers


def load_monitor(plan: Plan, model_id: str | None = None,
                 quantization: str | None = None) -> LoadedMonitor:
    model_id = model_id or plan.primary_model
    quantization = quantization if quantization is not None else plan.quantization
    assert_dtype_supported(plan.dtype)
    dtype = DTYPES[plan.dtype]

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": {"": 0} if plan.device == "cuda" else "cpu"}
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != "none":
        raise ValueError(f"unknown quantization {quantization!r}")

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    cfg = model.config
    return LoadedMonitor(
        model=model, tokenizer=tok, model_id=model_id, dtype=dtype,
        quantization=quantization, device=plan.device,
        n_layers=cfg.num_hidden_layers, hidden_size=cfg.hidden_size,
    )


def free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info()
    return round(free / 1024**3, 2)


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
