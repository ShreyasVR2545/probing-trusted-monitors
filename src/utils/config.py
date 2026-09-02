"""Config loading: the base research config merged with the hardware plan."""
from __future__ import annotations

import os
from typing import Any

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIGS = os.path.join(REPO_ROOT, "configs")


def load_config(smoke: bool = False) -> dict[str, Any]:
    with open(os.path.join(CONFIGS, "base.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    hw_path = os.path.join(CONFIGS, "hardware.yaml")
    if not os.path.exists(hw_path):
        raise FileNotFoundError(
            "configs/hardware.yaml is missing. Run "
            "`python scripts/run_all.py --stage -1` first."
        )
    with open(hw_path) as fh:
        cfg["hardware"] = yaml.safe_load(fh)
    cfg["smoke"] = smoke
    if smoke:
        # Smoke mode must exercise every stage end to end in under 15 minutes:
        # tiny sample, one seed, one layer stride, few bootstrap resamples.
        cfg["data"]["n_total"] = cfg["data"]["n_smoke"]
        cfg["data"]["n_signal_check"] = cfg["data"]["n_smoke"]
        cfg["data"]["max_per_cell"] = 4
        cfg["data"]["min_family_count"] = 2
        cfg["seeds"] = [cfg["seed"]]
        cfg["eval"]["bootstrap_resamples"] = 200
        cfg["probe"]["C_grid"] = [0.001, 0.1]
        cfg["head"]["epochs"] = 10
        cfg["hardware"]["plan"]["max_context_tokens"] = min(
            4096, cfg["hardware"]["plan"]["max_context_tokens"]
        )
        cfg["hardware"]["plan"]["layer_stride"] = max(
            4, cfg["hardware"]["plan"]["layer_stride"]
        )
    return cfg


def plan_of(cfg: dict[str, Any]):
    from src.utils.hardware import Plan

    return Plan(**cfg["hardware"]["plan"])


def results_dir(*parts: str) -> str:
    d = os.path.join(REPO_ROOT, "results", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def figures_dir() -> str:
    d = os.path.join(REPO_ROOT, "figures")
    os.makedirs(d, exist_ok=True)
    return d
