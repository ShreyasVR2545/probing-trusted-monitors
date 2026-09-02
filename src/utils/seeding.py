"""Global seeding and provenance capture.

Every manifest records the seed, the git SHA, the resolved config and the
versions of the libraries that could plausibly change a number.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULT_SEED = 0


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed python, numpy and torch. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # cuDNN determinism is cheap here: no training loops, only forward
            # passes and sklearn fits.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT,
                capture_output=True, text=True, timeout=30,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "-dirty"
            return sha
    except Exception as exc:  # noqa: BLE001 -- provenance only
        print(f"[seeding] git sha probe failed: {exc!r}")
    return "unknown"


def library_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "numpy", "scipy", "sklearn", "pandas",
                "datasets", "bitsandbytes", "matplotlib"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[mod] = "not-installed"
    return versions


def provenance(seed: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "seed": seed,
        "git_sha": git_sha(),
        "libraries": library_versions(),
        "config": config or {},
    }
