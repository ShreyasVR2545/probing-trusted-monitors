"""Console + file logging, and a stage timer that records wall-clock cost.

Cost discipline is a standing rule of the project, so every stage writes its
wall-clock and (where CUDA is present) peak GPU memory into results/timing.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
TIMING_PATH = os.path.join(RESULTS, "timing.json")

_CONFIGURED = False


def get_logger(name: str = "ptm") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("ptm")
    if not _CONFIGURED:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        os.makedirs(RESULTS, exist_ok=True)
        fh = logging.FileHandler(os.path.join(RESULTS, "run.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.propagate = False
        _CONFIGURED = True
    return logger if name == "ptm" else logger.getChild(name)


def _peak_gpu_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except ImportError:
        pass
    return None


@contextmanager
def stage_timer(stage: str, extra: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Time a stage and append the record to results/timing.json."""
    log = get_logger()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    log.info(f"=== stage {stage}: start ===")
    record: dict[str, Any] = {"stage": stage, "started": time.time()}
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record["wall_clock_s"] = round(time.perf_counter() - t0, 2)
        record["peak_gpu_gb"] = _peak_gpu_gb()
        record.update(extra or {})
        os.makedirs(RESULTS, exist_ok=True)
        history: list[dict[str, Any]] = []
        if os.path.exists(TIMING_PATH):
            try:
                with open(TIMING_PATH) as fh:
                    history = json.load(fh)
            except json.JSONDecodeError as exc:
                log.warning(f"timing.json unreadable ({exc}); starting a fresh file")
        history.append(record)
        with open(TIMING_PATH, "w") as fh:
            json.dump(history, fh, indent=2)
        log.info(
            f"=== stage {stage}: done in {record['wall_clock_s']}s "
            f"(peak GPU {record['peak_gpu_gb']} GB) ==="
        )
