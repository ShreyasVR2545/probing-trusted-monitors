"""Manifest-based checkpointing.

Each experiment cell writes a manifest before it starts and updates it on
completion. Re-runs skip cells whose manifest says `status == "complete"`
unless --force is passed. Activation caching is the expensive step and is
never recomputed unnecessarily.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(REPO_ROOT, "results")


def _safe_cell_dir(cell: str) -> str:
    """Map a cell name onto a filesystem path.

    Cell names are namespaced with "::" (e.g. "scores::primary") so one stage
    can hold several conditions. Windows rejects ":" in a path, so the separator
    becomes a subdirectory: results/scores/primary/manifest.json.
    """
    parts = [p for p in cell.split("::") if p]
    return os.path.join(RESULTS, *parts)


def manifest_path(cell: str) -> str:
    return os.path.join(_safe_cell_dir(cell), "manifest.json")


def read_manifest(cell: str) -> dict[str, Any] | None:
    path = manifest_path(cell)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def is_complete(cell: str) -> bool:
    man = read_manifest(cell)
    return bool(man and man.get("status") == "complete")


def write_manifest(cell: str, payload: dict[str, Any]) -> str:
    path = manifest_path(cell)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


@contextmanager
def cell(name: str, provenance: dict[str, Any], force: bool = False) -> Iterator[dict[str, Any]]:
    """Run an experiment cell under a manifest.

    Yields a mutable dict; whatever the body puts in it lands in the manifest
    under "outputs". Raising inside the body records status="failed" with the
    traceback and re-raises -- failures are never swallowed.
    """
    from src.utils.logging import get_logger

    log = get_logger()
    if is_complete(name) and not force:
        log.info(f"[cell {name}] already complete -- skipping (use --force to redo)")
        existing = read_manifest(name) or {}
        yield {"_skipped": True, **existing.get("outputs", {})}
        return

    outputs: dict[str, Any] = {}
    payload = {
        "cell": name,
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provenance": provenance,
        "outputs": outputs,
    }
    write_manifest(name, payload)
    t0 = time.perf_counter()
    try:
        yield outputs
    except BaseException as exc:
        payload["status"] = "failed"
        payload["error"] = repr(exc)
        payload["traceback"] = traceback.format_exc()
        payload["wall_clock_s"] = round(time.perf_counter() - t0, 2)
        payload["outputs"] = outputs
        write_manifest(name, payload)
        log.error(f"[cell {name}] FAILED: {exc!r}")
        raise
    payload["status"] = "complete"
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["wall_clock_s"] = round(time.perf_counter() - t0, 2)
    payload["outputs"] = outputs
    write_manifest(name, payload)
    log.info(f"[cell {name}] complete in {payload['wall_clock_s']}s")


def cell_dir(name: str) -> str:
    path = _safe_cell_dir(name)
    os.makedirs(path, exist_ok=True)
    return path
