"""Single entry point.

    python scripts/run_all.py --stage all --smoke     # full pipeline, tiny
    python scripts/run_all.py --stage 2               # one stage
    python scripts/run_all.py --stage all --force     # ignore manifests

Resumes from manifests: a stage whose cell is marked complete is skipped unless
`--force`. Halts at Gate 1 (after stage 1) and Gate 2 (after stage 4) and
prints the report the human is meant to read, unless `--pass-gates` is given
(used only by the smoke run, which is not a result).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.utils.logging import get_logger  # noqa: E402

PYTHON = sys.executable

STAGES: dict[str, tuple[str, str]] = {
    "-1": ("bootstrap", "src.utils.hardware"),
    "0": ("fetch data", "scripts/00_fetch_data.py"),
    "1": ("signal check (GATE 1)", "scripts/01_signal_check.py"),
    "2": ("score and cache activations", "scripts/02_score_and_cache.py"),
    "3": ("probe training", "scripts/03_probe_train.py"),
    "4": ("transfer (GATE 2)", "scripts/04_transfer.py"),
    "5": ("analysis", "scripts/05_analysis.py"),
    "6": ("figures", "scripts/06_figures.py"),
}
ORDER = ["-1", "0", "1", "2", "3", "4", "5", "6"]
GATES = {"1": "gate1", "4": "gate2"}


def run_stage(stage: str, smoke: bool, force: bool, extra: list[str]) -> int:
    log = get_logger()
    name, target = STAGES[stage]
    log.info(f"### stage {stage}: {name}")
    if target.startswith("src."):
        cmd = [PYTHON, "-m", target]
    else:
        cmd = [PYTHON, os.path.join(REPO_ROOT, target)]
        if smoke:
            cmd.append("--smoke")
        if force:
            cmd.append("--force")
        cmd += extra
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def gate_report(which: str) -> dict | None:
    path = {
        "gate1": os.path.join(REPO_ROOT, "results", "signal_check", "gate1_report.json"),
        "gate2": os.path.join(REPO_ROOT, "results", "transfer", "gate2_report.json"),
    }[which]
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def print_gate(which: str) -> None:
    rep = gate_report(which)
    bar = "=" * 78
    print(f"\n{bar}\n  {which.upper()} -- stopping for human review\n{bar}")
    if rep is None:
        print("  (no report found)")
        return
    skip = {"tie_table", "histogram", "per_family", "transfer_rows"}
    for k, v in rep.items():
        if k in skip:
            continue
        print(f"  {k}: {json.dumps(v, default=str)[:400]}")
    print(bar + "\n")


def import_results(zip_path: str) -> int:
    """Merge a results archive produced by the Kaggle runner."""
    import zipfile

    log = get_logger()
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    dest = os.path.join(REPO_ROOT, "results")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(dest)
    log.info(f"imported {len(names)} entries from {zip_path} into {dest}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    help="one of -1,0,1,2,3,4,5,6 or 'all'")
    ap.add_argument("--smoke", action="store_true",
                    help="30 trajectories, 1 seed, every stage, under 15 minutes")
    ap.add_argument("--force", action="store_true", help="ignore completed manifests")
    ap.add_argument("--pass-gates", action="store_true",
                    help="do not halt at gates (smoke runs only)")
    ap.add_argument("--import-results", default=None, metavar="ZIP",
                    help="merge a results archive from a Kaggle run")
    args, extra = ap.parse_known_args()

    log = get_logger()
    if args.import_results:
        return import_results(args.import_results)

    stages = ORDER if args.stage == "all" else [args.stage]
    for s in stages:
        if s not in STAGES:
            log.error(f"unknown stage {s!r}; expected one of {ORDER} or 'all'")
            return 2
        rc = run_stage(s, args.smoke, args.force, extra)
        if rc != 0:
            log.error(f"stage {s} failed with exit code {rc}")
            return rc
        if s in GATES and not args.pass_gates:
            print_gate(GATES[s])
            if args.stage == "all":
                log.info(f"halting at {GATES[s]}. Re-run with "
                         f"--stage {ORDER[ORDER.index(s)+1]} to continue.")
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
