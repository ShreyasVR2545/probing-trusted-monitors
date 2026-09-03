"""Stage 2 -- the three scoring channels, computed once per trajectory.

For every trajectory: the emitted integer, the logit-derived continuous score,
and the residual-stream activations at the scoring position (plus a mean-pooled
representation), all from the same forward pass.

Activation caching is the expensive step, so it is resumable at trajectory
granularity: a trajectory whose activation files already exist is skipped.

Writes:
  results/scores/scores.csv             one row per trajectory, all channels
  results/scores/activations/           gitignored .npy cache
  results/scores/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from src.monitor.activations import ActivationStore  # noqa: E402
from src.monitor.model import load_monitor  # noqa: E402
from src.monitor.score_emit import MonitorRunner  # noqa: E402
from src.utils.checkpoint import cell, cell_dir  # noqa: E402
from src.utils.config import load_config, plan_of  # noqa: E402
from src.utils.logging import get_logger, stage_timer  # noqa: E402
from src.utils.power import Heartbeat, PowerGuard  # noqa: E402
from src.utils.seeding import provenance, seed_everything  # noqa: E402

CELL = "scores"


def load_inputs() -> tuple[pd.DataFrame, dict[str, str]]:
    """Load metadata and rendered text, honouring an optional scoring subset.

    `results/data/score_subset.json`, when present, restricts scoring to a
    chosen set of trajectories. It exists for cost discipline: see DECISIONS.md
    D-029. Absent, everything in metadata.csv is scored.
    """
    d = cell_dir("data")
    meta = pd.read_csv(os.path.join(d, "metadata.csv"))
    subset_path = os.path.join(d, "score_subset.json")
    if os.path.exists(subset_path):
        with open(subset_path) as fh:
            keep = set(json.load(fh))
        before = len(meta)
        meta = meta[meta.traj_id.isin(keep)]
        get_logger().info(
            f"scoring subset active: {len(meta)} of {before} trajectories "
            f"({int((meta.label == 0).sum())} benign / {int((meta.label == 1).sum())} attack)"
        )
    texts: dict[str, str] = {}
    with open(os.path.join(d, "trajectories.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            texts[r["traj_id"]] = r["text"]
    return meta[meta.traj_id.isin(texts)].reset_index(drop=True), texts


def chosen_condition() -> dict:
    """The model and prompt variant that cleared Gate 1."""
    path = os.path.join(cell_dir("signal_check"), "gate1_report.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "results/signal_check/gate1_report.json is missing -- run stage 1 first."
        )
    with open(path) as fh:
        rep = json.load(fh)
    return {"model_id": rep["model_id"], "prompt_variant": rep["prompt_variant"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--prompt-variant", default=None)
    ap.add_argument("--tag", default="primary", help="sub-directory for this condition")
    args = ap.parse_args()

    log = get_logger()
    cfg = load_config(smoke=args.smoke)
    seed = seed_everything(cfg["seed"])
    plan = plan_of(cfg)

    cond = chosen_condition()
    model_id = args.model_id or cond["model_id"]
    variant = args.prompt_variant or cond["prompt_variant"]
    log.info(f"scoring with {model_id} / prompt={variant}")

    d = cell_dir(CELL)
    store = ActivationStore(os.path.join(d, "activations", args.tag))
    scores_path = os.path.join(d, f"scores_{args.tag}.csv")
    cell_name = f"{CELL}::{args.tag}"

    with stage_timer("02_score_and_cache"), cell(cell_name, provenance(seed, cfg),
                                                 force=args.force) as out:
        if out.get("_skipped"):
            return 0
        meta, texts = load_inputs()
        loaded = load_monitor(plan, model_id=model_id)
        runner = MonitorRunner(
            loaded, max_context_tokens=plan.max_context_tokens,
            max_new_tokens=args.max_new_tokens, layer_stride=plan.layer_stride,
            capture_activations=True, prompt_variant=variant,
            prefill_chunk_tokens=cfg["power"]["prefill_chunk_tokens"],
        )
        log.info(f"{loaded.n_layers} layers, capturing {len(runner.capture.layer_indices)} "
                 f"(stride {plan.layer_stride}); score trie {runner.trie.stats()}")

        # Resume: keep rows already written for trajectories whose activations exist.
        done: dict[str, dict] = {}
        if os.path.exists(scores_path) and not args.force:
            prev = pd.read_csv(scores_path)
            for r in prev.to_dict("records"):
                if store.has(r["traj_id"]):
                    done[r["traj_id"]] = r
            log.info(f"resuming: {len(done)} trajectories already cached")

        rows = list(done.values())
        todo = [r for r in meta.to_dict("records") if r["traj_id"] not in done]
        log.info(f"{len(todo)} trajectories to score")
        hb = Heartbeat()
        pcfg = cfg["power"]
        t0 = time.perf_counter()
        # Checkpointed every trajectory: this stage is the expensive one, and
        # the host has hard-reset under sustained GPU load before
        # (CRASH_MITIGATION.md). A reset must cost one trajectory, not the run.
        with PowerGuard(cooldown_s=pcfg["cooldown_s"], max_temp_c=pcfg["max_temp_c"],
                        cpu_max_pct=pcfg["cpu_max_pct"],
                        torch_threads=pcfg["torch_threads"],
                        enabled=pcfg["enabled"], logger=log) as guard:
            for i, r in enumerate(todo):
                hb.beat("02_score_and_cache", r["traj_id"], i, len(todo),
                        extra={"tag": args.tag, "model_id": model_id})
                res = runner.run(r["traj_id"], texts[r["traj_id"]])
                store.save(r["traj_id"], res.activations)
                rows.append({
                    **res.to_row(), "label": int(r["label"]), "task": r["task"],
                    "task_suite": r["task_suite"], "strategy": r["strategy"],
                    "hint_condition": r["hint_condition"], "agent_model": r["agent_model"],
                    "family_dir": r["family_dir"],
                    "original_monitor_verdict": r["original_monitor_verdict"],
                    "length_baseline": float(r["render_rendered_chars"]),
                    "model_id": model_id, "prompt_variant": variant,
                })
                pd.DataFrame(rows).to_csv(scores_path, index=False)
                guard.cooldown()
                if (i + 1) % 10 == 0 or i + 1 == len(todo):
                    rate = (time.perf_counter() - t0) / (i + 1)
                    log.info(f"  {i+1}/{len(todo)} ({rate:.1f}s/traj, "
                             f"~{rate*(len(todo)-i-1)/60:.1f} min left, "
                             f"cache {store.size_gb()} GB)")
        hb.clear()

        df = pd.DataFrame(rows)
        df.to_csv(scores_path, index=False)
        log.info(f"parse rate {df.parse_ok.mean():.1%}; "
                 f"truncated {df.truncated.mean():.1%}; cache {store.size_gb()} GB")
        out.update({
            "n": len(df), "model_id": model_id, "prompt_variant": variant,
            "tag": args.tag,
            "parse_rate": float(df.parse_ok.mean()),
            "parse_rate_by_label": df.groupby("label").parse_ok.mean().to_dict(),
            "truncation_rate": float(df.truncated.mean()),
            "layers_cached": store.layers().tolist(),
            "activation_cache_gb": store.size_gb(),
            "scores_csv": scores_path,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
