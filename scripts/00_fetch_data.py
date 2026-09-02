"""Stage 0 -- fetch and index a stratified sample of ScaleAI/mrt.

Writes:
  results/data/trajectories.jsonl   rendered monitor-view text + metadata
  results/data/metadata.csv         one row per trajectory (committed)
  results/data/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd  # noqa: E402

from src.data import mrt  # noqa: E402
from src.data.schema import render_for_monitor  # noqa: E402
from src.utils.checkpoint import cell, cell_dir  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, stage_timer  # noqa: E402
from src.utils.seeding import provenance, seed_everything  # noqa: E402

CELL = "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="override sample size")
    args = ap.parse_args()

    log = get_logger()
    cfg = load_config(smoke=args.smoke)
    seed = seed_everything(cfg["seed"])
    dcfg, mv = cfg["data"], cfg["monitor_view"]
    n_total = args.n or dcfg["n_total"]

    with stage_timer("00_fetch_data"), cell(CELL, provenance(seed, cfg), force=args.force) as out:
        if out.get("_skipped"):
            log.info("data cell already complete")
            return 0

        log.info(f"indexing {mrt.DATASET} file tree (cached after the first run)")
        paths = mrt.build_file_index(revision=dcfg["revision"])
        log.info(f"{len(paths)} transcripts on the Hub")

        recs = mrt.stratified_sample(
            paths, n_total=n_total, agent_models=dcfg["agent_models"],
            max_per_cell=dcfg["max_per_cell"],
            balance_labels=dcfg["balance_labels"], seed=seed,
        )
        log.info(f"sampled {len(recs)} paths; downloading")

        trajs, load_stats = mrt.load_many(recs, revision=dcfg["revision"], logger=log)
        log.info(f"loaded {load_stats['loaded']} "
                 f"({load_stats['duplicates']} content-hash duplicates dropped, "
                 f"{load_stats['errors']} errors)")
        if not trajs:
            raise RuntimeError("no trajectories loaded -- check network access to the Hub")

        d = cell_dir(CELL)
        rows = []
        with open(os.path.join(d, "trajectories.jsonl"), "w", encoding="utf-8") as fh:
            for t in trajs:
                text, stats = render_for_monitor(
                    t,
                    strip_agent_thinking=mv["strip_agent_thinking"],
                    strip_agent_system_prompt=mv["strip_agent_system_prompt"],
                    max_tool_result_chars=mv["max_tool_result_chars"],
                    max_message_chars=mv["max_message_chars"],
                )
                fh.write(json.dumps({"traj_id": t.traj_id, "text": text}) + "\n")
                rows.append({**t.to_meta(), **{f"render_{k}": v for k, v in stats.items()}})

        meta = pd.DataFrame(rows)
        meta.to_csv(os.path.join(d, "metadata.csv"), index=False)

        by_strategy = Counter(meta.loc[meta.label == 1, "strategy"])
        by_task = Counter(meta["task"])
        log.info(f"labels: {dict(Counter(meta['label']))}")
        log.info(f"attack families: {dict(by_strategy)}")
        log.info(f"tasks: {dict(by_task)}")

        # Elision rate by label -- a confound check, per the preregistration.
        elis = meta.groupby("label")[["render_tool_results_elided",
                                      "render_messages_elided",
                                      "render_rendered_chars"]].mean().round(2)
        log.info(f"mean elisions and rendered size by label:\n{elis}")

        out.update({
            "n_sampled": len(recs), **load_stats,
            "n_attack": int((meta.label == 1).sum()),
            "n_benign": int((meta.label == 0).sum()),
            "families": dict(by_strategy), "tasks": dict(by_task),
            "mean_rendered_chars": float(meta["render_rendered_chars"].mean()),
            "monitor_view": mv,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
