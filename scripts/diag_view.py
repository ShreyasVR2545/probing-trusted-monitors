"""Diagnostic: signal check for one monitor-view / prompt / model combination.

Stage 1's remediation ladder walks prompt variants and model sizes but cannot
vary what the monitor is *shown*, which the Gate 1 diagnosis pointed at as the
binding constraint. This runs a small, fast signal check for an arbitrary
combination so a candidate configuration can be rejected in minutes instead of
the ~45 a full stage-1 attempt costs.

    python scripts/diag_view.py --n 60 --keep-thinking --prompt specific

Writes results/diag/<tag>.csv and prints AUROC with a bootstrap CI. It never
touches the stage manifests, so it cannot corrupt a real run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import mrt  # noqa: E402
from src.data.schema import Trajectory, render_for_monitor  # noqa: E402
from src.eval.bootstrap import bootstrap_metric  # noqa: E402
from src.eval.metrics import auroc  # noqa: E402
from src.monitor.model import load_monitor  # noqa: E402
from src.monitor.score_emit import MonitorRunner  # noqa: E402
from src.utils.config import load_config, plan_of, results_dir  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.power import Heartbeat, PowerGuard  # noqa: E402
from src.utils.seeding import seed_everything  # noqa: E402


def build_texts(meta: pd.DataFrame, mv: dict, keep_thinking: bool,
                tool_chars: int, msg_chars: int, log) -> dict[str, str]:
    raw = os.path.join("results", "data", "raw")
    texts: dict[str, str] = {}
    for r in meta.itertuples():
        p = mrt.local_path_for(r.source_path, raw)
        if not os.path.exists(p):
            continue
        try:
            doc = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(f"skipping {r.traj_id}: {exc!r}")
            continue
        conv = doc.get("conversation") or []
        if not conv:
            continue
        t = Trajectory(
            traj_id=r.traj_id, task=r.task, task_suite=r.task_suite,
            family_dir=r.family_dir, strategy=r.strategy,
            hint_condition=r.hint_condition, agent_model=r.agent_model,
            label=int(r.label), side_task_success=None,
            original_monitor_verdict=None, n_messages=len(conv), messages=conv,
        )
        txt, _ = render_for_monitor(
            t, strip_agent_thinking=not keep_thinking,
            strip_agent_system_prompt=mv["strip_agent_system_prompt"],
            max_tool_result_chars=tool_chars, max_message_chars=msg_chars,
        )
        texts[r.traj_id] = txt
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--keep-thinking", action="store_true",
                    help="show the agent's <model_thinking> scratchpad to the monitor")
    ap.add_argument("--prompt", default="specific",
                    choices=["primary", "specific", "crossref", "terse"])
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--tool-chars", type=int, default=None)
    ap.add_argument("--msg-chars", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    log = get_logger()
    cfg = load_config()
    seed = seed_everything(cfg["seed"])
    plan = plan_of(cfg)
    mv = cfg["monitor_view"]
    tool_chars = args.tool_chars or mv["max_tool_result_chars"]
    msg_chars = args.msg_chars or mv["max_message_chars"]
    model_id = args.model_id or plan.primary_model
    tag = args.tag or (f"{'think' if args.keep_thinking else 'nothink'}"
                       f"_{args.prompt}_{os.path.basename(model_id)}")

    meta = pd.read_csv(os.path.join("results", "data", "metadata.csv"))
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in meta.groupby("label"):
        take = min(args.n // 2, len(grp))
        parts.append(grp.iloc[rng.permutation(len(grp))[:take]])
    meta = pd.concat(parts).sort_values("traj_id").reset_index(drop=True)

    log.info(f"[diag {tag}] {len(meta)} trajectories, keep_thinking={args.keep_thinking}, "
             f"prompt={args.prompt}, model={model_id}, elision={tool_chars}/{msg_chars}")
    texts = build_texts(meta, mv, args.keep_thinking, tool_chars, msg_chars, log)
    meta = meta[meta.traj_id.isin(texts)].reset_index(drop=True)

    loaded = load_monitor(plan, model_id=model_id)
    runner = MonitorRunner(
        loaded, max_context_tokens=plan.max_context_tokens,
        max_new_tokens=args.max_new_tokens, layer_stride=plan.layer_stride,
        capture_activations=False, prompt_variant=args.prompt,
        prefill_chunk_tokens=cfg["power"]["prefill_chunk_tokens"],
    )
    hb = Heartbeat()
    pc = cfg["power"]
    rows = []
    t0 = time.perf_counter()
    with PowerGuard(cooldown_s=pc["cooldown_s"], max_temp_c=pc["max_temp_c"],
                    cpu_max_pct=pc["cpu_max_pct"], torch_threads=pc["torch_threads"],
                    enabled=pc["enabled"], logger=log) as guard:
        for i, r in meta.iterrows():
            hb.beat(f"diag::{tag}", r.traj_id, int(i), len(meta))
            res = runner.run(r.traj_id, texts[r.traj_id])
            rows.append({**res.to_row(), "label": int(r.label), "strategy": r.strategy})
            guard.cooldown()
            if (i + 1) % 20 == 0:
                rate = (time.perf_counter() - t0) / (i + 1)
                log.info(f"  {i+1}/{len(meta)} ({rate:.1f}s/traj)")
    hb.clear()

    df = pd.DataFrame(rows)
    d = results_dir("diag")
    df.to_csv(os.path.join(d, f"{tag}.csv"), index=False)
    ok = df[df.parse_ok]
    out = {"tag": tag, "n": len(df), "parse_rate": float(df.parse_ok.mean()),
           "truncation_rate": float(df.truncated.mean()),
           "keep_thinking": args.keep_thinking, "prompt": args.prompt,
           "model_id": model_id}
    for ch in ("emitted", "logit_ev", "logit_logodds_50", "logit_logodds_80"):
        if ch in ok.columns and ok.label.nunique() > 1:
            ci = bootstrap_metric(ok[ch].to_numpy(float), ok.label.to_numpy(int),
                                  lambda a, b: auroc(a, b), n_resamples=2000, seed=seed)
            out[ch] = f"{ci.point:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]"
    out["mean_emitted_benign"] = float(ok[ok.label == 0].emitted.mean())
    out["mean_emitted_attack"] = float(ok[ok.label == 1].emitted.mean())
    log.info(f"[diag {tag}] {json.dumps(out, indent=2)}")
    with open(os.path.join(d, f"{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
