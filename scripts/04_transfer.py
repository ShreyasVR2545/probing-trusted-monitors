"""Stage 4 -- leave-one-family-out transfer, then GATE 2. The headline result.

Splits by attack family rather than at random, on two axes (red-team strategy
and task/environment), and evaluates a probe on families it has never seen. The
zero-shot channels go through the identical per-family evaluation, which is what
makes the comparison fair: their transfer ratio is 1 by construction, and that
is exactly the point.

Writes:
  results/transfer/transfer_probe.csv     per-family probe results, per seed
  results/transfer/transfer_zeroshot.csv  per-family zero-shot results
  results/transfer/test_scores.csv        per-trajectory held-out scores
  results/transfer/gate2_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.splits import (  # noqa: E402
    eligible_families, in_distribution_reference, leave_one_family_out,
)
from src.monitor.activations import ActivationStore  # noqa: E402
from src.probe.transfer import run_transfer, zero_shot_transfer  # noqa: E402
from src.utils.checkpoint import cell, cell_dir  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, stage_timer  # noqa: E402
from src.utils.seeding import provenance, seed_everything  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

_probe_train = import_module("03_probe_train")
as_trajectories = _probe_train.as_trajectories
load_scores = _probe_train.load_scores

CELL = "transfer"
AXES = ("strategy", "task")
ZERO_SHOT = ["emitted", "logit_ev", "logit_logodds_80", "length_baseline"]


def best_probe_config() -> dict:
    """Layer and representation selected on validation in stage 3.

    Reused here rather than re-selected per family: selecting a layer inside
    each transfer fold would tune on the held-out family, which is the exact
    leakage the transfer evaluation exists to avoid.
    """
    path = os.path.join(cell_dir("probe"), "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError("results/probe/manifest.json missing -- run stage 3 first.")
    with open(path) as fh:
        man = json.load(fh)
    outs = man["outputs"]
    layers = outs["best_layer_by_seed"]
    reps = outs["best_rep_by_seed"]
    first = sorted(layers)[0]
    return {"layer": int(layers[first]), "representation": reps[first]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args()

    log = get_logger()
    cfg = load_config(smoke=args.smoke)
    seed_everything(cfg["seed"])
    d = cell_dir(CELL)
    store = ActivationStore(os.path.join(cell_dir("scores"), "activations", args.tag))

    with stage_timer("04_transfer"), cell(CELL, provenance(cfg["seed"], cfg),
                                          force=args.force) as out:
        if out.get("_skipped"):
            return 0

        df = load_scores(args.tag, cfg)
        df = df[df.traj_id.map(store.has)].reset_index(drop=True)
        labels = dict(zip(df.traj_id, df.label.astype(int)))
        trajs = as_trajectories(df)

        cfgp = best_probe_config()
        layers = store.layers().tolist()
        if cfgp["layer"] not in layers:
            raise ValueError(f"probe layer {cfgp['layer']} not in cached layers {layers}")
        layer_pos = layers.index(cfgp["layer"])
        log.info(f"transfer using layer {cfgp['layer']} (pos {layer_pos}), "
                 f"representation {cfgp['representation']} -- selected in stage 3, "
                 f"held fixed across folds")

        probe_frames, zs_frames, score_frames = [], [], []
        summary: dict[str, dict] = {}

        for axis in AXES:
            fams = eligible_families(trajs, axis, cfg["data"]["min_family_count"])
            if len(fams) < 2:
                log.warning(f"axis {axis}: only {len(fams)} eligible families "
                            f"(min_family_count={cfg['data']['min_family_count']}) -- skipping")
                continue
            log.info(f"axis {axis}: {len(fams)} families -- {fams}")
            lofo = leave_one_family_out(trajs, axis, cfg["data"]["min_family_count"],
                                        seed=cfg["seed"])
            indist = in_distribution_reference(trajs, axis, cfg["data"]["min_family_count"],
                                               seed=cfg["seed"])
            for seed in cfg["seeds"]:
                seed_everything(seed)
                res, scores = run_transfer(
                    store, labels, lofo, indist, cfgp["representation"], layer_pos,
                    cfgp["layer"], cfg["probe"]["C_grid"], seed,
                    n_resamples=cfg["eval"]["bootstrap_resamples"],
                    ci=cfg["eval"]["ci"], max_iter=cfg["probe"]["max_iter"],
                    standardize=cfg["probe"]["standardize"], logger=log,
                )
                probe_frames.append(res)
                score_frames.append(scores)

            zs = zero_shot_transfer(
                df, axis, ZERO_SHOT, lofo, indist, cfg["eval"]["fpr_targets"],
                n_resamples=cfg["eval"]["bootstrap_resamples"],
                ci=cfg["eval"]["ci"], seed=cfg["seed"],
            )
            zs_frames.append(zs)

            non_empty = [f for f in probe_frames if not f.empty]
            pf = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
            axis_rows = pf[pf.axis == axis]
            if axis_rows.empty:
                raise RuntimeError(f"axis {axis}: no transfer results were produced")
            summary[axis] = {
                "families": fams,
                "mean_transfer_auroc": float(axis_rows.transfer_auroc.mean()),
                "mean_indist_auroc": float(axis_rows.indist_auroc.mean()),
                "mean_transfer_ratio": float(axis_rows.ratio.mean()),
                "min_transfer_ratio": float(axis_rows.ratio.min()),
                "max_transfer_ratio": float(axis_rows.ratio.max()),
                "families_with_ci_including_0.5": sorted(
                    axis_rows.loc[axis_rows.transfer_lo <= 0.5, "family"].unique().tolist()
                ),
                "zero_shot_mean_transfer_auroc": zs.groupby("channel").transfer_auroc
                    .mean().round(4).to_dict(),
            }

        probe_all = pd.concat(probe_frames, ignore_index=True)
        probe_all.to_csv(os.path.join(d, "transfer_probe.csv"), index=False)
        zs_all = pd.concat(zs_frames, ignore_index=True)
        zs_all.to_csv(os.path.join(d, "transfer_zeroshot.csv"), index=False)
        pd.concat(score_frames, ignore_index=True).to_csv(
            os.path.join(d, "test_scores.csv"), index=False)

        # Preregistered falsification check for RQ2.
        strat = summary.get("strategy", {})
        mean_ratio = strat.get("mean_transfer_ratio", float("nan"))
        rq2_falsified = bool(
            np.isfinite(mean_ratio) and (
                mean_ratio < 0.6 or strat.get("families_with_ci_including_0.5")
            )
        )
        report = {
            "probe_layer": cfgp["layer"], "representation": cfgp["representation"],
            "layer_tuned": True, "seeds": cfg["seeds"],
            "n_trajectories": len(df),
            "summary": summary,
            "rq2_prediction": "transfer ratio in 0.6-0.9",
            "rq2_falsified": rq2_falsified,
            "per_family": probe_all.groupby(["axis", "family"])[
                ["transfer_auroc", "indist_auroc", "ratio"]
            ].mean().round(4).reset_index().to_dict("records"),
        }
        with open(os.path.join(d, "gate2_report.json"), "w") as fh:
            json.dump(report, fh, indent=2, default=str)

        log.info(f"transfer summary:\n{json.dumps(summary, indent=2, default=str)}")
        out.update({k: v for k, v in report.items() if k != "per_family"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
