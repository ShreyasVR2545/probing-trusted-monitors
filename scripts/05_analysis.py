"""Stage 5 -- tie plateaus, calibration, ensembling, layer profile.

This is where RQ1 is actually answered. The tie analysis connects the project
back to the observation it started from (Lucassen & Kaufman, arXiv:2605.26047,
Appendix H.2): integer monitor scores collapse the tail into plateaus, so at
realistic audit budgets the audit set is chosen by tie-breaking rather than by
the monitor.

Writes:
  results/analysis/ties.csv               tie statistics per channel
  results/analysis/tie_table_emitted.csv  per-value counts for the integer channel
  results/analysis/signal_depth.csv       lowest FPR each channel can express
  results/analysis/calibration.csv        ECE and reliability bins per channel
  results/analysis/ensemble.csv           ensemble vs single channels
  results/analysis/layer_profile.csv      per-layer AUROC
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval.bootstrap import bootstrap_metric, paired_bootstrap_difference  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    auroc, expected_calibration_error, to_probability, tpr_at_fpr,
    tpr_at_fpr_random_tiebreak,
)
from src.eval.ties import (  # noqa: E402
    integer_tie_table, score_histogram, tie_stats, usable_signal_depth,
)
from src.probe.evaluate import ensemble_score  # noqa: E402
from src.utils.checkpoint import cell, cell_dir  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, stage_timer  # noqa: E402
from src.utils.seeding import provenance, seed_everything  # noqa: E402

CELL = "analysis"

FPR_GRID = [0.1, 0.05, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003, 0.0001]
CHANNELS = ["emitted", "logit_ev", "logit_logodds_50", "logit_logodds_80",
            "logit_logodds_90", "probe", "head", "length_baseline"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    log = get_logger()
    cfg = load_config(smoke=args.smoke)
    seed_everything(cfg["seed"])
    d = cell_dir(CELL)

    with stage_timer("05_analysis"), cell(CELL, provenance(cfg["seed"], cfg),
                                          force=args.force) as out:
        if out.get("_skipped"):
            return 0

        test = pd.read_csv(os.path.join(cell_dir("probe"), "test_scores.csv"))
        # One seed's worth of rows for the zero-shot channels (they are identical
        # across seeds); the trained channels are averaged over seeds by rank so
        # the tie analysis is not distorted by seed-to-seed scale differences.
        first_seed = sorted(test.seed.unique())[0]
        base = test[test.seed == first_seed].copy()
        for ch in ("probe", "head"):
            if ch in test.columns:
                ranks = (test.groupby("seed")[ch].rank(pct=True)
                         .groupby(test.traj_id).mean())
                base[ch] = base.traj_id.map(ranks)
        chans = [c for c in CHANNELS if c in base.columns]
        base["ensemble"] = ensemble_score(
            base, [c for c in ("emitted", "logit_ev", "probe") if c in base.columns])
        y = base.label.to_numpy(int)
        log.info(f"analysis on {len(base)} test trajectories, channels {chans}")

        # -- RQ1: tie plateaus ------------------------------------------------
        ties = pd.DataFrame([
            tie_stats(base[c].to_numpy(float), y, c).to_row() for c in chans + ["ensemble"]
        ])
        ties.to_csv(os.path.join(d, "ties.csv"), index=False)
        log.info(f"tie statistics:\n{ties[['channel','n_distinct','n_tied_at_max','distinct_in_top_1pct','resolution_floor_fpr']]}")

        pd.DataFrame(integer_tie_table(base.emitted.to_numpy(float), y)).to_csv(
            os.path.join(d, "tie_table_emitted.csv"), index=False)

        depth_rows = []
        rng = np.random.default_rng(cfg["seed"])
        for c in chans + ["ensemble"]:
            s = base[c].to_numpy(float)
            usable = usable_signal_depth(s, y, FPR_GRID)
            row = {"channel": c}
            for f, ok in usable.items():
                row[f"usable@{f:g}"] = ok
            for f in cfg["eval"]["fpr_targets"]:
                p = tpr_at_fpr(s, y, f)
                row[f"tpr@{f:g}"] = p.tpr
                row[f"tpr_randomtie@{f:g}"] = tpr_at_fpr_random_tiebreak(s, y, f, rng)
            depth_rows.append(row)
        pd.DataFrame(depth_rows).to_csv(os.path.join(d, "signal_depth.csv"), index=False)

        pd.DataFrame(score_histogram(base.emitted.to_numpy(float), y,
                                     bins=list(range(0, 105, 5)))).to_csv(
            os.path.join(d, "histogram_emitted.csv"), index=False)

        # -- Calibration ------------------------------------------------------
        cal_rows, bin_rows = [], []
        for c in chans:
            p = to_probability(base[c].to_numpy(float), c)
            ece, bins = expected_calibration_error(p, y)
            cal_rows.append({"channel": c, "ece": ece})
            for b in bins:
                bin_rows.append({"channel": c, **b})
        pd.DataFrame(cal_rows).to_csv(os.path.join(d, "calibration.csv"), index=False)
        pd.DataFrame(bin_rows).to_csv(os.path.join(d, "reliability_bins.csv"), index=False)

        # -- Ensembling -------------------------------------------------------
        ens_rows = []
        for c in chans:
            diff = paired_bootstrap_difference(
                base["ensemble"].to_numpy(float), base[c].to_numpy(float), y,
                lambda a, b: auroc(a, b),
                n_resamples=cfg["eval"]["bootstrap_resamples"],
                ci=cfg["eval"]["ci"], seed=cfg["seed"],
            )
            single = bootstrap_metric(base[c].to_numpy(float), y, lambda a, b: auroc(a, b),
                                      n_resamples=cfg["eval"]["bootstrap_resamples"],
                                      ci=cfg["eval"]["ci"], seed=cfg["seed"])
            ens_rows.append({
                "channel": c, "channel_auroc": single.point,
                "channel_lo": single.lo, "channel_hi": single.hi,
                "d_ensemble_minus_channel": diff.point,
                "d_lo": diff.lo, "d_hi": diff.hi,
                "ensemble_better": diff.excludes(0.0) and diff.point > 0,
            })
        pd.DataFrame(ens_rows).to_csv(os.path.join(d, "ensemble.csv"), index=False)

        # -- Layer profile ----------------------------------------------------
        sweep = pd.read_csv(os.path.join(cell_dir("probe"), "layer_sweep.csv"))
        prof = (sweep[sweep.kind == "linear"]
                .groupby(["representation", "layer"])[["val_auroc", "train_auroc"]]
                .agg(["mean", "std", "min", "max"]))
        prof.columns = [f"{a}_{b}" for a, b in prof.columns]
        prof.reset_index().to_csv(os.path.join(d, "layer_profile.csv"), index=False)

        best = sweep[sweep.kind == "linear"].sort_values("val_auroc", ascending=False).iloc[0]
        out.update({
            "n_test": len(base),
            "channels": chans,
            "tie_summary": ties.set_index("channel")[
                ["n_distinct", "n_tied_at_max", "distinct_in_top_1pct",
                 "resolution_floor_fpr"]].to_dict("index"),
            "ece": {r["channel"]: r["ece"] for r in cal_rows},
            "ensemble_beats": [r["channel"] for r in ens_rows if r["ensemble_better"]],
            "best_layer": int(best.layer),
            "best_layer_val_auroc": float(best.val_auroc),
            "layer_tuned": True,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
