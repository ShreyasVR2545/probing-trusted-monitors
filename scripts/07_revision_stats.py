"""Revision-pass statistics: cross-fitted structure baseline, length reversal.

CPU only. Reuses the cached scores; loads no model.

Two questions this answers, both raised by reviewing the first draft:

1. Does any channel actually beat a baseline that reads only trajectory
   *shape*? The first pass fitted `structure_baseline` on the training split
   and scored the 75-trajectory test split, then compared it against logit
   channels pooled over all 299 -- different denominators, not a like-for-like
   comparison. Here it is cross-fitted so every one of the 299 trajectories
   gets an out-of-fold score.

2. Why did `length_baseline` move from 0.514 on the full sample to 0.610 on the
   matched subset? Matching on thinking-block presence removed one confound; it
   is not guaranteed to be neutral with respect to others.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from importlib import import_module  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.eval.bootstrap import bootstrap_metric, paired_bootstrap_difference  # noqa: E402
from src.eval.metrics import auroc  # noqa: E402
from src.utils.config import load_config, results_dir  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

_pt = import_module("03_probe_train")

CHANNELS = ["logit_ev", "logit_logodds_50", "logit_logodds_80", "logit_logodds_90",
            "emitted", "length_baseline"]
N_BOOT = 4000


def crossfit_structure(df: pd.DataFrame, feats: list[str], k: int = 5,
                       seed: int = 0) -> np.ndarray:
    """Out-of-fold structure-only scores for every trajectory.

    Fitting on k-1 folds and scoring the held-out fold means no trajectory is
    scored by a model that saw it, so the resulting vector is directly
    comparable to the zero-shot channels on the same 299 trajectories.
    """
    cols = [c for c in feats if c in df.columns]
    X = df[cols].astype(float).fillna(0.0).to_numpy()
    y = df.label.to_numpy(int)
    oof = np.full(len(df), np.nan)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, random_state=seed).fit(
            sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    if np.isnan(oof).any():
        raise RuntimeError("cross-fitting left unscored trajectories")
    return oof


def main() -> int:
    log = get_logger()
    cfg = load_config()
    df = _pt.load_scores("primary", cfg)
    y = df.label.to_numpy(int)
    d = results_dir("revision")
    out: dict = {"n": len(df), "n_benign": int((y == 0).sum()),
                 "n_attack": int((y == 1).sum())}

    # ---- Item 2: cross-fitted structural baseline ------------------------
    feats = cfg["analysis"]["structure_features"]
    df = df.copy()
    df["structure_baseline_oof"] = crossfit_structure(df, feats, k=5, seed=cfg["seed"])
    ci = bootstrap_metric(df.structure_baseline_oof.to_numpy(float), y,
                          lambda a, b: auroc(a, b), n_resamples=N_BOOT, seed=cfg["seed"])
    out["structure_baseline_oof"] = {"auroc": ci.point, "lo": ci.lo, "hi": ci.hi,
                                     "features": feats, "k": 5}
    log.info(f"structure_baseline (5-fold out-of-fold, n={len(df)}): "
             f"{ci.point:.3f} [{ci.lo:.3f}, {ci.hi:.3f}]")

    rows = []
    for ch in CHANNELS:
        if ch not in df.columns:
            continue
        c = bootstrap_metric(df[ch].to_numpy(float), y, lambda a, b: auroc(a, b),
                             n_resamples=N_BOOT, seed=cfg["seed"])
        dif = paired_bootstrap_difference(
            df[ch].to_numpy(float), df.structure_baseline_oof.to_numpy(float), y,
            lambda a, b: auroc(a, b), n_resamples=N_BOOT, seed=cfg["seed"])
        rows.append({"channel": ch, "auroc": c.point, "lo": c.lo, "hi": c.hi,
                     "d_vs_structure": dif.point, "d_lo": dif.lo, "d_hi": dif.hi,
                     "beats_structure": bool(dif.excludes(0.0) and dif.point > 0)})
        log.info(f"  {ch:20s} {c.point:.3f} [{c.lo:.3f}, {c.hi:.3f}]  "
                 f"vs structure {dif.point:+.3f} [{dif.lo:+.3f}, {dif.hi:+.3f}]  "
                 f"beats={rows[-1]['beats_structure']}")
    sb = pd.DataFrame(rows)
    sb.to_csv(os.path.join(d, "structure_comparison.csv"), index=False)
    out["channels_beating_structure"] = sb.loc[sb.beats_structure, "channel"].tolist()

    # ---- Item 3: why length_baseline moved -------------------------------
    full = pd.read_csv(os.path.join("results", "scores", "scores_primary.csv"))
    meta = pd.read_csv(os.path.join("results", "data", "metadata.csv"))
    full = full.merge(meta[["traj_id", "render_thinking_blocks_stripped"]],
                      on="traj_id", how="left")
    full["zero_think"] = full.render_thinking_blocks_stripped.fillna(0) == 0
    a_full = auroc(full.length_baseline.to_numpy(float), full.label.to_numpy(int))
    a_match = auroc(df.length_baseline.to_numpy(float), y)
    out["length_baseline"] = {"full_sample": a_full, "matched_subset": a_match,
                              "n_full": len(full), "n_matched": len(df)}
    log.info(f"length_baseline: full {a_full:.3f} (n={len(full)}) -> "
             f"matched {a_match:.3f} (n={len(df)})")

    atk = full[full.label == 1]
    by_fam = atk.groupby(["strategy", "zero_think"]).length_baseline.agg(
        ["mean", "median", "size"]).round(0).reset_index()
    by_fam.to_csv(os.path.join(d, "length_by_family.csv"), index=False)
    ret = atk[atk.zero_think]
    drop = atk[~atk.zero_think]
    ben = full[full.label == 0]
    out["length_mechanism"] = {
        "benign_mean_chars": float(ben.length_baseline.mean()),
        "retained_attacks_mean_chars": float(ret.length_baseline.mean()),
        "dropped_attacks_mean_chars": float(drop.length_baseline.mean()),
        "n_retained": int(len(ret)), "n_dropped": int(len(drop)),
    }
    log.info(f"attack length: retained (zero-thinking) {ret.length_baseline.mean():.0f} "
             f"chars vs dropped {drop.length_baseline.mean():.0f} vs benign "
             f"{ben.length_baseline.mean():.0f}")

    # ---- Item 1: the threshold reversal, both samples ---------------------
    rev = []
    for ch in ["logit_ev", "logit_logodds_50", "logit_logodds_80", "logit_logodds_90"]:
        rev.append({"channel": ch,
                    "full_sample": auroc(full[ch].to_numpy(float), full.label.to_numpy(int)),
                    "matched_subset": auroc(df[ch].to_numpy(float), y)})
    rv = pd.DataFrame(rev).round(3)
    rv.to_csv(os.path.join(d, "threshold_reversal.csv"), index=False)
    out["threshold_reversal"] = rv.to_dict("records")
    log.info(f"threshold reversal:\n{rv.to_string(index=False)}")

    with open(os.path.join(d, "revision_stats.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
