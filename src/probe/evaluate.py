"""Scoring a fitted probe on a test split, and assembling the channel table.

Everything downstream -- metrics, bootstrap CIs, tie analysis, figures --
consumes the tidy frame this module produces: one row per trajectory, one
column per channel, plus the metadata needed to slice by family.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.eval.bootstrap import CI, bootstrap_metric, paired_bootstrap_difference
from src.eval.metrics import auroc, tpr_at_fpr
from src.monitor.activations import ActivationStore
from src.probe.train import ProbeFit, load_xy, score_probe

# The zero-shot channels, in the order they appear in tables and figures.
ZERO_SHOT_CHANNELS = ["emitted", "logit_ev", "logit_logodds_50", "logit_logodds_80",
                      "logit_logodds_90"]
TRAINED_CHANNELS = ["probe", "head"]


def probe_scores_for(store: ActivationStore, fit: ProbeFit, traj_ids: Sequence[str],
                     labels: dict[str, int]) -> tuple[np.ndarray, list[str]]:
    X, _, ids = load_xy(store, traj_ids, labels, fit.representation, fit.layer_pos)
    return score_probe(fit, X), ids


def attach_probe_column(df: pd.DataFrame, ids: Sequence[str], scores: np.ndarray,
                        column: str) -> pd.DataFrame:
    mapping = dict(zip(ids, scores.tolist()))
    df = df.copy()
    df[column] = df["traj_id"].map(mapping)
    return df


def available_channels(df: pd.DataFrame) -> list[str]:
    return [c for c in ZERO_SHOT_CHANNELS + TRAINED_CHANNELS if c in df.columns]


def evaluate_channels(
    df: pd.DataFrame, channels: Sequence[str], fpr_targets: Sequence[float],
    n_resamples: int = 10000, ci: float = 0.95, seed: int = 0,
    label_col: str = "label",
) -> pd.DataFrame:
    """AUROC and TPR@FPR with bootstrap CIs, one row per channel."""
    rows: list[dict[str, Any]] = []
    y = df[label_col].to_numpy(dtype=int)
    for ch in channels:
        s = df[ch].to_numpy(dtype=float)
        row: dict[str, Any] = {"channel": ch}
        a = bootstrap_metric(s, y, lambda ss, yy: auroc(ss, yy),
                             n_resamples=n_resamples, ci=ci, seed=seed)
        row.update({"auroc": a.point, "auroc_lo": a.lo, "auroc_hi": a.hi})
        row["n_missing"] = int((~np.isfinite(s)).sum())
        for t in fpr_targets:
            b = bootstrap_metric(
                s, y, lambda ss, yy, t=t: tpr_at_fpr(ss, yy, t).tpr,
                n_resamples=n_resamples, ci=ci, seed=seed,
            )
            row[f"tpr@{t:g}"] = b.point
            row[f"tpr@{t:g}_lo"] = b.lo
            row[f"tpr@{t:g}_hi"] = b.hi
            row[f"tpr@{t:g}_reachable"] = tpr_at_fpr(s, y, t).reachable
        rows.append(row)
    return pd.DataFrame(rows)


def compare_channels(
    df: pd.DataFrame, comparisons: Sequence[tuple[str, str]],
    fpr_targets: Sequence[float], n_resamples: int = 10000, ci: float = 0.95,
    seed: int = 0, label_col: str = "label",
) -> pd.DataFrame:
    """Paired-bootstrap CIs on the difference between channels.

    A CI on the difference is the right test here; comparing two independent
    CIs for overlap is weaker and would understate real differences.
    """
    y = df[label_col].to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for a_name, b_name in comparisons:
        if a_name not in df.columns or b_name not in df.columns:
            continue
        a = df[a_name].to_numpy(dtype=float)
        b = df[b_name].to_numpy(dtype=float)
        row: dict[str, Any] = {"channel_a": a_name, "channel_b": b_name}
        d = paired_bootstrap_difference(a, b, y, lambda ss, yy: auroc(ss, yy),
                                        n_resamples=n_resamples, ci=ci, seed=seed)
        row.update({"d_auroc": d.point, "d_auroc_lo": d.lo, "d_auroc_hi": d.hi,
                    "auroc_sig": d.excludes(0.0)})
        for t in fpr_targets:
            dt = paired_bootstrap_difference(
                a, b, y, lambda ss, yy, t=t: tpr_at_fpr(ss, yy, t).tpr,
                n_resamples=n_resamples, ci=ci, seed=seed,
            )
            row[f"d_tpr@{t:g}"] = dt.point
            row[f"d_tpr@{t:g}_lo"] = dt.lo
            row[f"d_tpr@{t:g}_hi"] = dt.hi
            row[f"d_tpr@{t:g}_sig"] = dt.excludes(0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def ensemble_score(df: pd.DataFrame, channels: Sequence[str],
                   method: str = "rank_mean") -> np.ndarray:
    """Combine channels without fitting anything on the test split.

    Rank-averaging is used rather than a fitted stack: the channels live on
    incomparable scales (0-100, log-odds, an unbounded decision function), and
    fitting a combiner would add another trained component to a comparison that
    already has a train/zero-shot asymmetry to explain.
    """
    if method != "rank_mean":
        raise ValueError(f"unknown ensemble method {method!r}")
    ranks = []
    for ch in channels:
        s = pd.Series(df[ch].to_numpy(dtype=float))
        ranks.append(s.rank(pct=True, na_option="keep").to_numpy())
    stacked = np.vstack(ranks)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def summarize_seeds(frames: Sequence[pd.DataFrame], key: Sequence[str],
                    value_cols: Sequence[str]) -> pd.DataFrame:
    """Mean and spread across probe-training seeds, keeping per-seed rows visible."""
    allf = pd.concat(frames, ignore_index=True)
    agg = allf.groupby(list(key))[list(value_cols)].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()
