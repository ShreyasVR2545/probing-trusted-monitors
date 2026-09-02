"""Discrimination metrics, with the tail behaviour handled explicitly.

TPR at a very low FPR is the metric this project is really about, and it is
where naive implementations quietly lie. Two things are handled deliberately:

* **Ties.** `sklearn.metrics.roc_curve` groups tied scores into a single
  operating point, which is correct, but it means a channel with a large tie
  plateau simply has no operating point at low FPR. Rather than interpolating
  through the plateau (which would invent resolution the channel does not
  have), `tpr_at_fpr` returns the TPR of the best operating point whose FPR
  does not exceed the target -- the honest answer to "if I audit this budget,
  what do I catch". `tpr_at_fpr_random_tiebreak` gives the alternative
  convention, where ties are broken at random, for comparison.
* **Budget floors.** With N negatives the smallest achievable non-zero FPR is
  1/N. A target below that is unreachable, and the result is reported as NaN
  with the reason rather than as 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class TailPoint:
    fpr_target: float
    tpr: float
    achieved_fpr: float
    threshold: float
    reachable: bool
    note: str


def _clean(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray, int]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} differ in length")
    ok = np.isfinite(s)
    return s[ok], y[ok], int((~ok).sum())


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    s, y, _ = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(scores: Sequence[float], labels: Sequence[int],
               fpr_target: float) -> TailPoint:
    s, y, _ = _clean(scores, labels)
    n_neg = int((y == 0).sum())
    if n_neg == 0 or int((y == 1).sum()) == 0:
        return TailPoint(fpr_target, float("nan"), float("nan"), float("nan"), False,
                         "one class absent")
    min_fpr = 1.0 / n_neg
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= fpr_target + 1e-12
    if not ok.any():
        return TailPoint(fpr_target, float("nan"), float("nan"), float("nan"), False,
                         f"no operating point at or below FPR={fpr_target}")
    i = int(np.argmax(np.where(ok, tpr, -np.inf)))
    reachable = fpr_target >= min_fpr - 1e-12
    note = "" if reachable else (
        f"FPR target below the resolution of {n_neg} negatives (floor {min_fpr:.2e}); "
        "the reported point is the zero-FPR operating point"
    )
    return TailPoint(fpr_target, float(tpr[i]), float(fpr[i]), float(thr[i]), reachable, note)


def tpr_at_fpr_random_tiebreak(scores: Sequence[float], labels: Sequence[int],
                               fpr_target: float, rng: np.random.Generator,
                               n_draws: int = 200) -> float:
    """TPR when ties are broken uniformly at random.

    This is the convention that makes an integer score look better than it is:
    it grants the score resolution it does not possess. Reported alongside the
    honest number so the size of the tie-breaking effect is visible -- it is
    also the third falsification criterion in the preregistration.
    """
    s, y, _ = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    span = float(np.ptp(s)) or 1.0
    jitter = span * 1e-9
    vals = []
    for _ in range(n_draws):
        noisy = s + rng.uniform(-jitter, jitter, size=s.shape)
        vals.append(tpr_at_fpr(noisy, y, fpr_target).tpr)
    return float(np.nanmean(vals))


def channel_metrics(scores: Sequence[float], labels: Sequence[int],
                    fpr_targets: Sequence[float]) -> dict[str, Any]:
    s, y, n_missing = _clean(scores, labels)
    out: dict[str, Any] = {
        "n": int(len(s)),
        "n_missing": n_missing,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "auroc": auroc(s, y),
    }
    for t in fpr_targets:
        p = tpr_at_fpr(s, y, t)
        out[f"tpr@fpr{t:g}"] = p.tpr
        out[f"achieved_fpr@{t:g}"] = p.achieved_fpr
        out[f"reachable@{t:g}"] = p.reachable
    return out


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def expected_calibration_error(probs: Sequence[float], labels: Sequence[int],
                               n_bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    p, y, _ = _clean(probs, labels)
    if len(p) == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        n = int(m.sum())
        if n == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "confidence": math.nan,
                         "accuracy": math.nan})
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        ece += (n / len(p)) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "n": n, "confidence": conf, "accuracy": acc})
    return float(ece), bins


def to_probability(scores: Sequence[float], channel: str) -> np.ndarray:
    """Map a channel's raw score onto [0, 1] for calibration analysis.

    Integer and expected-value scores are on 0-100 and divide down; log-odds
    channels go through a logistic. No fitting happens here -- this measures
    whether the channel's *native* scale is calibrated, which is the question
    a deployment cares about.
    """
    s = np.asarray(scores, dtype=float)
    if channel.startswith("logit_logodds"):
        return 1.0 / (1.0 + np.exp(-s))
    if channel in ("emitted", "logit_ev"):
        return np.clip(s / 100.0, 0.0, 1.0)
    return np.clip(s, 0.0, 1.0)
