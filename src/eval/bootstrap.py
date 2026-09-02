"""Bootstrap confidence intervals over trajectories.

The unit of analysis is the trajectory, so resampling is over trajectories,
stratified by label so every resample has the same class balance as the
original sample (without stratification, a low-FPR metric would be dominated by
variation in the number of negatives, which is not the uncertainty we care
about).

Channel comparisons use the **paired** bootstrap: the same resampled index set
is scored under both channels and the CI is formed on the difference. Comparing
two independently-bootstrapped CIs for overlap is a weaker and often misleading
test, and every headline comparison in the preregistration is a difference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class CI:
    point: float
    lo: float
    hi: float
    n_resamples: int
    n_valid: int

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.lo, self.hi

    def excludes(self, value: float) -> bool:
        return not (self.lo <= value <= self.hi)

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {f"{p}point": self.point, f"{p}lo": self.lo, f"{p}hi": self.hi,
                f"{p}n_valid": self.n_valid}


def stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for lab in np.unique(labels):
        pool = np.flatnonzero(labels == lab)
        idx.append(rng.choice(pool, size=len(pool), replace=True))
    return np.concatenate(idx)


def bootstrap_metric(
    scores: Sequence[float],
    labels: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> CI:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    point = metric(s, y)
    vals = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        i = stratified_indices(y, rng)
        vals[b] = metric(s[i], y[i])
    good = vals[np.isfinite(vals)]
    if good.size == 0:
        return CI(point, float("nan"), float("nan"), n_resamples, 0)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(good, [alpha, 1.0 - alpha])
    return CI(float(point), float(lo), float(hi), n_resamples, int(good.size))


def paired_bootstrap_difference(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    labels: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> CI:
    """CI on metric(a) - metric(b), resampling both channels on the same indices."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    y = np.asarray(labels, dtype=int)
    if not (a.shape == b.shape == y.shape):
        raise ValueError(
            f"paired bootstrap needs aligned arrays; got {a.shape}, {b.shape}, {y.shape}"
        )
    rng = np.random.default_rng(seed)
    point = metric(a, y) - metric(b, y)
    vals = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        i = stratified_indices(y, rng)
        vals[k] = metric(a[i], y[i]) - metric(b[i], y[i])
    good = vals[np.isfinite(vals)]
    if good.size == 0:
        return CI(point, float("nan"), float("nan"), n_resamples, 0)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(good, [alpha, 1.0 - alpha])
    return CI(float(point), float(lo), float(hi), n_resamples, int(good.size))


def ratio_ci(
    numerator_scores: Sequence[float], numerator_labels: Sequence[int],
    denominator_scores: Sequence[float], denominator_labels: Sequence[int],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10000, ci: float = 0.95, seed: int = 0,
) -> CI:
    """CI on the transfer ratio metric(held-out) / metric(in-distribution).

    The two evaluations run on different test sets (held-out family vs a slice
    of the same family seen in training), so they are resampled independently.
    """
    ns = np.asarray(numerator_scores, dtype=float)
    ny = np.asarray(numerator_labels, dtype=int)
    ds = np.asarray(denominator_scores, dtype=float)
    dy = np.asarray(denominator_labels, dtype=int)
    rng = np.random.default_rng(seed)
    den_point = metric(ds, dy)
    point = metric(ns, ny) / den_point if den_point else float("nan")
    vals = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        i = stratified_indices(ny, rng)
        j = stratified_indices(dy, rng)
        d = metric(ds[j], dy[j])
        vals[k] = metric(ns[i], ny[i]) / d if d else np.nan
    good = vals[np.isfinite(vals)]
    if good.size == 0:
        return CI(point, float("nan"), float("nan"), n_resamples, 0)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(good, [alpha, 1.0 - alpha])
    return CI(float(point), float(lo), float(hi), n_resamples, int(good.size))
