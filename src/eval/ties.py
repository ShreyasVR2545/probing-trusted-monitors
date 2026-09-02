"""Tie-plateau analysis (RQ1).

This is the analysis that connects the project back to the observation it
started from: Lucassen & Kaufman (arXiv:2605.26047, Appendix H.2) report that
under max-of-5 aggregation over 250 trajectories tie at a maximum score of 95,
so there is no signal to allocate an audit budget among them, and auditing in
the 4e-3 to 2e-4 FPR range suffers.

The quantities here make that concrete for each channel:

* `n_tied_at_max` -- how many trajectories share the highest score.
* `distinct_in_top_1pct` -- how many *different* values occur among the top 1%
  of trajectories. An integer channel typically collapses these to a handful;
  a continuous channel gives one per trajectory.
* `resolution_floor_fpr` -- the smallest FPR at which the channel still has a
  distinct operating point. Below this the channel cannot express an audit
  budget at all, and the decision falls to whatever the tie-break happens to be.
* `max_plateau_fraction` -- the tie plateau at the maximum as a fraction of all
  negatives, which is the FPR you are forced to accept if you audit anything
  from the plateau.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np


@dataclass
class TieStats:
    channel: str
    n: int
    n_distinct: int
    n_tied_at_max: int
    max_value: float
    distinct_in_top_1pct: int
    n_in_top_1pct: int
    max_plateau_fraction_of_negatives: float
    resolution_floor_fpr: float
    largest_plateau: int
    fraction_in_any_tie: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def tie_stats(scores: Sequence[float], labels: Sequence[int], channel: str,
              top_frac: float = 0.01) -> TieStats:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if s.size == 0:
        return TieStats(channel, 0, 0, 0, float("nan"), 0, 0,
                        float("nan"), float("nan"), 0, float("nan"))

    counts = Counter(s.tolist())
    mx = float(s.max())
    n_top = max(1, int(round(len(s) * top_frac)))
    order = np.argsort(-s, kind="stable")
    top = s[order[:n_top]]

    n_neg = int((y == 0).sum())
    neg_at_max = int(((y == 0) & (s == mx)).sum())

    # The smallest FPR expressible: audit only the single highest-scoring
    # distinct value, and accept every negative that ties with it.
    distinct_desc = sorted(counts, reverse=True)
    floor = float("nan")
    if n_neg:
        for v in distinct_desc:
            negs = int(((y == 0) & (s >= v)).sum())
            if negs > 0:
                floor = negs / n_neg
                break
        else:
            floor = 0.0

    in_tie = sum(c for c in counts.values() if c > 1)
    return TieStats(
        channel=channel,
        n=int(s.size),
        n_distinct=len(counts),
        n_tied_at_max=int(counts[mx]),
        max_value=mx,
        distinct_in_top_1pct=int(len(set(top.tolist()))),
        n_in_top_1pct=int(n_top),
        max_plateau_fraction_of_negatives=(neg_at_max / n_neg) if n_neg else float("nan"),
        resolution_floor_fpr=floor,
        largest_plateau=max(counts.values()),
        fraction_in_any_tie=in_tie / len(s),
    )


def score_histogram(scores: Sequence[float], labels: Sequence[int],
                    bins: int | Sequence[float] = 21) -> dict[str, Any]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    edges = np.histogram_bin_edges(s, bins=bins)
    return {
        "edges": edges.tolist(),
        "attack": np.histogram(s[y == 1], bins=edges)[0].tolist(),
        "benign": np.histogram(s[y == 0], bins=edges)[0].tolist(),
    }


def integer_tie_table(scores: Sequence[float], labels: Sequence[int]) -> list[dict[str, Any]]:
    """Per-value counts for an integer channel -- the tie table for Gate 1."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    rows: list[dict[str, Any]] = []
    for v in sorted(set(s.tolist()), reverse=True):
        m = s == v
        rows.append({
            "value": v,
            "n": int(m.sum()),
            "n_attack": int((m & (y == 1)).sum()),
            "n_benign": int((m & (y == 0)).sum()),
        })
    return rows


def usable_signal_depth(scores: Sequence[float], labels: Sequence[int],
                        fpr_grid: Sequence[float]) -> dict[float, bool]:
    """For each audit budget, can this channel express it at all?

    A channel "retains usable signal" at FPR f if it has an operating point
    with FPR <= f that flags at least one trajectory. This is the practical
    form of the tie-plateau problem: below a channel's resolution floor, the
    audit set is decided by tie-breaking rather than by the monitor.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_neg = int((y == 0).sum())
    out: dict[float, bool] = {}
    for f in fpr_grid:
        if n_neg == 0:
            out[f] = False
            continue
        budget = f * n_neg
        found = False
        for v in sorted(set(s.tolist()), reverse=True):
            negs = int(((y == 0) & (s >= v)).sum())
            flagged = int((s >= v).sum())
            if negs <= budget and flagged > 0:
                found = True
                break
        out[f] = found
    return out
