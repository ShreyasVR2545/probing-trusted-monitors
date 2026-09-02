"""Leave-one-family-out transfer evaluation -- the headline result (RQ2).

A trusted monitor earns its place by catching attacks nobody anticipated. So
the question is not whether a probe separates attacks from benign trajectories
it has seen the like of, but whether it does so for an attack *family* it has
never seen.

For each family F:

* **Transfer** -- train on every family except F, test on F.
* **In-distribution control** -- train on every family *including* F (minus a
  held-out slice), test on that held-out slice of F.
* **Transfer ratio** = transfer AUROC / in-distribution AUROC.

Numerator and denominator use the same test family and the same benign pool, so
the only thing that differs is whether F was in training (DECISIONS.md D-009).

The zero-shot channels (`emitted`, `logit_ev`) are run through exactly the same
per-family evaluation. Their transfer ratio is 1 by construction -- they have no
training distribution to be out of -- and that is precisely the comparison: if
the probe's ratio is well below 1 while its absolute held-out AUROC is no better
than the zero-shot channels', the probe is not buying anything for unanticipated
attacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.splits import Split, assert_disjoint
from src.eval.bootstrap import bootstrap_metric, ratio_ci
from src.eval.metrics import auroc, tpr_at_fpr
from src.monitor.activations import ActivationStore
from src.probe.train import ProbeFit, fit_linear_probe, load_xy, score_probe


@dataclass
class TransferResult:
    axis: str
    family: str
    seed: int
    representation: str
    layer: int
    C: float | None
    n_test_attack: int
    n_test_benign: int
    transfer_auroc: float
    transfer_lo: float
    transfer_hi: float
    indist_auroc: float
    indist_lo: float
    indist_hi: float
    ratio: float
    ratio_lo: float
    ratio_hi: float
    val_auroc: float

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _fit_on_split(store: ActivationStore, labels: dict[str, int], split: Split,
                  representation: str, layer_pos: int, layer: int,
                  C_grid: Sequence[float], seed: int, max_iter: int,
                  standardize: bool) -> ProbeFit:
    assert_disjoint(split)
    Xtr, ytr, _ = load_xy(store, split.train, labels, representation, layer_pos)
    Xva, yva, _ = load_xy(store, split.val, labels, representation, layer_pos)
    clf, scaler, C, va, tr = fit_linear_probe(
        Xtr, ytr, Xva, yva, C_grid, seed, max_iter, standardize
    )
    return ProbeFit(kind="linear", representation=representation, layer=layer,
                    layer_pos=layer_pos, C=C, seed=seed, val_auroc=va,
                    train_auroc=tr, layer_tuned=True, model=clf, scaler=scaler)


def run_transfer(
    store: ActivationStore,
    labels: dict[str, int],
    transfer_splits: Sequence[Split],
    indist_splits: Sequence[Split],
    representation: str,
    layer_pos: int,
    layer: int,
    C_grid: Sequence[float],
    seed: int,
    n_resamples: int = 10000,
    ci: float = 0.95,
    max_iter: int = 2000,
    standardize: bool = True,
    logger: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every leave-one-family-out direction.

    Returns (per-family results, per-trajectory test scores). The second frame
    is kept so the transfer figures can show the score distributions, not only
    the summary numbers.
    """
    by_family = {s.held_out_family: s for s in indist_splits}
    results: list[TransferResult] = []
    score_rows: list[dict[str, Any]] = []

    for split in transfer_splits:
        fam = split.held_out_family
        control = by_family.get(fam)
        if control is None:
            raise KeyError(f"no in-distribution control split for family {fam!r}")

        fit = _fit_on_split(store, labels, split, representation, layer_pos, layer,
                            C_grid, seed, max_iter, standardize)
        Xte, yte, ids_te = load_xy(store, split.test, labels, representation, layer_pos)
        s_te = score_probe(fit, Xte)

        ctrl = _fit_on_split(store, labels, control, representation, layer_pos, layer,
                             C_grid, seed, max_iter, standardize)
        Xin, yin, _ = load_xy(store, control.test, labels, representation, layer_pos)
        s_in = score_probe(ctrl, Xin)

        t_ci = bootstrap_metric(s_te, yte, lambda a, b: auroc(a, b),
                                n_resamples=n_resamples, ci=ci, seed=seed)
        i_ci = bootstrap_metric(s_in, yin, lambda a, b: auroc(a, b),
                                n_resamples=n_resamples, ci=ci, seed=seed)
        r_ci = ratio_ci(s_te, yte, s_in, yin, lambda a, b: auroc(a, b),
                        n_resamples=n_resamples, ci=ci, seed=seed)

        results.append(TransferResult(
            axis=split.axis or "?", family=fam or "?", seed=seed,
            representation=representation, layer=layer, C=fit.C,
            n_test_attack=int((yte == 1).sum()), n_test_benign=int((yte == 0).sum()),
            transfer_auroc=t_ci.point, transfer_lo=t_ci.lo, transfer_hi=t_ci.hi,
            indist_auroc=i_ci.point, indist_lo=i_ci.lo, indist_hi=i_ci.hi,
            ratio=r_ci.point, ratio_lo=r_ci.lo, ratio_hi=r_ci.hi,
            val_auroc=fit.val_auroc,
        ))
        for tid, sc, lab in zip(ids_te, s_te.tolist(), yte.tolist()):
            score_rows.append({"axis": split.axis, "held_out_family": fam,
                               "seed": seed, "traj_id": tid, "probe": sc, "label": lab})
        if logger:
            logger.info(
                f"[transfer:{split.axis}] {fam}: held-out AUROC {t_ci.point:.3f} "
                f"[{t_ci.lo:.3f}, {t_ci.hi:.3f}] | in-dist {i_ci.point:.3f} "
                f"| ratio {r_ci.point:.3f}"
            )
    return (pd.DataFrame([r.to_row() for r in results]),
            pd.DataFrame(score_rows))


def zero_shot_transfer(
    df: pd.DataFrame, axis: str, channels: Sequence[str],
    transfer_splits: Sequence[Split], indist_splits: Sequence[Split],
    fpr_targets: Sequence[float], n_resamples: int = 10000, ci: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """The same per-family evaluation for the zero-shot channels.

    These channels are not trained, so "held out" and "in-distribution" differ
    only in which trajectories are being scored. Evaluating them on exactly the
    same test sets as the probe is what makes the transfer comparison fair --
    otherwise the probe's held-out AUROC would be compared against a zero-shot
    number computed on a different population.
    """
    by_family = {s.held_out_family: s for s in indist_splits}
    idx = df.set_index("traj_id")
    rows: list[dict[str, Any]] = []
    for split in transfer_splits:
        fam = split.held_out_family
        control = by_family[fam]
        te = idx.loc[[t for t in split.test if t in idx.index]]
        ind = idx.loc[[t for t in control.test if t in idx.index]]
        for ch in channels:
            if ch not in df.columns:
                continue
            s_te = te[ch].to_numpy(dtype=float)
            y_te = te["label"].to_numpy(dtype=int)
            s_in = ind[ch].to_numpy(dtype=float)
            y_in = ind["label"].to_numpy(dtype=int)
            t_ci = bootstrap_metric(s_te, y_te, lambda a, b: auroc(a, b),
                                    n_resamples=n_resamples, ci=ci, seed=seed)
            i_ci = bootstrap_metric(s_in, y_in, lambda a, b: auroc(a, b),
                                    n_resamples=n_resamples, ci=ci, seed=seed)
            r_ci = ratio_ci(s_te, y_te, s_in, y_in, lambda a, b: auroc(a, b),
                            n_resamples=n_resamples, ci=ci, seed=seed)
            row: dict[str, Any] = {
                "axis": axis, "family": fam, "channel": ch,
                "n_test_attack": int((y_te == 1).sum()),
                "n_test_benign": int((y_te == 0).sum()),
                "transfer_auroc": t_ci.point, "transfer_lo": t_ci.lo,
                "transfer_hi": t_ci.hi, "indist_auroc": i_ci.point,
                "indist_lo": i_ci.lo, "indist_hi": i_ci.hi,
                "ratio": r_ci.point, "ratio_lo": r_ci.lo, "ratio_hi": r_ci.hi,
            }
            for t in fpr_targets:
                row[f"tpr@{t:g}"] = tpr_at_fpr(s_te, y_te, t).tpr
            rows.append(row)
    return pd.DataFrame(rows)
