"""Linear probe training, plus the stronger fine-tuned-head comparison.

Both models read the same cached activations and see exactly the same
trajectories, so the only difference between them is capacity.

Fairness constraints, all enforced here rather than left to the caller:

* Regularisation strength `C` is chosen on the **validation** split. The test
  split is scored once, with the already-chosen `C`.
* Standardisation statistics are fit on the training split only.
* The layer is likewise selected on validation, and every result that used a
  selected layer is marked `layer_tuned=True` so the writeup cannot forget it.
* The probe is fit on labels while the emitted and logit channels are
  zero-shot. That asymmetry is not correctable -- it is stated in the writeup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.monitor.activations import ActivationStore


@dataclass
class ProbeFit:
    kind: str                # "linear" | "head"
    representation: str      # "score_pos" | "mean_pool"
    layer: int
    layer_pos: int
    C: float | None
    seed: int
    val_auroc: float
    train_auroc: float
    layer_tuned: bool = False
    C_tuned: bool = True
    model: Any = field(default=None, repr=False)
    scaler: Any = field(default=None, repr=False)

    def to_row(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "representation": self.representation,
            "layer": self.layer, "layer_pos": self.layer_pos, "C": self.C,
            "seed": self.seed, "val_auroc": self.val_auroc,
            "train_auroc": self.train_auroc, "layer_tuned": self.layer_tuned,
            "C_tuned": self.C_tuned,
        }


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def load_xy(store: ActivationStore, traj_ids: Sequence[str], labels: dict[str, int],
            representation: str, layer_pos: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ids = [t for t in traj_ids if store.has(t)]
    if not ids:
        raise RuntimeError(
            f"no cached activations for any of {len(traj_ids)} requested trajectories "
            f"under {store.root}"
        )
    X = store.matrix(ids, representation, layer_pos)
    y = np.array([labels[t] for t in ids], dtype=int)
    return X, y, ids


def fit_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    C_grid: Sequence[float], seed: int, max_iter: int = 2000,
    standardize: bool = True,
) -> tuple[LogisticRegression, Any, float, float, float]:
    scaler = None
    Xtr, Xva = X_train, X_val
    if standardize:
        scaler = StandardScaler().fit(X_train)
        Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

    best: tuple[float, float, LogisticRegression] | None = None
    for C in C_grid:
        clf = LogisticRegression(
            C=C, penalty="l2", max_iter=max_iter, solver="lbfgs", random_state=seed,
        ).fit(Xtr, y_train)
        va = _safe_auroc(y_val, clf.decision_function(Xva))
        if best is None or (np.isfinite(va) and va > best[0]):
            best = (va, C, clf)
    if best is None:
        raise RuntimeError("no C in the grid produced a fit")
    val_auroc, C, clf = best
    train_auroc = _safe_auroc(y_train, clf.decision_function(Xtr))
    return clf, scaler, C, val_auroc, train_auroc


def score_probe(fit: ProbeFit, X: np.ndarray) -> np.ndarray:
    Xs = fit.scaler.transform(X) if fit.scaler is not None else X
    if fit.kind == "linear":
        return fit.model.decision_function(Xs)
    return _head_scores(fit.model, Xs)


# --------------------------------------------------------------------------
# Fine-tuned classification head
# --------------------------------------------------------------------------

def _head_scores(model: Any, X: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        return model(t).squeeze(-1).cpu().numpy()


def fit_head(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: dict[str, Any], seed: int, standardize: bool = True,
) -> tuple[Any, Any, float, float]:
    """A small MLP on the same activations.

    Exists so the linear probe is not compared only against zero-shot
    baselines: if a two-layer network on these activations does no better than
    the linear probe, then linear decodability is not the limiting factor, and
    that is worth knowing before concluding anything about the probe.
    Early stopping is on validation AUROC.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    scaler = None
    Xtr, Xva = X_train, X_val
    if standardize:
        scaler = StandardScaler().fit(X_train)
        Xtr, Xva = scaler.transform(X_train), scaler.transform(X_val)

    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], cfg["hidden_dim"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden_dim"], 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    n = len(Xt)
    bs = min(cfg["batch_size"], n)
    g = torch.Generator().manual_seed(seed)

    best_val, best_state, best_train = -np.inf, None, float("nan")
    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        va = _safe_auroc(y_val, _head_scores(model, Xva))
        if np.isfinite(va) and va > best_val:
            best_val = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_train = _safe_auroc(y_train, _head_scores(model, Xtr))
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, float(best_val), float(best_train)


# --------------------------------------------------------------------------
# Layer sweep
# --------------------------------------------------------------------------

def layer_sweep(
    store: ActivationStore, labels: dict[str, int],
    train_ids: Sequence[str], val_ids: Sequence[str],
    representation: str, C_grid: Sequence[float], seed: int,
    max_iter: int = 2000, standardize: bool = True,
) -> tuple[list[ProbeFit], ProbeFit]:
    """Fit a probe at every cached layer; return all fits and the val-best one."""
    layers = store.layers()
    fits: list[ProbeFit] = []
    for pos, layer in enumerate(layers):
        Xtr, ytr, _ = load_xy(store, train_ids, labels, representation, pos)
        Xva, yva, _ = load_xy(store, val_ids, labels, representation, pos)
        clf, scaler, C, va, tr = fit_linear_probe(
            Xtr, ytr, Xva, yva, C_grid, seed, max_iter, standardize
        )
        fits.append(ProbeFit(
            kind="linear", representation=representation, layer=int(layer),
            layer_pos=pos, C=C, seed=seed, val_auroc=va, train_auroc=tr,
            model=clf, scaler=scaler,
        ))
    best = max(fits, key=lambda f: (f.val_auroc if np.isfinite(f.val_auroc) else -np.inf))
    best.layer_tuned = True
    return fits, best
