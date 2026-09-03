"""Stage 3 -- probe training and the in-distribution channel comparison.

Trains an L2 logistic probe on the cached activations with a full layer sweep,
selecting both the layer and the regularisation strength on validation, and
fits a small MLP head on the same activations as a stronger comparison. Then
evaluates every channel on the test split with bootstrap CIs, and runs the
paired-bootstrap comparisons listed in the preregistration.

Writes:
  results/probe/layer_sweep.csv       per-layer, per-seed, per-representation
  results/probe/channels.csv          AUROC + TPR@FPR with CIs, per channel
  results/probe/comparisons.csv       paired-bootstrap differences
  results/probe/test_scores.csv       per-trajectory scores, all channels
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.schema import Trajectory  # noqa: E402
from src.data.splits import assert_disjoint, random_split  # noqa: E402
from src.monitor.activations import ActivationStore  # noqa: E402
from src.probe.evaluate import (  # noqa: E402
    attach_probe_column, available_channels, compare_channels, ensemble_score,
    evaluate_channels, probe_scores_for,
)
from src.probe.train import (  # noqa: E402
    ProbeFit, fit_head, layer_sweep, load_xy,
)
from src.utils.checkpoint import cell, cell_dir  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger, stage_timer  # noqa: E402
from src.utils.seeding import provenance, seed_everything  # noqa: E402

CELL = "probe"

# The exhaustive comparison list from PREREGISTRATION.md section 7.
COMPARISONS = [
    ("logit_ev", "emitted"),
    ("logit_logodds_80", "emitted"),
    ("probe", "emitted"),
    ("probe", "logit_ev"),
    ("head", "probe"),
    ("ensemble", "emitted"),
    ("ensemble", "logit_ev"),
    ("ensemble", "probe"),
    ("emitted", "length_baseline"),
    ("logit_ev", "length_baseline"),
    ("probe", "length_baseline"),
    ("probe", "structure_baseline"),
    ("logit_ev", "structure_baseline"),
    ("emitted", "structure_baseline"),
]


def load_scores(tag: str = "primary", cfg: dict | None = None) -> pd.DataFrame:
    """Scored trajectories, optionally restricted to chosen agent models.

    The restriction exists to remove the agent-identity shortcut documented in
    DECISIONS.md D-030: agent identity alone predicts the label at AUROC 0.805,
    and a probe on mean-pooled activations exploits it to reach 0.99 without
    detecting anything about attacks.
    """
    path = os.path.join(cell_dir("scores"), f"scores_{tag}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing -- run stage 2 first.")
    df = pd.read_csv(path)
    # Join the render/structure columns so the structural baseline and the
    # thinking-block matching can be computed here.
    meta = pd.read_csv(os.path.join(cell_dir("data"), "metadata.csv"))
    scols = [c for c in meta.columns
             if c.startswith("render_") or c == "n_messages"]
    df = df.merge(meta[["traj_id", *scols]], on="traj_id", how="left",
                  suffixes=("", "_meta"))

    acfg = (cfg or {}).get("analysis") or {}
    log = get_logger()
    keep = acfg.get("restrict_agent_models")
    if keep:
        before = len(df)
        df = df[df.agent_model.isin(keep)].reset_index(drop=True)
        log.info(f"agent restriction {keep}: {len(df)} of {before} trajectories "
                 f"({int((df.label == 0).sum())} benign / {int((df.label == 1).sum())} attack)")
    if acfg.get("require_zero_thinking"):
        before = len(df)
        df = df[df.render_thinking_blocks_stripped.fillna(0) == 0].reset_index(drop=True)
        log.info(f"zero-thinking matching: {len(df)} of {before} trajectories "
                 f"({int((df.label == 0).sum())} benign / {int((df.label == 1).sum())} attack)")
    return df


def add_structure_baseline(df: pd.DataFrame, train_ids, feats, seed: int,
                           k: int = 5) -> pd.DataFrame:
    """A logistic baseline on structural features alone, cross-fitted.

    Measures how far trajectory *shape* -- message count, rendered size, elision
    counts -- carries the label without reading any content. Any channel that
    does not clearly beat it has not demonstrated that it detects attacks.

    Cross-fitted rather than train/test split: the zero-shot channels are scored
    on every trajectory, so a baseline scored only on the held-out split would be
    compared against them on a different and much smaller denominator. K-fold
    out-of-fold scoring gives every trajectory a score from a model that never
    saw it, making the comparison like-for-like. `train_ids` is retained for
    signature compatibility and is no longer used for fitting.
    """
    import numpy as _np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in feats if c in df.columns]
    if not cols or df.label.nunique() < 2 or len(df) < 2 * k:
        return df
    X = df[cols].astype(float).fillna(0.0).to_numpy()
    y = df.label.to_numpy(int)
    oof = _np.full(len(df), _np.nan)
    for tr, te in StratifiedKFold(n_splits=k, shuffle=True,
                                  random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, random_state=seed).fit(
            sc.transform(X[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(X[te]))
    df = df.copy()
    df["structure_baseline"] = oof
    return df


def as_trajectories(df: pd.DataFrame) -> list[Trajectory]:
    """Minimal Trajectory objects, enough for the split functions."""
    return [
        Trajectory(
            traj_id=r["traj_id"], task=r["task"], task_suite=r.get("task_suite", ""),
            family_dir=r.get("family_dir", ""), strategy=r["strategy"],
            hint_condition=r.get("hint_condition", ""),
            agent_model=r.get("agent_model", ""), label=int(r["label"]),
            side_task_success=None, original_monitor_verdict=None, n_messages=0,
        )
        for r in df.to_dict("records")
    ]


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

    with stage_timer("03_probe_train"), cell(CELL, provenance(cfg["seed"], cfg),
                                             force=args.force) as out:
        if out.get("_skipped"):
            return 0

        df = load_scores(args.tag, cfg)
        df = df[df.traj_id.map(store.has)].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("no trajectories with cached activations -- rerun stage 2")
        labels = dict(zip(df.traj_id, df.label.astype(int)))
        log.info(f"{len(df)} trajectories with activations; "
                 f"{int(df.label.sum())} attack / {int((df.label == 0).sum())} benign")

        trajs = as_trajectories(df)
        split = random_split(trajs, **cfg["split"], seed=cfg["seed"])
        assert_disjoint(split)
        log.info(f"split sizes: {split.sizes()}")

        sweep_rows: list[dict] = []
        per_seed_channels: list[pd.DataFrame] = []
        per_seed_comparisons: list[pd.DataFrame] = []
        best_by_seed: dict[int, ProbeFit] = {}
        test_frames: list[pd.DataFrame] = []

        for seed in cfg["seeds"]:
            seed_everything(seed)
            best_overall: ProbeFit | None = None
            for rep in cfg["probe"]["representations"]:
                fits, best = layer_sweep(
                    store, labels, split.train, split.val, rep,
                    cfg["probe"]["C_grid"], seed,
                    max_iter=cfg["probe"]["max_iter"],
                    standardize=cfg["probe"]["standardize"],
                )
                for f in fits:
                    sweep_rows.append(f.to_row())
                log.info(f"seed {seed} rep {rep}: best layer {best.layer} "
                         f"(val AUROC {best.val_auroc:.3f}, C={best.C})")
                if best_overall is None or best.val_auroc > best_overall.val_auroc:
                    best_overall = best
            assert best_overall is not None
            best_by_seed[seed] = best_overall

            test_df = df[df.traj_id.isin(split.test)].copy()
            s, ids = probe_scores_for(store, best_overall, split.test, labels)
            test_df = attach_probe_column(test_df, ids, s, "probe")

            # Stronger comparison: an MLP head on the same activations.
            Xtr, ytr, _ = load_xy(store, split.train, labels,
                                  best_overall.representation, best_overall.layer_pos)
            Xva, yva, _ = load_xy(store, split.val, labels,
                                  best_overall.representation, best_overall.layer_pos)
            head, hscaler, hval, htrain = fit_head(
                Xtr, ytr, Xva, yva, cfg["head"], seed,
                standardize=cfg["probe"]["standardize"],
            )
            head_fit = ProbeFit(
                kind="head", representation=best_overall.representation,
                layer=best_overall.layer, layer_pos=best_overall.layer_pos, C=None,
                seed=seed, val_auroc=hval, train_auroc=htrain, layer_tuned=True,
                model=head, scaler=hscaler,
            )
            sweep_rows.append(head_fit.to_row())
            hs, hids = probe_scores_for(store, head_fit, split.test, labels)
            test_df = attach_probe_column(test_df, hids, hs, "head")
            log.info(f"seed {seed}: head val AUROC {hval:.3f} vs probe {best_overall.val_auroc:.3f}")

            # Fit on the full frame (which contains the training rows) and keep
            # only the test rows: passing test_df alone left no training rows to
            # fit on, and the baseline silently never appeared.
            scored = add_structure_baseline(
                df, split.train, cfg["analysis"].get("structure_features", []), seed)
            if "structure_baseline" in scored.columns:
                test_df = test_df.merge(
                    scored[["traj_id", "structure_baseline"]], on="traj_id", how="left")
            chans = available_channels(test_df) + ["length_baseline"]
            if "structure_baseline" in test_df.columns:
                chans.append("structure_baseline")
            test_df["ensemble"] = ensemble_score(
                test_df, [c for c in ("emitted", "logit_ev", "probe") if c in test_df]
            )
            chans = chans + ["ensemble"]

            ch = evaluate_channels(
                test_df, chans, cfg["eval"]["fpr_targets"],
                n_resamples=cfg["eval"]["bootstrap_resamples"],
                ci=cfg["eval"]["ci"], seed=seed,
            )
            ch["seed"] = seed
            ch["probe_layer"] = best_overall.layer
            ch["probe_representation"] = best_overall.representation
            per_seed_channels.append(ch)

            cmp_df = compare_channels(
                test_df, COMPARISONS, cfg["eval"]["fpr_targets"],
                n_resamples=cfg["eval"]["bootstrap_resamples"],
                ci=cfg["eval"]["ci"], seed=seed,
            )
            cmp_df["seed"] = seed
            per_seed_comparisons.append(cmp_df)

            test_df["seed"] = seed
            test_frames.append(test_df)

        sweep = pd.DataFrame(sweep_rows)
        sweep.to_csv(os.path.join(d, "layer_sweep.csv"), index=False)
        channels = pd.concat(per_seed_channels, ignore_index=True)
        channels.to_csv(os.path.join(d, "channels.csv"), index=False)
        comps = pd.concat(per_seed_comparisons, ignore_index=True)
        comps.to_csv(os.path.join(d, "comparisons.csv"), index=False)
        pd.concat(test_frames, ignore_index=True).to_csv(
            os.path.join(d, "test_scores.csv"), index=False)

        agg = channels.groupby("channel")[["auroc"]].agg(["mean", "std"]).round(4)
        log.info(f"AUROC by channel (across seeds):\n{agg}")

        with open(os.path.join(d, "split.json"), "w") as fh:
            json.dump({"train": split.train, "val": split.val, "test": split.test}, fh)

        out.update({
            "n": len(df), "split_sizes": split.sizes(),
            "seeds": cfg["seeds"],
            "best_layer_by_seed": {s: f.layer for s, f in best_by_seed.items()},
            "best_rep_by_seed": {s: f.representation for s, f in best_by_seed.items()},
            "layer_tuned": True,
            "auroc_by_channel": channels.groupby("channel").auroc.mean().round(4).to_dict(),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
