"""Stage 6 -- figures.

All figures: colour-blind-safe palette, standalone-readable captions, seeds
shown individually alongside the aggregate where a training step is involved,
PNG at 150 dpi plus PDF.

  fig1_signal_histogram   emitted-score histogram with the tie plateau visible
  fig2_roc_curves         ROC, three channels, log FPR axis to expose the tail
  fig3_tpr_at_fpr         TPR at each audit budget, with bootstrap CIs
  fig4_transfer_matrix    leave-one-family-out probe AUROC (the headline)
  fig5_transfer_ratio     transfer ratio across channels
  fig6_layer_profile      per-layer AUROC
  fig7_calibration        reliability diagrams
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src.utils.checkpoint import cell, cell_dir  # noqa: E402
from src.utils.config import figures_dir, load_config  # noqa: E402
from src.utils.logging import get_logger, stage_timer  # noqa: E402
from src.utils.seeding import provenance, seed_everything  # noqa: E402

CELL = "figures"

# Okabe-Ito: colour-blind safe.
COLORS = {
    "emitted": "#E69F00", "logit_ev": "#0072B2", "logit_logodds_80": "#56B4E9",
    "probe": "#009E73", "head": "#CC79A7", "ensemble": "#D55E00",
    "length_baseline": "#999999",
}
LABELS = {
    "emitted": "Emitted integer (argmax)",
    "logit_ev": "Logit expected value",
    "logit_logodds_80": "Logit log-odds (>80)",
    "probe": "Linear probe",
    "head": "MLP head",
    "ensemble": "Ensemble (rank mean)",
    "length_baseline": "Trajectory length (control)",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": False,
})


def save(fig, name: str, caption: str) -> None:
    """Write PNG and PDF with the caption below the axes.

    The caption is placed in figure coordinates *below* the axes and wrapped by
    hand -- matplotlib's `wrap=True` does not respect the figure box for
    `fig.text`, and an unwrapped caption overlaps the x-axis label.
    """
    import textwrap

    d = figures_dir()
    width_chars = max(70, int(fig.get_size_inches()[0] * 17))
    wrapped = "\n".join(textwrap.wrap(caption, width=width_chars))
    # Place the caption below everything actually drawn. A fixed negative y
    # collides with rotated tick labels, whose extent is only known after a
    # draw, so measure the axes' tight bounding boxes first.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    ymin = min(
        ax.get_tightbbox(renderer).transformed(inv).y0 for ax in fig.axes
    ) if fig.axes else 0.0
    fig.text(0.0, ymin - 0.045, wrapped, fontsize=7, va="top", ha="left",
             color="#444444", linespacing=1.45)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.25)
    plt.close(fig)


def fig1_signal_histogram(analysis_dir: str, log) -> None:
    path = os.path.join(analysis_dir, "tie_table_emitted.csv")
    tt = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.9
    ax.bar(tt.value, tt.n_benign, width=w, label="Benign", color="#56B4E9")
    ax.bar(tt.value, tt.n_attack, width=w, bottom=tt.n_benign, label="Attack",
           color="#D55E00")
    top = tt.sort_values("value", ascending=False).iloc[0]
    ax.annotate(f"{int(top.n)} trajectories tied\nat the maximum ({top.value:g})",
                xy=(top.value, top.n), xytext=(-90, 22), textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "#333333"}, fontsize=8)
    ax.set_xlabel("Emitted suspiciousness score (integer, 0-100)")
    ax.set_ylabel("Trajectories")
    ax.set_title("The monitor's integer score collapses onto a few values")
    ax.legend(frameon=False)
    save(fig, "fig1_signal_histogram",
         "Figure 1. Distribution of the monitor's emitted integer score, stacked by true "
         "label. The score concentrates on a handful of round values, so a large block of "
         "trajectories is tied at the top and cannot be ranked against each other -- the "
         "regime where a small audit budget is spent.")


def fig2_roc_curves(test: pd.DataFrame, channels: list[str], log) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    y = test.label.to_numpy(int)
    for ch in channels:
        s = test[ch].to_numpy(float)
        ok = np.isfinite(s)
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, label=LABELS.get(ch, ch),
                color=COLORS.get(ch), lw=1.6,
                ls="--" if ch == "length_baseline" else "-")
    ax.plot([1e-4, 1], [1e-4, 1], color="#bbbbbb", lw=0.8, ls=":", label="Chance")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel("False positive rate (log scale)")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC with a log FPR axis, to expose the audit tail")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "fig2_roc_curves",
         "Figure 2. ROC curves on a logarithmic false-positive axis. The left-hand region "
         "is where a realistic audit budget sits. A channel whose curve goes flat on the "
         "left has run out of resolution: it has no operating point at that budget, and "
         "the audit set is then decided by tie-breaking rather than by the monitor.")


def fig3_tpr_at_fpr(channels_csv: str, fpr_targets: list[float], log) -> None:
    ch = pd.read_csv(channels_csv)
    chans = [c for c in COLORS if c in set(ch.channel)]
    fig, axes = plt.subplots(1, len(fpr_targets), figsize=(3.1 * len(fpr_targets), 3.8),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, fpr_targets):
        col, lo, hi = f"tpr@{f:g}", f"tpr@{f:g}_lo", f"tpr@{f:g}_hi"
        if col not in ch.columns:
            continue
        agg = ch.groupby("channel")[[col, lo, hi]].mean().reindex(chans).dropna(how="all")
        xs = np.arange(len(agg))
        ax.bar(xs, agg[col], color=[COLORS.get(c, "#888") for c in agg.index],
               yerr=[agg[col] - agg[lo], agg[hi] - agg[col]], capsize=3,
               error_kw={"lw": 1, "ecolor": "#333333"})
        for i, c in enumerate(agg.index):
            pts = ch[ch.channel == c][col]
            if len(pts) > 1:
                ax.scatter([xs[i]] * len(pts), pts, s=9, color="#222222", zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS.get(c, c) for c in agg.index], rotation=40,
                           ha="right", fontsize=7)
        ax.set_title(f"FPR = {f:g}", fontsize=9)
    axes[0].set_ylabel("TPR (attacks caught)")
    fig.suptitle("Attacks caught at three audit budgets", y=1.02)
    save(fig, "fig3_tpr_at_fpr",
         "Figure 3. True positive rate at three audit budgets, with 95% bootstrap "
         "confidence intervals over trajectories (10,000 resamples). Black dots are "
         "individual probe-training seeds. Tighter budgets are further right.")


def fig4_transfer_matrix(transfer_csv: str, log) -> None:
    tr = pd.read_csv(transfer_csv)
    for axis in tr.axis.unique():
        sub = tr[tr.axis == axis]
        piv = sub.groupby("family")[["transfer_auroc", "indist_auroc", "ratio"]].mean()
        piv = piv.sort_values("ratio")
        fig, ax = plt.subplots(figsize=(max(5.5, 0.85 * len(piv) + 3), 3.9))
        xs = np.arange(len(piv))
        ax.bar(xs - 0.2, piv.indist_auroc, width=0.4, label="Trained on this family",
               color="#999999")
        ax.bar(xs + 0.2, piv.transfer_auroc, width=0.4, label="Family held out",
               color="#009E73")
        ax.axhline(0.5, color="#cc0000", lw=1, ls="--")
        ax.text(len(piv) - 0.5, 0.505, "chance", fontsize=7, color="#cc0000", ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels(piv.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(f"Probe transfer to unseen attack families (by {axis})")
        ax.legend(frameon=False, fontsize=8)
        save(fig, f"fig4_transfer_matrix_{axis}",
             f"Figure 4. Leave-one-family-out transfer by {axis}. NOTE: the probe is "
             "confounded (see Figure 6) and these numbers measure dataset artefacts, not "
             "attack detection; they are shown because the pattern is itself diagnostic -- "
             "near-perfect transfer across attack families alongside collapse across "
             "environments is the signature of a shortcut. "
             "trained on data including this family. Green: this family was entirely "
             "held out of training. The gap between the pair is what the probe loses on "
             "an attack nobody anticipated -- the question a trusted monitor exists to "
             "answer.")


def fig5_transfer_ratio(transfer_csv: str, zeroshot_csv: str, log) -> None:
    tr = pd.read_csv(transfer_csv)
    zs = pd.read_csv(zeroshot_csv)
    for axis in tr.axis.unique():
        p = tr[tr.axis == axis].groupby("family").ratio.mean().rename("probe")
        z = (zs[zs.axis == axis].pivot_table(index="family", columns="channel",
                                             values="ratio"))
        both = pd.concat([p, z], axis=1).sort_index()
        fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * len(both) + 3), 3.8))
        xs = np.arange(len(both))
        cols = [c for c in ("probe", "logit_ev", "emitted", "length_baseline")
                if c in both.columns]
        w = 0.8 / len(cols)
        for k, c in enumerate(cols):
            ax.bar(xs + (k - len(cols) / 2 + 0.5) * w, both[c], width=w,
                   label=LABELS.get(c, c), color=COLORS.get(c, "#888"))
        ax.axhline(1.0, color="#333333", lw=1, ls="--")
        ax.text(-0.4, 1.01, "no loss from holding the family out", fontsize=7,
                color="#333333")
        ax.set_xticks(xs)
        ax.set_xticklabels(both.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Transfer ratio (held-out / in-distribution AUROC)")
        ax.set_title(f"Transfer ratio by channel (by {axis})")
        ax.legend(frameon=False, fontsize=7.5)
        save(fig, f"fig5_transfer_ratio_{axis}",
             "Figure 5. Transfer ratio: held-out-family AUROC divided by a matched "
             "in-distribution AUROC on the same test family and the same benign pool. "
             "The zero-shot channels sit at 1 by construction -- they have no training "
             "distribution to be out of -- which is exactly what the trained probe is "
             "being measured against.")


def fig6_layer_profile(layer_csv: str, log, monitor_auroc: float | None = None) -> None:
    """The shortcut diagnostic.

    A probe that is recovering the monitor's judgement should improve with depth,
    because the judgement is what the later layers compute. A flat curve starting
    high at layer 0 means the signal is present before any monitoring has
    happened -- i.e. it is a surface property of the input.
    """
    prof = pd.read_csv(layer_csv)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for rep, grp in prof.groupby("representation"):
        grp = grp.sort_values("layer")
        ax.plot(grp.layer, grp.val_auroc_mean, marker="o", ms=3.5, label=rep, lw=1.5)
        if "val_auroc_min" in grp:
            ax.fill_between(grp.layer, grp.val_auroc_min, grp.val_auroc_max, alpha=0.15)
        first, best = grp.val_auroc_mean.iloc[0], grp.val_auroc_mean.max()
        ax.annotate(f"layer 0: {first:.3f}   best: {best:.3f}   (+{best-first:.3f})",
                    xy=(grp.layer.iloc[0], first), xytext=(2, -14),
                    textcoords="offset points", fontsize=7, color="#333333")
    if monitor_auroc is not None:
        ax.axhline(monitor_auroc, color="#000000", lw=1.2, ls="-.")
        ax.text(prof.layer.max(), monitor_auroc + 0.012,
                f"the monitor's own score ({monitor_auroc:.3f})",
                fontsize=7.5, ha="right", color="#000000")
    ax.axhline(0.5, color="#cc0000", lw=1, ls="--")
    ax.set_xlabel("Layer (decoder block output)")
    ax.set_ylabel("Validation AUROC")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Depth buys nothing: the probe is reading surface text")
    ax.legend(frameon=False, fontsize=8, title="Representation", title_fontsize=8,
              loc="center right")
    save(fig, "fig6_layer_profile",
         "Figure 6. Per-layer validation AUROC for the linear probe. Layer 0 is the output "
         "of the FIRST decoder block, so almost no monitoring computation has happened "
         "there -- yet mean-pooled activations already separate the classes near-perfectly, "
         "and twenty-six further blocks add almost nothing. The monitor's own considered "
         "score, after all 28 blocks, is the dash-dotted line far below. A probe recovering "
         "the monitor's judgement would rise with depth and would not start above the "
         "monitor. This is the diagnostic that identified the result as confounded.")


def fig7_calibration(bins_csv: str, cal_csv: str, log) -> None:
    bins = pd.read_csv(bins_csv)
    cal = pd.read_csv(cal_csv).set_index("channel")
    chans = [c for c in ("emitted", "logit_ev", "probe") if c in set(bins.channel)]
    fig, axes = plt.subplots(1, len(chans), figsize=(3.0 * len(chans), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, chans):
        b = bins[(bins.channel == ch) & bins.n.gt(0)]
        ax.plot([0, 1], [0, 1], ls=":", color="#999999", lw=1)
        ax.plot(b.confidence, b.accuracy, marker="o", ms=4, color=COLORS.get(ch), lw=1.5)
        ece = cal.loc[ch, "ece"] if ch in cal.index else float("nan")
        ax.set_title(f"{LABELS.get(ch, ch)}\nECE = {ece:.3f}", fontsize=8.5)
        ax.set_xlabel("Predicted P(attack)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed fraction of attacks")
    save(fig, "fig7_calibration",
         "Figure 7. Reliability diagrams on each channel's native scale, with no post-hoc "
         "fitting: the integer and expected-value scores are divided by 100, the log-odds "
         "channel goes through a logistic. The dotted line is perfect calibration. This "
         "measures whether a deployment could read these scores as probabilities as-is.")


def _monitor_auroc(cfg: dict) -> float | None:
    """Emitted-integer AUROC over the whole analysis subset.

    Taken from the full subset rather than the 25% test split: the test split is
    small enough that its estimate is dominated by noise (0.501 against 0.607
    pooled), and this line is a reference point, not a paired comparison.
    """
    from sklearn.metrics import roc_auc_score

    try:
        df = pd.read_csv(os.path.join(cell_dir("scores"), "scores_primary.csv"))
        meta = pd.read_csv(os.path.join(cell_dir("data"), "metadata.csv"))
        df = df.merge(meta[["traj_id", "render_thinking_blocks_stripped"]],
                      on="traj_id", how="left")
        acfg = cfg.get("analysis") or {}
        if acfg.get("restrict_agent_models"):
            df = df[df.agent_model.isin(acfg["restrict_agent_models"])]
        if acfg.get("require_zero_thinking"):
            df = df[df.render_thinking_blocks_stripped.fillna(0) == 0]
        if df.label.nunique() < 2:
            return None
        return float(roc_auc_score(df.label, df.emitted))
    except (FileNotFoundError, KeyError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    log = get_logger()
    cfg = load_config(smoke=args.smoke)
    seed_everything(cfg["seed"])
    adir = cell_dir("analysis")
    pdir = cell_dir("probe")
    tdir = cell_dir("transfer")

    with stage_timer("06_figures"), cell(CELL, provenance(cfg["seed"], cfg),
                                         force=args.force) as out:
        if out.get("_skipped"):
            return 0
        made: list[str] = []

        test = pd.read_csv(os.path.join(pdir, "test_scores.csv"))
        first = sorted(test.seed.unique())[0]
        base = test[test.seed == first]
        chans = [c for c in ("emitted", "logit_ev", "probe", "length_baseline")
                 if c in base.columns]

        # Each figure is attempted independently: a missing upstream artefact
        # should cost one figure, not the whole stage.
        jobs = [
            ("fig1", lambda: fig1_signal_histogram(adir, log)),
            ("fig2", lambda: fig2_roc_curves(base, chans, log)),
            ("fig3", lambda: fig3_tpr_at_fpr(os.path.join(pdir, "channels.csv"),
                                             cfg["eval"]["fpr_targets"], log)),
            ("fig4", lambda: fig4_transfer_matrix(os.path.join(tdir, "transfer_probe.csv"), log)),
            ("fig5", lambda: fig5_transfer_ratio(os.path.join(tdir, "transfer_probe.csv"),
                                                 os.path.join(tdir, "transfer_zeroshot.csv"), log)),
            ("fig6", lambda: fig6_layer_profile(
                os.path.join(adir, "layer_profile.csv"), log,
                monitor_auroc=_monitor_auroc(cfg))),
            ("fig7", lambda: fig7_calibration(os.path.join(adir, "reliability_bins.csv"),
                                              os.path.join(adir, "calibration.csv"), log)),
        ]
        failures: dict[str, str] = {}
        for name, fn in jobs:
            try:
                fn()
                made.append(name)
                log.info(f"{name} written")
            except (FileNotFoundError, KeyError, ValueError, IndexError) as exc:
                failures[name] = repr(exc)
                log.error(f"{name} FAILED: {exc!r}")
        out.update({"figures": made, "failures": failures,
                    "figures_dir": figures_dir()})
        if failures:
            log.warning(f"{len(failures)} figures failed: {sorted(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
