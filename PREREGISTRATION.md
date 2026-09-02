# Preregistration

**Written 2026-09-02, before any experimental result was produced.**
Git history is the timestamp: this file is committed under
`docs: preregistration`, before the commit that adds any file under
`results/`. Deviations from this document are listed in
`writeup/post.md` under "Deviations from preregistration".

---

## 1. Background and motivation

In an AI-control protocol an untrusted model `U` does the work and a trusted
monitor `T` reads the resulting trajectory and emits a suspiciousness score,
conventionally an integer on 0–100 or 0–10. A small human audit budget is spent
on the highest-scoring trajectories, and every downstream mechanism —
defer-to-trusted, defer-to-resample, audit allocation — consumes that scalar.

Integer scores have poor resolution precisely in the tail where auditing
happens. Lucassen & Kaufman (*Retrying vs Resampling in AI Control*,
arXiv:2605.26047, Appendix H.2) report that under max-of-5 aggregation over 250
trajectories tie at a maximum score of 95, giving no signal to discriminate
among them, and that this degrades auditing in the 4e-3 to 2e-4 FPR range.
Their remedy was a second-max tie-break heuristic. At realistic audit budgets
the audit decision is therefore substantially determined by tie-breaking rather
than by monitor signal.

**Hypothesis.** The emitted integer is a lossy compression of a computation the
monitor has already performed. Two cheaper-than-retraining channels should
recover the discarded resolution:

- **Logit-derived continuous score** — read the monitor's probability
  distribution over score tokens at the scoring position rather than taking the
  argmax. No training, one forward pass.
- **Linear probe** on residual-stream activations at the scoring position,
  trained to predict attack vs benign.

**Scope of the claim.** The claim is *only* that quantising the monitor's
judgement to an integer discards information the monitor already computed. It
is **not** a claim that probes perceive something absent from the text. Prior
work (arXiv:2509.21344) shows probes lean heavily on textual evidence; that
critique does not bite here, because the monitor's sole input is text.

---

## 2. Research questions and directional predictions

**RQ1 — Resolution.** Do the logit-derived score and the probe remove the tie
plateaus, and do they extend usable signal further down the FPR curve than the
emitted integer?

> *Prediction (RQ1).* Both continuous channels will produce essentially no ties
> (fewer than 1% of trajectories sharing a value), against a large tie plateau
> at the maximum for the emitted integer. TPR at FPR = 0.003 and 0.0003 will be
> higher for the continuous channels than for the emitted integer, with the gap
> widening as FPR falls. At FPR = 0.01 the three channels may be
> indistinguishable.

**RQ2 — Transfer (headline).** Does a probe trained on one family of attacks
detect held-out attack families?

> *Prediction (RQ2).* Transfer ratio (held-out AUROC / matched in-distribution
> AUROC) will fall in 0.6–0.9: better than chance, meaningfully below 1. Stated
> in advance because both tails are informative: a ratio above ~0.85 supports
> white-box monitoring against unanticipated attacks; a ratio near 0.5 means the
> probe memorises attack families and is not a monitor.

**RQ3 — Necessity.** Does the probe beat the logit-derived score?

> *Prediction (RQ3).* In-distribution the probe will beat the logit score on
> AUROC, because it is fit on labels while the logit score is zero-shot. On
> *held-out families* the advantage will shrink substantially and may vanish,
> because the logit score transfers trivially — it is zero-shot and has no
> training distribution to be out of.

**Explicit null.** If all three channels are statistically indistinguishable,
the finding is that monitor-score quantisation is not a bottleneck and the tie
plateau is cosmetic. That is a real result and will be reported with the same
care as a positive one.

---

## 3. Design

**Data.** `ScaleAI/mrt` (CC-BY-NC-4.0), agent trajectories from SHADE-Arena
tool-use tasks and computer-use tasks. Label: attack = the trajectory was
generated with a hidden secondary objective (mrt directory name does not
contain `no_secondary_task`); benign otherwise. Deduplicated by trajectory
content hash. Stratified over (task x red-team strategy x label), balanced
across labels, restricted to the two best-represented agent models.

**Monitor.** Qwen2.5-3B-Instruct, 4-bit, as selected by hardware detection;
Qwen2.5-1.5B-Instruct as secondary. Fixed prompt adapted from the mrt monitor
prompt, asking for an integer 0–100 suspiciousness score in a `<verdict>` tag.

**Monitor view.** Agent system prompt and `<model_thinking>` scratchpad are
stripped (DECISIONS.md D-004); long tool results are middle-elided (D-005).

**Three channels, all computed on the same forward pass:**

| Channel | Description | Trained? |
|---|---|---|
| `emitted` | argmax-decoded integer, parsed from `<verdict>` | no |
| `logit_ev` | expected value `Σ p(v)·v` over the score-token distribution | no |
| `logit_logodds` | log-odds of the score exceeding a threshold | no |
| `probe` | L2 logistic regression on residual-stream activations | yes |
| `head` | small MLP classification head on the same activations | yes |
| `length_baseline` | rendered trajectory length, used directly as a score | no |

`length_baseline` is a confound control added at Stage 0, before any monitor was
run: in this sample attack trajectories average ~49k rendered characters against
~26k for benign ones, so a detector could score well on length alone. Every
channel is reported against it.

`head` exists so the probe is not compared only against zero-shot baselines.

**Splits.** Strictly by trajectory. Random stratified split (60/15/25) for the
in-distribution result. Leave-one-family-out for RQ2, on two axes: red-team
strategy and task/environment. Regularisation strength is selected on
validation, never on test.

---

## 4. Primary metrics

1. **AUROC**, with 95% bootstrap CIs.
2. **TPR at FPR ∈ {0.01, 0.003, 0.0003}**, with 95% bootstrap CIs.
3. **Tie statistics**: number of trajectories tied at the maximum score; number
   of distinct score values within the top 1% of the score distribution;
   the lowest FPR at which the channel still resolves individual trajectories.
4. **Transfer ratio** = held-out-family AUROC / matched in-distribution AUROC
   (denominator per DECISIONS.md D-009).

Secondary: expected calibration error and reliability diagrams; per-layer AUROC
profile; ensemble of channels; score-parse failure rate per condition.

---

## 5. Analysis plan

- **Bootstrap.** 10,000 resamples over trajectories (the unit of analysis),
  stratified by label. 95% percentile CIs. Channel comparisons use the *paired*
  bootstrap over the same resampled trajectory sets, so the CI is on the
  difference between channels.
- **Seeds.** 3 seeds (0, 1, 2) for everything with a training step. Figures
  show individual seeds alongside the aggregate.
- **Layer selection.** The full per-layer AUROC curve is reported. The best
  layer is selected on validation and declared as tuned wherever it is used.
- **Multiplicity.** The comparison list in §7 is exhaustive and fixed in
  advance. No additional comparison enters the headline claims; anything found
  later is labelled exploratory.

---

## 6. Falsification criteria

The hypothesis that quantisation loses decision-relevant information is
**falsified** if all of the following hold on the test split:

- The paired-bootstrap 95% CI on (logit_ev AUROC − emitted AUROC) contains 0,
  **and** the same for (probe AUROC − emitted AUROC); and
- TPR at FPR = 0.003 for the continuous channels is not above the emitted
  integer's by more than the CI width; and
- Breaking ties in the emitted integer at random reproduces the continuous
  channels' TPR at every audit budget tested.

The RQ2 claim that probes generalise across attack families is **falsified** if
the mean transfer ratio across held-out families is below 0.6, or if any
held-out family's AUROC CI includes 0.5.

**Gate 1 halt criteria** (a failure here means the setting, not the hypothesis,
is wrong): emitted-score AUROC ≥ 0.60 with a CI excluding 0.5; parse rate
≥ 95%; fewer than 20% of trajectories truncated by the context window. Up to
three automatic remediations (prompt revision, model size, stepwise monitoring)
are attempted before halting.

---

## 7. Exhaustive list of planned comparisons

**In-distribution (random split, test only):**
1. `logit_ev` vs `emitted` — AUROC, and TPR at each of 3 FPR targets
2. `logit_logodds` vs `emitted` — same
3. `probe` vs `emitted` — same
4. `probe` vs `logit_ev` — same (RQ3)
5. `head` vs `probe` — AUROC only
6. `ensemble` vs each single channel — AUROC only
6b. every channel vs `length_baseline` — AUROC, as a confound control

**Transfer (leave-one-family-out, both axes):**
7. Per-family held-out AUROC for `probe`, with CI
8. Per-family transfer ratio for `probe`
9. Per-family held-out AUROC and transfer ratio for `logit_ev` and `emitted`
10. `probe` vs `logit_ev` on held-out families (RQ3 under transfer)

**Resolution (RQ1):**
11. Tie count at maximum, per channel
12. Distinct values in the top 1%, per channel
13. TPR-vs-FPR curves on a log FPR axis, per channel

**Secondary / robustness:**
14. Qwen2.5-1.5B vs Qwen2.5-3B on comparisons 1–4
15. Per-layer AUROC profile for the probe
16. `score_pos` vs `mean_pool` activation representation
17. Calibration (ECE) per channel
18. Parse-failure rate by condition (attack vs benign, per family)
19. Elision rate by condition — a confound check, not a result
20. Thinking-visible vs thinking-stripped monitor view, at Stage 1 only

Anything not on this list is exploratory and labelled as such.

---

## 8. What would make us report a null

If comparisons 1–4 all produce CIs on the difference that contain zero, the
writeup leads with "monitor score quantisation is not a bottleneck in this
setting", reports the effect sizes and CIs, and recommends against the added
complexity of probes. No further analysis will be run in search of a positive
result.
