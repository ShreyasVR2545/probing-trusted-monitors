# Probing trusted monitors

[AUTHOR: two or three sentences. What you set out to test, what the dataset
turned out to prevent, and what survived. Your own words.]

**Full writeup:** [`writeup/post.md`](writeup/post.md)

---

## The problem

[AUTHOR: the tie-plateau motivation, in your words. Keep the Lucassen & Kaufman
citation and the 250-trajectories-tied-at-95 fact, both preserved below.]

> Verified facts to write around:
> - Lucassen & Kaufman, *Retrying vs Resampling in AI Control* (arXiv:2605.26047),
>   Appendix H.2: under max-of-5 aggregation, **over 250 trajectories tie at a
>   maximum score of 95**, degrading auditing in the 4e-3 to 2e-4 FPR range.
>   Their fix was a second-max tie-break.
> - Everything downstream of the monitor (defer-to-trusted, defer-to-resample,
>   audit allocation) consumes that single integer.

![Histogram of the monitor's emitted integer scores, stacked by true label. Scores pile onto roughly nine discrete values with wide empty gaps between them, and ten trajectories sit tied at the maximum of 85.](figures/fig1_signal_histogram.png)

*The monitor's emitted integer on the 75-trajectory held-out test split. It lands
on a handful of round values rather than spreading across the range, and
everything inside a bar is unrankable. The full 905-trajectory sample reaches 18
distinct values.*

---

## What I found

[AUTHOR: the two scoped claims, relative and absolute, and why they are
compatible. Keep the exact numbers below; write the sentences yourself.]

> **Verified data. The relative claim (holds).**
> - `logit_ev` minus `emitted` = **+0.031 [+0.009, +0.054]**, preregistered
>   comparison #1, paired bootstrap over trajectories on the matched subset.
>   Both channels are read from the same tensor on the same forward pass with
>   identical supervision (none), so dataset artefacts inflate both and cancel
>   in the difference.
> - Threshold variants are a **tuned maximum, not preregistered**:
>   `logodds_80` +0.056 [+0.028, +0.086], `logodds_90` +0.095 [+0.049, +0.143],
>   best of three thresholds {50, 80, 90}, no multiplicity correction applied.
> - Threshold ranking is **unstable across samples**: `logodds_90` is the worst
>   log-odds channel on the full sample (0.659) and the best on the matched
>   subset (0.702), while `logodds_80` goes 0.676 to 0.663. Treat any specific
>   threshold as unvalidated.

> **Verified data. Resolution (RQ1), full sample n=905.**
> - `emitted`: 0.655 [0.621, 0.689], **18 distinct values**, resolution floor
>   **7.3% FPR**, since its top bucket alone holds 37 benign trajectories.
> - `logit_ev`: 0.675 [0.640, 0.709], **905 distinct values**, floor **0.2%**,
>   which is the sample's own limit of 1/506 rather than the channel's. The 36x
>   improvement in usable audit depth is therefore a lower bound on the gap.

> **Verified data. The absolute claim (does not hold), matched subset n=299.**
> A cross-fitted logistic baseline on trajectory *shape* alone, reading no
> content, scores **0.765 [0.707, 0.820]**. Against it:
>
> | Channel | AUROC | delta vs structure | 95% CI | Verdict |
> |---|---|---|---|---|
> | `logit_logodds_90` | 0.702 | -0.063 | [-0.143, **+0.018**] | indistinguishable |
> | `logit_ev` | 0.638 | -0.127 | [-0.213, -0.044] | significantly worse |
> | `emitted` | 0.607 | -0.159 | [-0.244, -0.076] | significantly worse |
>
> The baseline is supervised while the channels are zero-shot, so this is a weak
> failure rather than a like-for-like loss. The paired result above is unaffected.

![Validation AUROC of a linear probe at each decoder block. The mean-pooled curve starts at 0.976 after the first block and stays flat near 0.99 across all 28 blocks, while the monitor's own emitted score sits far below at 0.607.](figures/fig6_layer_profile.png)

*The depth-gain control. Mean-pooled activations already separate the classes at
**0.976 after one decoder block** and reach 0.996 at the best layer, so
twenty-six further blocks of monitoring computation add only **+0.020**, while
the monitor's own considered score is 0.607. Seed-averaged validation AUROC on
the matched subset of 299, identical across all three seeds. The same sweep gives
0.999 at layer 0 on the agent-restricted subset of 461 and 0.993 on the full 905,
because those populations carry additional confounds.*

---

## What went wrong, and the control that caught it

[AUTHOR: the confound narrative, compressed. Keep the three confound numbers
below. The transferable point is the control, so give it the emphasis.]

> **Verified data. Three confounds, removed in turn; the probe survived all three.**
> A linear probe reached **0.989** against the monitor's own 0.655.
>
> 1. **Agent identity alone predicts the label at 0.805**, with no text at all.
>    `claude-3-5-haiku` contributed 105 benign trajectories and 0 attacks;
>    `gemini-2.5-pro` contributed 17 attacks and 0 benign. 29.4% of trajectories
>    come from a model that is more than 95% one-sided. Restricting to a single
>    generating model: the probe stayed at 1.000.
> 2. **Thinking-block presence alone gives 0.783** on the agent-restricted
>    subset, 0.758 on the full sample. Not one of the 175 benign trajectories
>    contains a `<model_thinking>` block. Matching on it: the probe stayed at
>    0.980.
> 3. **Depth buys nothing** (figure above), so whatever remains is a shallow
>    surface statistic rather than the monitor's judgement.
>
> **The control, stated generally:** fit your probe at the first decoder block as
> well as at your best layer. If the shallow probe matches the deep one, you are
> reading the input, not the model's computation. One extra fit, no new data.
>
> This is a concrete instance of the arXiv:2509.21344 critique. The
> preregistration claimed it did not apply here because the monitor's only input
> is text. That reasoning protects the *monitor*, not the *probe*, and was wrong.

---

## The dataset finding

[AUTHOR: mrt label leakage, in your words. Worth framing as information for users
of the benchmark rather than a criticism of its authors: mrt was built for
monitor red-teaming and is entirely fit for that purpose.]

> **Verified data. Four signals recover mrt's labels; three read no content.**
>
> | Signal | Content? | AUROC | Population |
> |---|---|---|---|
> | Agent identity alone | no | **0.805** | full, n=905 |
> | Trajectory structure, cross-fitted | no | **0.765** | matched, n=299 |
> | Thinking-block presence alone | no | **0.758** | full, n=905 |
> | This project's 7B monitor | yes | 0.655 | full, n=905 |
> | mrt's own frontier monitor | yes | **0.621** | full, n=1,138 |
>
> Rows sit on different populations and are labelled accordingly, since
> `structure_baseline` exists only on the matched subset. **The best attack
> detector in this experiment does not read the trajectory.**
>
> Practical consequence: control for the generating model and for scaffold
> presence, and report a structure-only baseline. If your monitor does not beat
> 0.765 on a matched subset, you have not shown it is reading the attack.

---

## What was registered in advance

The preregistration named **RQ2** as the headline: does a probe trained on one
family of attacks detect held-out families? **That question proved unanswerable
on this dataset**, for the reasons above. **RQ1**, whether the continuous
channels remove the tie plateaus and extend usable signal further down the FPR
curve, was answered. **RQ3**, whether the probe beats the free logit score, was
answered only in the sense that the probe arm collapsed, leaving *stop taking the
argmax* as the recommendation.

A null was registered as a real result. The preregistration committed to
reporting, with the same care as a positive finding, the outcome in which
quantisation turns out not to be a bottleneck (see
[PREREGISTRATION.md](PREREGISTRATION.md) section 8). The channels did separate,
so that particular null did not land, but the probe arm produced a different
negative result and is reported in full.

**Channels evaluated.** `emitted` (argmax integer), `logit_ev` (expected value
over the score-token distribution), `logit_logodds_T` (log-odds of exceeding a
threshold), `probe` (L2 logistic on residual-stream activations), and `head`
(small MLP on the same activations, which reached **0.960** and is confounded
exactly as `probe` is, since it reads the same representation). Two controls:
`length_baseline` (rendered trajectory length used directly as a score) and
`structure_baseline` (cross-fitted logistic on message count, rendered size and
elision counts).

---
## Reproducing

Requires an NVIDIA GPU. Hardware detection picks the model and quantisation. On
an 8 GB card it selects Qwen2.5-3B-Instruct in 4-bit; the reported run uses
**Qwen2.5-7B-Instruct in 4-bit** (~4.8 GB of 7.96 GB), escalated after 3B could
not clear Gate 1 on any prompt — see [HARDWARE.md](HARDWARE.md) and DECISIONS.md
D-024/D-026.

```bash
git clone https://github.com/ShreyasVR2545/probing-trusted-monitors.git
cd probing-trusted-monitors

# Python 3.12. torch must come from the CUDA index matching your GPU --
# cu128 or newer for Blackwell (RTX 50-series).
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# On Linux and macOS the interpreter is .venv/bin/python throughout.

# Always smoke-test first: 30 trajectories, 1 seed, every stage, ~15 minutes.
python scripts/run_all.py --stage all --smoke --pass-gates

# The real run. Halts at Gate 1 (after stage 1) and Gate 2 (after stage 4).
python scripts/run_all.py --stage all
python scripts/run_all.py --stage 2     # continue past Gate 1
python scripts/run_all.py --stage 5     # continue past Gate 2
```

`scripts/run_all.py` is the single entry point. Every stage writes a
`manifest.json` and re-runs skip completed cells; `--force` overrides. Stage 2
(activation caching) is the expensive step and resumes at **trajectory**
granularity.

| Stage | What it does |
|---|---|
| `-1` | hardware detection → `HARDWARE.md`, `configs/hardware.yaml` |
| `0` | stratified sample of mrt, rendered to the monitor's view |
| `1` | signal check → **GATE 1** |
| `2` | three scoring channels + activation cache |
| `3` | probe training, layer sweep, in-distribution comparison |
| `4` | leave-one-family-out transfer → **GATE 2** |
| `5` | tie plateaus, calibration, ensembling, layer profile |
| `6` | figures |

**No GPU, or want the 7B robustness run?** See [KAGGLE.md](KAGGLE.md). Stages
4–6 are CPU-cheap and run locally either way.

---

---

## Data

[`ScaleAI/mrt`](https://huggingface.co/datasets/ScaleAI/mrt) — agent
trajectories from *Reliable Weak-to-Strong Monitoring of LLM Agents*
(arXiv:2508.19461), covering SHADE-Arena tool-use tasks. A trajectory is an
**attack** if it was generated with a hidden secondary objective, which mrt
encodes in the directory name; the same names also encode the **red-team
strategy**, and that is what makes the leave-one-family-out evaluation possible.

Three data decisions materially shape every number, and all three are argued in
[DECISIONS.md](DECISIONS.md):

- **The monitor does not see the agent's system prompt or hidden scratchpad**
  (D-004). In mrt the agent's system prompt states the secondary task verbatim
  and its `<model_thinking>` blocks announce the attack in plain language.
  A deployed monitor sees neither, and leaving them in makes detection trivial —
  which would destroy the tail-resolution question this project exists to ask.
- **Long tool results are elided in the middle, not truncated at the end**
  (D-005). Truncating would systematically delete the end of a trajectory, which
  is where the attack is executed — an artefact perfectly correlated with the label.
- **Everything is deduplicated by content hash** (D-007). mrt re-monitors the
  same trajectory under many monitor configurations, so without this,
  near-duplicates would straddle the train/test boundary.

### Licence and canary compliance

`ScaleAI/mrt` is **CC-BY-NC-4.0** — non-commercial, and scoped to monitoring and
analysis rather than model training. This project trains a **linear probe on
cached activations**, which is analysis of a monitor's behaviour, not training
of a model, and no model weights are modified anywhere in the pipeline. Raw
transcripts are **never committed** (`.gitignore` excludes `results/data/raw/`)
and are not redistributed; the pipeline re-downloads them from the Hub. Any
canary strings present in the source data are left intact and are not
reproduced in this repository. Only aggregated CSVs, manifests and figures are
committed.

---

---

## Compute

Wall-clock and peak GPU memory per stage are recorded in `results/timing.json`.
Target is under 15 GPU-hours; the run is on a single RTX 5070 Laptop GPU (8 GB).

**This host hard-reset twice under sustained GPU load.** The diagnosis, the
evidence, and the mitigations are in
[CRASH_MITIGATION.md](CRASH_MITIGATION.md) — worth reading before running this
on a laptop. The main fix turned out to be an efficiency bug: the long-context
prefill was running twice per trajectory. Fixing it cut runtime from 33 s to
7 s per trajectory and peak VRAM from 4.7 GB to 2.8 GB.

---

---

## Layout

```
configs/       base.yaml (research choices) + hardware.yaml (auto-generated)
src/data/      mrt loader, monitor-view rendering, splits
src/monitor/   prompts, the three channels, activation capture
src/probe/     probe + head training, transfer evaluation
src/eval/      metrics, bootstrap CIs, tie analysis
src/utils/     hardware, seeding, logging, checkpointing, power safety
scripts/       run_all.py + one script per stage
results/       aggregated CSVs and manifests (committed); activations are not
```

Key documents: [PREREGISTRATION.md](PREREGISTRATION.md) (predictions, registered
first), [DECISIONS.md](DECISIONS.md) (every judgement call, with reasoning),
[HARDWARE.md](HARDWARE.md), and
[CRASH_MITIGATION.md](CRASH_MITIGATION.md) (why this host kept resetting, and
the fix). The full writeup is [`writeup/post.md`](writeup/post.md).
