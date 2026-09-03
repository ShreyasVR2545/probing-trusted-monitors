# Probing trusted monitors

**Is the integer a trusted monitor emits a lossy compression of a judgement it
has already made — and if so, does recovering the lost resolution help where
auditing actually happens?**

> **Status: complete.** All stages ran; both gates were reported. The probe
> arm produced a negative result and is reported as such.

---

## The problem

In an AI-control protocol an untrusted model does the work and a **trusted
monitor** reads the trajectory and emits a suspiciousness score, conventionally
an integer on 0–100. A small human audit budget goes to the highest scores, and
everything downstream — defer-to-trusted, defer-to-resample, audit allocation —
consumes that one scalar.

Integer scores have poor resolution exactly where auditing happens. Lucassen &
Kaufman (*Retrying vs Resampling in AI Control*, arXiv:2605.26047, Appendix
H.2) report that under max-of-5 aggregation **over 250 trajectories tie at a
maximum score of 95**, leaving nothing to rank them by, and that auditing in the
4e-3 to 2e-4 FPR range suffers as a result. Their fix was a second-max
tie-break. At realistic audit budgets, then, the audit decision is
substantially made by *tie-breaking* rather than by monitor signal.

**The claim tested here:** that integer is a lossy compression of a computation
the monitor already performed. Two cheaper-than-retraining ways to recover it:

| Channel | What it is | Cost |
|---|---|---|
| `emitted` | the argmax-decoded integer — today's baseline | — |
| `logit_ev` / `logit_logodds` | read the model's *distribution* over score tokens at the scoring position instead of taking the argmax | one forward pass, no training |
| `probe` | L2 logistic probe on residual-stream activations at that same position | needs labels |
| `head` | small MLP on the same activations — a stronger comparison than zero-shot alone | needs labels |
| `length_baseline` | rendered trajectory length used directly as a score | a confound control |

**Research questions.** **RQ1** — do the continuous channels remove the tie
plateaus and extend usable signal further down the FPR curve? **RQ2 (headline)**
— does a probe trained on one family of attacks detect *held-out* families? A
monitor earns its place by catching attacks nobody anticipated; a probe that
only recognises attacks it was trained on is a memoriser, not a monitor.
**RQ3** — does the probe beat the free logit score at all? If not, the
recommendation is simply *stop taking the argmax*, which is a better result
because it is more deployable.

**What this is not.** This is not a claim that probes see something the text
does not contain. Prior work (arXiv:2509.21344) shows probes lean heavily on
textual evidence. That critique does not bite here: the monitor's only input
*is* text, and the claim is purely that quantising its judgement to an integer
discards information it already computed.

**A null is a real result.** If all three channels match, the finding is that
monitor-score quantisation is not a bottleneck and the tie plateau is cosmetic.
That is reported with the same care as a positive finding — see
[PREREGISTRATION.md](PREREGISTRATION.md) §8.

---

## Results

**Headline: stop taking the argmax.** Reading the monitor's distribution over
score tokens instead of its argmax costs one forward pass and beats the emitted
integer significantly.

On the confound-matched subset (299 trajectories, one generating model,
thinking-block presence matched), paired bootstrap over trajectories:

| Comparison | Δ AUROC | 95% CI |
|---|---|---|
| `logit_logodds_90` − `emitted` | **+0.095** | [+0.049, +0.143] |
| `logit_logodds_80` − `emitted` | **+0.056** | [+0.028, +0.086] |
| `logit_ev` − `emitted` | **+0.031** | [+0.009, +0.054] |

All three exclude zero. For scale: the emitted integer (0.607) does **not** beat
a trajectory-length baseline (0.610); the log-odds channel (0.702) clearly does.

**Resolution (RQ1), full sample (905):** the emitted integer takes **18 distinct
values** and cannot express any audit budget below **7.3% FPR** — its top bucket
alone holds 37 benign trajectories. The continuous channels take 905 distinct
values and reach **0.2% FPR**, a **36x improvement in usable audit depth**.

**Transfer (RQ2) is not answerable on this dataset — a negative result.** A
linear probe reaches AUROC 0.98–0.99, but it is reading dataset artefacts, not
the monitor's judgement. Three were found and controlled in turn, and the probe
survived all three. The decisive diagnostic: after **one** transformer block,
mean-pooled activations already separate the classes at **0.999**; twenty-six
further blocks add **+0.001**. Details in
[`writeup/post.md`](writeup/post.md) and DECISIONS.md D-030 to D-032.

| | Outcome |
|---|---|
| Gate 1 — signal check | passed on judgement at n=60, confirmed at full sample (0.655 [0.621, 0.689]) |
| RQ1 — tie plateaus and tail resolution | **confirmed**, 36x resolution-floor improvement |
| Gate 2 — transfer | **not answerable on mrt**; probe confounded by dataset artefacts |
| RQ3 — probe vs logit score | logit channels win on the deployable comparison; probe arm invalid |

---

## Reproducing

Requires an NVIDIA GPU. Hardware detection picks the model and quantisation; on
an 8 GB card that is Qwen2.5-3B-Instruct in 4-bit. See [HARDWARE.md](HARDWARE.md).

```bash
git clone https://github.com/ShreyasVR2545/probing-trusted-monitors.git
cd probing-trusted-monitors

# Python 3.12. torch must come from the CUDA index matching your GPU --
# cu128 or newer for Blackwell (RTX 50-series).
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

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
[HARDWARE.md](HARDWARE.md), [CRASH_MITIGATION.md](CRASH_MITIGATION.md),
`writeup/post.md` (the writeup).
