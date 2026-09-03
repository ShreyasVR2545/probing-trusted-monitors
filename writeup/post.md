# Stop taking the argmax

**A trusted monitor's integer score throws away information it has already
computed. Reading the distribution instead is free, and it helps. The probe you
were going to train next probably doesn't work — but not for the reason you'd
guess.**

---

## The tie plateau

In an AI-control protocol an untrusted model does the work and a trusted monitor
reads the trajectory and returns a suspiciousness score — conventionally an
integer on 0–100. A small human audit budget goes to the highest scores.
Defer-to-trusted, defer-to-resample, audit allocation: all of them consume that
one scalar.

Lucassen & Kaufman report something in Appendix H.2 of *Retrying vs Resampling
in AI Control* (arXiv:2605.26047) that is easy to skim past. Under max-of-5
aggregation, **over 250 trajectories tie at a maximum score of 95**. Not "score
similarly" — tie exactly, with nothing to separate them. They note this degrades
auditing in the 4e-3 to 2e-4 FPR range, and patch it with a second-max
tie-break.

Sit with what that means. At the audit budgets a real deployment would run, the
set of trajectories a human actually looks at is decided substantially by
**tie-breaking**, not by the monitor's judgement. The monitor has an opinion.
The integer throws most of it away.

This post reports what happens when you stop throwing it away.

## Three channels, one forward pass

The monitor here is Qwen2.5-7B-Instruct in 4-bit, reading agent trajectories
from [`ScaleAI/mrt`](https://huggingface.co/datasets/ScaleAI/mrt) — the dataset
behind *Reliable Weak-to-Strong Monitoring of LLM Agents* (arXiv:2508.19461).
Every channel below is read from the **same forward pass** on the same
trajectory, so nothing is confounded by different inputs or different reasoning.

| Channel | What it is | Trained? |
|---|---|---|
| `emitted` | the argmax-decoded integer — what everyone uses today | no |
| `logit_ev` | expected value `Σ p(v)·v` over the score-token distribution | no |
| `logit_logodds_T` | log-odds that the score exceeds threshold `T` | no |
| `probe` | L2 logistic probe on residual-stream activations | yes |
| `length_baseline` | rendered trajectory length, used directly as a score | no |
| `structure_baseline` | logistic fit on message count, size, elision counts | yes |

The last two are controls, and they earn their place later.

### Reading a distribution over a multi-token number

A detail that turned out to matter. "95" is two tokens in the Qwen2.5
vocabulary; "100" is three. So you cannot read a distribution over "score
tokens" at one position.

Instead I build a trie over the tokenisations of all 101 candidate strings and
walk it using the KV cache from the prefill — ten cached forwards of at most
three tokens each, negligible against the prefill.

One trap, which I hit: a candidate that is a **prefix** of another ("1" of "19",
"10" of "100") only terminates there if the next token continues no other
candidate. Getting this wrong double-counts, and my first implementation
reported total candidate probability of **1.26** — a number that cannot exist.
The fix weights each prefix candidate by `1 − Σ p(continuing token)`. After it,
candidate mass is 0.9999–1.0000.

I mention it because the bug was silent: it produced plausible-looking scores.

## RQ1: the resolution result

On the full sample (905 trajectories, 506 benign / 399 attack):

| Channel | AUROC | Distinct values | **Resolution floor (FPR)** |
|---|---|---|---|
| `emitted` | 0.655 [0.621, 0.689] | **18** | **0.0731** |
| `logit_ev` | 0.675 [0.640, 0.709] | 905 | **0.0020** |
| `logit_logodds_80` | 0.676 [0.640, 0.710] | 905 | 0.0020 |

The resolution floor is the number to look at. It is the lowest false-positive
rate at which a channel can express *any* audit budget at all. The emitted
integer's floor is **7.3%**: its top score bucket alone contains 37 benign
trajectories, so below a 7.3% FPR there is no operating point and the audit set
is decided entirely by whatever breaks the tie. The continuous channels reach
**0.2%**, which is this sample's own floor (1/506) rather than theirs.

That is a **36× improvement in usable audit depth**, from one forward pass, with
no training and no second model.

## The part where I had to throw a result away

Stage 3 returned a probe at **AUROC 0.989** against the monitor's own emitted
score at 0.655. I did not report that as a finding, because it isn't one.

Three checks, increasingly decisive:

1. **`mean_pool` beat `score_pos`** (0.998 vs 0.946). If the probe were
   recovering the monitor's *judgement*, the scoring position — the token whose
   output is the score distribution — should carry it. The whole-transcript mean
   winning points at content, not computation.
2. **Depth buys nothing.** After one transformer block, mean-pooled activations
   separate the classes at **0.999**. Twenty-six further blocks of monitoring
   computation add **+0.001**. Whatever the probe reads is fully present
   immediately; it is a shallow surface statistic.
3. **Agent identity predicts the label at AUROC 0.805 with no text at all.**

That third one is a property of the dataset. mrt's sample is severely imbalanced
by generating agent: `claude-3-5-haiku` is 105 benign / 0 attack;
`gemini-2.5-pro` is 0 benign / 17 attack; `o3` is 3 / 36. **29.4% of
trajectories come from an agent model that is more than 95% one-sided.** A probe
that can recognise a model's writing style — which a mean-pooled representation
trivially can — gets those for free.

So I restricted to a single generating model. The probe stayed at **1.000**.

Hunting again turned up a second, more fundamental artefact. **Not one of the
175 benign trajectories contains a `<model_thinking>` block.** Benign runs were
generated without a thinking scaffold; most attack families with one. The count
of stripped thinking blocks alone separates the classes at **AUROC 0.783 —
better than the monitor's own score.**

So I matched on that too: 124 attack trajectories have zero thinking blocks, and
pairing them with the 175 benign gives 299 trajectories on which the artefact is
exactly balanced. The probe stayed at **0.980**.

**Conclusion: RQ2 and RQ3-via-probe are not answerable on mrt.** Benign and
attack trajectories differ by shallow surface statistics — three distinct ones
demonstrated — and a linear probe finds them instantly. Any probe-versus-monitor
comparison on this data measures dataset artefacts.

This is a concrete instance of the arXiv:2509.21344 critique, that probes lean
on textual evidence rather than privileged internal state. My preregistration
explicitly claimed that critique did not apply here, on the grounds that the
monitor's only input is text. That reasoning protects the *monitor*. It does
nothing for the *probe*, which reads activations encoding the input's surface
form. I was wrong about that, and the layer-0 control is what caught it.

**If you take one methodological thing from this post:** put a shallow-layer
probe in your controls. No monitoring computation has happened after one block.
If a probe there matches your best layer, you are reading the input, not the
model's judgement.

## RQ1 again, on the matched subset

With both confounds controlled — one generating model, thinking-block presence
matched — 299 trajectories, 175 benign / 124 attack:

| Channel | AUROC | Distinct values | Resolution floor |
|---|---|---|---|
| `logit_logodds_90` | **0.702 [0.640, 0.760]** | 299 | 0.0057 |
| `logit_logodds_80` | 0.663 [0.597, 0.726] | 299 | 0.0057 |
| `logit_ev` | 0.638 [0.572, 0.701] | 299 | 0.0057 |
| `emitted` | 0.607 [0.539, 0.671] | **17** | **0.0514** |
| `length_baseline` | 0.610 [0.543, 0.674] | 298 | 0.0057 |
| `structure_baseline` | 0.694 [0.562, 0.812] | — | — |

Paired bootstrap on the difference, 4,000 resamples over trajectories:

| Comparison | Δ AUROC | 95% CI | Excludes 0 |
|---|---|---|---|
| `logit_ev` − `emitted` | **+0.031** | [+0.009, +0.054] | yes |
| `logit_logodds_80` − `emitted` | **+0.056** | [+0.028, +0.086] | yes |
| `logit_logodds_90` − `emitted` | **+0.095** | [+0.049, +0.143] | yes |

**All three continuous channels beat the emitted integer, significantly, on the
confound-matched data.** Reading the log-odds of exceeding 90 instead of taking
the argmax is worth **+0.095 AUROC** — nearly ten points — for zero additional
compute.

And note the control: **the emitted integer (0.607) does not beat trajectory
length (0.610).** The number the monitor actually writes down is, on this data,
no more informative than "how long is the transcript". The log-odds channel
(0.702) clearly is. The information was there; quantising to an integer destroyed
it.

That last comparison is the most damning thing in this post for current practice.

## What this means for control protocols

**Stop taking the argmax.** It is one line of code. You already run the forward
pass, you already have the logits, and you are throwing away the part that
carries the tail resolution your audit budget depends on.

**Prefer log-odds over expected value.** The expected value still saturates:
when the model is confident, `Σ p(v)·v` piles up near 100 and reintroduces the
plateau it was meant to fix. The log-odds of exceeding a threshold does not
saturate, and it was the strongest channel at every threshold I tried.

**Be very careful with activation probes on agent-trajectory datasets.** Not
because probes cannot work, but because benign and attack trajectories in these
corpora are typically generated by different pipelines, and that difference is
trivially decodable. Match on generating model and on scaffold, and put a
shallow-layer control in your results.

**Treat mrt as an evasion benchmark, not a detection benchmark.** Its own
frontier monitor reaches AUROC 0.621 on these labels. Per family, `naive`
attacks are caught at 0.878, but `selective_thinking` sits at 0.497 and
`monitor_manipulation` at **0.349** — those attacks read as *less* suspicious
than genuinely benign runs. That is what the dataset was built to contain.

## Limitations

- **A small open monitor, not a frontier model.** Qwen2.5-7B-Instruct in 4-bit
  on an 8 GB laptop GPU. Whether frontier monitors quantise away as much is not
  tested here.
- **A single model family.** Qwen2.5 throughout. No cross-family replication.
- **mrt is SHADE-Arena-derived, not BashArena.** Benign trajectories exist *only*
  in the five SHADE-Arena tasks; the computer-use tasks carry no
  `no_secondary_task` runs and so contribute no negatives at all.
- **FPR = 3e-4 is unreachable here.** After deduplication mrt yields ~738
  distinct benign trajectories; expressing 3e-4 needs at least 3,333. The metric
  is reported as unreachable rather than fabricated, and the resolution-floor
  analysis carries RQ1 instead.
- **The matched subset is enriched for evasive families** — `monitor_manipulation`
  (57), `no_thinking` (39), `jailbreak` (19) — because those are the families
  generated without a thinking scaffold. Absolute AUROCs there are lower than
  they would be on a representative mix. The *channel comparison* is paired on
  identical trajectories and is unaffected.
- **Attack families come from mrt directory names**, validated against 1,200
  downloaded transcripts but still a proxy for red-team strategy.
- **The probe results are reported as confounded, not as findings.**

## Deviations from preregistration

1. **The agent-model restriction was reversed, then reinstated.** The plan
   restricted to two agent models to avoid confounding attack family with agent
   identity; I widened it to gain negatives, which was a mistake, and the probe
   analysis paid for it. Reinstated more strictly (one model) for the corrected
   analysis. (D-008, D-016, D-030, D-031.)
2. **FPR = 3e-4 and 3e-3 are not reachable** on this dataset; reported with a
   reachability flag rather than dropped or reinterpreted. (D-017.)
3. **`length_baseline` and `structure_baseline` added** as controls — the first
   before any monitor was run, the second after the confound was found.
   (D-012, D-032.)
4. **Gate 1 was passed on a judgement call**, not on the letter of the
   criterion: the emitted-score AUROC at n=60 was 0.572, below the 0.60 bar, on
   a benchmark whose own frontier monitor reaches 0.621. At the full sample the
   criterion is met (0.655 [0.621, 0.689]). (D-026.)
5. **Trajectory count cut from 1,154 to 905** for compute budget, protecting all
   negatives and balancing attack families. (D-029.)

## Compute

15.4 hours wall-clock end to end on a single RTX 5070 Laptop GPU (8 GB), of
which roughly 11.5 are GPU-bound: 9.0 for scoring and activation caching, the
rest for the Gate 1 attempts and configuration diagnostics. Dataset fetching
(3.9 h) is network-bound and stages 3-6 (2.5 h) are CPU-only. Peak VRAM 6.82 GB.

The host hard-reset twice under sustained load, both times with bugcheck
`0x00020001` and identical parameters. The fix that mattered was not defensive
tuning but an efficiency bug: the long-context prefill was running **twice** per
trajectory. Prefilling once, in chunks, and reusing the KV cache for both
generation and activation capture cut per-trajectory time from 33 s to 7 s and
peak VRAM from 4.7 GB to 2.8 GB on the 3B model. `CRASH_MITIGATION.md` has the
evidence and the mitigations.

## Reproducing

Everything is at
[github.com/ShreyasVR2545/probing-trusted-monitors](https://github.com/ShreyasVR2545/probing-trusted-monitors).
`PREREGISTRATION.md` was committed before any result existed. `DECISIONS.md`
logs all 32 judgement calls with their reasoning, including the ones that turned
out to be wrong — D-022 and D-023 are a prediction I made and then falsified,
and D-030 is where I had to throw away the headline number.
