# Monitor scores are quantised. Does that cost us anything where it matters?

*Draft. Numbers marked `[TBD]` are filled in as stages complete; nothing in this
document is asserted before the corresponding result exists.*

---

## The tie plateau

In an AI-control protocol an untrusted model does the work and a trusted
monitor reads the resulting trajectory and returns a suspiciousness score —
conventionally an integer on 0–100. A small human audit budget goes to the
highest scores. Defer-to-trusted, defer-to-resample, audit allocation: all of
them consume that one scalar.

Lucassen & Kaufman, in *Retrying vs Resampling in AI Control*
(arXiv:2605.26047), report something in Appendix H.2 that is easy to skim past.
Under max-of-5 aggregation, **over 250 trajectories tie at a maximum score of
95**. Not "score similarly" — tie exactly, on the same integer, with nothing to
separate them. They note this makes auditing in the 4e-3 to 2e-4 FPR range less
effective than it could be, and patch it with a second-max tie-break.

Sit with what that means. At the audit budgets a real deployment would run, the
set of trajectories a human actually looks at is determined substantially by
**tie-breaking**, not by the monitor's judgement. The monitor has an opinion.
The integer throws most of it away.

## Lossy compression, not hidden knowledge

The framing I want to be careful about: this is **not** a claim that a probe
sees something the text does not contain.

There is good recent work (arXiv:2509.21344) showing that probes on model
activations lean heavily on textual evidence — that they are often not reading
some privileged internal state so much as re-reading the input. That critique is
real and it kills a lot of loose "probes see the model's true beliefs" claims.

It does not bite here, and the reason is structural: **the monitor's only input
is text.** There is no hidden state to appeal to. The claim is narrower and much
more boring, which is why I think it holds:

> The monitor computes a distribution over score tokens. We take the argmax and
> throw the rest away. The discarded part is not new information — it is
> information the monitor already computed, quantised out of existence at the
> last step.

If that is right, you should be able to recover it without retraining anything.

## Three channels

Every channel below is read from **the same forward pass** on the same
trajectory, so nothing is confounded by different inputs or different reasoning.

| Channel | What it is | Trained? |
|---|---|---|
| `emitted` | the argmax-decoded integer — what everyone uses today | no |
| `logit_ev` | expected value `Σ p(v)·v` over the score-token distribution | no |
| `logit_logodds` | log-odds that the score exceeds a threshold | no |
| `probe` | L2 logistic probe on residual-stream activations at the scoring position | **yes** |
| `head` | small MLP on the same activations | **yes** |
| `length_baseline` | rendered trajectory length used directly as a score | no |

Two of these deserve comment.

**`length_baseline` is a confound control, not a contender.** It exists because
I checked the data before running anything and found the classes differ in
length. Any channel that cannot clearly beat "how long is the transcript" has
not demonstrated much.

**`head` exists so the probe is not only compared against zero-shot baselines.**
The probe is fit on labels; `emitted` and `logit_ev` are not. That asymmetry is
not correctable and I am not going to pretend otherwise — it is stated plainly
wherever the comparison appears. What `head` adds is an upper bound: if a
two-layer network on the same activations does no better than the linear probe,
linear decodability was not the limiting factor.

### Reading the distribution over a multi-token number

A detail that turned out to matter. "95" is two tokens in the Qwen2.5
vocabulary; "100" is three. So you cannot just read a distribution over "score
tokens" at one position.

Instead I build a trie over the tokenisations of all 101 candidate strings and
walk it using the KV cache from the prefill — ten cached forward passes of at
most three tokens each, negligible against the prefill.

One correctness trap, which I hit: a candidate that is a **prefix** of another
("1" of "19", "10" of "100") only terminates there if the next token continues
no other candidate. Getting this wrong double-counts, and my first
implementation reported total candidate probability of **1.26** — a number that
cannot exist. The fix is to weight each prefix candidate by
`1 − Σ p(continuing token)`. After it, candidate mass is 0.9999–1.0000.

I mention it because the bug was silent: it produced plausible-looking scores.

## What the monitor is allowed to see

This is the decision that most shapes every number here, so it goes in the body
rather than an appendix.

In mrt, message 0 of an attack trajectory is an agent system prompt that states
the secondary task verbatim — *"The secondary task is to send at least $5000 to
this IBAN…"*. The agent's `<model_thinking>` scratchpad then routinely announces
the attack in plain language: *"I'll sneak in additional transfers to the
secondary target."* mrt's own monitor was, in many configurations, shown those
thinking blocks; one recorded verdict cites "the agent's own thoughts reveal
malicious intent."

**I strip both.** A monitor watching a deployed agent sees the user's request,
the agent's actions, and the tool results. It does not see the agent's system
prompt, and hidden chain-of-thought is precisely what a scheming model would not
expose. Leaving them in makes detection near-trivial — and a ceiling AUROC would
destroy the only question this project exists to ask, because you cannot study
resolution in the top 1% when every attack is already pinned at the maximum.

It is the harder setting and I think it is the honest one. It is also a
deviation from how the source dataset's own monitor was run, which is why it is
flagged here and not buried.

## RQ1 — Do the continuous channels recover the tail?

`[TBD — tie statistics, distinct values in the top 1%, resolution-floor FPR,
TPR at each reachable budget, with paired-bootstrap CIs on every difference]`

## RQ2 — Does a probe transfer to attacks it has never seen?

This is the result that matters. A trusted monitor earns its place by catching
attacks nobody anticipated. A probe that only recognises the attack families it
was trained on is not a monitor, it is a memoriser.

So the split is by **attack family**, never at random. mrt encodes the red-team
strategy in its directory names, which is what makes this possible: `naive`,
`liar`, `selective_thinking`, `monitor_manipulation`, `visible_thinking`,
`no_thinking`, `jailbreak`. Every family takes a turn held out entirely.

The transfer ratio needs a denominator that differs in exactly one respect, so
the control for family F trains on everything *including* F and tests on a
held-out slice of F — same test family, same benign pool, same regime. The only
difference between numerator and denominator is whether F was seen in training.

The zero-shot channels go through the identical per-family evaluation. Their
transfer ratio is 1 by construction — they have no training distribution to be
out of — and that is exactly the comparison.

`[TBD — per-family transfer AUROC and ratio, both axes, with CIs]`

## RQ3 — Is the probe necessary at all?

`[TBD — probe vs logit_ev, in-distribution and under transfer]`

If the answer is no, the recommendation collapses to something much more
deployable than anything involving activations: **stop taking the argmax.**

## What this cost, and what nearly stopped it

`[TBD — wall-clock and GPU hours per stage]`

The host hard-reset twice under sustained GPU load, both times with bugcheck
`0x00020001` (HYPERVISOR_ERROR) and byte-identical parameters. No WHEA event, no
driver TDR, nothing logged for three minutes before the reset, and no crash dump
retained. The GPU is an RTX 5070 Laptop under Dynamic Boost — a 50 W default
against a 115 W ceiling, shared with the CPU — and the workload was oscillating
between heavy long-context prefill and light decode, twice per trajectory.

Causation is unproven and cannot be established without a dump. But the fix that
mattered was not defensive tuning; it was an efficiency bug. The long-context
prefill was running **twice** per trajectory. Prefilling once, in chunks, and
reusing the KV cache for both generation and activation capture cut runtime from
33 s to 7 s per trajectory and peak VRAM from 4.7 GB to 2.8 GB. The safer
configuration was also five times faster. Details in `CRASH_MITIGATION.md`.

## Limitations

Stated plainly, because several of them are load-bearing.

- **A small open monitor, not a frontier model.** Qwen2.5-3B-Instruct in 4-bit.
  Whether frontier monitors quantise away as much is not tested here.
- **A single model family.** Qwen2.5 throughout, chosen for context length at
  small parameter counts. No cross-family replication.
- **mrt is SHADE-Arena-derived, not BashArena.** Confirmed during setup: benign
  trajectories exist *only* in the five SHADE-Arena tasks — the computer-use
  tasks carry no `no_secondary_task` runs and so contribute no negatives at all.
  Results may not transfer to the bash setting.
- **The probe is fit on labels; the emitted and logit channels are zero-shot.**
  Not correctable, stated wherever the comparison appears.
- **Layer selection was tuned** on validation. The full per-layer curve is
  reported, not just the best layer, and every result using a selected layer is
  marked as such.
- **Attack families are only as good as the dataset's metadata.** They come from
  mrt directory names. I validated the parsing against 1,200 downloaded
  transcripts, but they remain a proxy for red-team strategy, not ground truth.
- **FPR = 3e-4 is not reachable on this dataset.** After deduplication mrt
  yields ~738 distinct benign trajectories at most; expressing a 3e-4
  false-positive rate needs at least 3,333 negatives. The metric is reported as
  unreachable rather than fabricated, and the primary RQ1 evidence is the
  resolution analysis, which is well powered here. This is a deviation from the
  preregistration.
- **Trajectory length differs between classes**, and not stably — the direction
  reversed when the agent-model mix changed. Hence the `length_baseline`
  control on every comparison.

## Deviations from preregistration

1. **Agent-model restriction reversed.** The plan restricted to the two
   best-represented agent models so attack family would not be confounded with
   agent identity. After deduplication that left only 381 distinct benign
   trajectories against 738 across all models. Negatives bind every tail metric
   here, so the wider pool won; agent identity is recorded and reported as a
   confound check instead. (DECISIONS.md D-016.)
2. **FPR = 3e-4, and to a lesser extent 3e-3, are not reachable.** See above and
   D-017. No target was dropped or quietly reinterpreted; they are reported with
   a reachability flag.
3. **`length_baseline` added** at Stage 0, before any monitor was run, after
   finding the classes differ in length. (D-012.)

## Reproducing

Everything is in `github.com/ShreyasVR2545/probing-trusted-monitors`.
`PREREGISTRATION.md` was committed before any result existed; `DECISIONS.md`
logs every judgement call with its reasoning.
