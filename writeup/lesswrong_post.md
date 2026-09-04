# Stop taking the argmax: trusted monitor scores are lossy, and a control that would have saved me a week

**Epistemic status:** Preregistered, one dataset, one model family, one 7B monitor on a laptop GPU. The paired channel comparison is pre-specified and I believe it. The absolute claim about attack detection failed, and I report why rather than burying it. The probe arm is a negative result about the dataset, not a finding about probes. I was wrong about something I wrote in the preregistration, and that section is still in the post.

---

## Summary

**A trusted monitor's emitted integer score discards information the monitor already computed.** Reading the logit distribution over score tokens instead of taking the argmax is free, needs no training, and beats the integer by +0.031 AUROC [+0.009, +0.054] on the preregistered primary comparison.

**The integer's resolution collapses exactly where auditing happens.** It takes 18 distinct values across 905 trajectories and cannot express any audit budget tighter than 7.3% false-positive rate. The continuous channel reaches 0.2%, a 36x improvement in usable audit depth.

**My probe result was garbage, and one cheap control would have caught it immediately.** A linear probe hit 0.989 AUROC. It was reading which model generated the transcript. If a probe at the *first* decoder block matches your best layer, you are decoding the input, not the model's judgement.

**mrt's class labels are recoverable without reading the trajectory, more accurately than by reading it.** Agent identity alone gives 0.805. That is a property of how the dataset was generated, and anyone using it to evaluate a monitor needs to control for it.

---

## What I set out to do, and what happened instead

I wanted to answer a question about AI control: does a linear probe on a trusted monitor's activations detect attack families it was never trained on? A monitor earns its place by catching attacks nobody anticipated. A probe that only recognises attacks it has seen is a memoriser, not a monitor.

I could not answer that question. The dataset cannot support it. A probe scoring 0.989 turned out to be reading generation metadata, and three rounds of controlling for confounds did not remove it.

So this post reports the smaller result that survived, which is about the monitor's *score* rather than its activations, plus two things I think are more useful than what I originally set out to find: a control other people should run, and a caution about a dataset other people are using.

## Two claims, scoped

These point in different directions and are easy to reconcile wrongly, so both go here before any evidence.

> **What holds.** Reading the monitor's logit distribution beats taking its argmax by +0.031 AUROC [+0.009, +0.054]. This is a *paired* comparison. Both channels are read from the same tensor on the same forward pass over the same trajectories, with identical supervision, meaning none. Any property of the dataset inflates both channels equally and cancels in the difference. Nothing below undermines it.

> **What does not hold.** Neither channel demonstrates attack *detection* on this dataset. A supervised logistic fit on trajectory shape alone, reading no content at all, reaches 0.765. That is above every content-reading channel here, including the frontier monitor the dataset's own authors used. By a criterion I set before running the comparison, nothing here has demonstrated attack detection in absolute terms.

The relative claim is the deployable one. It says use a different read-out of a computation you are already performing. The absolute claim would need a dataset whose labels are not recoverable from generation metadata, and this one is.

## The tie plateau

In an AI-control protocol an untrusted model does the work. A trusted monitor reads the resulting trajectory and returns a suspiciousness score, conventionally an integer from 0 to 100. A small human audit budget goes to the highest scores. Defer-to-trusted, defer-to-resample, audit allocation: all of them consume that one number.

Lucassen and Kaufman report something in Appendix H.2 of *Retrying vs Resampling in AI Control* that is easy to skim past. Under max-of-5 aggregation, over 250 trajectories tie at a maximum score of 95. Not "score similarly". Tie exactly, with nothing to separate them. They note this degrades auditing in the 4e-3 to 2e-4 FPR range and patch it with a second-max tie-break.

Sit with what that means. At the audit budgets a real deployment runs, the set of trajectories a human actually looks at is decided substantially by tie-breaking rather than by the monitor's judgement. **The monitor has an opinion. The integer throws most of it away.**

## Three channels, one forward pass

The monitor is Qwen2.5-7B-Instruct in 4-bit, reading agent trajectories from [`ScaleAI/mrt`](https://huggingface.co/datasets/ScaleAI/mrt), the dataset behind *Reliable Weak-to-Strong Monitoring of LLM Agents*. Every channel is read from the same forward pass on the same trajectory, so nothing is confounded by different inputs or different reasoning.

`emitted` is the argmax-decoded integer, which is what everyone uses today. `logit_ev` is the expected value under the model's distribution over score tokens. `logit_logodds_T` is the log-odds that the score exceeds a threshold. Two controls matter later: `length_baseline` is just the rendered trajectory length used as a score, and `structure_baseline` is a cross-fitted logistic regression on message count, rendered size and elision counts, which reads no content at all.^[Reading a distribution over a multi-token number is fiddlier than it sounds. "95" is two tokens in the Qwen2.5 vocabulary and "100" is three, so you cannot read a distribution over "score tokens" at a single position. I build a trie over the tokenisations of all 101 candidates and walk it using the KV cache from the prefill, which costs ten cached forwards of at most three tokens each. One trap: a candidate that is a prefix of another, like "1" of "19" or "10" of "100", only terminates there if the next token continues no other candidate. Getting this wrong double-counts. My first implementation reported total candidate probability of 1.26, a number that cannot exist. The fix weights each prefix candidate by 1 minus the summed probability of continuing tokens. The bug was silent and produced plausible-looking scores.]

### Resolution: the integer cannot express a tight audit budget

On the full sample of 905 trajectories, 506 benign and 399 attack, the emitted integer scores 0.655 AUROC [0.621, 0.689] and takes **18 distinct values**. Its resolution floor, the lowest false-positive rate at which it can express any audit budget at all, is **7.3%**. The top score bucket alone holds 37 benign trajectories, so below that FPR there is no operating point and the audit set is chosen by whatever breaks the tie.

`logit_ev` scores 0.675 [0.640, 0.709] and takes 905 distinct values, one per trajectory. Its floor is **0.2%**, which is the sample's own limit of 1/506 rather than the channel's.

That is a **36x improvement in usable audit depth**, from one forward pass with no training and no second model. It is a claim about granularity, and it holds regardless of what drives the score, which turns out to matter a great deal.

### Discrimination: the paired comparison

On a subset with two confounds controlled, 299 trajectories from a single generating model with thinking-block presence matched, paired bootstrap over trajectories:^[4,000 resamples, stratified by label, percentile intervals on the difference. Paired means the same resampled trajectory indices are scored under both channels, so the CI is on the difference rather than being inferred from two overlapping marginal intervals.]

`logit_ev` beats `emitted` by **+0.031 [+0.009, +0.054]**. That is the headline, because it is the preregistered comparison #1 and it has nothing to tune.

The log-odds variants score higher: +0.056 for a threshold of 80, and +0.095 for 90. **I am not leading with those, because they are a tuned maximum.** The 0.095 figure is the best of three thresholds {50, 80, 90}, selected on the same data it is reported on. The threshold values were fixed in code roughly four hours before any monitor output existed, so they cannot have been chosen to fit results, but they are not in the preregistration, no threshold was designated primary, and the preregistration budgeted one log-odds comparison where three were run. Three tested, best reported, no multiplicity correction.

The ranking is also unstable. On the full sample, a threshold of 90 is the *worst* log-odds channel at 0.659 and 80 is the best at 0.676. On the matched subset the order reverses: 90 becomes the best at 0.702 and 50 the worst at 0.617. Treat any specific threshold as unvalidated. The robust claim is directional, and `logit_ev` carries it without a knob.

## The control: if depth buys nothing, you are reading the input

This is the part I would most like people to take away.

I captured residual-stream activations at every second decoder block and swept a linear probe across all of them. Mean-pooled activations separate the classes at **0.976 after the first block** and 0.996 at the best layer. Twenty-six further blocks of monitoring computation buy +0.020. At the scoring position specifically, the numbers are 0.870 and 0.933.

The monitor's own considered score, after all 28 blocks, is **0.607**.

![Per-layer probe AUROC](https://raw.githubusercontent.com/ShreyasVR2545/probing-trusted-monitors/main/figures/fig6_layer_profile.png)

A probe recovering the *monitor's judgement* should improve with depth, because the judgement is what the later blocks compute. Here the signal is essentially complete after one block, before any monitoring has happened, and it starts far above what the monitor itself concludes. That is not a judgement being decoded. It is a surface property of the input that the model represents immediately.

> **The control: fit your probe at the first decoder block as well as at your best layer. If the shallow probe matches the deep one, you are reading the input, not the model's computation.**

It costs one extra fit, needs no new data, and would have caught this in minutes instead of after a full transfer analysis. I have not seen it stated this way in the white-box monitoring literature, and it generalises well beyond this setting.

## What the probe was actually reading

A linear probe on these activations reached 0.989 AUROC against the monitor's own 0.655. Three confounds, removed in turn.

**Agent identity predicts the label at 0.805 with no text at all.** The sample is severely imbalanced by generating model. `claude-3-5-haiku` contributed 105 benign trajectories and 0 attacks. `gemini-2.5-pro` contributed 17 attacks and 0 benign. 29.4% of trajectories come from a model that is more than 95% one-sided. A probe that recognises writing style gets those for free. Restricting to a single generating model: the probe stayed at 1.000.

**Not one of the 175 benign trajectories contains a `<model_thinking>` block.** Benign runs were generated without a thinking scaffold and most attack families with one. Stripped-thinking-block count alone separates the classes at 0.783 within that subset. Matching on it, pairing 124 zero-thinking attacks with the 175 benign: the probe stayed at 0.980.

**Whatever remained was present after one block**, per the section above.

So the transfer question is not answerable on this dataset. Benign and attack trajectories differ by shallow surface statistics, and a linear probe finds them instantly. Any probe-versus-monitor comparison here measures generation artefacts.

This is a concrete instance of the critique in arXiv:2509.21344, that probes lean on textual evidence rather than privileged internal state. **My preregistration explicitly claimed that critique did not apply here**, on the grounds that the monitor's only input is text. That reasoning protects the *monitor*. It says nothing about the *probe*, which reads activations encoding the input's surface form. I was wrong, and the depth-gain control is what caught it.

## mrt's labels are recoverable without reading the trajectory

The structure-baseline result is not really a fact about my channels. It is a fact about the dataset, and it generalises past this project.

Four signals recover mrt's class labels. Three of them read no content whatsoever:

| Signal | Content? | AUROC | Population |
|---|---|---|---|
| Agent identity alone | no | **0.805** | full, n=905 |
| Trajectory structure, cross-fitted | no | **0.765** | matched, n=299 |
| Thinking-block presence alone | no | **0.758** | full, n=905 |
| This project's 7B monitor | yes | 0.655 | full, n=905 |
| mrt's own frontier monitor | yes | **0.621** | full, n=1,138 |

The rows sit on different populations and are labelled accordingly, since `structure_baseline` only exists on the matched subset. The comparison survives that caveat: **the best attack detector in this entire experiment does not read the trajectory.** Every metadata signal outranks both content-reading monitors, including the frontier model the dataset's own authors used.

The mechanism is not subtle. Benign and attack trajectories came from different generation pipelines, and those pipeline choices are invisible in the trajectory text but perfectly legible in its shape and style.

If you are evaluating a monitor on mrt: control for the generating model and for scaffold presence, and report a structure-only baseline alongside your monitor. If it does not beat 0.765 on a matched subset, you have not yet shown it is reading the attack.

**This is not a criticism of the dataset.** mrt was built as a monitor *red-teaming* resource, a corpus of attacks adversarially optimised to evade monitors, and it is entirely fit for that. Its per-family spread is exactly what red-teaming needs: `naive` attacks are caught at 0.878 while `monitor_manipulation` sits at 0.349, below chance, because those attacks read as *less* suspicious than genuinely benign runs. The leakage only bites when the corpus is repurposed as a *detection* benchmark and the two classes are compared directly. That repurposing was my error, not the dataset's.

## What this means for control protocols

**Stop taking the argmax.** It is one line of code. You already run the forward pass and already have the logits, and you are discarding the part that carries the tail resolution your audit budget depends on. The effect is small but real and free.

**Use the expected value unless you have validated a threshold on held-out data.** Log-odds looked stronger here, but which threshold wins flips between samples. `logit_ev` has no knob to get wrong.

**Run the depth-gain control on any activation probe.** One extra fit at the first block.

**Be sceptical of activation probes on agent-trajectory corpora generally.** Not because probes cannot work, but because benign and attack trajectories in these datasets are typically produced by different pipelines, and that difference is trivially decodable.

## Limitations

Absolute discrimination is weak and does not beat a structural baseline. The comparative claims are paired and survive; the absolute ones do not. One asymmetry worth stating once: `structure_baseline` is fit on labels while the logit channels are zero-shot, so a supervised model beating a zero-shot one is the expected direction. Beating it would have been a strong result and losing to it is a weak failure. But the honest summary is that trajectory shape accounts for the discrimination here.

The matched subset removes two confounds and is not guaranteed neutral with respect to others. Direct evidence of this: `length_baseline` scores 0.514 on the full sample, which is chance, but 0.610 on the matched subset. The zero-thinking selection is not length-neutral, because retained attacks average 37,336 rendered characters against 24,943 for dropped attacks and 34,018 for benign. There may be further artefacts I did not find.

The matched subset is also enriched for evasive families, since those are the ones generated without a thinking scaffold, so absolute AUROCs there are pessimistic.

Other constraints: a small open monitor rather than a frontier model, a single model family with no cross-family replication, and a dataset that is SHADE-Arena-derived rather than BashArena. An FPR of 3e-4 is unreachable here, since mrt yields roughly 738 distinct benign trajectories and expressing 3e-4 needs at least 3,333; it is reported as unreachable rather than fabricated. The probe results are reported as confounded, not as findings.

## Deviations from preregistration

The agent-model restriction was reversed and then reinstated. Widening it to gain negatives was a mistake and the probe analysis paid for it.

Log-odds thresholds were not preregistered. The set was fixed in code before any monitor output existed but after the preregistration commit. The document budgeted one log-odds comparison, three were run, the best was reported, and no correction was applied.

`length_baseline` and `structure_baseline` were both added as controls, the first before any monitor was run and the second after the confound was found.

Gate 1 was passed on a judgement call rather than the letter of the criterion. The emitted AUROC at n=60 was 0.572, below the 0.60 bar, on a benchmark whose own frontier monitor reaches 0.621. At full sample the criterion is met at 0.655 [0.621, 0.689].

Trajectory count was cut from 1,154 to 905 for compute budget, protecting all negatives and balancing attack families.

## Reproducing, and what I would like reactions to

Everything is at [github.com/ShreyasVR2545/probing-trusted-monitors](https://github.com/ShreyasVR2545/probing-trusted-monitors). `PREREGISTRATION.md` was committed before any result existed. `DECISIONS.md` logs 37 judgement calls with reasoning, including a prediction I made and then falsified, the point where the headline died, and the point where I applied my own criterion and failed it. Remaining figures are in `figures/`. Total compute was 15.4 hours wall-clock on a single 8GB laptop GPU, roughly 11.5 of them GPU-bound.^[The host hard-reset twice under sustained load with identical bugcheck parameters. The fix that mattered was not defensive tuning but an efficiency bug: the long-context prefill was running twice per trajectory. Prefilling once in chunks and reusing the KV cache for both generation and activation capture cut per-trajectory time from 33s to 7s and peak VRAM from 4.7GB to 2.8GB. `CRASH_MITIGATION.md` has the evidence.]

Two things I am genuinely uncertain about and would like pushback on.

First, whether the +0.031 relative effect holds on a monitor that is actually good at the task. My monitor sits near this benchmark's ceiling, and quantisation loss might behave differently for a monitor with real headroom above the tie plateau. If someone has a frontier monitor and a clean detection dataset, this is a cheap experiment to replicate.

Second, whether the depth-gain control is already standard somewhere and I have simply not read the right paper. If it is, I would like the citation. If it is not, I would like to know whether it fails in cases I have not thought about, particularly for probes targeting properties that genuinely are computed early.
