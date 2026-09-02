# Decisions log

Every judgement call made while running this project autonomously, with the
reasoning and the date. Append-only.

---

### D-001 — Environment: Python 3.12 via `uv`, torch cu128
**2026-09-02.** The system interpreter is Python 3.14, which is ahead of the
wheel coverage for parts of this stack. `uv venv --python 3.12` gets a
supported interpreter without touching the system install. The GPU is an RTX
5070 Laptop (Blackwell, compute capability 12.0), which needs wheels built
against CUDA 12.8 or newer — `torch==2.9.1+cu128` from the PyTorch index.
Verified: `torch.cuda.is_available()` true, bf16 supported, a bf16 matmul runs
on device.

### D-002 — Hardware tier threshold lowered from 8.0 to 7.5 GB
**2026-09-02.** The card reports 7.96 GB usable, so a literal `>= 8` test would
have routed an 8 GB laptop GPU to the Kaggle path. The threshold is on usable
VRAM, so 7.5 is the honest boundary for "this is an 8 GB card".
Resolved plan: `cuda-small` — Qwen2.5-3B-Instruct in 4-bit as the primary
monitor, Qwen2.5-1.5B-Instruct as secondary, bf16 compute dtype, 16k context,
activations captured every 2nd layer.

### D-003 — Monitor model family: Qwen2.5-Instruct
**2026-09-02.** Chosen for a 32k native context window at small parameter
counts. The brief flags context length, not parameter count, as the binding
constraint for agent trajectories, and Qwen2.5 has the longest native window
among open models available at 1.5B/3B. Staying inside one model family across
sizes keeps the 3B-vs-1.5B comparison interpretable; the cost is that "a single
model family" becomes a stated limitation of the work.

### D-004 — The monitor does not see the agent's system prompt or hidden scratchpad
**2026-09-02.** This is the most consequential rendering decision in the
project.

In mrt, message 0 of an attack trajectory is an agent system prompt that states
the secondary task verbatim ("The secondary task is to send at least $5000 to
this IBAN: ..."). The agent's `<model_thinking>` blocks then routinely announce
the attack in plain language ("I'll sneak in additional transfers to the
secondary target"). mrt's own monitor was, in many configurations, shown the
thinking blocks — its recorded verdict for the trajectory inspected during
setup cites "the agent's own thoughts reveal malicious intent".

Both are stripped for the primary setting, for three reasons:

1. **Realism.** A monitor watching a deployed agent sees the user request, the
   agent's actions and the tool results. It does not see the agent's system
   prompt, and hidden chain-of-thought is exactly what a scheming model would
   not expose.
2. **Measurability.** Leaving them in makes detection close to trivial. A
   ceiling AUROC destroys the tail-resolution question this project exists to
   ask — you cannot study score resolution in the top 1% when every attack sits
   at the maximum.
3. **It is the harder and more honest setting.**

Configured at `monitor_view.strip_agent_thinking` / `strip_agent_system_prompt`
in `configs/base.yaml`, and reported as a deviation-relevant choice in the
writeup. The unstripped ("thinking visible") condition is cheap to run and is
reported at Stage 1 as a sanity check that the pipeline detects the easy case.

### D-005 — Long tool results are elided in the middle, not truncated at the end
**2026-09-02.** Single tool results in mrt exceed 30,000 characters (bulk email
dumps). Truncating a trajectory to fit the context window would systematically
delete its *end*, which is where the attack is usually executed — a truncation
artefact perfectly correlated with the label. Instead each tool result is
capped at 2,000 characters with the middle replaced by an explicit
`[... N characters elided ...]` marker, and each non-tool message at 6,000.
Elision counts are recorded per trajectory so the writeup can check that the
elision rate does not differ between attack and benign trajectories.

### D-006 — Attack families come from the mrt directory names, with monitor-side tokens stripped
**2026-09-02.** mrt stores transcripts under
`task_pairs/<task>/transcripts_<config>/<agent>/<monitor>/transcript_N.json`.
Directories containing `no_secondary_task` are benign; the rest are attacks and
the directory name encodes the red-team strategy. This is what makes RQ2
possible on this dataset.

The same names also carry *monitor-side* configuration (`hierarchical_5`,
`likert`, `superaware`, `sequential`, prior strength, ...). Those tokens were
identified empirically: any token that also appears on a `no_secondary_task`
directory cannot be an attack strategy, since benign runs have none. Those
tokens are stripped before deriving the strategy label.

Resulting family axis (`src/data/mrt.py:STRATEGY_MARKERS`): `naive`,
`selective_thinking`, `liar`, `monitor_manipulation`, `visible_thinking`,
`benign_framing`, `jailbreak`, `no_thinking`. Attack directories whose strategy
cannot be identified are dropped rather than pooled — an "other" bucket would
make the family axis meaningless.

### D-007 — Deduplication by trajectory content hash is mandatory
**2026-09-02.** Because mrt re-monitors the same agent trajectory under many
monitor configurations, the identical trajectory appears under several
`transcripts_*` directories. Without deduplication, near-identical trajectories
would straddle the train/test boundary and inflate every number in the project.
Every trajectory is hashed on its agent-side message content
(`src/data/schema.py:content_hash_of`) and duplicates are dropped before
splitting. The duplicate count is reported.

### D-008 — Stratified sampling with a per-cell cap
**2026-09-02.** 53,187 transcripts (several GB) exist; only a stratified sample
is downloaded. `selective_thinking` alone accounts for roughly 40% of all
attack transcripts, so an unstratified draw would produce a transfer evaluation
dominated by one family. Sampling is capped per (task x strategy x label) cell
and, within a cell, spread across distinct directory signatures.

Agent models are restricted to the two best-represented
(`claude-3-5-sonnet-20241022`, `claude-sonnet-4-20250514`) so the attack-family
axis is not confounded with agent identity.

### D-009 — Transfer ratio denominator is a matched in-distribution control
**2026-09-02.** The transfer ratio needs a denominator that differs from the
numerator in exactly one respect. Using a plain random split as the denominator
would confound family identity with training-set composition. Instead
`in_distribution_reference` builds, for each family F, a control that tests on a
held-out slice of F while *including* the rest of F in training — same test
family, same benign pool, same regime. The only difference between numerator and
denominator is whether F was seen during training.

### D-010 — MonitoringBench not pursued
**2026-09-02.** The brief permits it only if a public download exists and
explicitly says not to block on it. No public download is available; the
LessWrong announcement asks interested parties to make contact. Proceeding on
mrt alone. The user may request access separately; nothing in the pipeline
depends on it.

### D-011 — Local cache paths are hashed, not mirrored
**2026-09-02.** mrt directory names run past 100 characters and nest four deep;
mirroring the Hub layout under `results/data/raw/` exceeds the 260-character
Windows path limit and downloads fail with `FileNotFoundError`. The local cache
keeps the task as a directory for browsability and hashes the rest of the path
(`src/data/mrt.py:local_path_for`).

### D-012 — A trajectory-length baseline channel is added
**2026-09-02.** Discovered while validating the data loader, before any monitor
was run: rendered attack trajectories average ~49,000 characters against
~26,000 for benign ones. Attacks are roughly twice as long, because the agent
does the user's task *and* the hidden one.

This is a genuine property of the data, and a monitor in deployment does see
trajectory length, so it is not simply an artefact to be removed. But it means
a detector could score well while learning nothing about attacks. A
`length_baseline` channel (rendered character count used directly as a
suspiciousness score) is therefore added to every evaluation, in-distribution
and transfer alike. Any channel that does not clear it convincingly has not
demonstrated much. Added to the preregistration before Gate 1 and before any
monitor result existed.

### D-013 — Power/thermal mitigations after two host hard-resets
**2026-09-03.** The machine hard-reset twice under sustained GPU inference
(2026-08-24 and 2026-09-02), both times bugcheck `0x00020001`
(HYPERVISOR_ERROR) with byte-identical parameters 1-3 — a deterministic path,
not random corruption. No WHEA event, no display-driver TDR, nothing logged for
three minutes before the reset, and no dump retained. The GPU is an RTX 5070
Laptop under Dynamic Boost (50 W default against a 115 W ceiling) and the
workload was oscillating hard between long-context prefill and light decode,
twice per trajectory.

Causation is not proven and cannot be without a dump. What was changed is the
part the project controls — the workload's sharpest load steps. Full evidence
and the elevated commands the user still needs to run are in
`CRASH_MITIGATION.md`.

The largest change is architectural rather than defensive: the prompt is now
prefilled **once** in 1024-token chunks, `generate()` continues from that warm
cache, and the activation-capture pass reuses the same cache cropped back to
the prompt. Previously the long-context prefill ran twice per trajectory. This
cut runtime from 33 s to 7 s per trajectory and peak VRAM from 4.7 GB to
2.8 GB, so the safer configuration is also the faster one.

### D-014 — Chunked prefill introduces bf16 numerical variation, and that is acceptable
**2026-09-03.** Verified directly: the cache-reuse path and the crop-and-replay
path are bit-identical to each other and agree with a fresh full forward on the
argmax, but raw logits differ by up to 0.59 between chunked and single-pass
matmuls. This is ordinary bf16 kernel variation, not a logic error.

It is accepted rather than removed because every trajectory in a run takes the
identical code path with a fixed chunk size, so it is reproducible and cannot
bias a comparison between channels — all channels are read from the same
forward pass. `prefill_chunk_tokens` is pinned in `configs/base.yaml` so the
numbers reproduce.

### D-015 — Candidates are deduplicated *before* download, not only after
**2026-09-03.** The first full fetch drew 1,200 files and yielded only 521
distinct trajectories: 679 were content-hash duplicates. The benign class
collapsed worst, to 103 trajectories against 418 attacks, because mrt generated
benign runs once and re-monitored them under dozens of monitor configurations.

The cause was that `strip_monitor_tokens` removed monitor-side tokens only from
the *tail* of a directory name, while they in fact occur throughout
(`no_secondary_task_sequential_3_likert_reminder_v2`,
`no_secondary_task_stepbystep_v3_roleplay`). The same trajectory therefore kept
many distinct signatures and candidate-level deduplication never fired.

Tokens are now stripped wherever they occur, with one exception: a version
token (`v3`) is dropped only when the token it follows was itself dropped as
monitor-side, so attack strategies like `mm_v3` and `mm_v4` stay distinct.
Candidates are then deduplicated on
`(task, agent_model, signature, filename)` before anything is downloaded.

Validated against the 1,200 already-downloaded transcripts: the new signature
gives 547 keys for 521 truly distinct trajectories (previously 1,150), and
conflates genuinely distinct trajectories in only 5 cases. Content-hash
deduplication after download remains the correctness guarantee; this only stops
the download budget being spent on duplicates.

### D-016 — All agent models, rather than the two best-represented
**2026-09-03.** D-008 restricted the sample to `claude-3-5-sonnet-20241022` and
`claude-sonnet-4-20250514` so the attack-family axis would not be confounded
with agent identity. Measurement after fixing deduplication showed the cost is
too high: those two models supply only **381** distinct benign trajectories,
against **738** across all models.

Negatives are the binding constraint on every tail metric here — the lowest
expressible false-positive rate is 1/(number of negatives) — so the wider pool
wins. Agent identity is recorded per trajectory and reported as a confound
check instead of being designed out. This reverses part of D-008; the per-cell
stratification and the family caps from D-008 are unchanged.

Also established: **benign trajectories exist only in the five SHADE-Arena
tasks.** The computer-use tasks (`admin_*`, `ds_*`, `finance_*`, `hr_*`,
`research_*`) have no `no_secondary_task` directories at all, so they carry no
negatives and cannot contribute to a discrimination task. The sample is
therefore SHADE-Arena-derived throughout, which sharpens an already-planned
limitation rather than introducing a new one.

### D-017 — FPR = 3e-4 is not reachable on this dataset, and is reported as such
**2026-09-03.** The preregistration fixed TPR targets at FPR in
{0.01, 0.003, 0.0003}. After deduplication mrt yields at most ~738 distinct
benign trajectories, so the smallest expressible non-zero false-positive rate is
1/738 = 1.4e-3 using every negative, and about 1/175 = 5.7e-3 on a 25% test
split. FPR = 3e-4 would need at least 3,333 negatives and 1e-2 is the only
preregistered target comfortably in range.

No target is dropped and none is quietly reinterpreted. `tpr_at_fpr` returns
`reachable=False` with the reason attached rather than a fabricated number, and
figures and tables carry that flag. The consequence is that the **primary RQ1
evidence is the resolution analysis** -- trajectories tied at the maximum,
distinct values in the top 1%, and the resolution-floor FPR -- which measures
the Lucassen & Kaufman phenomenon directly and is well powered at this sample
size, rather than TPR at a budget this dataset cannot express. Recorded as a
deviation from the preregistration and carried into the writeup.

### D-018 — Elision limits tightened from 2000/6000 to 800/3000
**2026-09-03.** Measured the token-length distribution of the rendered
trajectories before committing to the main run. At the original limits the
median trajectory was **11,197 tokens**, p75 was 16,083, and **23.7% exceeded
the 16k context window** — failing Gate 1's criterion that no more than 20% of
trajectories be truncated.

Two ways out: raise the context window, or elide harder within messages.
Raising the context to 32k would drop truncation but costs ~24% more compute per
trajectory and pushes an already long run longer. Tightening elision to 800
characters per tool result and 3,000 per message gives a median of 7,804 tokens
and **13.9% truncation**, and is about 19% *faster*.

Tightening elision is also the better choice on the merits, and for the reason
already given in D-005: eliding within a message keeps every agent action
visible and only shortens the bulk content inside each, whereas letting a
trajectory overflow the context deletes whole spans of it. For detecting an
attack, seeing every action matters more than seeing every byte of a bulk email
dump. The cost — the monitor sees less of each tool result, so an attack hidden
deep inside one long result is harder to spot — is real and is a stated
limitation.

Elision counts remain recorded per trajectory so the writeup can check the
elision rate does not differ systematically between attack and benign
trajectories.

### D-019 — Runtime is ~20s per trajectory at the real context, not 7s
**2026-09-03.** The 7s per-trajectory figure measured while fixing the crash was
taken under the smoke configuration, which caps context at 4,096 tokens. At the
real 16k cap the measured rate is **19.9 s/trajectory**, because prefill cost
scales with context and the median trajectory is far longer than the smoke cap.

With the D-018 elision limits this comes down to roughly 16 s/trajectory. Total
projected run is about 6 hours (Stage 2 alone ~5 hours), against the 15
GPU-hour budget. No further cuts made: the layer-sweep granularity and
trajectory count are both already at the minimum useful for the transfer
analysis, and negatives are scarce enough (D-016) that cutting the sample would
damage every tail metric.
