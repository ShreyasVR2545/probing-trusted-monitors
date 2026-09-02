# Running the GPU stages on Kaggle

Hardware detection resolved this machine to the `cuda-small` tier and the
experiment runs **locally** (see `HARDWARE.md`). This path exists for two
reasons:

1. **Fallback** — if the local host keeps hard-resetting under sustained GPU
   load (`CRASH_MITIGATION.md`), the GPU stages can be moved off it entirely.
2. **Robustness check** — a Kaggle P100 (16 GB) or T4 x2 clears the
   `cuda-large` tier, so the same pipeline can run **Qwen2.5-7B-Instruct**
   unquantised at a 32k context. That answers "is this a small-model artefact?",
   which is otherwise a standing limitation of the work.

Only stages 1–3 need a GPU. **Stages 4–6 (transfer, analysis, figures) are
CPU-cheap and always run locally**, on top of whichever activation cache is
present.

## Steps

1. Open <https://www.kaggle.com/code> → **New Notebook**.
2. **File → Import Notebook** → upload `notebooks/kaggle_runner.ipynb`.
3. Right sidebar → **Accelerator** → `GPU T4 x2` or `GPU P100`.
   Right sidebar → **Internet** → **On** (needed to clone the repo and pull the
   dataset from the Hub).
4. Edit the first cell if you want a non-default configuration:
   - `REPO_URL` — defaults to this repository.
   - `STAGES` — defaults to `["0", "1", "2", "3"]`.
   - `SMOKE` — set `True` for a 30-trajectory end-to-end check first. Do this
     once before committing a 9-hour session.
   - `FORCE_MODEL` — set to `"Qwen/Qwen2.5-7B-Instruct"` for the 7B robustness
     run. Leave `None` to let hardware detection choose.
5. **Run All**. A smoke run takes minutes; the full grid takes a few hours.
6. When it finishes, download `results.zip` from the **Output** panel.

## Bringing the results back

```bash
python scripts/run_all.py --import-results ~/Downloads/results.zip
python scripts/run_all.py --stage 4      # transfer, GATE 2
python scripts/run_all.py --stage 5
python scripts/run_all.py --stage 6
```

`--import-results` unpacks the archive into `results/`, and the manifests
inside it mark the GPU stages complete, so stage 4 picks up from the imported
activation cache rather than recomputing it.

## What the archive contains

`results.zip` holds `results/data/metadata.csv`, `results/signal_check/`,
`results/scores/scores_*.csv`, `results/probe/`, every `manifest.json`, and
`results/scores/activations/` — the cache is the expensive artefact and is the
reason to bring it back rather than regenerate it. Raw mrt transcripts are
excluded; they are re-downloadable and are not ours to redistribute.

## Session limits worth knowing

- Kaggle GPU sessions are capped at 9 hours and the weekly GPU quota is ~30
  hours. The full grid fits, but run `SMOKE = True` first.
- A session dies if the browser tab is closed for too long; the notebook writes
  `results.zip` incrementally after each stage so a dead session does not lose
  everything.
- Stage 2 is resumable at trajectory granularity. Re-running the notebook after
  a lost session skips every trajectory whose activations are already cached,
  provided you re-upload the previous `results.zip` as a dataset and unzip it
  into `results/` first.

## Licence note

`ScaleAI/mrt` is CC-BY-NC-4.0. Kaggle notebooks default to public — set the
notebook to **Private** if you would rather not republish anything derived from
it. Nothing in the archive contains raw transcripts, but the cached activations
are derived from them.
