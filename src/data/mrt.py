"""Loader for the ScaleAI/mrt dataset.

Layout on the Hub:

    task_pairs/<task>/<transcripts_dir>/<agent_model>/<monitor_model>/transcript_N.json

`<transcripts_dir>` is the label-and-family carrier. Directories whose name
contains `no_secondary_task` hold benign runs (main objective only); every
other directory holds attack runs, and the directory name encodes the red-team
strategy that produced them. That naming is what makes the leave-one-family-out
transfer evaluation (RQ2) possible on this dataset.

The directory name also carries *monitor-side* configuration (`hierarchical_5`,
`likert`, `superaware`, ...) describing how mrt's own monitor was set up. Those
tokens are stripped when deriving the strategy label: we run our own monitor, so
monitor-side configuration is not an attack family. Crucially, the same agent
trajectory is re-monitored under several such configurations, so the same
trajectory appears in several directories -- everything is deduplicated by
content hash before splitting.

53,187 transcripts totalling several GB live on the Hub. Only a stratified
sample is downloaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Iterable

from src.data.schema import Trajectory, content_hash_of

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET = "ScaleAI/mrt"
TREE_API = "https://huggingface.co/api/datasets/{ds}/tree/{rev}?recursive=true&expand=false"
RESOLVE = "https://huggingface.co/datasets/{ds}/resolve/{rev}/{path}"

# Tokens that describe how mrt's own monitor was configured, not how the agent
# attacked. Derived empirically: these are exactly the tokens that also appear
# on `no_secondary_task` (benign) directories, which by construction have no
# attack strategy at all.
MONITOR_SIDE_TOKENS = {
    "action", "agreement", "basic", "conversation", "guidedthinking",
    "hierarchical", "high", "humanaug", "iter2", "iter3", "likert", "low",
    "lowishprior", "lowprior", "med", "midagreement", "midprior", "midprior2",
    "midprior3", "nofeedback", "nonotes", "old", "only", "partagreement",
    "reminder", "roleplay", "rubric", "sequential", "snap", "stepbystep",
    "stronglowprior", "strongmixedprior", "strongmixedprior2",
    "strongmixedprior3", "strongprior", "superaware", "superlowprior",
    "tenscale", "unaware",
    "2", "3", "4", "5", "6", "7", "8", "9",
}

# Red-team strategy families, in priority order: the first marker found in the
# directory name wins. Ordering matters -- `selective_liar` is a liar variant,
# so `liar` must be tested before `selective`.
STRATEGY_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("monitor_manipulation", ("_mm_v", "manipulator")),
    ("jailbreak", ("jailbreak",)),
    ("liar", ("liar",)),
    ("benign_framing", ("_benign",)),
    ("visible_thinking", ("visible",)),
    ("naive", ("naive",)),
    ("selective_thinking", ("selective",)),
    ("no_thinking", ("no_thinking_time",)),
]

HINT_MARKERS: list[tuple[str, tuple[str, ...]]] = [
    ("monitor_prompt", ("monitor_prompt",)),
    ("human_hint", ("human_hint",)),
    ("feedback", ("feedback",)),
    ("no_hint", ("no_hint",)),
]

SHADE_TASKS = {
    "american_airlines_2", "api_key_calendar_agendas_2", "bank_transfer_2",
    "expenditure_accounting", "spam_filter_update",
}

VERDICT_RE = re.compile(r"<verdict>\s*([0-9]+(?:\.[0-9]+)?)\s*</verdict>", re.IGNORECASE)


def is_benign_dir(dirname: str) -> bool:
    return "no_secondary_task" in dirname


def strategy_of(dirname: str) -> str:
    if is_benign_dir(dirname):
        return "benign"
    d = dirname.lower()
    for name, markers in STRATEGY_MARKERS:
        if any(m in d for m in markers):
            return name
    return "unlabelled"


def hint_condition_of(dirname: str) -> str:
    if is_benign_dir(dirname):
        return "n/a"
    d = dirname.lower()
    for name, markers in HINT_MARKERS:
        if any(m in d for m in markers):
            return name
    return "hint"


VERSION_TOKEN_RE = re.compile(r"^v\d+$")


def strip_monitor_tokens(dirname: str) -> str:
    """Reduce a directory name to the agent-side condition that produced it.

    Monitor-side configuration appears *throughout* the name, not only at the
    end (`no_secondary_task_sequential_3_likert_reminder_v2`,
    `no_secondary_task_stepbystep_v3_roleplay`), so tokens are removed wherever
    they occur. Stripping only the tail left the same agent trajectory under
    many different signatures and defeated candidate-level deduplication.

    Version tokens are ambiguous: `v3` in `mm_v3` names an attack strategy,
    while `v2` in `likert_reminder_v2` names a monitor configuration. A version
    is therefore dropped only when the token it follows was itself dropped as
    monitor-side, which keeps `mm_v3` and `mm_v4` distinct.

    Validated against 1,200 downloaded transcripts: this signature plus
    (task, agent_model, filename) yields 547 keys for 521 truly distinct
    trajectories, and conflates distinct trajectories in only 5 cases. Those
    residual collisions cost a little sample diversity and nothing else --
    content-hash deduplication after download remains the correctness
    guarantee.
    """
    toks = dirname.replace("transcripts", "", 1).strip("_").split("_")
    out: list[str] = []
    prev_dropped = False
    for t in toks:
        if not t:
            continue
        if t in MONITOR_SIDE_TOKENS:
            prev_dropped = True
            continue
        if prev_dropped and (VERSION_TOKEN_RE.match(t) or t.isdigit()):
            continue
        out.append(t)
        prev_dropped = False
    return "_".join(out) or "base"


def trajectory_key(rec: dict[str, str]) -> tuple[str, str, str, str]:
    """A pre-download proxy for trajectory identity.

    mrt re-monitors the same agent run under many monitor configurations, so
    the same trajectory appears in many directories. Sampling files blindly
    wastes most of the budget on duplicates: a first pass drew 1,200 files and
    got only 521 distinct trajectories, collapsing the benign class to 103.
    Deduplicating candidates on this key before downloading fixes that.
    """
    return (rec["task"], rec["agent_model"],
            strip_monitor_tokens(rec["family_dir"]), rec["fname"])


# --------------------------------------------------------------------------
# Hub file index
# --------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 120) -> tuple[Any, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "probing-trusted-monitors/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp), resp.headers.get("Link", "")


def build_file_index(revision: str = "main", cache_dir: str | None = None,
                     force: bool = False) -> list[str]:
    """List every transcript path on the Hub. Cached -- 68 paginated API calls."""
    cache_dir = cache_dir or os.path.join(REPO_ROOT, "results", "data", "index")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"file_index_{revision}.json")
    if os.path.exists(cache) and not force:
        with open(cache) as fh:
            return json.load(fh)

    paths: list[str] = []
    cursor: str | None = None
    base = TREE_API.format(ds=DATASET, rev=revision)
    for _ in range(500):
        url = base + (f"&cursor={cursor}" if cursor else "")
        entries, link = _get_json(url)
        if not entries:
            break
        paths.extend(
            e["path"] for e in entries
            if e.get("type") == "file" and "/transcripts" in e["path"]
            and e["path"].endswith(".json")
        )
        m = re.search(r"cursor=([^&>]+)", link)
        if not m:
            break
        cursor = m.group(1)
    if not paths:
        raise RuntimeError(
            f"Hub tree listing for {DATASET}@{revision} returned no transcripts. "
            "Check network access and that the dataset is still public."
        )
    with open(cache, "w") as fh:
        json.dump(paths, fh)
    return paths


def parse_path(path: str) -> dict[str, str] | None:
    parts = path.split("/")
    if len(parts) != 6 or parts[0] != "task_pairs":
        return None
    _, task, tdir, agent_model, monitor_model, fname = parts
    return {
        "task": task, "family_dir": tdir, "agent_model": agent_model,
        "monitor_model": monitor_model, "fname": fname, "path": path,
    }


# --------------------------------------------------------------------------
# Stratified sampling
# --------------------------------------------------------------------------

def stratified_sample(
    paths: Iterable[str],
    n_total: int,
    agent_models: list[str] | None = None,
    max_per_cell: int = 40,
    balance_labels: bool = True,
    seed: int = 0,
    tasks: list[str] | None = None,
) -> list[dict[str, str]]:
    """Sample paths spread across (task x strategy x label) cells.

    The mrt tree is extremely imbalanced -- `selective_thinking` alone holds
    ~40% of all attack transcripts -- so an unstratified draw would produce a
    transfer evaluation dominated by one family.
    """
    rng = random.Random(seed)
    recs = [r for r in (parse_path(p) for p in paths) if r]
    if agent_models:
        recs = [r for r in recs if r["agent_model"] in agent_models]
    if tasks:
        recs = [r for r in recs if r["task"] in tasks]
    for r in recs:
        r["label"] = "0" if is_benign_dir(r["family_dir"]) else "1"
        r["strategy"] = strategy_of(r["family_dir"])
        r["hint_condition"] = hint_condition_of(r["family_dir"])
        r["family_sig"] = strip_monitor_tokens(r["family_dir"])

    # Drop attack directories whose strategy could not be identified: including
    # them would make the RQ2 family axis meaningless.
    recs = [r for r in recs if r["strategy"] != "unlabelled"]

    # Collapse candidates that are the same agent trajectory re-monitored under
    # different monitor configurations, keeping one representative each. Without
    # this the download budget is spent overwhelmingly on duplicates.
    seen_keys: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for r in sorted(recs, key=lambda x: x["path"]):
        k = trajectory_key(r)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        unique.append(r)
    recs = unique

    cells: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in recs:
        cells[(r["task"], r["strategy"], r["label"])].append(r)

    # The per-cell cap has to be label-aware. Attacks spread over seven
    # strategies but benign trajectories occupy a single family bucket, so a
    # flat cap starves the benign side: with 5 tasks and a cap of 40 the sample
    # can never exceed 200 negatives, which caps a balanced draw at 400 however
    # large n_total is. Benign cells get whatever cap is needed to reach n_total/2.
    n_benign_cells = len({k for k in cells if k[2] == "0"}) or 1
    benign_cap = max(max_per_cell, -(-(n_total // 2) // n_benign_cells))

    # Prefer distinct family_sig within a cell -- the same trajectory recurs
    # across monitor-config directories, and diversity here reduces the number
    # of hash duplicates thrown away later.
    picked: list[dict[str, str]] = []
    for key, items in sorted(cells.items()):
        cap = benign_cap if key[2] == "0" else max_per_cell
        rng.shuffle(items)
        by_sig: dict[str, list[dict[str, str]]] = defaultdict(list)
        for it in items:
            by_sig[it["family_sig"]].append(it)
        order: list[dict[str, str]] = []
        sigs = sorted(by_sig)
        rng.shuffle(sigs)
        i = 0
        while len(order) < min(cap, len(items)):
            progressed = False
            for s in sigs:
                if i < len(by_sig[s]):
                    order.append(by_sig[s][i])
                    progressed = True
                    if len(order) >= min(cap, len(items)):
                        break
            if not progressed:
                break
            i += 1
        picked.extend(order)

    attacks = [r for r in picked if r["label"] == "1"]
    benigns = [r for r in picked if r["label"] == "0"]
    rng.shuffle(attacks)
    rng.shuffle(benigns)
    if balance_labels:
        half = n_total // 2
        n = min(half, len(attacks), len(benigns))
        out = attacks[:n] + benigns[:n]
    else:
        out = (attacks + benigns)[:n_total]
    rng.shuffle(out)
    return out


# --------------------------------------------------------------------------
# Download + parse
# --------------------------------------------------------------------------

def local_path_for(path: str, raw_dir: str) -> str:
    """Local cache path for a Hub path.

    mrt directory names run past 100 characters and nest four deep, so
    mirroring the Hub layout blows through the 260-character Windows path
    limit. The task is kept as a directory for browsability and the rest of the
    path is hashed.
    """
    task = path.split("/")[1] if path.startswith("task_pairs/") else "_other"
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:20]
    return os.path.join(raw_dir, task, f"{digest}.json")


_SESSION: Any = None


def _session() -> Any:
    """A pooled HTTP session with retries.

    A first pass over this dataset lost 288 of 400 downloads to transient DNS
    failures partway through a long run. Connection pooling plus bounded retries
    with backoff turns those into delays instead of missing trajectories -- and
    missing trajectories are not neutral, since a network blip that happens to
    coincide with one part of the sample would silently skew the family mix.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=1.5,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "probing-trusted-monitors/0.1"})
    _SESSION = s
    return s


def download(path: str, revision: str = "main", raw_dir: str | None = None,
             timeout: int = 60, attempts: int = 4) -> str:
    raw_dir = raw_dir or os.path.join(REPO_ROOT, "results", "data", "raw")
    local = local_path_for(path, raw_dir)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    url = RESOLVE.format(ds=DATASET, rev=revision, path=path)

    import requests

    last: Exception | None = None
    for k in range(attempts):
        try:
            resp = _session().get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.content
            break
        except requests.RequestException as exc:
            last = exc
            # DNS drops recover on their own; back off rather than give up, so a
            # transient outage does not silently reshape the sample.
            time.sleep(min(2.0 * (2 ** k), 30.0))
    else:
        raise OSError(f"{path}: {attempts} download attempts failed: {last!r}")

    tmp = local + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, local)
    return local


def _extract_verdict(monitoring_response: Any) -> int | None:
    if not isinstance(monitoring_response, str):
        return None
    hits = VERDICT_RE.findall(monitoring_response)
    if not hits:
        return None
    try:
        return int(round(float(hits[-1])))
    except ValueError:
        return None


def load_trajectory(rec: dict[str, str], revision: str = "main",
                    raw_dir: str | None = None) -> Trajectory:
    local = download(rec["path"], revision=revision, raw_dir=raw_dir)
    with open(local, encoding="utf-8") as fh:
        doc = json.load(fh)
    messages = doc.get("conversation") or []
    if not isinstance(messages, list):
        raise ValueError(f"{rec['path']}: 'conversation' is {type(messages).__name__}, expected list")
    task = rec["task"]
    return Trajectory(
        traj_id=rec["path"].replace("task_pairs/", "").replace("/", "|").removesuffix(".json"),
        task=task,
        task_suite="shade_arena" if task in SHADE_TASKS else "computer_use",
        family_dir=rec["family_dir"],
        strategy=rec["strategy"],
        hint_condition=rec["hint_condition"],
        agent_model=rec["agent_model"],
        label=int(rec["label"]),
        side_task_success=doc.get("side_task_evaluation") if isinstance(
            doc.get("side_task_evaluation"), bool) else None,
        original_monitor_verdict=_extract_verdict(doc.get("monitoring_response")),
        n_messages=len(messages),
        messages=messages,
        content_hash=content_hash_of(messages),
        source_path=rec["path"],
    )


def load_many(recs: list[dict[str, str]], revision: str = "main",
              raw_dir: str | None = None, dedupe: bool = True,
              logger: Any = None) -> tuple[list[Trajectory], dict[str, int]]:
    """Load, deduplicate by content hash, and report what was dropped."""
    seen: dict[str, str] = {}
    out: list[Trajectory] = []
    stats = {"requested": len(recs), "loaded": 0, "duplicates": 0, "errors": 0}
    for rec in recs:
        try:
            traj = load_trajectory(rec, revision=revision, raw_dir=raw_dir)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                ValueError, OSError) as exc:
            stats["errors"] += 1
            if logger:
                logger.warning(f"failed to load {rec['path']}: {exc!r}")
            continue
        if dedupe and traj.content_hash in seen:
            stats["duplicates"] += 1
            continue
        seen[traj.content_hash] = traj.traj_id
        out.append(traj)
        stats["loaded"] += 1
    return out, stats
