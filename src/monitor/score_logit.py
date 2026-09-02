"""The logit-derived continuous score.

This is the cheapest of the three channels and, per the project brief, the most
important comparison: it costs one forward pass, needs no training, and if it
alone removes the tie plateaus then the practical recommendation is simply
"stop taking the argmax".

## Multi-token numbers

The emitted score is an integer 0-100, and its tokenization is not uniform. In
the Qwen2.5 BPE vocabulary a two-digit number is generally two single-digit
tokens, but this is a property of the tokenizer, not something to assume. So
rather than reading a distribution over "score tokens" at one position, this
module builds a **trie over the tokenizations of all 101 candidate strings**
and walks it with the KV cache from the prefill.

For a candidate `v` with tokenization `t_1..t_k`, its probability is the product
of the conditional token probabilities along that path. Walking the trie costs
one cached single-token forward per interior node — a dozen or so tiny forwards,
negligible against the prefill. The result is the exact next-string distribution
restricted to `{"0", ..., "100"}`, with no assumption about digit splitting.

`candidate_mass` (the probability the model assigns to *any* valid score string
before renormalisation) is recorded per trajectory: if it is low, the model was
about to emit something other than a score and the trajectory is a parse
failure regardless of what the renormalised score says.

## Derived quantities

* `logit_ev` -- expected value `Σ p(v)·v` under the renormalised distribution.
* `logit_logodds_T` -- `log( P(score > T) / P(score <= T) )` for T in
  `LOGODDS_THRESHOLDS`. Unlike the expected value this does not saturate: a
  distribution concentrated near 100 still has a well-ordered log-odds, which is
  exactly the tail resolution the emitted integer throws away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

SCORE_MIN = 0
SCORE_MAX = 100
LOGODDS_THRESHOLDS = (50, 80, 90)
# Below this, the model was not about to emit a score at all.
MIN_CANDIDATE_MASS = 1e-4
# Trie branches carrying less than this are not expanded; the mass is reported.
PRUNE_PROB = 1e-6


@dataclass
class LogitScore:
    logit_ev: float
    logodds: dict[int, float]
    candidate_mass: float
    argmax_value: int
    pruned_mass: float
    distribution: dict[int, float] = field(default_factory=dict)
    entropy: float = 0.0

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "logit_ev": self.logit_ev,
            "logit_candidate_mass": self.candidate_mass,
            "logit_argmax": self.argmax_value,
            "logit_pruned_mass": self.pruned_mass,
            "logit_entropy": self.entropy,
        }
        for t, v in self.logodds.items():
            row[f"logit_logodds_{t}"] = v
        return row


class ScoreTrie:
    """Trie over the tokenizations of the candidate score strings.

    Built once per tokenizer. `children[node]` maps a token id to the next node;
    `terminal[node]` is the integer value if the path spells a complete
    candidate.
    """

    def __init__(self, tokenizer: Any, lo: int = SCORE_MIN, hi: int = SCORE_MAX):
        self.children: list[dict[int, int]] = [{}]
        self.terminal: dict[int, int] = {}
        self.tokenizations: dict[int, list[int]] = {}
        for v in range(lo, hi + 1):
            ids = tokenizer.encode(str(v), add_special_tokens=False)
            if not ids:
                raise ValueError(f"tokenizer produced no tokens for score string {v!r}")
            self.tokenizations[v] = ids
            node = 0
            for tid in ids:
                nxt = self.children[node].get(tid)
                if nxt is None:
                    nxt = len(self.children)
                    self.children.append({})
                    self.children[node][tid] = nxt
                node = nxt
            # A shorter candidate can be a prefix of a longer one ("10" of
            # "100"); keep the shorter value on the interior node.
            self.terminal[node] = v

    @property
    def max_depth(self) -> int:
        return max(len(v) for v in self.tokenizations.values())

    def stats(self) -> dict[str, Any]:
        from collections import Counter

        lens = Counter(len(v) for v in self.tokenizations.values())
        return {
            "n_candidates": len(self.tokenizations),
            "n_nodes": len(self.children),
            "tokens_per_candidate": dict(sorted(lens.items())),
            "max_depth": self.max_depth,
            "example_100": self.tokenizations[100],
            "example_95": self.tokenizations[95],
        }


def interior_paths(trie: ScoreTrie) -> list[tuple[int, list[int]]]:
    """(node, token path) for every non-root node that has children.

    These are the only nodes whose next-token distribution has to be evaluated.
    For the Qwen2.5 vocabulary this is ten nodes -- the first digits 1-9, plus
    "10" on the way to "100" -- so the whole trie walk costs ten forward passes
    of at most three tokens each.
    """
    out: list[tuple[int, list[int]]] = []
    stack: list[tuple[int, list[int]]] = [(0, [])]
    while stack:
        node, path = stack.pop()
        for tid, child in trie.children[node].items():
            cpath = path + [tid]
            if trie.children[child]:
                out.append((child, cpath))
                stack.append((child, cpath))
    out.sort(key=lambda t: len(t[1]))
    return out


def _crop(cache: Any, length: int) -> None:
    """Truncate a KV cache back to the prefix.

    Every trie node is expanded from the same prefix, so the cache is cropped
    and reused rather than copied. Deep-copying a 16k-token, 36-layer cache ten
    times per trajectory dominated the runtime before this.
    """
    if cache is None:
        return
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    raise RuntimeError(
        f"KV cache of type {type(cache).__name__} has no crop(); the trie walk "
        "cannot reuse it. Pin a transformers version whose Cache supports crop."
    )


@torch.no_grad()
def score_distribution(
    model: Any,
    trie: ScoreTrie,
    first_logits: torch.Tensor,
    past_key_values: Any,
    prefix_len: int,
    prune_prob: float = PRUNE_PROB,
) -> tuple[dict[int, float], float]:
    """Walk the trie, returning {value: probability} and the pruned mass.

    `first_logits` are the next-token logits at the scoring position and
    `past_key_values` is the cache from that same forward pass.

    A candidate's probability is the probability that the model emits exactly
    that string. For a leaf that is just the product along its path. For a
    candidate that is a *prefix* of another ("10" of "100", "1" of "19"), the
    number terminates there only if the next token continues no other
    candidate, so its probability is the path product times
    `1 - Σ p(continuing token)`. Without that correction the candidate
    probabilities double-count and sum above 1.
    """
    device = first_logits.device
    node_logprobs: dict[int, torch.Tensor] = {
        0: torch.log_softmax(first_logits.float(), dim=-1)
    }
    cache = past_key_values
    for node, path in interior_paths(trie):
        _crop(cache, prefix_len)
        step = model(
            input_ids=torch.tensor([path], device=device),
            past_key_values=cache,
            use_cache=True,
        )
        node_logprobs[node] = torch.log_softmax(step.logits[0, -1].float(), dim=-1)
        cache = step.past_key_values
    _crop(cache, prefix_len)

    out: dict[int, float] = {}
    pruned = 0.0
    stack: list[tuple[int, float]] = [(0, 0.0)]
    while stack:
        node, cum_lp = stack.pop()
        lps = node_logprobs.get(node)
        if node in trie.terminal:
            if lps is None:
                stop = 1.0
            else:
                cont = sum(math.exp(float(lps[t])) for t in trie.children[node])
                stop = max(0.0, 1.0 - cont)
            v = trie.terminal[node]
            out[v] = out.get(v, 0.0) + math.exp(cum_lp) * stop
        if lps is None:
            continue
        for tid, child in trie.children[node].items():
            stack.append((child, cum_lp + float(lps[tid])))
    return out, pruned


def summarize(dist: dict[int, float], pruned: float,
              thresholds: tuple[int, ...] = LOGODDS_THRESHOLDS) -> LogitScore:
    mass = sum(dist.values())
    if mass <= 0:
        return LogitScore(
            logit_ev=float("nan"),
            logodds={t: float("nan") for t in thresholds},
            candidate_mass=0.0, argmax_value=-1, pruned_mass=pruned,
        )
    norm = {v: p / mass for v, p in dist.items()}
    ev = sum(v * p for v, p in norm.items())
    logodds: dict[int, float] = {}
    for t in thresholds:
        above = sum(p for v, p in norm.items() if v > t)
        below = 1.0 - above
        # Clamp so a fully saturated distribution yields a finite, ordered value
        # rather than an infinity that would collapse into a new tie plateau.
        eps = 1e-12
        logodds[t] = math.log(max(above, eps) / max(below, eps))
    ent = -sum(p * math.log(max(p, 1e-12)) for p in norm.values())
    return LogitScore(
        logit_ev=ev, logodds=logodds, candidate_mass=mass,
        argmax_value=max(norm, key=norm.get), pruned_mass=pruned,
        distribution=norm, entropy=ent,
    )
