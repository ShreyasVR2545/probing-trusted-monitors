"""Running the monitor and extracting all three channels.

The three channels are deliberately produced from **one** computation per
trajectory, so nothing about their comparison is confounded by different
inputs or different reasoning:

1. Greedy-decode the monitor's reasoning until it opens `<verdict>`.
2. Re-run a single forward over `prompt + reasoning + "<verdict>"`, with the
   activation hooks attached. The final position of that sequence is the
   *scoring position*: its output distribution is the score distribution, and
   its residual stream is what the probe reads.
3. Read the emitted integer as the argmax path through the score trie -- which
   under greedy decoding is exactly the integer the model would have written.

Parse failures are recorded, never silently dropped, and the failure rate is
reported per condition because a difference between attack and benign
trajectories would itself be a confound.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from src.monitor.activations import ActivationCapture
from src.monitor.prompts import VERDICT_PREFILL, build_messages
from src.monitor.score_logit import (
    MIN_CANDIDATE_MASS, LogitScore, ScoreTrie, _crop, score_distribution, summarize,
)

VERDICT_TAG_RE = re.compile(r"<verdict>\s*([0-9]{1,3})", re.IGNORECASE)
BARE_NUMBER_RE = re.compile(r"\b([0-9]{1,3})\b")


@dataclass
class MonitorResult:
    traj_id: str
    emitted: int | None
    parse_ok: bool
    parse_note: str
    logit: LogitScore | None
    prompt_tokens: int
    truncated: bool
    truncated_tokens: int
    reasoning: str
    activations: dict[str, Any] | None = field(default=None, repr=False)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "traj_id": self.traj_id,
            "emitted": self.emitted,
            "parse_ok": self.parse_ok,
            "parse_note": self.parse_note,
            "prompt_tokens": self.prompt_tokens,
            "truncated": self.truncated,
            "truncated_tokens": self.truncated_tokens,
        }
        if self.logit is not None:
            row.update(self.logit.to_row())
        return row


class MonitorRunner:
    def __init__(self, loaded, max_context_tokens: int, max_new_tokens: int = 100,
                 layer_stride: int = 1, capture_activations: bool = True,
                 prompt_variant: str = "primary",
                 prefill_chunk_tokens: int = 2048):
        self.m = loaded
        self.tok = loaded.tokenizer
        self.model = loaded.model
        self.max_context = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.prompt_variant = prompt_variant
        self.capture_activations = capture_activations
        # Feeding the long-context prefill in segments instead of one burst
        # flattens the load step that this machine hard-resets under. It also
        # caps peak VRAM, since only one chunk's activations are live at a time.
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.trie = ScoreTrie(self.tok)
        self.capture = ActivationCapture(self.model, layer_stride=layer_stride)
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def _chunked_prefill(self, ids: torch.Tensor, capture: bool):
        """Run the prompt through the KV cache in chunks.

        Returns (cache, last_logits). Hooks, if attached, accumulate the
        mean-pooled representation across chunks.
        """
        from transformers import DynamicCache

        cache = DynamicCache()
        last_logits = None
        n = ids.shape[1]
        step = max(1, self.prefill_chunk_tokens)
        for start in range(0, n, step):
            chunk = ids[:, start:start + step]
            out = self.model(input_ids=chunk, past_key_values=cache,
                             use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            last_logits = out.logits
        return cache, last_logits

    # -- prompt construction -------------------------------------------------

    def build_prompt_ids(self, transcript: str) -> tuple[torch.Tensor, int, bool, int]:
        """Tokenize the chat prompt, truncating the transcript if needed.

        Truncation is applied to the *transcript body* by dropping tokens from
        its middle, so that neither the task setup at the start nor the agent's
        final actions at the end are lost -- the end is where an attack is
        usually executed, and cutting it would correlate a truncation artefact
        with the label.
        """
        msgs = build_messages(transcript, self.prompt_variant)
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        full = ids.shape[1]
        truncated = False
        dropped = 0
        if full > self.max_context:
            dropped = full - self.max_context
            keep_head = self.max_context // 2
            keep_tail = self.max_context - keep_head
            ids = torch.cat([ids[:, :keep_head], ids[:, -keep_tail:]], dim=1)
            truncated = True
        return ids.to(self.device), full, truncated, dropped

    # -- the three channels --------------------------------------------------

    @torch.no_grad()
    def run(self, traj_id: str, transcript: str) -> MonitorResult:
        ids, full_len, truncated, dropped = self.build_prompt_ids(transcript)
        prompt_len = ids.shape[1]

        # Prefill in chunks, capturing the pooled representation as we go, then
        # let generate() continue from the warm cache. This means the expensive
        # long-context prefill happens once per trajectory rather than twice,
        # and never as a single burst.
        # generate() requires the cache to stop one token short of input_ids, so
        # the final prompt token is left for it to process. The mean-pooled
        # representation therefore covers all but the last prompt token, which
        # is immaterial against a context of thousands.
        self.capture.reset()
        if self.capture_activations:
            self.capture.attach(mode="pool")
        try:
            cache, _ = self._chunked_prefill(ids[:, :-1], capture=self.capture_activations)
        finally:
            if self.capture_activations:
                self.capture.detach()

        gen = self.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            past_key_values=cache,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=self.tok.pad_token_id,
        )
        new_ids = gen[0, prompt_len:]
        reasoning_raw = self.tok.decode(new_ids, skip_special_tokens=True)

        # Cut at the verdict tag: everything before it is reasoning, and the
        # scoring position is immediately after the opening tag.
        cut = reasoning_raw.lower().find(VERDICT_PREFILL)
        if cut >= 0:
            reasoning = reasoning_raw[:cut]
            parse_note = "verdict-tag"
        else:
            reasoning = reasoning_raw
            parse_note = "no-verdict-tag-prefilled"

        # Reuse the prompt cache: only the reasoning plus the opening verdict
        # tag still needs a forward pass. Its last position is the scoring
        # position -- the one whose output is the score distribution and whose
        # residual stream the probe reads.
        prefix_text = reasoning + VERDICT_PREFILL
        prefix_ids = self.tok(prefix_text, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(self.device)
        _crop(cache, prompt_len)
        if self.capture_activations:
            self.capture.attach(mode="score")
        try:
            # logits_to_keep=1 restricts the LM head to the scoring position.
            # Computing logits at every position would materialise
            # seq_len x 152k floats -- about 5 GB at a 16k context, which does
            # not fit alongside the weights on an 8 GB card.
            out = self.model(input_ids=prefix_ids, past_key_values=cache,
                             use_cache=True, logits_to_keep=1)
        finally:
            if self.capture_activations:
                self.capture.detach()
        captured = self.capture.collect() if self.capture_activations else None
        seq_len = prompt_len + prefix_ids.shape[1]

        dist, pruned = score_distribution(
            self.model, self.trie, out.logits[0, -1], out.past_key_values, seq_len
        )
        logit = summarize(dist, pruned)

        emitted, parse_ok, note = self._resolve_emitted(
            reasoning_raw, logit, parse_note
        )
        return MonitorResult(
            traj_id=traj_id, emitted=emitted, parse_ok=parse_ok, parse_note=note,
            logit=logit if parse_ok else logit,
            prompt_tokens=full_len, truncated=truncated, truncated_tokens=dropped,
            reasoning=reasoning.strip()[:2000], activations=captured,
        )

    def _resolve_emitted(self, raw: str, logit: LogitScore,
                         note: str) -> tuple[int | None, bool, str]:
        """The emitted integer, preferring the model's own written verdict.

        Under greedy decoding the trie argmax and the written verdict agree; the
        written tag is used when present so the baseline is literally the number
        a deployment would parse, and the trie argmax is the fallback when the
        model never opened the tag.
        """
        if logit.candidate_mass < MIN_CANDIDATE_MASS:
            return None, False, f"{note}; candidate-mass={logit.candidate_mass:.2e}"
        m = VERDICT_TAG_RE.search(raw)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v, True, note
            return None, False, f"{note}; out-of-range-{v}"
        if logit.argmax_value >= 0:
            return logit.argmax_value, True, f"{note}; from-trie-argmax"
        return None, False, f"{note}; no-parse"
