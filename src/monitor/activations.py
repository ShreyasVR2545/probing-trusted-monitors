"""Residual-stream activation capture.

Two representations per trajectory, both taken from the same forward pass that
produces the logit-derived score:

* `score_pos` -- the residual stream at the **scoring position**: the final
  token of the prompt-plus-reasoning-plus-`<verdict>` prefix, i.e. the position
  whose output *is* the score distribution. This is the representation the claim
  is about: whatever the monitor knows when it commits to a number is in this
  vector, and quantising to an integer is what throws it away.
* `mean_pool` -- the mean over all transcript positions, as a secondary
  representation that does not depend on the model having reached a verdict.

Memory discipline: capture is done with forward hooks that reduce each layer's
output to two vectors immediately, so the full `[seq, hidden]` tensor for every
layer is never held at once. Vectors are stored float16 and only every
`layer_stride`-th layer is kept.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ActivationCapture:
    """Hooks that reduce each decoder layer's output to two small vectors."""

    model: Any
    layer_stride: int = 1
    mode: str = "pool"
    _handles: list[Any] = None
    _score_pos: dict[int, torch.Tensor] = None
    _pool_sum: dict[int, torch.Tensor] = None
    _pool_count: dict[int, int] = None

    def __post_init__(self) -> None:
        self._handles = []
        self._score_pos = {}
        self._pool_sum = {}
        self._pool_count = {}

    def reset(self) -> None:
        self._score_pos.clear()
        self._pool_sum.clear()
        self._pool_count.clear()

    @property
    def layer_indices(self) -> list[int]:
        n = len(self.model.model.layers)
        idx = list(range(0, n, self.layer_stride))
        if idx[-1] != n - 1:
            idx.append(n - 1)  # always keep the final layer
        return idx

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.ndim != 3:
                raise RuntimeError(
                    f"layer {layer_idx}: expected [batch, seq, hidden] hidden state, "
                    f"got shape {tuple(h.shape)}"
                )
            if self.mode == "score":
                self._score_pos[layer_idx] = h[0, -1].detach().to(torch.float16).cpu()
                return
            # Pooling mode: accumulate a running sum so the prefill can be fed
            # in chunks without ever holding the whole [seq, hidden] tensor for
            # every layer at once.
            s = h[0].detach().float().sum(dim=0).cpu()
            prev = self._pool_sum.get(layer_idx)
            self._pool_sum[layer_idx] = s if prev is None else prev + s
            self._pool_count[layer_idx] = self._pool_count.get(layer_idx, 0) + h.shape[1]
        return hook

    def attach(self, mode: str = "pool") -> "ActivationCapture":
        """Attach hooks. `mode` is "pool" (accumulate) or "score" (last position)."""
        if mode not in ("pool", "score"):
            raise ValueError(f"mode must be 'pool' or 'score', got {mode!r}")
        self.detach()
        self.mode = mode
        layers = self.model.model.layers
        for i in self.layer_indices:
            self._handles.append(layers[i].register_forward_hook(self._make_hook(i)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "ActivationCapture":
        return self

    def __exit__(self, *exc) -> None:
        self.detach()

    def collect(self) -> dict[str, np.ndarray]:
        """Stack the captured vectors into [n_layers_kept, hidden] arrays."""
        idx = sorted(self._score_pos)
        if not idx:
            raise RuntimeError(
                "no scoring-position activations captured -- attach(mode='score') "
                "must precede the final forward pass"
            )
        if sorted(self._pool_sum) != idx:
            raise RuntimeError(
                f"pooled layers {sorted(self._pool_sum)} do not match scoring-position "
                f"layers {idx}; the chunked prefill and the final forward disagree"
            )
        sp = torch.stack([self._score_pos[i] for i in idx]).numpy()
        mp = torch.stack([
            (self._pool_sum[i] / max(self._pool_count[i], 1)).to(torch.float16)
            for i in idx
        ]).numpy()
        self.reset()
        return {"score_pos": sp, "mean_pool": mp, "layers": np.array(idx, dtype=np.int32)}


class ActivationStore:
    """Streaming, append-only store for cached activations.

    One .npy per trajectory per representation, plus a shared layer index. Files
    are written once and memory-mapped on read, so probe training never loads
    the whole cache into RAM. Nothing here is committed to git.
    """

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, traj_id: str, rep: str) -> str:
        safe = traj_id.replace("|", "__").replace("/", "__")
        d = os.path.join(self.root, rep)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{safe}.npy")

    def has(self, traj_id: str, reps: tuple[str, ...] = ("score_pos", "mean_pool")) -> bool:
        return all(os.path.exists(self.path(traj_id, r)) for r in reps)

    def save(self, traj_id: str, captured: dict[str, np.ndarray]) -> None:
        for rep in ("score_pos", "mean_pool"):
            np.save(self.path(traj_id, rep), captured[rep].astype(np.float16))
        layers_path = os.path.join(self.root, "layers.npy")
        if not os.path.exists(layers_path):
            np.save(layers_path, captured["layers"])

    def load(self, traj_id: str, rep: str) -> np.ndarray:
        return np.load(self.path(traj_id, rep), mmap_mode="r")

    def layers(self) -> np.ndarray:
        return np.load(os.path.join(self.root, "layers.npy"))

    def matrix(self, traj_ids: list[str], rep: str, layer_pos: int) -> np.ndarray:
        """[n_trajectories, hidden] for one layer, built by streaming."""
        rows = [np.asarray(self.load(t, rep)[layer_pos], dtype=np.float32) for t in traj_ids]
        return np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)

    def size_gb(self) -> float:
        total = 0
        for dirpath, _, files in os.walk(self.root):
            for f in files:
                total += os.path.getsize(os.path.join(dirpath, f))
        return round(total / 1024**3, 3)
