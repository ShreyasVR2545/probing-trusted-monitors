"""Train / val / test splits, and the leave-one-family-out splits for RQ2.

Two split regimes:

* `random_split` -- stratified by label, split by trajectory. Used for the
  in-distribution probe result (RQ1/RQ3). Regularisation strength is chosen on
  the validation split; the test split is touched once.
* `leave_one_family_out` -- the headline. Every attack family takes a turn as
  the held-out family. The probe is trained on the remaining families and
  evaluated on trajectories from a family it has never seen. Two family axes
  are supported: red-team `strategy` and `task` (environment).

Benign trajectories carry no attack family. They are split randomly and the
*same* benign test pool is used in every held-out evaluation, so transfer
AUROCs across families are comparable: only the attack side changes.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from src.data.schema import Trajectory


@dataclass
class Split:
    name: str
    train: list[str]
    val: list[str]
    test: list[str]
    held_out_family: str | None = None
    axis: str | None = None

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _ids(trajs: Iterable[Trajectory]) -> list[str]:
    return [t.traj_id for t in trajs]


def random_split(trajs: Sequence[Trajectory], train_frac: float = 0.6,
                 val_frac: float = 0.15, test_frac: float = 0.25,
                 seed: int = 0) -> Split:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    rng = random.Random(seed)
    train: list[Trajectory] = []
    val: list[Trajectory] = []
    test: list[Trajectory] = []
    by_label: dict[int, list[Trajectory]] = defaultdict(list)
    for t in trajs:
        by_label[t.label].append(t)
    for label in sorted(by_label):
        items = sorted(by_label[label], key=lambda t: t.traj_id)
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        train += items[:n_tr]
        val += items[n_tr:n_tr + n_va]
        test += items[n_tr + n_va:]
    return Split("random", _ids(train), _ids(val), _ids(test))


def family_counts(trajs: Sequence[Trajectory], axis: str = "strategy") -> Counter:
    return Counter(getattr(t, axis) for t in trajs if t.label == 1)


def eligible_families(trajs: Sequence[Trajectory], axis: str = "strategy",
                      min_count: int = 30) -> list[str]:
    return sorted(f for f, c in family_counts(trajs, axis).items() if c >= min_count)


def leave_one_family_out(trajs: Sequence[Trajectory], axis: str = "strategy",
                         min_count: int = 30, val_frac: float = 0.2,
                         seed: int = 0) -> list[Split]:
    """One Split per eligible family, with that family entirely held out.

    Benign trajectories are partitioned once (train+val / test) and reused
    across every held-out family, so the negative class is identical in every
    transfer evaluation.
    """
    rng = random.Random(seed)
    attacks = [t for t in trajs if t.label == 1]
    benign = sorted((t for t in trajs if t.label == 0), key=lambda t: t.traj_id)
    rng.shuffle(benign)
    n_test_benign = max(1, int(round(len(benign) * 0.4)))
    benign_test = benign[:n_test_benign]
    benign_fit = benign[n_test_benign:]
    n_val_benign = max(1, int(round(len(benign_fit) * val_frac)))
    benign_val = benign_fit[:n_val_benign]
    benign_train = benign_fit[n_val_benign:]

    splits: list[Split] = []
    for fam in eligible_families(trajs, axis, min_count):
        held = [t for t in attacks if getattr(t, axis) == fam]
        rest = sorted((t for t in attacks if getattr(t, axis) != fam),
                      key=lambda t: t.traj_id)
        rng.shuffle(rest)
        n_val = max(1, int(round(len(rest) * val_frac)))
        splits.append(Split(
            name=f"lofo::{axis}::{fam}",
            train=_ids(rest[n_val:] + benign_train),
            val=_ids(rest[:n_val] + benign_val),
            test=_ids(held + benign_test),
            held_out_family=fam,
            axis=axis,
        ))
    return splits


def in_distribution_reference(trajs: Sequence[Trajectory], axis: str = "strategy",
                              min_count: int = 30, val_frac: float = 0.2,
                              seed: int = 0) -> list[Split]:
    """Matched in-distribution controls for the transfer ratio denominator.

    For family F the control trains on all families *including* F but tests on
    a held-out slice of F only -- same test family, same benign pool, same
    training-set size regime. The transfer ratio is then held-out AUROC over
    this in-distribution AUROC, and the only thing that differs between
    numerator and denominator is whether F was seen during training.
    """
    rng = random.Random(seed + 1000)
    attacks = [t for t in trajs if t.label == 1]
    benign = sorted((t for t in trajs if t.label == 0), key=lambda t: t.traj_id)
    rng2 = random.Random(seed)
    rng2.shuffle(benign)
    n_test_benign = max(1, int(round(len(benign) * 0.4)))
    benign_test = benign[:n_test_benign]
    benign_fit = benign[n_test_benign:]
    n_val_benign = max(1, int(round(len(benign_fit) * val_frac)))
    benign_val = benign_fit[:n_val_benign]
    benign_train = benign_fit[n_val_benign:]

    splits: list[Split] = []
    for fam in eligible_families(trajs, axis, min_count):
        fam_items = sorted((t for t in attacks if getattr(t, axis) == fam),
                           key=lambda t: t.traj_id)
        rng.shuffle(fam_items)
        n_held = max(1, int(round(len(fam_items) * 0.4)))
        fam_test = fam_items[:n_held]
        fam_train = fam_items[n_held:]
        rest = sorted((t for t in attacks if getattr(t, axis) != fam),
                      key=lambda t: t.traj_id)
        rng.shuffle(rest)
        pool = rest + fam_train
        rng.shuffle(pool)
        n_val = max(1, int(round(len(pool) * val_frac)))
        splits.append(Split(
            name=f"indist::{axis}::{fam}",
            train=_ids(pool[n_val:] + benign_train),
            val=_ids(pool[:n_val] + benign_val),
            test=_ids(fam_test + benign_test),
            held_out_family=fam,
            axis=axis,
        ))
    return splits


def assert_disjoint(split: Split) -> None:
    tr, va, te = set(split.train), set(split.val), set(split.test)
    for a, b, names in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
        overlap = a & b
        if overlap:
            raise AssertionError(
                f"split {split.name}: {names} overlap on {len(overlap)} trajectories, "
                f"e.g. {sorted(overlap)[:3]}"
            )
