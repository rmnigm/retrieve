"""Shared numeric helpers across the dataset CLIs.

The arxiv and goodreads CLIs both build a per-target narrow query-attribute
tensor and a rare-biased wide-shelf sample. The numeric ops are identical;
only the input vocab and per-clause semantics differ. Keep the math here, the
domain plumbing in the dataset modules.
"""

from __future__ import annotations

import numpy as np
import torch


def dense_remap_ids(values: list[int], *, start: int = 1) -> dict[int, int]:
    """Build a 1-indexed (or `start`-indexed) dense int map from sorted unique values."""
    return {int(v): i + start for i, v in enumerate(values)}


def synthesize_qa_narrow(
    target_first: list[int],
    narrow_t: torch.Tensor,
    n_clauses: int,
) -> torch.Tensor:
    """Build per-user `query_attrs_narrow` of shape `[n_users, n_clauses]`.

    For each clause `c`, take the first non-pad value of
    `narrow_t[target - 1, c, :]` (i.e. the most-frequent or first-listed
    attribute the held-out target carries). Users with `target <= 0`
    (no held-out item) get all `-1`.

    ``narrow_t`` is the ``[N, n_clauses, A]`` 0-indexed dense item-attr
    tensor (row ``i`` = item_id ``i + 1``). ``target_first[u]`` is a
    1-indexed item_id (matching ``item_id_map.json``).
    """
    n_users = len(target_first)
    qa = torch.full((n_users, n_clauses), -1, dtype=torch.long)
    for u, tgt in enumerate(target_first):
        if tgt <= 0:
            continue
        for c in range(n_clauses):
            row = narrow_t[tgt - 1, c]
            for v in row.tolist():
                if v != -1:
                    qa[u, c] = v
                    break
    return qa


def sample_rare_biased_wide(
    target_first: list[int],
    wide_t: torch.Tensor,
    wide_global_freq: list[int] | np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sample 1-shelf rare and 2-shelf (common+rare) wide attrs per user.

    Returns `(qa_wide_1, qa_wide_2, n_drop_1, n_drop_2)`.

    1-shelf: ``p ∝ 1/sqrt(global_freq)`` — biases toward rare shelves so the
    filter is selective.
    2-shelf: one common (``p ∝ sqrt(freq)``) and one rare (``p ∝ 1/sqrt(freq)``
    among the remaining bag entries). Both `-1` if the bag is empty / has
    only a single entry / target is missing.
    """
    rng = np.random.default_rng(seed)
    n_users = len(target_first)
    freq_np = np.asarray(wide_global_freq, dtype=np.int64)
    qa_wide_1 = np.full((n_users,), -1, dtype=np.int64)
    qa_wide_2 = np.full((n_users, 2), -1, dtype=np.int64)
    n_drop_1 = 0
    n_drop_2 = 0
    wide_np = wide_t.numpy()
    # ``wide_t`` is ``[N, 1, BAG]`` 0-indexed dense; ``tgt`` is a 1-indexed
    # item_id, so index at ``tgt - 1``.
    for u, tgt in enumerate(target_first):
        if tgt <= 0:
            n_drop_1 += 1
            n_drop_2 += 1
            continue
        bag = wide_np[tgt - 1, 0]
        bag = bag[bag != -1]
        if len(bag) == 0:
            n_drop_1 += 1
            n_drop_2 += 1
            continue
        freq = freq_np[bag].astype(np.float64)
        w_rare = 1.0 / np.sqrt(np.maximum(freq, 1.0))
        w_rare = w_rare / w_rare.sum()
        qa_wide_1[u] = int(rng.choice(bag, p=w_rare))
        if len(bag) < 2:
            n_drop_2 += 1
            continue
        w_common = np.sqrt(freq)
        w_common = w_common / w_common.sum()
        common_idx = int(rng.choice(len(bag), p=w_common))
        common_id = int(bag[common_idx])
        rest = np.array([i for i in range(len(bag)) if i != common_idx])
        if len(rest) == 0:
            n_drop_2 += 1
            continue
        w_rare2 = 1.0 / np.sqrt(np.maximum(freq[rest], 1.0))
        w_rare2 = w_rare2 / w_rare2.sum()
        rare_id = int(bag[rest[int(rng.choice(len(rest), p=w_rare2))]])
        qa_wide_2[u, 0] = common_id
        qa_wide_2[u, 1] = rare_id
    return qa_wide_1, qa_wide_2, n_drop_1, n_drop_2


__all__ = ["dense_remap_ids", "synthesize_qa_narrow", "sample_rare_biased_wide"]
