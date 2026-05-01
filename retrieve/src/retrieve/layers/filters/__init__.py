from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from retrieve.interfaces import FilterModule
from retrieve.layers.filters.bloom import BloomFilter
from retrieve.layers.filters.clause import ClauseIndex


def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """Element-wise AND of N optional ``[B, N]`` masks.

    ``None`` inputs are ignored. Returns ``None`` if every input is ``None``.
    """
    out: Tensor | None = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else (out & m)
    return out


def combine_indices(
    filters: Sequence[FilterModule],
    query_clause_attrs: Sequence[Tensor],
) -> tuple[Tensor, Tensor]:
    """Sparse cascade across multiple filters.

    The first filter produces ``(ids, counts)`` via its native compact path;
    each subsequent filter is applied to those ids via ``evaluate_subset`` and
    the survivors are re-compacted. No ``[B, N]`` is materialized by the cascade
    itself — only the first filter's ``evaluate_indices`` may, depending on its
    implementation.

    Caller orders filters most-selective first.
    """
    if len(filters) == 0:
        raise ValueError("combine_indices requires at least one filter")
    if len(filters) != len(query_clause_attrs):
        raise ValueError(
            f"filters / queries length mismatch: {len(filters)} vs {len(query_clause_attrs)}",
        )

    f0, q0 = filters[0], query_clause_attrs[0]
    ids, counts = f0.evaluate_indices(q0)

    for f, q in zip(filters[1:], query_clause_attrs[1:]):
        b, p = ids.shape
        if p == 0:
            return ids, counts
        sub_mask = f.evaluate_subset(q, ids)  # [B, P] bool
        valid = torch.arange(p, device=ids.device).unsqueeze(0) < counts.unsqueeze(1)
        sub_mask = sub_mask & valid
        new_counts = sub_mask.sum(dim=1)
        new_p = int(new_counts.max().item())
        if new_p == 0:
            ids = torch.empty(b, 0, dtype=torch.long, device=ids.device)
            counts = new_counts
            continue
        sorted_idx = sub_mask.float().argsort(dim=1, descending=True, stable=True)[:, :new_p]
        ids = ids.gather(1, sorted_idx)
        counts = new_counts

    return ids, counts


__all__ = [
    "BloomFilter",
    "ClauseIndex",
    "combine_indices",
    "combine_masks",
]
