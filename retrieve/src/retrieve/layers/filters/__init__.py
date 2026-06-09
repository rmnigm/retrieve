from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from retrieve.interfaces import FilterModule
from retrieve.layers.filters.bloom import BloomFilter
from retrieve.layers.filters.exact_attribute import ExactAttributeFilter


def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """Element-wise AND of N optional ``[B, N]`` masks.

    ``None`` inputs are ignored. Returns ``None`` if every input is ``None``."""
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
    """Sparse cascade across filters: the first produces ``(ids, counts)`` via its compact path,
    then each subsequent filter is applied to those ids via ``evaluate_subset`` and survivors
    re-compacted (no ``[B, N]`` materialized by the cascade). Order filters most-selective first."""
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
        valid = torch.arange(p, device=ids.device).unsqueeze(0) < counts.unsqueeze(1)
        # ids past counts[b] are scratch from clause_compact's torch.empty(); gather as 0, then
        # sub_mask & valid zeroes them out.
        safe_ids = torch.where(valid, ids, ids.new_zeros(()))
        sub_mask = f.evaluate_subset(q, safe_ids)  # [B, P] bool
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
    "ExactAttributeFilter",
    "combine_indices",
    "combine_masks",
]
