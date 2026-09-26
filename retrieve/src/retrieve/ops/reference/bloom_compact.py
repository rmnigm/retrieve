from __future__ import annotations

from torch import Tensor

from retrieve.functional import compact_mask
from retrieve.ops.reference.bloom_match import bloom_match


def bloom_compact(qb: Tensor, sigs: Tensor) -> tuple[Tensor, Tensor]:
    """``compact_mask(bloom_match(qb, sigs))`` → (positive_indices [B, N] int64, counts [B]
    int64); the tail past ``counts[b]`` is the argsort's (see ``compact_mask``)."""
    return compact_mask(bloom_match(qb, sigs))
