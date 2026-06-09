from __future__ import annotations

from torch import Tensor


def compact_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compact a [B, N] bool mask to (positive_indices [B, N], counts [B]); rows are right-padded
    with arbitrary ids, so callers must use ``counts`` to bound valid reads.

    Matches the triton ``bloom_compact``/``clause_compact`` contract so torch and triton paths
    stay interchangeable."""
    counts = mask.sum(dim=1)
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    return sorted_idx, counts
