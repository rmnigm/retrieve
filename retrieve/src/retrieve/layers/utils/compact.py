from __future__ import annotations

from torch import Tensor


def compact_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compact a [B, N] bool mask to (positive_indices [B, N], counts [B]).

    Same contract as the triton ``bloom_compact``/``clause_compact`` kernels, so torch and
    triton paths stay interchangeable: all three return full-width ``[B, N]`` indices with
    only the first ``counts[b]`` entries of each row meaningful. The tails differ — arbitrary
    ids (argsort tail) here vs ``-1`` (prefilled) for the kernels — so consumers must bound
    reads by ``counts`` either way."""
    counts = mask.sum(dim=1)
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    return sorted_idx, counts
