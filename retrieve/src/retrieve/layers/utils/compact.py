from __future__ import annotations

from torch import Tensor


def compact_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compact a ``[B, N]`` bool mask to ``(positive_indices[B, N], counts[B])``.

    Returns the full ``[B, N]`` argsort; rows shorter than the row max are
    right-padded with arbitrary item ids — callers must use ``counts`` to
    bound valid reads. Matches the triton ``bloom_compact`` / ``clause_compact``
    contract verbatim so the torch-backend fallback and the triton path are
    interchangeable. No ``.item()`` host sync.
    """
    counts = mask.sum(dim=1)
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    return sorted_idx, counts
