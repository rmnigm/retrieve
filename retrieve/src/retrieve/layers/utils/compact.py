from __future__ import annotations

import torch
from torch import Tensor


def compact_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compact a ``[B, N]`` bool mask to ``(positive_indices[B, P], counts[B])``.

    P = max passing count across the batch; rows shorter than P are right-padded
    with arbitrary item ids — callers must use ``counts`` to bound valid reads.
    """
    counts = mask.sum(dim=1)
    p = int(counts.max().item())
    if p == 0:
        b = mask.shape[0]
        return (
            torch.zeros(b, 0, dtype=torch.long, device=mask.device),
            counts,
        )
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    return sorted_idx[:, :p], counts
