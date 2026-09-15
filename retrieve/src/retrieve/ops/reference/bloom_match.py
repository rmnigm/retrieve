from __future__ import annotations

from torch import Tensor


def bloom_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """``(qb & sig) == qb`` across W int64 words; qb [B, W], sigs [N, W] int64 → [B, N] bool."""
    match = (qb.unsqueeze(1) & sigs.unsqueeze(0)) == qb.unsqueeze(1)
    return match.all(dim=-1)
