from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import bloom_subset_match


def bloom_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """``(qb & sig) == qb`` across W int64 words; qb [B, W], sigs [N, W] int64 → [B, N] bool, the
    shared ``bloom_subset_match`` over every item."""
    return bloom_subset_match(qb, sigs.unsqueeze(0).expand(qb.shape[0], *sigs.shape))


def bloom_match_scores(scores: Tensor, qb: Tensor, sigs: Tensor) -> Tensor:
    """``scores`` [B, N] where the subset test passes, ``-inf`` elsewhere."""
    return torch.where(bloom_match(qb, sigs), scores, float("-inf"))
