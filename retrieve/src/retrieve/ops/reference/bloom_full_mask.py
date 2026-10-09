from __future__ import annotations

import torch
from torch import Tensor


def bloom_full_mask(query_bit_positions: Tensor, bloom_transposed: Tensor) -> Tensor:
    """The AND of the query's rows of ``bloom_transposed`` (``-1`` slots skipped):
    ``[B, ceil(N / 64)]`` int64, bit ``n % 64`` of word ``n // 64`` set iff item ``n`` passes."""
    rows = bloom_transposed[query_bit_positions.clamp_min(0)]  # [B, C·k, words]
    rows = torch.where((query_bit_positions < 0).unsqueeze(2), -1, rows)
    out = torch.full_like(rows[:, 0], -1)
    for i in range(rows.shape[1]):
        out = out & rows[:, i]
    return out
