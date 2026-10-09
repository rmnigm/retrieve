"""The per-bit frequency table behind the bloom scorer's pass-rate bound (kernels.md § SilverTorch
kernels, "Gated tile skip"): an item passes only if every queried bit is set, so a query's pass
rate is at most its rarest bit's frequency, and a scorer program can tell from a few table loads
whether its row is selective enough to vote on skipping tiles."""

from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import popcount_int64


def bloom_bit_freq(bloom_transposed: Tensor, n: int) -> Tensor:
    """``[m_bits]`` fp32: the fraction of the ``n`` items whose signature has bit m set, from the
    transposed index ``[m_bits, ceil(n / 64)]`` (padding bits are 0)."""
    return popcount_int64(bloom_transposed).sum(dim=1).to(torch.float32) / n
