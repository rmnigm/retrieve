"""The per-bit frequency table behind the bloom scorer's pass-rate bound (kernels.md § SilverTorch
kernels, "Gated tile skip"): an item passes only if every queried bit is set, so a query's pass
rate is at most its rarest bit's frequency, and a scorer program can tell from a few table loads
whether its row is selective enough to vote on skipping tiles."""

from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import popcount_int64


def bloom_cluster_counts(bloom_transposed: Tensor, cluster_offsets: Tensor, n: int) -> Tensor:
    """``[n_lists, m_bits]`` int32: per cluster, how many of its items (cluster-sorted, as the
    transposed index) have bit m set. One bit row at a time: an ``[n]`` int32 temporary."""
    n_lists = cluster_offsets.numel() - 1
    cluster_of = torch.repeat_interleave(
        torch.arange(n_lists, device=bloom_transposed.device), cluster_offsets.diff()
    )
    lane = torch.arange(64, device=bloom_transposed.device)
    counts = torch.zeros(n_lists, bloom_transposed.shape[0], dtype=torch.int32, device=lane.device)
    for m, words in enumerate(bloom_transposed):
        bits = ((words.unsqueeze(1) >> lane) & 1).reshape(-1)[:n].to(torch.int32)
        counts[:, m].index_add_(0, cluster_of, bits)
    return counts


def bloom_bit_freq(bloom_transposed: Tensor, n: int) -> Tensor:
    """``[m_bits]`` fp32: the fraction of the ``n`` items whose signature has bit m set, from the
    transposed index ``[m_bits, ceil(n / 64)]`` (padding bits are 0)."""
    return popcount_int64(bloom_transposed).sum(dim=1).to(torch.float32) / n
