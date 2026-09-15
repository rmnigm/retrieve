from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import counts_to_valid, masked_topk, popcount_int64


def _score_full_bits(query_bits: Tensor, item_bits: Tensor) -> Tensor:
    """Loop-free xor + popcount + reduce; ``d_total`` is computed inside the body so it stays
    symbolic under a ``dynamic=True`` parent compile."""
    d_total = 64 * item_bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
    hamming = popcount_int64(xor).sum(dim=-1)
    return d_total - 2 * hamming.to(torch.float32)


def oporp_1bit_match_topk_full(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Full-scan Hamming top-K: ``64·W − 2·popcount(q ^ item)`` over every row of ``item_bits``."""
    scores = _score_full_bits(query_bits, item_bits)
    topk_scores, topk_ids = torch.topk(scores, k, dim=1)
    return topk_ids, topk_scores


def oporp_1bit_match_topk_indirect(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
    positive_indices: Tensor,
    counts: Tensor,
) -> tuple[Tensor, Tensor]:
    """Hamming top-K over ``positive_indices[b, :counts[b]]`` only. ``pad_to_k=False``: returns
    ``min(k, P)`` columns (no ``-1``/``-inf`` tail) — frozen behaviour; callers bound short rows
    by ``counts``."""
    d_total = 64 * item_bits.shape[1]
    cand_bits = item_bits[positive_indices]  # [B, P, W]
    xor = query_bits.unsqueeze(1) ^ cand_bits
    hamming = popcount_int64(xor).sum(dim=-1)
    scores = (d_total - 2 * hamming).to(torch.float32)
    valid = counts_to_valid(counts, positive_indices.shape[1])
    return masked_topk(scores, k, valid=valid, gather_ids=positive_indices, pad_to_k=False)
