"""Pure-torch SilverTorch phases 2+3 (the ``"torch"`` backend and the parity oracle): int8 × int8
→ int32 dot with one global scale and a per-row query scale, optionally gated by the row-wise
bloom subset test, then ``masked_topk``. The dot runs in fp32 (not ``torch._int_mm``) — at these
``D`` the integer products fit the fp32 mantissa exactly, so it is bit-identical to an int32
accumulator, and ``(dot · q_scale) · global_scale`` is the same two left-associated fp32
multiplies as the Triton kernels and the official epilogue (kernels.md → Numerics)."""

from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import bloom_subset_match, masked_topk
from retrieve.indexing.quantize import quantize_int8


def _probe_score(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
    keep: Tensor,
) -> tuple[Tensor, Tensor]:
    safe = flat_probed_items.clamp_min(0)
    q_codes, q_scales = quantize_int8(query)  # [B, D] int8, [B] fp32
    codes = item_codes[safe].to(torch.float32)  # [B, P, D]
    scores = torch.einsum("bd,bpd->bp", q_codes.to(torch.float32), codes)
    scores = scores * q_scales.unsqueeze(1) * global_scale
    return masked_topk(scores, k, valid=keep, gather_ids=flat_probed_items)


def codesigned_probe_score(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Plain int8 ANN scoring over the probed items (``-1`` = padding)."""
    keep = flat_probed_items >= 0
    return _probe_score(query, flat_probed_items, item_codes, global_scale, k, keep)


def codesigned_probe_score_bloom(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    query_bits: Tensor,
    bloom_sigs: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Int8 ANN scoring gated by the bloom subset test ``(qb & sig) == qb`` per probed item."""
    valid = flat_probed_items >= 0
    keep = valid & bloom_subset_match(query_bits, bloom_sigs[flat_probed_items.clamp_min(0)])
    return _probe_score(query, flat_probed_items, item_codes, global_scale, k, keep)
