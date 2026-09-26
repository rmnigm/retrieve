"""Pure-torch SilverTorch phases 2+3 (the ``"torch"`` backend and the parity oracle) over the
compact CSR probe layout: int8 × int8 → int32 dot with one global scale and a per-row query
scale, optionally gated by the transposed-bloom subset test, then ``masked_topk``. The dot runs
in fp32 (not ``torch._int_mm``) — at these ``D`` the integer products fit the fp32 mantissa
exactly, so it is bit-identical to an int32 accumulator, and ``(dot · q_scale) · global_scale``
is the same two left-associated fp32 multiplies as the Triton kernels and the official epilogue
(kernels.md → Numerics)."""

from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import masked_topk
from retrieve.indexing.quantize import quantize_int8


def probe_positions(
    probe_ids: Tensor, cluster_offsets: Tensor, width: int
) -> tuple[Tensor, Tensor]:
    """Every compact slot of every row → ``(pos [B, width]`` cluster-sorted position, ``0`` past
    the row's items, ``valid [B, width])``. Row ``b`` lists its probed clusters back to back:
    slot ``p`` belongs to probe ``j``, the count of cluster ends at or below ``p``."""
    lo = cluster_offsets[probe_ids]
    sizes = cluster_offsets[probe_ids + 1] - lo
    ends = sizes.cumsum(1)
    slots = torch.arange(width, device=probe_ids.device).expand(probe_ids.shape[0], width)
    j = torch.searchsorted(ends, slots.contiguous(), right=True)
    valid = j < ends.shape[1]
    j = j.clamp_max(ends.shape[1] - 1)
    pos = lo.gather(1, j) + slots - (ends - sizes).gather(1, j)
    return torch.where(valid, pos, 0), valid


def _probe_score(
    query: Tensor,
    pos: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    k: int,
    keep: Tensor,
) -> tuple[Tensor, Tensor]:
    q_codes, q_scales = quantize_int8(query)  # [B, D] int8, [B] fp32
    codes = item_codes[pos].to(torch.float32)  # [B, width, D]
    scores = torch.einsum("bd,bpd->bp", q_codes.to(torch.float32), codes)
    scores = scores * q_scales.unsqueeze(1) * global_scale
    return masked_topk(scores, k, valid=keep, gather_ids=sort_perm[pos])


def codesigned_probe_score(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    k: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """Plain int8 ANN scoring over the probed clusters."""
    pos, valid = probe_positions(probe_ids, cluster_offsets, width)
    return _probe_score(query, pos, item_codes, sort_perm, global_scale, k, valid)


def codesigned_probe_score_bloom(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    query_bit_positions: Tensor,
    bloom_transposed: Tensor,
    global_scale: float,
    k: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """Int8 ANN scoring gated by the bloom subset test: every set query bit ``m`` (``-1`` =
    none) must be set for the item, bit ``pos % 64`` of ``bloom_transposed[m, pos // 64]``."""
    pos, valid = probe_positions(probe_ids, cluster_offsets, width)
    m = query_bit_positions.clamp_min(0).unsqueeze(2)  # [B, n_bits, 1]
    words = bloom_transposed[m, (pos >> 6).unsqueeze(1)]  # [B, n_bits, width]
    bits = ((words >> (pos & 63).unsqueeze(1)) & 1).bool()
    keep = valid & (bits | (query_bit_positions < 0).unsqueeze(2)).all(dim=1)
    return _probe_score(query, pos, item_codes, sort_perm, global_scale, k, keep)
