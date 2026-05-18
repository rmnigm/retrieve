"""Fused bloom subset-test + stream compaction.

Avoids materializing the dense ``[B, N]`` bool that
``compact_mask(bloom_match(.))`` would otherwise produce. One kernel launch:

  per program (b, tile):
      subset-test BLOCK_N items against query bloom signature qb[b]
      tl.cumsum within tile to get intra-tile write offsets
      tl.atomic_add into counts[b] to get the row's base offset
      tl.store passing item ids at positive_indices[b, base + intra]

Output ordering within a row is unspecified (atomics across tiles) — same
convention as ``clause_compact``. V2's ``fused_masked_knn_topk`` only consumes
the *set*, not the order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# NOTE: no @triton.autotune — atomic_add accumulates across autotune trials and
# corrupts ``counts``. Same gotcha that bit ``clause_compact``.
_BLOCK_N = 256
_NUM_WARPS = 4


@triton.jit
def _bloom_compact_kernel(
    qb_ptr,  # [B, W] int64
    sigs_ptr,  # [N, W] int64
    out_indices_ptr,  # [B, N] int64 (worst-case scratch)
    counts_ptr,  # [B] int64 (init 0)
    N,
    W: tl.constexpr,
    stride_qb_b,
    stride_qb_w,
    stride_s_n,
    stride_s_w,
    stride_ob,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(1)

    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    n_valid = n_offsets < N

    w_off = tl.arange(0, W)
    qb = tl.load(qb_ptr + bid * stride_qb_b + w_off * stride_qb_w)  # [W]

    sigs = tl.load(
        sigs_ptr + n_offsets[:, None] * stride_s_n + w_off[None, :] * stride_s_w,
        mask=n_valid[:, None],
        other=0,
    )  # [BLOCK_N, W]

    masked = qb[None, :] & sigs
    eq_per_word = (masked == qb[None, :]).to(tl.int32)
    all_eq = tl.min(eq_per_word, axis=1)  # [BLOCK_N] int32
    pass_mask = (all_eq != 0) & n_valid

    # Stream compaction: cumsum gives intra-tile offsets; atomic_add gives base.
    pass_int = tl.where(pass_mask, 1, 0).to(tl.int32)
    intra = tl.cumsum(pass_int, axis=0) - 1  # 0-indexed inclusive position
    tile_sum = tl.sum(pass_int)

    base = tl.atomic_add(counts_ptr + bid, tile_sum.to(tl.int64))
    write_pos = base + intra.to(tl.int64)

    tl.store(
        out_indices_ptr + bid * stride_ob + write_pos * stride_on,
        n_offsets.to(tl.int64),
        mask=pass_mask,
    )


@torch._dynamo.disable
def bloom_compact(qb: Tensor, sigs: Tensor) -> tuple[Tensor, Tensor]:
    """Fused bloom subset-test + compaction.

    Inputs:
        qb:   [B, W] int64 — packed query bloom signatures.
        sigs: [N, W] int64 — packed item bloom signatures.

    Returns ``(positive_indices [B, P] int64, counts [B] int64)`` where
    ``P = max(counts.max(), 1)``. Indices within a row are unordered.
    """
    if qb.dim() != 2:
        raise ValueError("qb must be [B, W]")
    if sigs.dim() != 2:
        raise ValueError("sigs must be [N, W]")
    if qb.dtype != torch.int64 or sigs.dtype != torch.int64:
        raise TypeError("qb and sigs must be int64")
    if qb.shape[1] != sigs.shape[1]:
        raise ValueError(f"qb has W={qb.shape[1]} but sigs has W={sigs.shape[1]}")

    b, w = qb.shape
    n = sigs.shape[0]

    qb = qb.contiguous()
    sigs = sigs.contiguous()

    # Initialise to -1 sentinel; see clause_compact.py for the rationale.
    # Positions past counts[bid] are not written by the kernel, and -1
    # propagates as "no item" through fused_masked_knn_topk's gather when
    # counts[b] < k.
    out_indices = torch.full((b, n), -1, dtype=torch.int64, device=qb.device)
    counts = torch.zeros((b,), dtype=torch.int64, device=qb.device)

    grid = (b, triton.cdiv(n, _BLOCK_N))

    _bloom_compact_kernel[grid](
        qb,
        sigs,
        out_indices,
        counts,
        N=n,
        W=w,
        stride_qb_b=qb.stride(0),
        stride_qb_w=qb.stride(1),
        stride_s_n=sigs.stride(0),
        stride_s_w=sigs.stride(1),
        stride_ob=out_indices.stride(0),
        stride_on=out_indices.stride(1),
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )

    p = max(int(counts.max().item()), 1)
    return out_indices[:, :p].contiguous(), counts
