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

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@dataclass(frozen=True)
class BloomCompactConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Single default the library ships with. Re-tune on a new arch by running
# ``uv run tune-kernels bloom-compact`` and pasting the resulting line in.
# Same atomic_add gotcha as ``clause_compact`` is avoided the same way: the
# host wrapper allocates fresh ``out_indices`` and ``counts`` per call, so
# tuner reps don't accumulate.
# Tuned on A100 (sm_80) against real-eval shapes (Goodreads N=797K /
# arXiv N=3M / arXiv-synth N=15M, W=16 from m_bits=1024 default,
# B∈{1, 16}): block_n=256, num_warps=8 wins all batched regimes; the
# B=1 regimes prefer block_n=128 by ~5%. Wide W=16 still causes severe
# register spill at block_n≥512 with num_warps≤4 (3–15× slowdown).
DEFAULT_CONFIG = BloomCompactConfig(block_n=256, num_warps=8)


@triton.jit
def _bloom_compact_kernel(
    qb_ptr,  # [B, W] int64
    sigs_ptr,  # [N, W] int64
    out_indices_ptr,  # [B, N] int64 (worst-case scratch)
    counts_ptr,  # [B] int64 (init 0)
    N,
    tiles_y,
    W: tl.constexpr,
    stride_qb_b,
    stride_qb_w,
    stride_s_n,
    stride_s_w,
    stride_ob,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    # 3D grid: batch on grid_x (small, restores L2 reuse on sigs because
    # adjacent dispatched programs share the same tile), tiles split
    # across grid_y × grid_z to dodge the 65535 cap on a single axis.
    # `tile_id = tile_x * tiles_y + tile_y` keeps tiles contiguous.
    bid = tl.program_id(0)
    tile_y = tl.program_id(1)
    tile_x = tl.program_id(2)
    tile_id = tile_x * tiles_y + tile_y

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


def _bloom_compact_impl(
    qb: Tensor,
    sigs: Tensor,
    *,
    config: BloomCompactConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Direct-launch body used by the offline tuner and unit tests. Takes an
    optional ``config=`` so tile parameters can be swept; the public
    ``@triton_op``-wrapped ``bloom_compact`` always uses ``DEFAULT_CONFIG``."""
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

    cfg = config if config is not None else DEFAULT_CONFIG
    # 3D grid (batch, tiles_y, tiles_x) — see kernel comment.
    tiles = triton.cdiv(n, cfg.block_n)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    grid = (b, tiles_y, tiles_x)

    _bloom_compact_kernel[grid](
        qb,
        sigs,
        out_indices,
        counts,
        N=n,
        tiles_y=tiles_y,
        W=w,
        stride_qb_b=qb.stride(0),
        stride_qb_w=qb.stride(1),
        stride_s_n=sigs.stride(0),
        stride_s_w=sigs.stride(1),
        stride_ob=out_indices.stride(0),
        stride_on=out_indices.stride(1),
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    return out_indices, counts


@triton_op("retrieve::bloom_compact", mutates_args=())
def bloom_compact(qb: Tensor, sigs: Tensor) -> tuple[Tensor, Tensor]:
    """Fused bloom subset-test + compaction.

    Inputs:
        qb:   [B, W] int64 — packed query bloom signatures.
        sigs: [N, W] int64 — packed item bloom signatures.

    Returns ``(positive_indices [B, N] int64, counts [B] int64)``. The full
    item-width ``[B, N]`` indices buffer is returned with ``-1`` sentinels
    in the unused tail; downstream consumers (``fused_masked_knn_topk``,
    ``oporp_1bit_match_topk``) row-bound by ``counts[b]`` so the wider
    buffer never costs a re-read. Indices within a row are unordered
    (atomic-add writes).

    Registered as ``triton_op`` so the kernel launch is captured as a HOP
    that ``torch.compile(dynamic=True, mode="reduce-overhead")`` can stitch
    into a single cudagraph_trees graph; the ``torch.full``/``torch.zeros``
    allocations inside the body act as the fake/meta kernel. Mirrors
    ``_bloom_compact_impl`` but routes the launch through ``wrap_triton``
    for HOP capture and hard-codes ``DEFAULT_CONFIG``.
    """
    cfg = DEFAULT_CONFIG

    b, w = qb.shape
    n = sigs.shape[0]

    qb = qb.contiguous()
    sigs = sigs.contiguous()

    # ``-1`` sentinel + zero counts: see initialisation note in
    # ``_bloom_compact_impl``.
    out_indices = torch.full((b, n), -1, dtype=torch.int64, device=qb.device)
    counts = torch.zeros((b,), dtype=torch.int64, device=qb.device)

    tiles = triton.cdiv(n, cfg.block_n)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    grid = (b, tiles_y, tiles_x)

    wrap_triton(_bloom_compact_kernel)[grid](
        qb,
        sigs,
        out_indices,
        counts,
        N=n,
        tiles_y=tiles_y,
        W=w,
        stride_qb_b=qb.stride(0),
        stride_qb_w=qb.stride(1),
        stride_s_n=sigs.stride(0),
        stride_s_w=sigs.stride(1),
        stride_ob=out_indices.stride(0),
        stride_on=out_indices.stride(1),
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    return out_indices, counts
