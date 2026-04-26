"""Fused clause evaluation + stream compaction.

Avoids materializing the dense ``[B, N]`` bool that ``ClauseIndex.evaluate_mask``
would otherwise produce. One kernel launch:

  per program (b, tile):
      evaluate clauses for BLOCK_N items against query_clause_attrs[b]
      tl.cumsum within tile to get intra-tile write offsets
      tl.atomic_add into counts[b] to get the row's base offset
      tl.store passing item ids at positive_indices[b, base + intra]

Output ordering within a row is unspecified (atomics across tiles). V2's
``fused_masked_knn_topk`` only consumes the *set*, not the order — callers that
care must sort.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# NOTE: no @triton.autotune — atomic_add accumulates across autotune trials
# and corrupts ``counts``. A single fixed config is correct and fast enough
# for now; if a larger sweep matters, re-enable autotune with
# ``reset_to_zero`` covering ``counts_ptr`` AND ``out_indices_ptr`` and verify
# behaviour across both the autotune phase and steady-state calls.
_BLOCK_N = 256
_NUM_WARPS = 4


@triton.jit
def _clause_compact_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] bool (stored as int8 in torch)
    query_attrs_ptr,  # [B, C] int64
    out_indices_ptr,  # [B, N] int64 (worst-case scratch)
    counts_ptr,  # [B] int64 (init 0)
    N,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,  # item_attrs.stride(0)
    stride_ic,  # item_attrs.stride(1)
    stride_ia,  # item_attrs.stride(2)
    stride_qb,
    stride_qc,
    stride_ob,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(1)

    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    n_valid = n_offsets < N

    # pass_mask starts as all True; AND in each clause's verdict.
    pass_mask = tl.full([BLOCK_N], 1, tl.int1)

    for c in tl.static_range(C):
        q_c = tl.load(query_attrs_ptr + bid * stride_qb + c * stride_qc)  # scalar
        rev_c = tl.load(is_reverse_ptr + c).to(tl.int1)  # scalar

        clause_match = tl.full([BLOCK_N], 0, tl.int1)
        for a in tl.static_range(A_MAX):
            ia = tl.load(
                item_attrs_ptr + n_offsets * stride_in + c * stride_ic + a * stride_ia,
                mask=n_valid,
                other=-1,
            )
            clause_match = clause_match | (ia == q_c)

        # reverse: flip clause_match if rev_c is True.
        # XOR with broadcast scalar bool flips per element.
        clause_match = clause_match ^ rev_c

        # inactive (q_c == -1) → clause always passes. Overrides reverse.
        inactive = q_c == -1
        clause_match = clause_match | inactive

        pass_mask = pass_mask & clause_match

    pass_mask = pass_mask & n_valid

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


def clause_compact(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> tuple[Tensor, Tensor]:
    """Fused clause evaluation + compaction.

    Returns ``(positive_indices [B, P] int64, counts [B] int64)`` where
    ``P = max(counts.max(), 1)``. Indices within a row are unordered.
    """
    if item_clause_attrs.dim() != 3:
        raise ValueError("item_clause_attrs must be [N, C, A_max]")
    if query_clause_attrs.dim() != 2:
        raise ValueError("query_clause_attrs must be [B, C]")

    n, c, a_max = item_clause_attrs.shape
    b, c_q = query_clause_attrs.shape
    if c != c_q:
        raise ValueError(f"clause-count mismatch: items C={c}, query C={c_q}")

    device = query_clause_attrs.device
    item_clause_attrs = item_clause_attrs.contiguous()
    clause_is_reverse = clause_is_reverse.contiguous().to(torch.int8)
    query_clause_attrs = query_clause_attrs.contiguous()

    out_indices = torch.empty((b, n), dtype=torch.int64, device=device)
    counts = torch.zeros((b,), dtype=torch.int64, device=device)

    grid = (b, triton.cdiv(n, _BLOCK_N))

    _clause_compact_kernel[grid](
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        out_indices,
        counts,
        N=n,
        C=c,
        A_MAX=a_max,
        stride_in=item_clause_attrs.stride(0),
        stride_ic=item_clause_attrs.stride(1),
        stride_ia=item_clause_attrs.stride(2),
        stride_qb=query_clause_attrs.stride(0),
        stride_qc=query_clause_attrs.stride(1),
        stride_ob=out_indices.stride(0),
        stride_on=out_indices.stride(1),
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )

    p = max(int(counts.max().item()), 1)
    return out_indices[:, :p].contiguous(), counts
