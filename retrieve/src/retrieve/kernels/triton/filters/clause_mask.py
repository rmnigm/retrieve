"""Fused clause evaluation emitting ``[B, N]`` bool directly.

Avoids the ``[B, N, C, A_max]`` intermediate that
``ExactAttributeFilter.evaluate_mask``'s pure-torch broadcast materializes.
Same inner loop as ``clause_compact`` minus the cumsum + atomic_add epilogue.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# No atomics — autotune would be safe but compile-time cost is not yet
# justified. Start with a fixed config; benchmark before tuning.
_BLOCK_N = 256
_NUM_WARPS = 4


@triton.jit
def _clause_mask_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] bool (stored as int8 in torch)
    query_attrs_ptr,  # [B, C] int64
    out_ptr,  # [B, N] bool
    N,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,
    stride_ic,
    stride_ia,
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

        clause_match = clause_match ^ rev_c

        inactive = q_c == -1
        clause_match = clause_match | inactive

        pass_mask = pass_mask & clause_match

    pass_mask = pass_mask & n_valid

    tl.store(
        out_ptr + bid * stride_ob + n_offsets * stride_on,
        pass_mask,
        mask=n_valid,
    )


@torch._dynamo.disable
def clause_mask(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """Fused clause evaluation → ``[B, N]`` bool. No intermediate.

    Int args (shapes, strides) are cast through ``int(...)`` at the kernel
    launch site: under ``torch.compile(dynamic=True)`` they arrive as
    ``torch.SymInt`` and ``triton.jit`` can't construct ``ConstantVariable``
    from those. The cast forces specialization at trace time; values are
    static per index instance so it costs nothing.
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

    out = torch.empty((b, n), dtype=torch.bool, device=device)

    grid = (b, triton.cdiv(n, _BLOCK_N))

    _clause_mask_kernel[grid](
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        out,
        N=n,
        C=c,
        A_MAX=a_max,
        stride_in=item_clause_attrs.stride(0),
        stride_ic=item_clause_attrs.stride(1),
        stride_ia=item_clause_attrs.stride(2),
        stride_qb=query_clause_attrs.stride(0),
        stride_qc=query_clause_attrs.stride(1),
        stride_ob=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )

    return out
