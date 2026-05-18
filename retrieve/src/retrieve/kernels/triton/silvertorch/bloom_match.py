from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _bloom_match_kernel(
    qb_ptr,
    sigs_ptr,
    out_ptr,
    N: tl.constexpr,
    W: tl.constexpr,
    stride_qb_b,
    stride_qb_w,
    stride_s_n,
    stride_s_w,
    stride_o_b,
    stride_o_n,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    n_block = tl.program_id(1)

    n_off = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = n_off < N

    w_off = tl.arange(0, W)
    qb = tl.load(qb_ptr + bid * stride_qb_b + w_off * stride_qb_w)

    sigs = tl.load(
        sigs_ptr + n_off[:, None] * stride_s_n + w_off[None, :] * stride_s_w,
        mask=valid[:, None],
        other=0,
    )

    masked = qb[None, :] & sigs
    eq_per_word = (masked == qb[None, :]).to(tl.int32)
    all_eq = tl.min(eq_per_word, axis=1)
    pass_all = all_eq != 0

    tl.store(
        out_ptr + bid * stride_o_b + n_off * stride_o_n,
        pass_all,
        mask=valid,
    )


@torch._dynamo.disable
def bloom_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """Compute (qb & sig) == qb across W int64 words.

    Inputs:
        qb:   [B, W] int64 — packed query bloom signatures.
        sigs: [N, W] int64 — packed item bloom signatures.

    Returns BoolTensor [B, N].
    """
    b, w = qb.shape
    n = sigs.shape[0]
    if qb.dtype != torch.int64 or sigs.dtype != torch.int64:
        raise TypeError("qb and sigs must be int64")
    if sigs.shape[1] != w:
        raise ValueError(f"qb has W={w} but sigs has W={sigs.shape[1]}")

    out = torch.empty(b, n, dtype=torch.bool, device=qb.device)
    block_n = 128 if n >= 128 else triton.next_power_of_2(int(n))
    grid = (b, triton.cdiv(n, block_n))

    _bloom_match_kernel[grid](
        qb,
        sigs,
        out,
        N=n,
        W=w,
        stride_qb_b=qb.stride(0),
        stride_qb_w=qb.stride(1),
        stride_s_n=sigs.stride(0),
        stride_s_w=sigs.stride(1),
        stride_o_b=out.stride(0),
        stride_o_n=out.stride(1),
        BLOCK_N=block_n,
    )
    return out
