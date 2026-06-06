from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


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


@triton_op("retrieve::bloom_match", mutates_args=())
def bloom_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """Compute (qb & sig) == qb across W int64 words; qb [B, W] int64, sigs [N, W] int64 →
    BoolTensor [B, N].

    Registered as a ``triton_op`` so the launch is captured into a single cudagraph under
    ``torch.compile``; ``BLOCK_N`` is constexpr 128 with a tile-tail mask for ``N < 128``."""
    b, w = qb.shape
    n = sigs.shape[0]

    out = torch.empty(b, n, dtype=torch.bool, device=qb.device)
    grid = (b, triton.cdiv(n, 128))

    wrap_triton(_bloom_match_kernel)[grid](
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
        BLOCK_N=128,
    )
    return out
