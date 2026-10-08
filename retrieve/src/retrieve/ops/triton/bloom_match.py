from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.ops.triton._host import check_contiguous, grid_batch_tiles, wide
from retrieve.ops.triton.common import bloom_subset_pass, row_base, tile_rows


@triton.jit
def _bloom_match_kernel(
    qb_ptr,
    sigs_ptr,
    out_ptr,
    N: tl.constexpr,
    tiles_y,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    stride_qb_b,
    stride_qb_w,
    stride_s_n,
    stride_s_w,
    stride_o_b,
    stride_o_n,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(2) * tiles_y + tl.program_id(1)
    row0 = tile_id * BLOCK_N
    lane = tl.arange(0, BLOCK_N)
    n_off = row0 + lane
    valid = n_off < N

    # Words [W, W_PAD) load qb = 0, which every signature contains (kernels.md § Padding).
    w_off = tl.arange(0, W_PAD)
    w_in = w_off < W
    qb = tl.load(qb_ptr + bid * stride_qb_b + w_off * stride_qb_w, mask=w_in, other=0)

    sig_base, ids = tile_rows(sigs_ptr, row0, lane, stride_s_n, WIDE)
    sigs = tl.load(
        sig_base + ids[:, None] * stride_s_n + w_off[None, :] * stride_s_w,
        mask=valid[:, None] & w_in[None, :],
        other=0,
    )

    # OOB lanes never leak: the store below is masked with `valid`.
    pass_all = bloom_subset_pass(qb, sigs)

    tl.store(row_base(out_ptr, bid, stride_o_b, WIDE) + n_off * stride_o_n, pass_all, mask=valid)


@triton_op("retrieve::bloom_match", mutates_args=())
def bloom_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """Compute (qb & sig) == qb across W int64 words; qb [B, W] int64, sigs [N, W] int64 →
    BoolTensor [B, N].

    Registered as a ``triton_op`` so the launch is captured into a single cudagraph under
    ``torch.compile``; ``BLOCK_N`` is constexpr 128 with a tile-tail mask for ``N < 128``."""
    b, w = qb.shape
    n = sigs.shape[0]
    check_contiguous(sigs=sigs)
    qb = qb.contiguous()

    out = torch.empty(b, n, dtype=torch.bool, device=qb.device)
    grid, tiles_y = grid_batch_tiles(b, n, 128)

    wrap_triton(_bloom_match_kernel)[grid](
        qb,
        sigs,
        out,
        N=n,
        tiles_y=tiles_y,
        W=w,
        W_PAD=triton.next_power_of_2(w),
        stride_qb_b=qb.stride(0),
        stride_qb_w=qb.stride(1),
        stride_s_n=sigs.stride(0),
        stride_s_w=sigs.stride(1),
        stride_o_b=out.stride(0),
        stride_o_n=out.stride(1),
        BLOCK_N=128,
        WIDE=wide(sigs, out),
    )
    return out
