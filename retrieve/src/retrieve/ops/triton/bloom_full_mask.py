from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.ops.triton._host import check_contiguous

BLOCK_W = 1024


@triton.jit
def _bloom_full_mask_kernel(
    qpos_ptr,
    bloom_t_ptr,
    out_ptr,
    n_words,
    n_qbits,
    stride_qpos,
    stride_tm,
    stride_ob,
    BLOCK_W: tl.constexpr,
):
    bid = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
    in_w = offs < n_words
    acc = tl.full([BLOCK_W], -1, tl.int64)
    for i in range(n_qbits):
        m = tl.load(qpos_ptr + bid * stride_qpos + i)  # -1: an inactive clause's slot
        acc = acc & tl.load(bloom_t_ptr + m * stride_tm + offs, mask=in_w & (m >= 0), other=-1)
    tl.store(out_ptr + bid.to(tl.int64) * stride_ob + offs, acc, mask=in_w)


@triton_op("retrieve::bloom_full_mask", mutates_args=())
def bloom_full_mask(query_bit_positions: Tensor, bloom_transposed: Tensor) -> Tensor:
    """The bloom subset test over every item, packed: ``[B, C·k]`` query bit positions (``-1`` =
    inactive) and the transposed index ``[m_bits, ceil(N / 64)]`` → ``[B, ceil(N / 64)]`` int64,
    bit ``n % 64`` of word ``n // 64`` set iff item ``n`` (cluster-sorted) passes. The AND of the
    query's index rows; SilverTorch's ``bloom_path="full"`` (kernels.md § bloom_full_mask)."""
    check_contiguous(bloom_transposed=bloom_transposed)
    qpos = query_bit_positions.contiguous()
    b, n_words = qpos.shape[0], bloom_transposed.shape[1]
    out = torch.empty(b, n_words, dtype=torch.int64, device=qpos.device)
    grid = (triton.cdiv(n_words, BLOCK_W), b)
    wrap_triton(_bloom_full_mask_kernel)[grid](
        qpos_ptr=qpos,
        bloom_t_ptr=bloom_transposed,
        out_ptr=out,
        n_words=n_words,
        n_qbits=qpos.shape[1],
        stride_qpos=qpos.stride(0),
        stride_tm=bloom_transposed.stride(0),
        stride_ob=out.stride(0),
        BLOCK_W=BLOCK_W,
        num_warps=4,
    )  # keep inline (export)
    return out
