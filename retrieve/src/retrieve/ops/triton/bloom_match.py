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
    scores_ptr,  # [B, N] fp32, read only when HAS_SCORES
    out_ptr,  # [B, N] bool, or fp32 when HAS_SCORES
    N: tl.constexpr,
    tiles_y,
    W: tl.constexpr,
    W_PAD: tl.constexpr,
    stride_qb_b,
    stride_qb_w,
    stride_s_n,
    stride_s_w,
    stride_sc_b,
    stride_sc_n,
    stride_o_b,
    stride_o_n,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
    HAS_SCORES: tl.constexpr,
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
    out = row_base(out_ptr, bid, stride_o_b, WIDE) + n_off * stride_o_n
    if HAS_SCORES:
        sc = tl.load(row_base(scores_ptr, bid, stride_sc_b, WIDE) + n_off * stride_sc_n, mask=valid)
        tl.store(out, tl.where(pass_all, sc, float("-inf")), mask=valid)
    else:
        tl.store(out, pass_all, mask=valid)


def _prep(
    qb: Tensor, sigs: Tensor, scores: Tensor | None, *, block_n: int = 128, num_warps: int = 4
) -> tuple[tuple, dict, Tensor]:
    """Validation, output buffer and the launch args shared by both ops."""
    b, w = qb.shape
    n = sigs.shape[0]
    check_contiguous(sigs=sigs)
    qb = qb.contiguous()
    if scores is not None:
        if scores.shape != (b, n) or scores.dtype != torch.float32:
            raise ValueError(
                f"scores must be [B, N] = {[b, n]} float32, got {scores.shape} {scores.dtype}"
            )
        scores = scores.contiguous()
    out = torch.empty(b, n, dtype=torch.bool if scores is None else torch.float32, device=qb.device)
    src = out if scores is None else scores  # an unread pointer when not HAS_SCORES
    grid, tiles_y = grid_batch_tiles(b, n, block_n)
    kwargs = {
        "qb_ptr": qb,
        "sigs_ptr": sigs,
        "scores_ptr": src,
        "out_ptr": out,
        "N": n,
        "tiles_y": tiles_y,
        "W": w,
        "W_PAD": triton.next_power_of_2(w),
        "stride_qb_b": qb.stride(0),
        "stride_qb_w": qb.stride(1),
        "stride_s_n": sigs.stride(0),
        "stride_s_w": sigs.stride(1),
        "stride_sc_b": src.stride(0),
        "stride_sc_n": src.stride(1),
        "stride_o_b": out.stride(0),
        "stride_o_n": out.stride(1),
        "BLOCK_N": block_n,
        "WIDE": wide(sigs, out),
        "HAS_SCORES": scores is not None,
        "num_warps": num_warps,
    }
    return grid, kwargs, out


@triton_op("retrieve::bloom_match", mutates_args=())
def bloom_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """Compute (qb & sig) == qb across W int64 words; qb [B, W] int64, sigs [N, W] int64 →
    BoolTensor [B, N].

    Registered as a ``triton_op`` so the launch is captured into a single cudagraph under
    ``torch.compile``; ``BLOCK_N`` is constexpr 128 with a tile-tail mask for ``N < 128``."""
    grid, kwargs, out = _prep(qb, sigs, None)
    wrap_triton(_bloom_match_kernel)[grid](**kwargs)  # keep inline (export)
    return out


@triton_op("retrieve::bloom_match_scores", mutates_args=())
def bloom_match_scores(scores: Tensor, qb: Tensor, sigs: Tensor) -> Tensor:
    """``where(bloom_match(qb, sigs), scores, -inf)`` in one pass, scores [B, N] fp32: the
    predicate never leaves the kernel as a ``[B, N]`` bool. Functional (a new buffer), so
    inductor has no in-place mask to copy."""
    grid, kwargs, out = _prep(qb, sigs, scores)
    wrap_triton(_bloom_match_kernel)[grid](**kwargs)  # keep inline (export)
    return out


def _bloom_match_scores_impl(
    scores: Tensor, qb: Tensor, sigs: Tensor, *, block_n: int, num_warps: int
) -> Tensor:
    """Direct launch of the ``bloom_match_scores`` form at a given tile, for sweeps."""
    grid, kwargs, out = _prep(qb, sigs, scores, block_n=block_n, num_warps=num_warps)
    _bloom_match_kernel[grid](**kwargs)
    return out
