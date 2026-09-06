"""Fused bloom subset-test + stream compaction in one launch (cumsum for intra-tile offsets,
atomic_add for the per-row base), avoiding the dense ``[B, N]`` bool of
``compact_mask(bloom_match(.))``. Output ordering within a row is unspecified (atomics across
tiles); callers that need order must sort."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

# By name, not `common.<fn>` — see the note in clause_mask.py.
from retrieve.kernels.common import bloom_subset_pass, compact_store


@dataclass(frozen=True)
class BloomCompactConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Default tile config (tuned on A100/sm_80). Same atomic_add gotcha as clause_compact (host
# allocates fresh buffers per call). Wide W=16 spills registers badly at block_n≥512 with
# num_warps≤4.
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
    # 3D grid: batch on grid_x (L2 reuse on sigs), tiles split across grid_y × grid_z to dodge the
    # 65535 single-axis cap; tile_id = tile_x * tiles_y + tile_y keeps tiles contiguous.
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

    # Helper leaves masking to the caller (lanes loaded with other=0 pass iff qb == 0), so AND
    # with n_valid here.
    pass_mask = bloom_subset_pass(qb, sigs) & n_valid

    compact_store(pass_mask, n_offsets, counts_ptr, out_indices_ptr, bid, stride_ob, stride_on)


@dataclass(frozen=True)
class _BloomCompactLaunch:
    grid: tuple[int, int, int]
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    out_indices: Tensor
    counts: Tensor


def _bloom_compact_prep(
    qb: Tensor,  # [B, W] int64
    sigs: Tensor,  # [N, W] int64
    *,
    cfg: BloomCompactConfig,
) -> _BloomCompactLaunch:
    """Validation + contiguity + buffers + the full launch-arg dict. THE single place input
    checking happens — shared by ``_bloom_compact_impl`` and the public op.

    ``out_indices`` is init'd to the ``-1`` sentinel (see ``clause_compact._clause_compact_prep``):
    positions past ``counts[bid]`` aren't written and ``-1`` propagates as "no item" through the
    gather when ``counts[b] < k``. ``counts`` must be zero-initialized — the kernel's atomic_add
    accumulates into it."""
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

    out_indices = torch.full((b, n), -1, dtype=torch.int64, device=qb.device)
    counts = torch.zeros((b,), dtype=torch.int64, device=qb.device)

    # 3D grid (batch, tiles_y, tiles_x) — see kernel comment.
    tiles = triton.cdiv(n, cfg.block_n)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    grid = (b, tiles_y, tiles_x)

    kwargs = dict(
        qb_ptr=qb,
        sigs_ptr=sigs,
        out_indices_ptr=out_indices,
        counts_ptr=counts,
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
    return _BloomCompactLaunch(grid, kwargs, out_indices, counts)


def _bloom_compact_impl(
    qb: Tensor,
    sigs: Tensor,
    *,
    config: BloomCompactConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Direct-launch body used by the offline tuner and unit tests. Takes an optional ``config=`` so
    tile parameters can be swept; the public ``@triton_op``-wrapped ``bloom_compact`` always uses
    ``DEFAULT_CONFIG``."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _bloom_compact_prep(qb, sigs, cfg=cfg)
    _bloom_compact_kernel[launch.grid](**launch.kwargs)
    return launch.out_indices, launch.counts


@torch.library.custom_op("retrieve::bloom_compact", mutates_args=(), device_types="cuda")
def bloom_compact(qb: Tensor, sigs: Tensor) -> tuple[Tensor, Tensor]:
    """Fused bloom subset-test + compaction; qb [B, W] int64, sigs [N, W] int64 → (positive_indices
    [B, N] int64, counts [B] int64). The ``[B, N]`` buffer has ``-1`` sentinels in the unused
    tail (consumers row-bound by ``counts[b]``); within-row order is unspecified (atomic writes).
    Mirrors ``_bloom_compact_impl`` with ``DEFAULT_CONFIG``. An opaque ``custom_op`` rather than
    a ``triton_op`` for the reason given on ``clause_compact``: the compaction address depends
    on the pass mask, so inductor's TTIR mutation analysis flags ``sigs`` (an index buffer) as
    mutated and cudagraph trees skip the forward."""
    return _bloom_compact_impl(qb, sigs)


@bloom_compact.register_fake
def _(qb, sigs):
    b, n = qb.shape[0], sigs.shape[0]
    return qb.new_empty((b, n), dtype=torch.int64), qb.new_empty((b,), dtype=torch.int64)
