"""Fused bloom subset-test + stream compaction without the dense ``[B, N]`` bool of
``compact_mask(bloom_match(.))``. Same two-phase shape as ``clause_compact`` (plan L3): the
subset-test launch writes per-tile survivor counts and stashes the survivors' ids tile-locally,
``_host.compact_finish`` scans and scatters — so a row's ids are in **ascending item order**,
``torch.equal`` to ``ops.reference.bloom_compact`` and reproducible across launches and
processes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

# By name, not `common.<fn>` — see the note in clause_mask.py.
from retrieve.ops.triton._host import compact_finish, grid_batch_tiles
from retrieve.ops.triton.common import bloom_subset_pass, compact_stash


@dataclass(frozen=True)
class BloomCompactConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Tuned on A100/sm_80 for the one-pass atomic kernel (`tune-kernels bloom-compact`); not re-tuned
# for the two-phase shape (roadmap Phase G). Wide W=16 spills registers badly at block_n≥512 with
# num_warps≤4.
DEFAULT_CONFIG = BloomCompactConfig(block_n=256, num_warps=8)


@triton.jit
def _bloom_compact_kernel(
    qb_ptr,  # [B, W] int64
    sigs_ptr,  # [N, W] int64
    tile_counts_ptr,  # [B, T] int64
    scratch_ptr,  # [B, T * BLOCK_N] int32
    N,
    tiles_y,
    W: tl.constexpr,
    stride_qb_b,
    stride_qb_w,
    stride_s_n,
    stride_s_w,
    stride_tb,
    stride_sb,
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

    compact_stash(
        pass_mask,
        n_offsets,
        tile_counts_ptr,
        scratch_ptr,
        bid,
        tile_id,
        stride_tb,
        stride_sb,
        BLOCK_N,
    )


@dataclass(frozen=True)
class _BloomCompactLaunch:
    grid: tuple[int, int, int]
    tiles_y: int
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    tile_counts: Tensor
    scratch: Tensor


def _bloom_compact_prep(
    qb: Tensor,  # [B, W] int64
    sigs: Tensor,  # [N, W] int64
    *,
    cfg: BloomCompactConfig,
) -> _BloomCompactLaunch:
    """Validation + contiguity + the phase-1 buffers + the launch-arg dict. THE single place input
    checking happens — shared by ``_bloom_compact_impl`` and the public op (see
    ``clause_compact._clause_compact_prep`` for the buffer contract)."""
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

    grid, tiles_y = grid_batch_tiles(b, n, cfg.block_n)
    tiles = tiles_y * grid[2]
    tile_counts = torch.empty((b, tiles), dtype=torch.int64, device=qb.device)
    scratch = torch.empty((b, tiles * cfg.block_n), dtype=torch.int32, device=qb.device)

    kwargs = dict(
        qb_ptr=qb,
        sigs_ptr=sigs,
        tile_counts_ptr=tile_counts,
        scratch_ptr=scratch,
        N=n,
        tiles_y=tiles_y,
        W=w,
        stride_qb_b=qb.stride(0),
        stride_qb_w=qb.stride(1),
        stride_s_n=sigs.stride(0),
        stride_s_w=sigs.stride(1),
        stride_tb=tile_counts.stride(0),
        stride_sb=scratch.stride(0),
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return _BloomCompactLaunch(grid, tiles_y, kwargs, tile_counts, scratch)


def _bloom_compact_impl(
    qb: Tensor,
    sigs: Tensor,
    *,
    config: BloomCompactConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Direct-launch body used by the offline tuner and unit tests. Takes an optional ``config=`` so
    tile parameters can be swept; the public ``custom_op``-wrapped ``bloom_compact`` always uses
    ``DEFAULT_CONFIG``."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _bloom_compact_prep(qb, sigs, cfg=cfg)
    _bloom_compact_kernel[launch.grid](**launch.kwargs)
    return compact_finish(
        launch.grid,
        launch.tiles_y,
        launch.tile_counts,
        launch.scratch,
        sigs.shape[0],
        block_n=cfg.block_n,
        num_warps=cfg.num_warps,
    )


@torch.library.custom_op("retrieve::bloom_compact", mutates_args=(), device_types="cuda")
def bloom_compact(qb: Tensor, sigs: Tensor) -> tuple[Tensor, Tensor]:
    """Fused bloom subset-test + compaction; qb [B, W] int64, sigs [N, W] int64 → (positive_indices
    [B, N] int64, counts [B] int64). The ``[B, N]`` buffer has ``-1`` sentinels in the unused
    tail (consumers row-bound by ``counts[b]``); within a row the ids are in **ascending item
    order**, bit-equal to ``ops.reference.bloom_compact``. Mirrors ``_bloom_compact_impl`` with
    ``DEFAULT_CONFIG``. An opaque ``custom_op`` rather than a ``triton_op`` for the reason given
    on ``clause_compact``: the compaction address depends on the pass mask, so inductor's TTIR
    mutation analysis flags ``sigs`` (an index buffer) as mutated and cudagraph trees skip the
    forward."""
    return _bloom_compact_impl(qb, sigs)


@bloom_compact.register_fake
def _(qb, sigs):
    b, n = qb.shape[0], sigs.shape[0]
    return qb.new_empty((b, n), dtype=torch.int64), qb.new_empty((b,), dtype=torch.int64)
