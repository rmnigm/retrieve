"""Fused bloom subset-test + stream compaction without the dense ``[B, N]`` bool of
``compact_mask(bloom_match(.))``. Same two-phase shape as ``clause_compact``: the
subset-test launch writes per-tile survivor counts and stashes the survivors' ids tile-locally,
``_host.compact_finish`` scans and scatters — so a row's ids are in **ascending item order**,
``torch.equal`` to ``ops.reference.bloom_compact`` on ``[:counts]`` and reproducible across
launches and processes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

# By name, not `common.<fn>` — see the note in clause_mask.py.
from retrieve.ops.triton._host import (
    CompactLaunch,
    check_contiguous,
    compact_finish,
    grid_batch_tiles,
    wide,
)
from retrieve.ops.triton.common import bloom_subset_pass, compact_stash, tile_rows


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
    W_PAD: tl.constexpr,
    stride_qb_b,
    stride_qb_w,
    stride_s_n,
    stride_s_w,
    stride_tb,
    stride_sb,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(2) * tiles_y + tl.program_id(1)
    row0 = tile_id * BLOCK_N
    lane = tl.arange(0, BLOCK_N)
    n_offsets = row0 + lane
    n_valid = n_offsets < N

    # Words [W, W_PAD) load qb = 0, which every signature contains (kernels.md § Padding).
    w_off = tl.arange(0, W_PAD)
    w_in = w_off < W
    qb = tl.load(qb_ptr + bid * stride_qb_b + w_off * stride_qb_w, mask=w_in, other=0)

    sig_base, ids = tile_rows(sigs_ptr, row0, lane, stride_s_n, WIDE)
    sigs = tl.load(
        sig_base + ids[:, None] * stride_s_n + w_off[None, :] * stride_s_w,
        mask=n_valid[:, None] & w_in[None, :],
        other=0,
    )

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
        WIDE,
    )


def _bloom_compact_prep(
    qb: Tensor,  # [B, W] int64
    sigs: Tensor,  # [N, W] int64
    *,
    cfg: BloomCompactConfig,
) -> CompactLaunch:
    """Validation + contiguity + the phase-1 buffers + the launch-arg dict. The one place inputs
    are checked — shared by ``_bloom_compact_impl`` and the public op (see
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

    check_contiguous(sigs=sigs)
    qb = qb.contiguous()

    grid, tiles_y = grid_batch_tiles(b, n, cfg.block_n)
    tiles = tiles_y * grid[2]
    tile_counts = torch.empty((b, tiles), dtype=torch.int64, device=qb.device)
    scratch = torch.empty((b, tiles * cfg.block_n), dtype=torch.int32, device=qb.device)

    kwargs = {
        "qb_ptr": qb,
        "sigs_ptr": sigs,
        "tile_counts_ptr": tile_counts,
        "scratch_ptr": scratch,
        "N": n,
        "tiles_y": tiles_y,
        "W": w,
        "W_PAD": triton.next_power_of_2(w),
        "stride_qb_b": qb.stride(0),
        "stride_qb_w": qb.stride(1),
        "stride_s_n": sigs.stride(0),
        "stride_s_w": sigs.stride(1),
        "stride_tb": tile_counts.stride(0),
        "stride_sb": scratch.stride(0),
        "BLOCK_N": cfg.block_n,
        "WIDE": wide(sigs, scratch),
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
    return CompactLaunch(grid, tiles_y, kwargs, tile_counts, scratch)


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
    return compact_finish(launch, sigs.shape[0], block_n=cfg.block_n, num_warps=cfg.num_warps)


@torch.library.custom_op("retrieve::bloom_compact", mutates_args=(), device_types="cuda")
def bloom_compact(qb: Tensor, sigs: Tensor) -> tuple[Tensor, Tensor]:
    """Fused bloom subset-test + compaction; qb [B, W] int64, sigs [N, W] int64 → (positive_indices
    [B, N] int64, counts [B] int64). Nothing is written past ``counts[b]`` (consumers row-bound
    by it); within a row the ids are in **ascending item order**, bit-equal to
    ``ops.reference.bloom_compact``. Mirrors ``_bloom_compact_impl`` with
    ``DEFAULT_CONFIG``. An opaque ``custom_op`` rather than a ``triton_op`` for the reason given
    on ``clause_compact``: the compaction address depends on the pass mask, so inductor's TTIR
    mutation analysis flags ``sigs`` (an index buffer) as mutated and cudagraph trees skip the
    forward."""
    return _bloom_compact_impl(qb, sigs)


@bloom_compact.register_fake
def _(qb, sigs):
    b, n = qb.shape[0], sigs.shape[0]
    return qb.new_empty((b, n), dtype=torch.int64), qb.new_empty((b,), dtype=torch.int64)
