"""Fused clause evaluation + stream compaction without the dense ``[B, N]`` bool of the pure-torch
path, in two phases (plan L3): the predicate launch writes each tile's survivor count and stashes
its surviving ids, compacted, in the tile's own slot range of an int32 scratch;
``_host.compact_finish`` scans the counts and a second, predicate-free launch moves every run to
``tile_offset``. A row's ids therefore come out in **ascending item order**, ``torch.equal`` to
``ops.reference.clause_compact`` and to itself launch after launch — the ``atomic_add`` row base
this replaced ordered them by tile completion, which the downstream tie-breakers turned into
run-to-run quality noise."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

# By name, not `common.<fn>` — see the note in clause_mask.py: inductor's
# re-compilation of a @triton_op kernel captures @triton.jit callees from
# the kernel's globals by name, and a module object is not one.
from retrieve.ops.triton._host import compact_finish, grid_batch_tiles, wide
from retrieve.ops.triton.common import clause_pass, compact_stash, tile_rows


@dataclass(frozen=True)
class ClauseCompactConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Tuned on A100/sm_80 for the one-pass atomic kernel (`tune-kernels clause-compact`); not
# re-tuned for the two-phase shape (roadmap Phase G).
DEFAULT_CONFIG = ClauseCompactConfig(block_n=512, num_warps=2)


@triton.jit
def _clause_compact_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] bool (stored as int8 in torch)
    query_attrs_ptr,  # [B, C] int64
    tile_counts_ptr,  # [B, T] int64
    scratch_ptr,  # [B, T * BLOCK_N] int32
    N,
    tiles_y,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,  # item_attrs.stride(0)
    stride_ic,  # item_attrs.stride(1)
    stride_ia,  # item_attrs.stride(2)
    stride_qb,
    stride_qc,
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

    # Result is already ANDed with n_valid inside the helper (keep seeds from load_mask).
    attrs_base, ids = tile_rows(item_attrs_ptr, row0, lane, stride_in, WIDE)
    pass_mask = clause_pass(
        attrs_base,
        is_reverse_ptr,
        query_attrs_ptr,
        ids,
        n_valid,
        bid,
        stride_in,
        stride_ic,
        stride_ia,
        stride_qb,
        stride_qc,
        C=C,
        A_MAX=A_MAX,
    )

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


@dataclass(frozen=True)
class _ClauseCompactLaunch:
    grid: tuple[int, int, int]
    tiles_y: int
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    tile_counts: Tensor
    scratch: Tensor


def _clause_compact_prep(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
    *,
    cfg: ClauseCompactConfig,
) -> _ClauseCompactLaunch:
    """Validation + contiguity + the phase-1 buffers + the launch-arg dict. THE single place input
    checking happens — shared by ``_clause_compact_impl`` and the public op. ``tile_counts`` and
    ``scratch`` cover the grid's ``T = tiles_y * tiles_x`` tiles, every one of which the launch
    writes (tiles past ``cdiv(N, BLOCK_N)`` write zeros)."""
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

    grid, tiles_y = grid_batch_tiles(b, n, cfg.block_n)
    tiles = tiles_y * grid[2]
    tile_counts = torch.empty((b, tiles), dtype=torch.int64, device=device)
    scratch = torch.empty((b, tiles * cfg.block_n), dtype=torch.int32, device=device)

    kwargs = dict(
        item_attrs_ptr=item_clause_attrs,
        is_reverse_ptr=clause_is_reverse,
        query_attrs_ptr=query_clause_attrs,
        tile_counts_ptr=tile_counts,
        scratch_ptr=scratch,
        N=n,
        tiles_y=tiles_y,
        C=c,
        A_MAX=a_max,
        stride_in=item_clause_attrs.stride(0),
        stride_ic=item_clause_attrs.stride(1),
        stride_ia=item_clause_attrs.stride(2),
        stride_qb=query_clause_attrs.stride(0),
        stride_qc=query_clause_attrs.stride(1),
        stride_tb=tile_counts.stride(0),
        stride_sb=scratch.stride(0),
        BLOCK_N=cfg.block_n,
        WIDE=wide(item_clause_attrs, scratch),
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return _ClauseCompactLaunch(grid, tiles_y, kwargs, tile_counts, scratch)


def _clause_compact_impl(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
    *,
    config: ClauseCompactConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Direct-launch body used by the offline tuner and unit tests. Takes an optional ``config=`` so
    tile parameters can be swept; the public ``custom_op``-wrapped ``clause_compact`` always uses
    ``DEFAULT_CONFIG``."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _clause_compact_prep(item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=cfg)
    _clause_compact_kernel[launch.grid](**launch.kwargs)
    return compact_finish(
        launch.grid,
        launch.tiles_y,
        launch.tile_counts,
        launch.scratch,
        item_clause_attrs.shape[0],
        block_n=cfg.block_n,
        num_warps=cfg.num_warps,
    )


@torch.library.custom_op("retrieve::clause_compact", mutates_args=(), device_types="cuda")
def clause_compact(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> tuple[Tensor, Tensor]:
    """Fused clause evaluation + compaction → (positive_indices [B, N] int64, counts [B] int64). The
    full ``[B, N]`` buffer has ``-1`` sentinels in the unused tail; consumers row-bound by
    ``counts[b]``. Within a row the ids are in **ascending item order** — bit-equal to
    ``ops.reference.clause_compact`` and to itself across launches and processes. Mirrors
    ``_clause_compact_impl`` with ``DEFAULT_CONFIG``.

    Registered as an *opaque* ``custom_op``, not a ``triton_op`` (roadmap C4, 2026-09-06): under
    ``triton_op`` inductor analyses the kernel's TTIR for mutated pointers, and its provenance
    walk follows a store address back through every argument of the ``tt.call`` that produced
    it. ``compact_store``'s address is ``base + intra`` — data-dependent on the pass mask, which
    is the result of the ``clause_pass`` call whose arguments include ``item_attrs_ptr`` — so
    the index buffer (a graph input) is reported as mutated and cudagraph trees skip the whole
    forward ("skipping cudagraphs due to mutated inputs"). ``clause_mask`` stores at a
    data-independent address and is unaffected. As a custom op the kernel launch is opaque to
    inductor, the op is functional as declared, and the forward captures. The cost is one
    custom-op dispatch instead of an inlined launch; the eager path is unchanged."""
    return _clause_compact_impl(item_clause_attrs, clause_is_reverse, query_clause_attrs)


@clause_compact.register_fake
def _(item_clause_attrs, clause_is_reverse, query_clause_attrs):
    b, n = query_clause_attrs.shape[0], item_clause_attrs.shape[0]
    return (
        query_clause_attrs.new_empty((b, n), dtype=torch.int64),
        query_clause_attrs.new_empty((b,), dtype=torch.int64),
    )
