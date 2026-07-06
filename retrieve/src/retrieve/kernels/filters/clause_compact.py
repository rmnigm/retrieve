"""Fused clause evaluation + stream compaction in one launch (cumsum for intra-tile offsets,
atomic_add for the per-row base), avoiding the dense ``[B, N]`` bool of the pure-torch path.
Output ordering within a row is unspecified (atomics across tiles); callers that need order must
sort."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.kernels import common


@dataclass(frozen=True)
class ClauseCompactConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Default tile config (tuned on A100/sm_80). In-kernel @triton.autotune is unsafe here: atomic_add
# into counts accumulates across trials and out_indices is written in-place; the offline tuner
# allocates fresh buffers per call.
DEFAULT_CONFIG = ClauseCompactConfig(block_n=512, num_warps=2)


@triton.jit
def _clause_compact_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] bool (stored as int8 in torch)
    query_attrs_ptr,  # [B, C] int64
    out_indices_ptr,  # [B, N] int64 (worst-case scratch)
    counts_ptr,  # [B] int64 (init 0)
    N,
    tiles_y,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,  # item_attrs.stride(0)
    stride_ic,  # item_attrs.stride(1)
    stride_ia,  # item_attrs.stride(2)
    stride_qb,
    stride_qc,
    stride_ob,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    # 3D grid: batch on grid_x (L2 reuse on item_attrs), tiles split across grid_y × grid_z to dodge
    # the 65535 single-axis cap; tile_id = tile_x * tiles_y + tile_y keeps tiles contiguous.
    bid = tl.program_id(0)
    tile_y = tl.program_id(1)
    tile_x = tl.program_id(2)
    tile_id = tile_x * tiles_y + tile_y

    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    n_valid = n_offsets < N

    # Result is already ANDed with n_valid inside the helper (keep seeds from load_mask).
    pass_mask = common.clause_pass(
        item_attrs_ptr,
        is_reverse_ptr,
        query_attrs_ptr,
        n_offsets,
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

    # Stream compaction: cumsum gives intra-tile offsets; atomic_add gives base.
    common.compact_store(
        pass_mask, n_offsets, counts_ptr, out_indices_ptr, bid, stride_ob, stride_on
    )


@dataclass(frozen=True)
class _ClauseCompactLaunch:
    grid: tuple[int, int, int]
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    out_indices: Tensor
    counts: Tensor


def _clause_compact_prep(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
    *,
    cfg: ClauseCompactConfig,
) -> _ClauseCompactLaunch:
    """Validation + contiguity + buffers + the full launch-arg dict. THE single place input
    checking happens — shared by ``_clause_compact_impl`` and the public op.

    ``out_indices`` is init'd to the ``-1`` sentinel: the kernel writes only
    ``[base, base + tile_sum)`` per tile, so positions beyond ``counts[bid]`` stay ``-1``. With
    ``torch.empty`` they'd hold uninitialised memory that leaks into the gather when
    ``counts[b] < k``. ``-1`` is the canonical "no item" sentinel. ``counts`` must be
    zero-initialized — the kernel's atomic_add accumulates into it."""
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

    out_indices = torch.full((b, n), -1, dtype=torch.int64, device=device)
    counts = torch.zeros((b,), dtype=torch.int64, device=device)

    # 3D grid (batch, tiles_y, tiles_x) — see kernel comment.
    tiles = triton.cdiv(n, cfg.block_n)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    grid = (b, tiles_y, tiles_x)

    kwargs = dict(
        item_attrs_ptr=item_clause_attrs,
        is_reverse_ptr=clause_is_reverse,
        query_attrs_ptr=query_clause_attrs,
        out_indices_ptr=out_indices,
        counts_ptr=counts,
        N=n,
        tiles_y=tiles_y,
        C=c,
        A_MAX=a_max,
        stride_in=item_clause_attrs.stride(0),
        stride_ic=item_clause_attrs.stride(1),
        stride_ia=item_clause_attrs.stride(2),
        stride_qb=query_clause_attrs.stride(0),
        stride_qc=query_clause_attrs.stride(1),
        stride_ob=out_indices.stride(0),
        stride_on=out_indices.stride(1),
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return _ClauseCompactLaunch(grid, kwargs, out_indices, counts)


def _clause_compact_impl(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
    *,
    config: ClauseCompactConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Direct-launch body used by the offline tuner and unit tests. Takes an optional ``config=`` so
    tile parameters can be swept; the public ``@triton_op``-wrapped ``clause_compact`` always
    uses ``DEFAULT_CONFIG``."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _clause_compact_prep(
        item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=cfg
    )
    _clause_compact_kernel[launch.grid](**launch.kwargs)
    return launch.out_indices, launch.counts


@triton_op("retrieve::clause_compact", mutates_args=())
def clause_compact(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> tuple[Tensor, Tensor]:
    """Fused clause evaluation + compaction → (positive_indices [B, N] int64, counts [B] int64). The
    full ``[B, N]`` buffer has ``-1`` sentinels in the unused tail; consumers row-bound by
    ``counts[b]``, and within-row order is unspecified (atomic writes). Registered as a
    ``triton_op`` for ``torch.compile``; mirrors ``_clause_compact_impl`` with
    ``DEFAULT_CONFIG``."""
    launch = _clause_compact_prep(
        item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=DEFAULT_CONFIG
    )
    wrap_triton(_clause_compact_kernel)[launch.grid](**launch.kwargs)  # inline, K2 invariant
    return launch.out_indices, launch.counts
