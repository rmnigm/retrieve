"""Host-side launch scaffold shared by the kernel files: the boundary checks and the ``WIDE``
predicate every prep calls, the 3-D grid split of the filter kernels, and the scan + scatter
tail of the compaction op. ``common.py`` is ``@triton.jit`` only; this is plain Python."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
from torch import Tensor

from retrieve.ops.triton.common import compact_scatter_kernel


@dataclass(frozen=True)
class CompactLaunch:
    grid: tuple[int, int, int]
    tiles_y: int
    kwargs: dict[str, object]  # every predicate-kernel arg: tensors, strides, constexprs, cfg
    tile_counts: Tensor
    scratch: Tensor


def check_contiguous(**tables: Tensor) -> None:
    """Item-side tables must arrive contiguous: a per-call ``.contiguous()`` would copy the
    whole index on every forward (kernels.md, conventions). Query-side tensors are small and
    are still copied."""
    for name, t in tables.items():
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (copy it once at index build)")


def wide(*tensors: Tensor) -> bool:
    """A kernel's ``WIDE`` constexpr: some tensor it addresses has ``>= 2**31`` elements, so its
    row bases need int64 (kernels.md § Addressing)."""
    return any(t.numel() >= 2**31 for t in tensors)


def grid_batch_tiles(b: int, n: int, block: int) -> tuple[tuple[int, int, int], int]:
    """``((b, tiles_y, tiles_x), tiles_y)``: batch on grid_x (L2 reuse on the index), the
    ``cdiv(n, block)`` tiles split across grid_y × grid_z to dodge the 65535 single-axis cap;
    the kernel rebuilds ``tile_id = tile_x * tiles_y + tile_y`` from the returned ``tiles_y``."""
    tiles = triton.cdiv(n, block)
    tiles_x = max(1, triton.cdiv(tiles, 65535))  # n == 0: an empty grid, not cdiv(0, 0)
    tiles_y = triton.cdiv(tiles, tiles_x)
    return (b, tiles_y, tiles_x), tiles_y


def compact_finish(
    launch: CompactLaunch, n: int, *, block_n: int, num_warps: int
) -> tuple[Tensor, Tensor]:
    """Phases 2-3 of the two-phase compaction: exclusive-scan the ``[B, T]`` tile counts the
    predicate launch wrote (``torch.cumsum`` over int64, exact), then one program per ``(row,
    tile)`` on the same grid moves the tile's stashed ids to the scanned offset. Returns
    ``(positive_indices [B, N] int64, counts [B] int64)``; each row is defined on ``[0,
    counts[b])`` only (kernels.md § ``clause_compact``). ``counts`` is a fresh tensor, not a view
    into the scan: inductor asserts custom-op outputs are aligned, and at ``B == 1`` the last
    column is a view at an odd offset."""
    tile_counts, scratch = launch.tile_counts, launch.scratch
    b = tile_counts.shape[0]
    tile_ends = tile_counts.cumsum(1)
    counts = tile_ends[:, -1].clone()
    out_indices = torch.empty((b, n), dtype=torch.int64, device=tile_counts.device)
    compact_scatter_kernel[launch.grid](
        scratch,
        tile_counts,
        tile_ends - tile_counts,
        out_indices,
        launch.tiles_y,
        scratch.stride(0),
        tile_counts.stride(0),
        out_indices.stride(0),
        out_indices.stride(1),
        BLOCK_N=block_n,
        WIDE=wide(scratch, out_indices),
        num_warps=num_warps,
    )
    return out_indices, counts
