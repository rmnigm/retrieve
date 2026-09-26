"""Host-side launch scaffold shared by the kernel files (review A5): the probe-scorer launch
record and epilogue used by both ``codesigned_probe_score*`` files, the 3-D grid split the
three filter kernels use, and the scan + scatter tail of the two compaction ops.
``common.py`` is ``@triton.jit`` only; this is plain Python."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
from torch import Tensor

from retrieve.ops.triton.common import compact_scatter_kernel


@dataclass(frozen=True)
class ProbeLaunch:
    p: int  # probe width — grid_x tiles over it
    b: int  # batch — grid_y
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    all_scores: Tensor
    flat_probed_items: Tensor  # post-contiguous, for the epilogue gather


def probe_finish(launch: ProbeLaunch, k: int) -> tuple[Tensor, Tensor]:
    # P = n_probe × max_cluster_size >= k by layer-construction assert, so topk(k) needs no min/pad
    # path.
    topk_scores, topk_local = torch.topk(launch.all_scores, k, dim=1)
    topk_ids = launch.flat_probed_items.gather(1, topk_local)
    # -1 at every -inf slot — a filter-rejected item or a -1 padding lane of the probe pool — so
    # the Triton backend meets interfaces.py's "-1 / -inf are the no-item sentinels" the same
    # way ``masked_topk`` does for the torch and official backends (O §14.7). One capture-safe
    # elementwise op: no host sync, no data-dependent branch.
    topk_ids = torch.where(torch.isfinite(topk_scores), topk_ids, -1)
    return topk_ids, topk_scores


def wide(*tensors: Tensor) -> bool:
    """A kernel's ``WIDE`` constexpr: some tensor it addresses has ``>= 2**31`` elements, so its
    row bases need int64 (kernels.md § Addressing)."""
    return any(t.numel() >= 2**31 for t in tensors)


def grid_batch_tiles(b: int, n: int, block: int) -> tuple[tuple[int, int, int], int]:
    """``((b, tiles_y, tiles_x), tiles_y)``: batch on grid_x (L2 reuse on the index), the
    ``cdiv(n, block)`` tiles split across grid_y × grid_z to dodge the 65535 single-axis cap;
    the kernel rebuilds ``tile_id = tile_x * tiles_y + tile_y`` from the returned ``tiles_y``."""
    tiles = triton.cdiv(n, block)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    return (b, tiles_y, tiles_x), tiles_y


def compact_finish(
    grid: tuple[int, int, int],
    tiles_y: int,
    tile_counts: Tensor,
    scratch: Tensor,
    n: int,
    *,
    block_n: int,
    num_warps: int,
) -> tuple[Tensor, Tensor]:
    """Phases 2-3 of the two-phase compaction shared by ``clause_compact`` and ``bloom_compact``:
    exclusive-scan the ``[B, T]`` tile counts the predicate launch wrote (``torch.cumsum`` over
    int64 — exact, hence deterministic), then one program per ``(row, tile)`` on the same grid
    moves the tile's stashed ids to the scanned offset. Returns ``(positive_indices [B, N] int64
    with -1 tails, counts [B] int64)``; ``counts`` is a fresh tensor, not a view into the scan
    (inductor asserts custom-op outputs are aligned, and at ``B == 1`` the last column *is*
    contiguous)."""
    b = tile_counts.shape[0]
    tile_ends = tile_counts.cumsum(1)
    out_indices = torch.full((b, n), -1, dtype=torch.int64, device=tile_counts.device)
    compact_scatter_kernel[grid](
        scratch,
        tile_counts,
        tile_ends - tile_counts,
        out_indices,
        tiles_y,
        scratch.stride(0),
        tile_counts.stride(0),
        out_indices.stride(0),
        out_indices.stride(1),
        BLOCK_N=block_n,
        WIDE=wide(scratch, out_indices),
        num_warps=num_warps,
    )
    return out_indices, tile_ends[:, -1].clone()
