"""Host-side launch scaffold shared by the kernel files (review A5): the probe-scorer launch
record and epilogue used by both ``codesigned_probe_score*`` files, and the 3-D grid split the
three filter kernels use. ``common.py`` is ``@triton.jit`` only; this is plain Python."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
from torch import Tensor


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


def grid_batch_tiles(b: int, n: int, block: int) -> tuple[tuple[int, int, int], int]:
    """``((b, tiles_y, tiles_x), tiles_y)``: batch on grid_x (L2 reuse on the index), the
    ``cdiv(n, block)`` tiles split across grid_y × grid_z to dodge the 65535 single-axis cap;
    the kernel rebuilds ``tile_id = tile_x * tiles_y + tile_y`` from the returned ``tiles_y``."""
    tiles = triton.cdiv(n, block)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    return (b, tiles_y, tiles_x), tiles_y
