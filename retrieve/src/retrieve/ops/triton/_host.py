"""Host-side launch scaffold shared by the kernel files: the probe scorers' shared prep, launch
record and epilogue (both ``codesigned_probe_score*`` files), the boundary checks and ``WIDE``
predicate every prep calls, the 3-D grid split of the filter kernels, and the scan + scatter
tail of the two compaction ops.
``common.py`` is ``@triton.jit`` only; this is plain Python."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TypeVar

import torch
import triton
from torch import Tensor

from retrieve import functional
from retrieve.ops.triton.common import compact_scatter_kernel

Cfg = TypeVar("Cfg")


# Fewer programs than this leave the A100's 108 SMs short of work on a narrow probe width: the
# tile falls back to a smaller one (kernels.md § SilverTorch kernels, "Tile config").
MIN_PROGRAMS = 1024

# The id epilogue's [BLOCK_K, BLOCK_N] tile caps (kernels.md § SilverTorch kernels, "Ids after the
# top-k").
IDS_BLOCK_K = 16
IDS_BLOCK_N = 512

# From this many (row, probe) pairs the scorers read their tile's cluster from a per-row table
# that the prep launch writes, instead of rebuilding the row's layout in every tile (kernels.md §
# SilverTorch kernels, "Probe table").
TABLE_MIN_PAIRS = 512


def gate_pays(launch: ProbeLaunch) -> bool:
    """Whether a probe scorer's pass-rate-gated tile skip can pay: at least ``MIN_PROGRAMS``
    programs, so skipped tiles shorten the run; below that the vote is pure cost (kernels.md §
    SilverTorch kernels, "Gated tile skip")."""
    b, tiles_y, tiles_x = launch.grid
    return b * tiles_y * tiles_x >= MIN_PROGRAMS


def sm_count(device: torch.device) -> int:
    """The device's SM count, the persistent kernels' program unit."""
    return _sm_count(torch.device(device).index or 0)


@functools.cache
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def width_tiles(configs: dict[int, tuple[Cfg, ...]], d: int) -> tuple[Cfg, ...]:
    """The probe scorers' tiles at embedding width ``d``, largest first: the entry of the smallest
    ``D_PAD`` bound in ``configs`` at or above ``next_power_of_2(d)``, the widest entry past the
    last bound (kernels.md § SilverTorch kernels, "Tile config")."""
    d_pad = triton.next_power_of_2(d)
    return configs[min((b for b in configs if b >= d_pad), default=max(configs))]


def tile_for_width(configs: dict[int, tuple[Cfg, ...]], d: int, b: int, width: int) -> Cfg:
    """The shipped tile for a ``[b, width]`` probe launch at width ``d``: the first of
    ``width_tiles`` giving ``b · cdiv(width, block_p) >= MIN_PROGRAMS``, else the smallest.
    ``b`` and ``width`` are static Python ints, so the choice is fixed under capture."""
    tiles = width_tiles(configs, d)
    return next((c for c in tiles if b * triton.cdiv(width, c.block_p) >= MIN_PROGRAMS), tiles[-1])


@dataclass(frozen=True)
class ProbeLaunch:
    grid: tuple[int, int, int]  # (B, tiles_y, tiles_x) over the cluster-aligned + tail tiles
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    all_scores: Tensor  # [B, width], padded to whole TOPK_BLOCKs (functional.two_level_topk)
    prep: ProbePrep  # launched first: the int8 query and, with "TABLE", the per-row table


@dataclass(frozen=True)
class ProbePrep:
    grid: tuple[int]  # (B,)
    kwargs: dict[str, object]  # common.probe_prep_kernel's args


@dataclass(frozen=True)
class CompactLaunch:
    grid: tuple[int, int, int]
    tiles_y: int
    kwargs: dict[str, object]  # every predicate-kernel arg: tensors, strides, constexprs, cfg
    tile_counts: Tensor
    scratch: Tensor


@dataclass(frozen=True)
class ProbeIds:
    grid: tuple[int, int]  # (B, cdiv(k, BLOCK_K)): one program per row and k chunk
    kwargs: dict[str, object]  # common.probe_ids_kernel's args
    ids: Tensor  # [B, k] int64, written by the launch
    scores: Tensor  # [B, k] fp32


def probe_prep(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    width: int,
    *,
    block_p: int,
    num_warps: int,
    num_stages: int,
    block_d: int,
    skip: bool,
) -> ProbeLaunch:
    """The half of both probe scorers' prep that is the same: validation, the per-row int8
    query, the ``[B, width]`` score buffer (``torch.empty``: the kernel writes every slot, a
    dot or ``-inf``) and the launch kwargs they share. The probe layout is built inside the
    kernel (``common.probe_tile``), or, from ``TABLE_MIN_PAIRS`` pairs, once per row by the
    ``prep`` launch the caller runs first (``common.probe_prep_kernel``), which also writes the
    int8 query."""
    if query.dim() != 2 or probe_ids.dim() != 2:
        raise ValueError("query must be [B, D] and probe_ids [B, n_probe]")
    if item_codes.dtype != torch.int8:
        raise TypeError(f"item_codes must be int8, got {item_codes.dtype}")
    b, d = query.shape
    n_probe = probe_ids.shape[1]
    check_contiguous(item_codes=item_codes, sort_perm=sort_perm)
    if query.dtype != torch.float32:
        raise TypeError(f"query must be fp32, got {query.dtype}")
    query = query.contiguous()
    q_codes = torch.empty((b, d), dtype=torch.int8, device=query.device)
    q_scales = torch.empty(b, dtype=torch.float32, device=query.device)
    if b * width >= functional.TOPK_MIN_SLOTS:
        # The scorer's -inf tail covers the pad: the grid and the tail mask follow the width.
        width = triton.cdiv(width, functional.TOPK_BLOCK) * functional.TOPK_BLOCK
    all_scores = torch.empty((b, width), dtype=torch.float32, device=query.device)
    # Cluster-aligned tiles: at most one partial tile per probe, so n_probe extra tiles cover
    # the rounding and the -inf tail past the row's items.
    grid, tiles_y = grid_batch_tiles(b, triton.cdiv(width, block_p) + n_probe, 1)
    npp = triton.next_power_of_2(n_probe)
    probe_ids = probe_ids.contiguous()
    use_table = b * n_probe >= TABLE_MIN_PAIRS
    # Without the table the prep's table loads compile out, so any int64 tensor stands in.
    table_ptr = (
        torch.empty((b, 3, n_probe), dtype=torch.int64, device=query.device)
        if use_table
        else probe_ids
    )
    prep = ProbePrep(
        (b,),
        {
            "query_ptr": query,
            "q_codes_ptr": q_codes,
            "q_scales_ptr": q_scales,
            "probe_ids_ptr": probe_ids,
            "offsets_ptr": cluster_offsets,
            "table_ptr": table_ptr,
            "count_ptr": table_ptr,  # the bloom two-pass's list count, zeroed here
            "stride_qb": query.stride(0),
            "n_probe": n_probe,
            "D": d,
            "D_PAD": triton.next_power_of_2(d),
            "NPP": npp,
            "BLOCK_P": block_p,
            "TABLE": use_table,
            "ZERO_COUNT": False,
            "num_warps": 4,
        },
    )
    kwargs: dict[str, object] = {
        "q_codes_ptr": q_codes,
        "q_scales_ptr": q_scales,
        "probe_ids_ptr": probe_ids,
        "offsets_ptr": cluster_offsets,
        "table_ptr": table_ptr,
        "item_codes_ptr": item_codes,
        "out_scores_ptr": all_scores,
        "global_scale": float(global_scale),
        "n_probe": n_probe,
        "width": width,
        "tiles_y": tiles_y,
        "D": d,
        "D_PAD": triton.next_power_of_2(d),
        "NPP": npp,
        "FAN": 1 << (npp.bit_length() // 2),  # FAN ** 2 >= NPP
        "TABLE": use_table,
        "stride_qcb": q_codes.stride(0),
        "stride_cn": item_codes.stride(0),
        "stride_ob": all_scores.stride(0),
        "BLOCK_P": block_p,
        "BLOCK_D": block_d,
        "SKIP": skip,
        "WIDE": wide(all_scores),
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    return ProbeLaunch(grid, kwargs, all_scores, prep)


def probe_topk(
    launch: ProbeLaunch, k: int, probe_ids: Tensor, cluster_offsets: Tensor, sort_perm: Tensor
) -> ProbeIds:
    """``torch.topk`` over the compact slots (``width >= k`` by the layer's probe-pool check)
    and the ``common.probe_ids_kernel`` launch that turns the winning slots into original ids,
    ``-1`` at every ``-inf`` slot — a rejected item or a slot past the row's items — so the
    Triton backend meets ``interfaces.py``'s "``-1`` / ``-inf`` are the no-item sentinels" as
    ``masked_topk`` does for the other backends. The caller launches it (the op bodies keep
    their ``wrap_triton`` lines inline). Capture-safe: no host sync, no data-dependent branch."""
    scores, slots = functional.two_level_topk(launch.all_scores, k)
    block_k = min(triton.next_power_of_2(k), IDS_BLOCK_K)
    ids = torch.empty_like(slots)
    n_probe = probe_ids.shape[1]
    kwargs: dict[str, object] = {
        "slots_ptr": slots,
        "scores_ptr": scores,
        "probe_ids_ptr": probe_ids.contiguous(),
        "offsets_ptr": cluster_offsets,
        "sort_perm_ptr": sort_perm,
        "ids_ptr": ids,
        "n_probe": n_probe,
        "k": k,
        "BLOCK_K": block_k,
        "BLOCK_N": min(triton.next_power_of_2(n_probe), IDS_BLOCK_N),
        "num_warps": 4,
    }
    return ProbeIds((scores.shape[0], triton.cdiv(k, block_k)), kwargs, ids, scores)


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
    """Phases 2-3 of the two-phase compaction shared by ``clause_compact`` and ``bloom_compact``:
    exclusive-scan the ``[B, T]`` tile counts the predicate launch wrote (``torch.cumsum`` over
    int64 — exact, hence deterministic), then one program per ``(row, tile)`` on the same grid
    moves the tile's stashed ids to the scanned offset. Returns ``(positive_indices [B, N]
    int64, counts [B] int64)``; each row is defined on ``[0, counts[b])`` only, the rest is
    whatever ``torch.empty`` held (kernels.md § ``clause_compact``). ``counts`` is a fresh
    tensor, not a view into the scan (inductor asserts custom-op outputs are aligned, and at
    ``B == 1`` the last column *is* contiguous)."""
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
