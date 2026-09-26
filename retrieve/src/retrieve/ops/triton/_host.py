"""Host-side launch scaffold shared by the kernel files: the probe scorers' shared prep, launch
record and epilogue (both ``codesigned_probe_score*`` files), the boundary checks and ``WIDE``
predicate every prep calls, the 3-D grid split of the filter kernels, and the scan + scatter
tail of the two compaction ops.
``common.py`` is ``@triton.jit`` only; this is plain Python."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
from torch import Tensor

from retrieve.indexing.quantize import quantize_int8
from retrieve.ops.triton.common import compact_scatter_kernel


@dataclass(frozen=True)
class ProbeLaunch:
    grid: tuple[int, int, int]  # (B, tiles_y, tiles_x) over the cluster-aligned + tail tiles
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    all_scores: Tensor


@dataclass(frozen=True)
class ProbeIds:
    grid: tuple[int]  # one program per row
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
) -> ProbeLaunch:
    """The half of both probe scorers' prep that is the same: validation, the per-row int8
    query, the ``[B, width]`` score buffer (``torch.empty``: the kernel writes every slot, a
    dot or ``-inf``) and the launch kwargs they share. The probe layout itself is built inside
    the kernel (``common.probe_tile``)."""
    if query.dim() != 2 or probe_ids.dim() != 2:
        raise ValueError("query must be [B, D] and probe_ids [B, n_probe]")
    if item_codes.dtype != torch.int8:
        raise TypeError(f"item_codes must be int8, got {item_codes.dtype}")
    b, d = query.shape
    n_probe = probe_ids.shape[1]
    check_contiguous(item_codes=item_codes, sort_perm=sort_perm)
    check_pow2(D=d)
    q_codes, q_scales = quantize_int8(query)
    all_scores = torch.empty((b, width), dtype=torch.float32, device=query.device)
    # Cluster-aligned tiles: at most one partial tile per probe, so n_probe extra tiles cover
    # the rounding and the -inf tail past the row's items.
    grid, tiles_y = grid_batch_tiles(b, triton.cdiv(width, block_p) + n_probe, 1)
    kwargs: dict[str, object] = dict(
        q_codes_ptr=q_codes.contiguous(),
        q_scales_ptr=q_scales.contiguous(),
        probe_ids_ptr=probe_ids.contiguous(),
        offsets_ptr=cluster_offsets,
        item_codes_ptr=item_codes,
        out_scores_ptr=all_scores,
        global_scale=float(global_scale),
        n_probe=n_probe,
        width=width,
        tiles_y=tiles_y,
        D=d,
        NPP=triton.next_power_of_2(n_probe),
        stride_qcb=q_codes.stride(0),
        stride_cn=item_codes.stride(0),
        stride_ob=all_scores.stride(0),
        BLOCK_P=block_p,
        WIDE=wide(all_scores),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return ProbeLaunch(grid, kwargs, all_scores)


def probe_topk(
    launch: ProbeLaunch, k: int, probe_ids: Tensor, cluster_offsets: Tensor, sort_perm: Tensor
) -> ProbeIds:
    """``torch.topk`` over the compact slots (``width >= k`` by the layer's probe-pool check)
    and the ``common.probe_ids_kernel`` launch that turns the winning slots into original ids,
    ``-1`` at every ``-inf`` slot — a rejected item or a slot past the row's items — so the
    Triton backend meets ``interfaces.py``'s "``-1`` / ``-inf`` are the no-item sentinels" as
    ``masked_topk`` does for the other backends. The caller launches it (the op bodies keep
    their ``wrap_triton`` lines inline). Capture-safe: no host sync, no data-dependent branch."""
    scores, slots = torch.topk(launch.all_scores, k, dim=1)
    ids = torch.empty_like(slots)
    n_probe = probe_ids.shape[1]
    kwargs: dict[str, object] = dict(
        slots_ptr=slots,
        scores_ptr=scores,
        probe_ids_ptr=probe_ids.contiguous(),
        offsets_ptr=cluster_offsets,
        sort_perm_ptr=sort_perm,
        ids_ptr=ids,
        n_probe=n_probe,
        k=k,
        NPP=triton.next_power_of_2(n_probe),
        KP=triton.next_power_of_2(k),
        num_warps=4,
    )
    return ProbeIds((scores.shape[0],), kwargs, ids, scores)


def check_contiguous(**tables: Tensor) -> None:
    """Item-side tables must arrive contiguous: a per-call ``.contiguous()`` would copy the
    whole index on every forward (kernels.md, conventions). Query-side tensors are small and
    are still copied."""
    for name, t in tables.items():
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (copy it once at index build)")


def check_pow2(**extents: int) -> None:
    """A ``tl.arange`` extent must be a power of two; Triton otherwise fails inside the
    compiler."""
    for name, v in extents.items():
        if v <= 0 or v & (v - 1):
            raise ValueError(f"{name}={v} must be a power of two")


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
    moves the tile's stashed ids to the scanned offset and writes ``-1`` over its slice of the
    row's tail. Returns ``(positive_indices [B, N] int64 with -1 tails, counts [B] int64)``;
    ``counts`` is a fresh tensor, not a view into the scan (inductor asserts custom-op outputs
    are aligned, and at ``B == 1`` the last column *is* contiguous)."""
    b = tile_counts.shape[0]
    tile_ends = tile_counts.cumsum(1)
    counts = tile_ends[:, -1].clone()
    out_indices = torch.empty((b, n), dtype=torch.int64, device=tile_counts.device)
    compact_scatter_kernel[grid](
        scratch,
        tile_counts,
        tile_ends - tile_counts,
        counts,
        out_indices,
        n,
        tiles_y,
        scratch.stride(0),
        tile_counts.stride(0),
        out_indices.stride(0),
        out_indices.stride(1),
        BLOCK_N=block_n,
        WIDE=wide(scratch, out_indices),
        num_warps=num_warps,
    )
    return out_indices, counts
