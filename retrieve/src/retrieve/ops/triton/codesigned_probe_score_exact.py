"""The exact-clause sibling of ``codesigned_probe_score``: the same compact CSR probe layout and
int8 dot, gated by the AND-of-OR clause predicate over the cluster-sorted attrs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.ops.triton._host import (
    ProbeLaunch,
    check_contiguous,
    probe_candidates,
    probe_prep,
    probe_topk,
    tile_for_width,
)
from retrieve.ops.triton.common import (
    clause_pass,
    probe_candidates_kernel,
    probe_dots,
    probe_ids_kernel,
    probe_prep_kernel,
    probe_tile,
    probe_tile_table,
    row_base,
)


@dataclass(frozen=True)
class CodesignedProbeScoreExactConfig:
    block_p: int
    num_warps: int
    num_stages: int = 3
    block_d: int = 256  # the D loop's chunk; unused at D_PAD <= 256 (one dot)
    skip: bool = False  # skip a filtered tile with no passing lane


# Tiles per D_PAD bound, largest first, tuned on A100/sm_80 (kernels.md § SilverTorch kernels,
# "Tile config").
CONFIGS = {
    256: (CodesignedProbeScoreExactConfig(block_p=256, num_warps=4),),
    1024: (
        CodesignedProbeScoreExactConfig(
            block_p=128, num_warps=4, num_stages=2, block_d=128, skip=True
        ),
    ),
}


@triton.jit
def _codesigned_probe_score_exact_kernel(
    q_codes_ptr,
    q_scales_ptr,
    probe_ids_ptr,
    offsets_ptr,
    table_ptr,
    item_codes_ptr,
    item_attrs_ptr,
    is_reverse_ptr,
    query_attrs_ptr,
    out_scores_ptr,
    tile_max_ptr,
    global_scale,
    n_probe,
    width,
    tiles_y,
    n_tiles_grid,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    NPP: tl.constexpr,
    FAN: tl.constexpr,
    TABLE: tl.constexpr,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_qcb,
    stride_cn,
    stride_ian,
    stride_iac,
    stride_iaa,
    stride_qab,
    stride_qac,
    stride_ob,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SKIP: tl.constexpr,
    WIDE: tl.constexpr,
    TILE_MAX: tl.constexpr,
):
    # Batch on grid_x, so the rows' early probes run together and share clusters in L2; tiles
    # split across grid_y × grid_z (kernels.md § SilverTorch kernels).
    bid = tl.program_id(0)
    t = tl.program_id(2) * tiles_y + tl.program_id(1)
    if TABLE:
        pos, slot, valid, tail = probe_tile_table(table_ptr, bid, t, n_probe, FAN, BLOCK_P)
    else:
        pos, slot, valid, tail = probe_tile(
            probe_ids_ptr, offsets_ptr, bid, t, n_probe, NPP, BLOCK_P
        )
    out_row = row_base(out_scores_ptr, bid, stride_ob, WIDE)
    if tail:
        # Past the row's clusters (most of the width on a skewed IVF): the -inf tail.
        tl.store(out_row + slot, tl.full([BLOCK_P], float("-inf"), tl.float32), mask=slot < width)
        if TILE_MAX:
            tl.store(tile_max_ptr + bid * n_tiles_grid + t, tl.full([], float("-inf"), tl.float32))
    else:
        keep = clause_pass(
            item_attrs_ptr,
            is_reverse_ptr,
            query_attrs_ptr,
            ids=pos,
            load_mask=valid,
            bid=bid,
            stride_in=stride_ian,
            stride_ic=stride_iac,
            stride_ia=stride_iaa,
            stride_qb=stride_qab,
            stride_qc=stride_qac,
            C=C,
            A_MAX=A_MAX,
        )

        any_pass = 1  # a Python int when the skip is off, so the branch folds away
        if SKIP:
            any_pass = tl.max(keep.to(tl.int32), axis=0)
        if any_pass == 0:
            # No lane passes the filter: no code load, no dot (kernels.md § SilverTorch
            # kernels, "Tile skip").
            tl.store(out_row + slot, tl.full([BLOCK_P], float("-inf"), tl.float32), mask=valid)
            if TILE_MAX:
                tl.store(
                    tile_max_ptr + bid * n_tiles_grid + t, tl.full([], float("-inf"), tl.float32)
                )
        else:
            dots_i32, q_scale = probe_dots(
                q_codes_ptr + bid * stride_qcb,
                q_scales_ptr,
                bid,
                item_codes_ptr,
                pos,
                stride_cn,
                keep,
                D,
                D_PAD,
                BLOCK_D,
                BLOCK_P,
            )
            # fp32 pinned: Inductor passes global_scale as a Python float (kernels.md § Numerics).
            dots = (
                dots_i32.to(tl.float32)
                * tl.cast(q_scale, tl.float32)
                * tl.cast(global_scale, tl.float32)
            )
            dots = tl.where(keep, dots, float("-inf"))
            # Lanes past the cluster's end hold the next cluster's slots: leave them to its tile.
            tl.store(out_row + slot, dots, mask=valid)
            if TILE_MAX:
                # The two-level top-k's tile max (kernels.md § SilverTorch kernels, "Top-k").
                tile_max = tl.max(tl.where(valid, dots, float("-inf")), axis=0)
                tl.store(tile_max_ptr + bid * n_tiles_grid + t, tile_max)


def _cpse_prep(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    k: int,
    width: int,
    *,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    cfg: CodesignedProbeScoreExactConfig,
) -> ProbeLaunch:
    """``_host.probe_prep`` plus the clause arguments. The one place inputs are checked —
    shared by ``_codesigned_probe_score_exact_impl`` and the ``@triton_op`` wrapper (which keeps
    only its textually-inline ``wrap_triton`` launch)."""
    if item_clause_attrs.dim() != 3:
        raise ValueError("item_clause_attrs must be [N, C, A_max]")
    if query_clause_attrs.dim() != 2:
        raise ValueError("query_clause_attrs must be [B, C]")
    _, c, a_max = item_clause_attrs.shape
    b_q, c_q = query_clause_attrs.shape
    if c != c_q:
        raise ValueError(f"clause-count mismatch: items C={c}, query C={c_q}")
    if b_q != query.shape[0]:
        raise ValueError(f"batch mismatch: query B={query.shape[0]}, query_clause_attrs B={b_q}")
    if clause_is_reverse.shape != (c,):
        raise ValueError(f"clause_is_reverse must be [{c}], got {tuple(clause_is_reverse.shape)}")
    check_contiguous(item_clause_attrs=item_clause_attrs)
    launch = probe_prep(
        query,
        probe_ids,
        cluster_offsets,
        item_codes,
        sort_perm,
        global_scale,
        width,
        block_p=cfg.block_p,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        block_d=cfg.block_d,
        skip=cfg.skip,
        k=k,
    )
    query_clause_attrs = query_clause_attrs.contiguous()
    launch.kwargs.update(
        item_attrs_ptr=item_clause_attrs,
        # Triton can't load native torch.bool; mirror clause_mask.py.
        is_reverse_ptr=clause_is_reverse.contiguous().to(torch.int8),
        query_attrs_ptr=query_clause_attrs,
        C=c,
        A_MAX=a_max,
        stride_ian=item_clause_attrs.stride(0),
        stride_iac=item_clause_attrs.stride(1),
        stride_iaa=item_clause_attrs.stride(2),
        stride_qab=query_clause_attrs.stride(0),
        stride_qac=query_clause_attrs.stride(1),
    )
    return launch


def _codesigned_probe_score_exact_impl(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    k: int,
    width: int,
    *,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    config: CodesignedProbeScoreExactConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused phase-2+3 with an exact-clause filter instead of Bloom — sibling of
    ``codesigned_probe_score`` with the same int8×int8 → int32 IMMA path and ``-inf`` + top-K
    epilogue. The predicate (from ``clause_mask``) is AND across ``C`` clauses, OR across
    ``A_max`` values per clause, XOR with ``clause_is_reverse``, OR with the ``q_c == -1``
    inactive sentinel; the ``[BLOCK_P, C, A_max]`` gather stays off HBM.

    Inputs as ``_codesigned_probe_score_impl`` plus item_clause_attrs [N, C, A_max] int64
    cluster-sorted (-1 pad), clause_is_reverse [C] bool, query_clause_attrs [B, C] int64 (-1
    inactive). Returns (ids [B, K], scores [B, K]); requires width >= k.

    Eager entry point for tune scripts / parity tests; the compiled path goes through the
    ``@triton_op`` wrapper."""
    cfg = (
        config
        if config is not None
        else tile_for_width(CONFIGS, query.shape[1], query.shape[0], width)
    )
    launch = _cpse_prep(
        query,
        probe_ids,
        cluster_offsets,
        item_codes,
        sort_perm,
        global_scale,
        k,
        width,
        item_clause_attrs=item_clause_attrs,
        clause_is_reverse=clause_is_reverse,
        query_clause_attrs=query_clause_attrs,
        cfg=cfg,
    )
    probe_prep_kernel[launch.prep.grid](**launch.prep.kwargs)
    _codesigned_probe_score_exact_kernel[launch.grid](**launch.kwargs)
    cands = probe_candidates(launch, k)
    if cands is not None:
        probe_candidates_kernel[cands.grid](**cands.kwargs)
    fin = probe_topk(launch, k, probe_ids, cluster_offsets, sort_perm, cands)
    probe_ids_kernel[fin.grid](**fin.kwargs)
    return fin.ids, fin.scores


@triton_op("retrieve::codesigned_probe_score_exact", mutates_args=())
def codesigned_probe_score_exact(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    global_scale: float,
    k: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """Production ``@triton_op`` for exact-clause-filtered int8 ANN scoring; shares
    ``_cpse_prep``/``probe_topk`` with ``_codesigned_probe_score_exact_impl``, keeping the
    launch inline (``wrap_triton`` must appear textually in the decorated source for
    torch.export). Requires width >= k."""
    launch = _cpse_prep(
        query,
        probe_ids,
        cluster_offsets,
        item_codes,
        sort_perm,
        global_scale,
        k,
        width,
        item_clause_attrs=item_clause_attrs,
        clause_is_reverse=clause_is_reverse,
        query_clause_attrs=query_clause_attrs,
        cfg=tile_for_width(CONFIGS, query.shape[1], query.shape[0], width),
    )
    wrap_triton(probe_prep_kernel)[launch.prep.grid](**launch.prep.kwargs)
    wrap_triton(_codesigned_probe_score_exact_kernel)[launch.grid](**launch.kwargs)
    cands = probe_candidates(launch, k)
    if cands is not None:
        wrap_triton(probe_candidates_kernel)[cands.grid](**cands.kwargs)
    fin = probe_topk(launch, k, probe_ids, cluster_offsets, sort_perm, cands)
    wrap_triton(probe_ids_kernel)[fin.grid](**fin.kwargs)
    return fin.ids, fin.scores
