"""The exact-clause sibling of ``codesigned_probe_score``: the same compact CSR probe layout and
int8 dot, gated by the AND-of-OR clause predicate over the cluster-sorted attrs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.ops.triton._host import ProbeLaunch, check_contiguous, probe_prep, probe_topk
from retrieve.ops.triton.common import clause_pass, probe_ids_kernel, probe_tile, row_base


@dataclass(frozen=True)
class CodesignedProbeScoreExactConfig:
    block_p: int
    num_warps: int
    num_stages: int = 3


# Default tile config (mirrors codesigned_probe_score's A100 tuning).
DEFAULT_CONFIG = CodesignedProbeScoreExactConfig(block_p=256, num_warps=4)


@triton.jit
def _codesigned_probe_score_exact_kernel(
    q_codes_ptr,
    q_scales_ptr,
    probe_ids_ptr,
    offsets_ptr,
    item_codes_ptr,
    item_attrs_ptr,
    is_reverse_ptr,
    query_attrs_ptr,
    out_scores_ptr,
    global_scale,
    n_probe,
    width,
    tiles_y,
    D: tl.constexpr,
    NPP: tl.constexpr,
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
    WIDE: tl.constexpr,
):
    # Batch on grid_x, so the rows' early probes run together and share clusters in L2; tiles
    # split across grid_y × grid_z (kernels.md § SilverTorch kernels).
    bid = tl.program_id(0)
    t = tl.program_id(2) * tiles_y + tl.program_id(1)
    pos, slot, valid, total, tail = probe_tile(
        probe_ids_ptr, offsets_ptr, bid, t, n_probe, NPP, BLOCK_P
    )
    out_row = row_base(out_scores_ptr, bid, stride_ob, WIDE)
    if tail:
        # Past the row's clusters (most of the width on a skewed IVF): the -inf tail.
        tl.store(out_row + slot, tl.full([BLOCK_P], float("-inf"), tl.float32), mask=slot < width)
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

        d_off = tl.arange(0, D)
        q_codes = tl.load(q_codes_ptr + bid * stride_qcb + d_off)
        q_scale = tl.load(q_scales_ptr + bid)
        codes = tl.load(
            item_codes_ptr + pos[:, None] * stride_cn + d_off[None, :],
            mask=keep[:, None],
            other=0,
        )
        # int8 × int8 → int32 (paper §4.2): q[1,D] @ codes^T[D,BLOCK_P]. M=1 can't use IMMA
        # tensor cores, so Triton lowers to the dp4a int8 path the paper claims.
        dots_2d = tl.dot(q_codes[None, :], tl.trans(codes), out_dtype=tl.int32)
        # Squeeze the length-1 M axis: tl.sum over length-1 (no reshape to drop a dim).
        dots_i32 = tl.sum(dots_2d, axis=0)
        # fp32 pinned: Inductor passes global_scale as a Python float (kernels.md § Numerics).
        dots = (
            dots_i32.to(tl.float32)
            * tl.cast(q_scale, tl.float32)
            * tl.cast(global_scale, tl.float32)
        )
        dots = tl.where(keep, dots, float("-inf"))
        # Lanes past the cluster's end hold the next cluster's slots: leave them to its tile.
        tl.store(out_row + slot, dots, mask=valid)


def _cpse_prep(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    width: int,
    *,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    cfg: CodesignedProbeScoreExactConfig,
) -> ProbeLaunch:
    """``_host.probe_prep`` plus the clause arguments. THE single place input checking happens —
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
    ``codesigned_probe_score`` with the same int8×int8 → int32 dp4a path and ``-inf`` + top-K
    epilogue. The predicate (from ``clause_mask``) is AND across ``C`` clauses, OR across
    ``A_max`` values per clause, XOR with ``clause_is_reverse``, OR with the ``q_c == -1``
    inactive sentinel; the ``[BLOCK_P, C, A_max]`` gather stays off HBM.

    Inputs as ``_codesigned_probe_score_impl`` plus item_clause_attrs [N, C, A_max] int64
    cluster-sorted (-1 pad), clause_is_reverse [C] bool, query_clause_attrs [B, C] int64 (-1
    inactive). Returns (ids [B, K], scores [B, K]); requires width >= k.

    Eager entry point for tune scripts / parity tests; the compiled path goes through the
    ``@triton_op`` wrapper."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _cpse_prep(
        query,
        probe_ids,
        cluster_offsets,
        item_codes,
        sort_perm,
        global_scale,
        width,
        item_clause_attrs=item_clause_attrs,
        clause_is_reverse=clause_is_reverse,
        query_clause_attrs=query_clause_attrs,
        cfg=cfg,
    )
    _codesigned_probe_score_exact_kernel[launch.grid](**launch.kwargs)
    fin = probe_topk(launch, k, probe_ids, cluster_offsets, sort_perm)
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
        width,
        item_clause_attrs=item_clause_attrs,
        clause_is_reverse=clause_is_reverse,
        query_clause_attrs=query_clause_attrs,
        cfg=DEFAULT_CONFIG,
    )
    wrap_triton(_codesigned_probe_score_exact_kernel)[launch.grid](**launch.kwargs)
    fin = probe_topk(launch, k, probe_ids, cluster_offsets, sort_perm)
    wrap_triton(probe_ids_kernel)[fin.grid](**fin.kwargs)
    return fin.ids, fin.scores
