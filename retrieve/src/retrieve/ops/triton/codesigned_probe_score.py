"""SilverTorch phases 2+3 fused over the compact CSR probe layout (kernels.md § SilverTorch
kernels): each row's probed clusters back to back, width = the sum of the ``n_probe`` largest
clusters, read from the cluster-sorted ``item_codes``. The bloom variant tests the query's set
bits against the transposed index, one word per bit per 64 items."""

from __future__ import annotations

from dataclasses import dataclass

import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.ops.triton._host import (
    ProbeLaunch,
    check_contiguous,
    probe_prep,
    probe_topk,
)
from retrieve.ops.triton.common import probe_ids_kernel, probe_tile, row_base


@dataclass(frozen=True)
class CodesignedProbeScoreConfig:
    block_p: int
    num_warps: int
    num_stages: int = 3


# Default tile config (tuned on A100/sm_80); pass config= to _codesigned_probe_score_impl to
# override.
DEFAULT_CONFIG = CodesignedProbeScoreConfig(block_p=256, num_warps=4)


@triton.jit
def _codesigned_probe_score_kernel(
    q_codes_ptr,
    q_scales_ptr,
    probe_ids_ptr,
    offsets_ptr,
    item_codes_ptr,
    qpos_ptr,
    bloom_t_ptr,
    out_scores_ptr,
    global_scale,
    n_probe,
    width,
    tiles_y,
    n_qbits,
    D: tl.constexpr,
    NPP: tl.constexpr,
    stride_qcb,
    stride_cn,
    stride_qpos,
    stride_tm,
    stride_ob,
    HAS_QB: tl.constexpr,
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
        keep = valid
        if HAS_QB:
            # Transposed bloom: bit `pos % 64` of word `pos // 64` of row m is item pos's bit m.
            word = pos >> 6
            bit = pos & 63
            for i in range(n_qbits):
                m = tl.load(qpos_ptr + bid * stride_qpos + i)  # -1: an inactive clause's slot
                w = tl.load(bloom_t_ptr + m * stride_tm + word, mask=keep & (m >= 0), other=-1)
                keep = keep & (((w >> bit) & 1) != 0)

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


def _cps_prep(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    width: int,
    *,
    query_bit_positions: Tensor | None,
    bloom_transposed: Tensor | None,
    cfg: CodesignedProbeScoreConfig,
) -> ProbeLaunch:
    """``_host.probe_prep`` plus the bloom arguments. THE single place input checking happens —
    shared by ``_codesigned_probe_score_impl`` and both ``@triton_op`` wrappers (which keep only
    their textually-inline ``wrap_triton`` launch)."""
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
    has_qb = query_bit_positions is not None
    if has_qb != (bloom_transposed is not None):
        raise ValueError("query_bit_positions and bloom_transposed go together")
    if has_qb:
        check_contiguous(bloom_transposed=bloom_transposed)
        qpos = query_bit_positions.contiguous()
    else:
        # HAS_QB=False gates every load through these pointers, so any int64 tensor stands in.
        qpos = bloom_transposed = probe_ids
    launch.kwargs.update(
        qpos_ptr=qpos,
        bloom_t_ptr=bloom_transposed,
        n_qbits=qpos.shape[1],
        stride_qpos=qpos.stride(0),
        stride_tm=bloom_transposed.stride(0),
        HAS_QB=has_qb,
    )
    return launch


def _codesigned_probe_score_impl(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    k: int,
    width: int,
    *,
    query_bit_positions: Tensor | None = None,
    bloom_transposed: Tensor | None = None,
    config: CodesignedProbeScoreConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused phase-2+3 of SilverTorch's co-designed int8 ANN + optional bloom filter (paper
    Algorithm 1, §4.2): query and items stay int8 through an int32-accumulated dot, dequantized
    by ``q_scale[b] * global_scale``; the transposed bloom test is fused in (failing items and
    slots past a row's items score ``-inf``).

    Inputs: query [B, D] fp32 (int8-quantized here), probe_ids [B, n_probe], cluster_offsets
    [n_lists + 1], item_codes [N, D] int8 cluster-sorted, sort_perm [N], global_scale, k,
    width (the compact probe width, >= k), query_bit_positions [B, C·k_hash] (-1 = none) and
    bloom_transposed [m_bits, ceil(N/64)] (both or neither). Returns (ids [B, K], scores
    [B, K]).

    Eager entry point for tune scripts / parity tests; the compiled path goes through the
    ``@triton_op`` wrappers."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _cps_prep(
        query,
        probe_ids,
        cluster_offsets,
        item_codes,
        sort_perm,
        global_scale,
        width,
        query_bit_positions=query_bit_positions,
        bloom_transposed=bloom_transposed,
        cfg=cfg,
    )
    _codesigned_probe_score_kernel[launch.grid](**launch.kwargs)
    fin = probe_topk(launch, k, probe_ids, cluster_offsets, sort_perm)
    probe_ids_kernel[fin.grid](**fin.kwargs)
    return fin.ids, fin.scores


@triton_op("retrieve::codesigned_probe_score", mutates_args=())
def codesigned_probe_score(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    global_scale: float,
    k: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """Plain int8 ANN scoring (no attribute filter); shares ``_cps_prep``/``probe_topk`` with
    ``_codesigned_probe_score_impl``, keeping the launch inline (``wrap_triton`` must appear
    textually in the decorated source for torch.export). Requires width >= k."""
    launch = _cps_prep(
        query,
        probe_ids,
        cluster_offsets,
        item_codes,
        sort_perm,
        global_scale,
        width,
        query_bit_positions=None,
        bloom_transposed=None,
        cfg=DEFAULT_CONFIG,
    )
    wrap_triton(_codesigned_probe_score_kernel)[launch.grid](**launch.kwargs)
    fin = probe_topk(launch, k, probe_ids, cluster_offsets, sort_perm)
    wrap_triton(probe_ids_kernel)[fin.grid](**fin.kwargs)
    return fin.ids, fin.scores


@triton_op("retrieve::codesigned_probe_score_bloom", mutates_args=())
def codesigned_probe_score_bloom(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    query_bit_positions: Tensor,
    bloom_transposed: Tensor,
    global_scale: float,
    k: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    """Int8 ANN scoring fused with the paper's bloom subset test over the transposed index —
    sibling of ``codesigned_probe_score``, split into a separate op (not one op with an
    Optional/flag) so the layer just routes to the right op."""
    launch = _cps_prep(
        query,
        probe_ids,
        cluster_offsets,
        item_codes,
        sort_perm,
        global_scale,
        width,
        query_bit_positions=query_bit_positions,
        bloom_transposed=bloom_transposed,
        cfg=DEFAULT_CONFIG,
    )
    wrap_triton(_codesigned_probe_score_kernel)[launch.grid](**launch.kwargs)
    fin = probe_topk(launch, k, probe_ids, cluster_offsets, sort_perm)
    wrap_triton(probe_ids_kernel)[fin.grid](**fin.kwargs)
    return fin.ids, fin.scores
