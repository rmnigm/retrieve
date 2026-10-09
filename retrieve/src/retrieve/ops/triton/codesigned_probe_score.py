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
    gate_pays,
    probe_prep,
    probe_topk,
    tile_for_width,
)
from retrieve.ops.triton.common import probe_dots, probe_ids_kernel, probe_tile, row_base


@dataclass(frozen=True)
class CodesignedProbeScoreConfig:
    block_p: int
    num_warps: int
    num_stages: int = 3
    block_d: int = 256  # the D loop's chunk; unused at D_PAD <= 256 (one dot)
    skip: bool = False  # skip a filtered tile with no passing lane


# Tiles per D_PAD bound, largest first, tuned on A100/sm_80 (kernels.md § SilverTorch kernels,
# "Tile config").
CONFIGS = {
    256: (CodesignedProbeScoreConfig(block_p=256, num_warps=4),),
    1024: (
        CodesignedProbeScoreConfig(block_p=256, num_warps=4, num_stages=2, block_d=128, skip=True),
        CodesignedProbeScoreConfig(block_p=128, num_warps=4, num_stages=2, block_d=256, skip=True),
        CodesignedProbeScoreConfig(block_p=64, num_warps=4, num_stages=2, block_d=256, skip=True),
    ),
}


@triton.jit
def _codesigned_probe_score_kernel(
    q_codes_ptr,
    q_scales_ptr,
    probe_ids_ptr,
    offsets_ptr,
    item_codes_ptr,
    qpos_ptr,
    bloom_t_ptr,
    bit_freq_ptr,
    out_scores_ptr,
    global_scale,
    n_probe,
    width,
    tiles_y,
    n_qbits,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    NPP: tl.constexpr,
    stride_qcb,
    stride_cn,
    stride_qpos,
    stride_tm,
    stride_ob,
    HAS_QB: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SKIP: tl.constexpr,
    GATED: tl.constexpr,
    NQB: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Batch on grid_x, so the rows' early probes run together and share clusters in L2; tiles
    # split across grid_y × grid_z (kernels.md § SilverTorch kernels).
    bid = tl.program_id(0)
    t = tl.program_id(2) * tiles_y + tl.program_id(1)
    pos, slot, valid, tail = probe_tile(probe_ids_ptr, offsets_ptr, bid, t, n_probe, NPP, BLOCK_P)
    out_row = row_base(out_scores_ptr, bid, stride_ob, WIDE)
    if GATED and HAS_QB:
        # Every queried bit must be set: p <= the rarest bit's frequency. Loaded first, so the
        # latency overlaps the tile's own loads.
        qi = tl.arange(0, NQB)
        qbits = tl.load(qpos_ptr + bid * stride_qpos + qi, mask=qi < n_qbits, other=-1)
        bound = tl.min(tl.load(bit_freq_ptr + qbits, mask=qbits >= 0, other=1.0), axis=0)
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
                # Masked by `valid`, not the running `keep`, so the bits' loads do not wait on
                # each other (kernels.md § codesigned_probe_score).
                w = tl.load(bloom_t_ptr + m * stride_tm + word, mask=valid & (m >= 0), other=-1)
                keep = keep & (((w >> bit) & 1) != 0)

        any_pass = 1  # a Python int when the skip is off, so the branch folds away
        if SKIP and HAS_QB:
            any_pass = tl.max(keep.to(tl.int32), axis=0)
        elif GATED and HAS_QB:
            # Vote only where under one passing lane per tile is expected (kernels.md § SilverTorch
            # kernels, "Gated tile skip").
            any_pass = tl.full([], 1, tl.int32)
            if bound * BLOCK_P < 1.0:
                any_pass = tl.max(keep.to(tl.int32), axis=0)
        if any_pass == 0:
            # No lane passes the filter: no code load, no dot (kernels.md § SilverTorch
            # kernels, "Tile skip").
            tl.store(out_row + slot, tl.full([BLOCK_P], float("-inf"), tl.float32), mask=valid)
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
    bit_freq: Tensor | None = None,
) -> ProbeLaunch:
    """``_host.probe_prep`` plus the bloom arguments. The one place inputs are checked —
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
        block_d=cfg.block_d,
        skip=cfg.skip,
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
    # The pass-rate gate where the tile skip is not on unconditionally (D_PAD <= 256), and only on
    # grids of several waves: below that a skipped tile does not shorten the critical path.
    gated = has_qb and bit_freq is not None and not cfg.skip and gate_pays(launch)
    launch.kwargs.update(
        qpos_ptr=qpos,
        bloom_t_ptr=bloom_transposed,
        # GATED=False compiles the loads out, so any fp32 tensor stands in.
        bit_freq_ptr=bit_freq if gated else launch.kwargs["q_scales_ptr"],
        GATED=gated,
        NQB=triton.next_power_of_2(max(qpos.shape[1], 1)),
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
    bloom_bit_freq: Tensor | None = None,
    config: CodesignedProbeScoreConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused phase-2+3 of SilverTorch's co-designed int8 ANN + optional bloom filter (paper
    Algorithm 1, §4.2): query and items stay int8 through an int32-accumulated dot, dequantized
    by ``q_scale[b] * global_scale``; the transposed bloom test is fused in (failing items and
    slots past a row's items score ``-inf``).

    Inputs: query [B, D] fp32 (int8-quantized here), probe_ids [B, n_probe], cluster_offsets
    [n_lists + 1], item_codes [N, D] int8 cluster-sorted, sort_perm [N], global_scale, k,
    width (the compact probe width, >= k), query_bit_positions [B, C·k_hash] (-1 = none) and
    bloom_transposed [m_bits, ceil(N/64)] (both or neither), and optionally bloom_bit_freq [m_bits]
    (``indexing.selectivity.bloom_bit_freq``; None: no pass-rate gate). Returns (ids [B, K], scores
    [B, K]).

    Eager entry point for tune scripts / parity tests; the compiled path goes through the
    ``@triton_op`` wrappers."""
    cfg = (
        config
        if config is not None
        else tile_for_width(CONFIGS, query.shape[1], query.shape[0], width)
    )
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
        bit_freq=bloom_bit_freq,
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
        cfg=tile_for_width(CONFIGS, query.shape[1], query.shape[0], width),
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
    bloom_bit_freq: Tensor,
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
        cfg=tile_for_width(CONFIGS, query.shape[1], query.shape[0], width),
        bit_freq=bloom_bit_freq,
    )
    wrap_triton(_codesigned_probe_score_kernel)[launch.grid](**launch.kwargs)
    fin = probe_topk(launch, k, probe_ids, cluster_offsets, sort_perm)
    wrap_triton(probe_ids_kernel)[fin.grid](**fin.kwargs)
    return fin.ids, fin.scores
