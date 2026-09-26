from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.indexing.quantize import quantize_int8
from retrieve.ops.triton._host import (
    ProbeLaunch,
    check_contiguous,
    check_pow2,
    probe_finish,
    wide,
)
from retrieve.ops.triton.common import bloom_subset_pass, row_base


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
    qb_ptr,
    flat_items_ptr,
    item_codes_ptr,
    bloom_sigs_ptr,
    out_scores_ptr,
    global_scale,
    P: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    stride_qcb,
    stride_qcd,
    stride_qs,
    stride_qbb,
    stride_qbw,
    stride_fb,
    stride_fp,
    stride_cn,
    stride_cd,
    stride_bn,
    stride_bw,
    stride_ob,
    stride_op,
    HAS_QB: tl.constexpr,
    BLOCK_P: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Tile on grid_x (≤ 2³¹), batch on grid_y (≤ 65535): P = n_probe × max_cluster_size can overflow
    # grid_y.
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)

    p_off = tile_id * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p_off < P

    d_off = tl.arange(0, D)
    # Query is pre-quantized in the wrapper (per-row int8 + fp32 scale); quantizing in-kernel would
    # redundantly recompute amax per P-tile.
    q_codes = tl.load(q_codes_ptr + bid * stride_qcb + d_off * stride_qcd)
    q_scale = tl.load(q_scales_ptr + bid * stride_qs)

    item_ids = tl.load(
        row_base(flat_items_ptr, bid, stride_fb, WIDE) + p_off * stride_fp,
        mask=p_valid,
        other=-1,
    ).to(tl.int64)
    # Masked load returned -1 for OOB lanes, so id>=0 already implies p_valid.
    valid = item_ids >= 0
    safe_ids = tl.where(valid, item_ids, 0)

    keep = valid

    if HAS_QB:
        w_off = tl.arange(0, W)
        qb = tl.load(qb_ptr + bid * stride_qbb + w_off * stride_qbw)
        sigs = tl.load(
            bloom_sigs_ptr + safe_ids[:, None] * stride_bn + w_off[None, :] * stride_bw,
            mask=valid[:, None],
            other=0,
        )
        # Shared subset test (qb & ~sig OR-reduce form); AND with `keep` supplies the
        # caller-side validity mask the helper contract requires.
        keep = keep & bloom_subset_pass(qb, sigs)

    codes = tl.load(
        item_codes_ptr + safe_ids[:, None] * stride_cn + d_off[None, :] * stride_cd,
        mask=keep[:, None],
        other=0,
    )

    # int8 × int8 → int32 (paper §4.2): q[1,D] @ codes^T[D,BLOCK_P]. M=1 can't use IMMA tensor
    # cores, so Triton lowers to the dp4a int8 path the paper claims.
    q_codes_2d = q_codes[None, :]
    codes_T = tl.trans(codes)
    dots_2d = tl.dot(q_codes_2d, codes_T, out_dtype=tl.int32)
    # Squeeze the length-1 M axis: tl.sum over length-1 (Triton has no reshape to drop a dim).
    dots_i32 = tl.sum(dots_2d, axis=0)

    # fp32 pinned: Inductor passes global_scale as a Python float (kernels.md § Numerics).
    dots = (
        dots_i32.to(tl.float32) * tl.cast(q_scale, tl.float32) * tl.cast(global_scale, tl.float32)
    )
    dots = tl.where(keep, dots, float("-inf"))

    tl.store(
        row_base(out_scores_ptr, bid, stride_ob, WIDE) + p_off * stride_op,
        dots,
        mask=p_valid,
    )


def _cps_prep(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    *,
    query_bits: Tensor | None,
    bloom_sigs: Tensor | None,
    cfg: CodesignedProbeScoreConfig,
) -> ProbeLaunch:
    """Validation + contiguity + buffers + the full launch-kwarg dict.

    THE single place input checking happens — shared by ``_codesigned_probe_score_impl`` and
    both ``@triton_op`` wrappers (which keep only their textually-inline ``wrap_triton``
    launch)."""
    if query.dim() != 2 or flat_probed_items.dim() != 2:
        raise ValueError("query must be [B, D] and flat_probed_items [B, P]")
    if item_codes.dtype != torch.int8:
        raise TypeError(f"item_codes must be int8, got {item_codes.dtype}")
    has_qb = query_bits is not None
    if has_qb and bloom_sigs is None:
        raise ValueError("bloom_sigs is required when query_bits is provided")

    b, d = query.shape
    p = flat_probed_items.shape[1]
    check_contiguous(item_codes=item_codes, flat_probed_items=flat_probed_items)
    check_pow2(D=d)
    if has_qb:
        check_contiguous(bloom_sigs=bloom_sigs)
        check_pow2(W=query_bits.shape[1])

    q_codes, q_scales = quantize_int8(query)
    q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
    if has_qb:
        query_bits = query_bits.contiguous()
        w = query_bits.shape[1]
    else:
        # HAS_QB=False gates every load through these pointers, so any int64 tensor stands in
        # (review D3: no per-call allocation).
        query_bits = bloom_sigs = flat_probed_items
        w = 1

    # torch.empty is safe: the kernel writes every slot in [0, P) (real dot or -inf), so topk sees
    # deterministic values.
    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)

    kwargs: dict[str, object] = dict(
        q_codes_ptr=q_codes,
        q_scales_ptr=q_scales,
        qb_ptr=query_bits,
        flat_items_ptr=flat_probed_items,
        item_codes_ptr=item_codes,
        bloom_sigs_ptr=bloom_sigs,
        out_scores_ptr=all_scores,
        global_scale=float(global_scale),
        P=p,
        D=d,
        W=w,
        stride_qcb=q_codes.stride(0),
        stride_qcd=q_codes.stride(1),
        stride_qs=q_scales.stride(0),
        stride_qbb=query_bits.stride(0),
        stride_qbw=query_bits.stride(1),
        stride_fb=flat_probed_items.stride(0),
        stride_fp=flat_probed_items.stride(1),
        stride_cn=item_codes.stride(0),
        stride_cd=item_codes.stride(1),
        stride_bn=bloom_sigs.stride(0),
        stride_bw=bloom_sigs.stride(1),
        stride_ob=all_scores.stride(0),
        stride_op=all_scores.stride(1),
        HAS_QB=has_qb,
        BLOCK_P=cfg.block_p,
        WIDE=wide(flat_probed_items, all_scores),
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return ProbeLaunch(
        p=p, b=b, kwargs=kwargs, all_scores=all_scores, flat_probed_items=flat_probed_items
    )


def _codesigned_probe_score_impl(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
    *,
    query_bits: Tensor | None = None,
    bloom_sigs: Tensor | None = None,
    config: CodesignedProbeScoreConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused phase-2+3 of SilverTorch's co-designed int8 ANN + optional bloom filter (paper
    Algorithm 1, §4.2): query and items stay int8 through an int32-accumulated dot, dequantized
    by ``q_scale[b] * global_scale``; the bloom subset test ``(qb & sig) == qb`` is fused in
    (failing or padding items score ``-inf``). The ``[B, P, W]`` sigs and ``[B, P, D]`` code tile
    never touch HBM.

    Inputs: query [B, D] fp32 (int8-quantized in the wrapper), flat_probed_items [B, P] int64 (-1
    pad), item_codes [N, D] int8, global_scale float, query_bits [B, W] int64 (optional; skips
    bloom when None), bloom_sigs [N, W] int64 (required iff query_bits given). Returns (ids [B,
    K], scores [B, K]); requires P >= k.

    Eager entry point for tune scripts / parity tests; the compiled path goes through the
    ``@triton_op`` wrappers."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _cps_prep(
        query,
        flat_probed_items,
        item_codes,
        global_scale,
        query_bits=query_bits,
        bloom_sigs=bloom_sigs,
        cfg=cfg,
    )
    # grid_x tiles P (≤ 2³¹); P can exceed grid_y's 65535 limit.
    grid = (triton.cdiv(int(launch.p), cfg.block_p), int(launch.b))
    _codesigned_probe_score_kernel[grid](**launch.kwargs)
    return probe_finish(launch, k)


@triton_op("retrieve::codesigned_probe_score", mutates_args=())
def codesigned_probe_score(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Plain int8 ANN scoring (no attribute filter); shares ``_cps_prep``/``probe_finish`` with
    ``_codesigned_probe_score_impl``, keeping the launch inline (``wrap_triton`` must appear
    textually in the decorated source for torch.export). Requires P >= k."""
    launch = _cps_prep(
        query,
        flat_probed_items,
        item_codes,
        global_scale,
        query_bits=None,
        bloom_sigs=None,
        cfg=DEFAULT_CONFIG,
    )
    p, b = launch.p, launch.b

    def grid(meta):
        return (triton.cdiv(p, meta["BLOCK_P"]), b)

    wrap_triton(_codesigned_probe_score_kernel)[grid](**launch.kwargs)
    return probe_finish(launch, k)


@triton_op("retrieve::codesigned_probe_score_bloom", mutates_args=())
def codesigned_probe_score_bloom(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    query_bits: Tensor,
    bloom_sigs: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Int8 ANN scoring fused with the paper's bloom subset test — sibling of
    ``codesigned_probe_score``, split into a separate op (not one op with an Optional/flag) so
    the layer just routes to the right op."""
    launch = _cps_prep(
        query,
        flat_probed_items,
        item_codes,
        global_scale,
        query_bits=query_bits,
        bloom_sigs=bloom_sigs,
        cfg=DEFAULT_CONFIG,
    )
    p, b = launch.p, launch.b

    def grid(meta):
        return (triton.cdiv(p, meta["BLOCK_P"]), b)

    wrap_triton(_codesigned_probe_score_kernel)[grid](**launch.kwargs)
    return probe_finish(launch, k)
