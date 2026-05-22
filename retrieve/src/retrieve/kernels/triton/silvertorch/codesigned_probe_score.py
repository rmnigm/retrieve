from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import custom_op

from retrieve.layers.utils.quantize import quantize_int8


@triton.jit
def _or_combine(a, b):
    return a | b


@dataclass(frozen=True)
class CodesignedProbeScoreConfig:
    block_p: int
    num_warps: int
    num_stages: int = 3


# Single default the library ships with. Re-tune on a new arch by
# running ``evaluation/scripts/tune_kernels.py`` and pasting the
# resulting line in. Callers who want a different tile config pass
# ``config=`` through to ``_codesigned_probe_score_impl``.
# Tuned on A100 (sm_80) against the int8×int8 ``tl.dot`` path:
# block_p=256 wins plurality (3/6 regimes); num_warps=4 wins all 6.
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
):
    # tile on axis-0 (CUDA grid_x ≤ 2^31), batch on axis-1 (grid_y ≤ 65535):
    # n_probe × max_cluster_size can be millions, which would overflow grid_y.
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)

    p_off = tile_id * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p_off < P

    d_off = tl.arange(0, D)
    # Query is pre-quantized in the wrapper — symmetric per-row int8 with a
    # fp32 scale, paying one amax + div per batch row outside the loop.
    # Quantizing inside the kernel would recompute the amax in every
    # (P_tile, bid) program, redundant by a factor of cdiv(P, BLOCK_P).
    q_codes = tl.load(q_codes_ptr + bid * stride_qcb + d_off * stride_qcd)
    q_scale = tl.load(q_scales_ptr + bid * stride_qs)

    item_ids = tl.load(
        flat_items_ptr + bid * stride_fb + p_off * stride_fp,
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
        # (qb & sig) == qb  ⇔  qb & ~sig == 0 (per word). OR-reduce → 0 iff all
        # words pass — saves the int32 cast + min reduction the equality form
        # required.
        diff = qb[None, :] & ~sigs
        any_diff = tl.reduce(diff, axis=1, combine_fn=_or_combine)
        bloom_pass = any_diff == 0
        keep = keep & bloom_pass

    codes = tl.load(
        item_codes_ptr + safe_ids[:, None] * stride_cn + d_off[None, :] * stride_cd,
        mask=keep[:, None],
        other=0,
    )

    # int8 × int8 → int32 matmul (paper §4.2). Shapes: q[1, D] @ codes^T[D, BLOCK_P]
    # → out[1, BLOCK_P] int32. With M=1 (single query row per program) IMMA
    # tensor cores can't fit, so Triton lowers this to the dp4a / int8 CUDA-core
    # path — the exact instruction the paper claims. Win over the old
    # dequant-then-fp32 design: 4× less code bandwidth (int8 stays narrow
    # through the reduction) and dp4a's 4-mul-add throughput per CUDA core.
    #
    # To trade global-scale quality loss for an [N] gather, pass per-item
    # ``item_scales`` instead of ``global_scale`` and replace the final
    # ``* global_scale`` with ``* tl.load(item_scales_ptr + safe_ids ...)``.
    q_codes_2d = q_codes[None, :]
    codes_T = tl.trans(codes)
    dots_2d = tl.dot(q_codes_2d, codes_T, out_dtype=tl.int32)
    # Squeeze the length-1 M axis. ``tl.sum`` over length-1 is the standard
    # Triton idiom — no reshape primitive for dropping a dim.
    dots_i32 = tl.sum(dots_2d, axis=0)

    # Two scalar fp32 multiplies per item: per-row q_scale and the global
    # tensor scale. With per-item scales this would be three.
    dots = dots_i32.to(tl.float32) * q_scale * global_scale
    dots = tl.where(keep, dots, float("-inf"))

    tl.store(
        out_scores_ptr + bid * stride_ob + p_off * stride_op,
        dots,
        mask=p_valid,
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
    """Fused phase-2+3 of SilverTorch's co-designed ANN+bloom (Algorithm 1).

    Paper-faithful int8 ANN scoring: both query and items stay in int8 through
    the dot product (int32 accumulator), with one global per-tensor scale and
    a per-row query scale collapsed in the dequant epilogue. The bloom subset
    test is fused into the same kernel — items that fail get ``-inf`` without
    a separate scratch pass.

    For each (query, probed-item) cell: optionally evaluate
    ``(qb & sig) == qb`` against the item's bloom signature, and (if it
    passes) score ``(item_codes[id] · q_codes[b])_i32 * q_scale[b] * global_scale``.
    Items failing the filter or with id == -1 (cluster padding) get score
    ``-inf``.

    The bloom intermediate (``[B, P, W]`` sigs / bool match) and the int8
    item-code tile (``[B, P, D]``) never touch HBM — they live in registers/SRAM.

    Inputs:
        query:             [B, D]  fp32 — quantized to int8 in the wrapper.
        flat_probed_items: [B, P]  int64 (-1 padding for empty cluster slots).
        item_codes:        [N, D]  int8 — symmetric per-tensor codes.
        global_scale:      Python float — single scale, paired with item_codes.
        query_bits:        [B, W]  int64, optional (skip bloom filter when None).
        bloom_sigs:        [N, W]  int64, required iff query_bits is not None.

    Returns ``(ids[B, K], scores[B, K])``; pads with ``-1`` / ``-inf`` when
    fewer than K candidates pass.

    Internal tune/test entry point — production callers go through the
    ``@custom_op`` wrapper ``codesigned_probe_score`` below, which always
    passes concrete tensors + ``has_bloom`` and ``config=None``.

    **Quality knob.** Global scale is paper-default; for higher recall on
    non-uniform-norm indexes, switch to per-item scales (``quantize_int8`` in
    place of ``quantize_int8_global``) and add an ``item_scales [N]`` arg —
    the kernel needs a ``tl.load(item_scales_ptr + safe_ids ...)`` plus a
    third scalar multiply in the epilogue. See bench results in
    ``_agent_scratch/bench_results/silvertorch_int8mm_quality.json``.
    """
    if query.dim() != 2 or flat_probed_items.dim() != 2:
        raise ValueError("query must be [B, D] and flat_probed_items [B, P]")
    if item_codes.dtype != torch.int8:
        raise TypeError(f"item_codes must be int8, got {item_codes.dtype}")
    b, d = query.shape
    p = flat_probed_items.shape[1]

    has_qb = query_bits is not None
    if has_qb and bloom_sigs is None:
        raise ValueError("bloom_sigs is required when query_bits is provided")

    # Per-batch query int8 quantization. One amax + scalar div per row.
    q_codes, q_scales = quantize_int8(query)
    q_codes = q_codes.contiguous()
    q_scales = q_scales.contiguous()

    flat_probed_items = flat_probed_items.contiguous()
    item_codes = item_codes.contiguous()
    if has_qb:
        query_bits = query_bits.contiguous()
        bloom_sigs = bloom_sigs.contiguous()
        w = query_bits.shape[1]
    else:
        # 1×1 int64 placeholders so Triton has a valid pointer to bind.
        # ``HAS_QB`` constexpr gates every load, so the pointers are never
        # dereferenced. Production callers (the ``SilverTorch`` layer) pass
        # layer-owned dummies through the ``@custom_op`` wrapper, avoiding
        # this alloc on the cudagraph path — this branch fires only when
        # tune scripts / parity tests call ``_impl`` with ``query_bits=None``.
        query_bits = torch.empty(1, 1, dtype=torch.int64, device=query.device)
        bloom_sigs = torch.empty(1, 1, dtype=torch.int64, device=query.device)
        w = 1

    # `torch.empty` is safe: the kernel writes every slot in [0, P) — either a
    # real dot product or -inf for filtered/padding lanes — so downstream topk
    # sees deterministic values.
    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)

    cfg = config if config is not None else DEFAULT_CONFIG

    # Tile axis on grid_x (≤ 2^31) since num_tiles can exceed grid_y/grid_z's
    # 65535 limit at large n_probe × max_cluster_size.
    grid = (triton.cdiv(int(p), cfg.block_p), int(b))

    _codesigned_probe_score_kernel[grid](
        q_codes,
        q_scales,
        query_bits,
        flat_probed_items,
        item_codes,
        bloom_sigs,
        all_scores,
        float(global_scale),
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
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    actual_k = min(k, p)
    topk_scores, topk_local = torch.topk(all_scores, actual_k, dim=1)
    topk_ids = flat_probed_items.gather(1, topk_local)

    if actual_k == k:
        return topk_ids, topk_scores

    # P < K: pad to the requested width with sentinel rows. Pre-allocate once
    # and slice-assign instead of cat'ing — 2 allocs vs 4.
    out_ids = torch.full((b, k), -1, dtype=torch.long, device=query.device)
    out_scores = torch.full((b, k), float("-inf"), dtype=torch.float32, device=query.device)
    out_ids[:, :actual_k] = topk_ids
    out_scores[:, :actual_k] = topk_scores
    return out_ids, out_scores


@custom_op("retrieve::codesigned_probe_score", mutates_args=())
def codesigned_probe_score(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Plain int8 ANN scoring — no attribute filter. Registered as an opaque
    ``custom_op`` so dynamo can stitch the caller's compiled forward into a
    single cudagraph (no graph break per call). The ``config=`` keyword is
    dropped because ``custom_op``'s schema inference doesn't accept dataclass
    args; tune scripts and parity tests that need a non-default config call
    ``_codesigned_probe_score_impl`` directly.
    """
    return _codesigned_probe_score_impl(
        query, flat_probed_items, item_codes, global_scale, k, config=None
    )


@codesigned_probe_score.register_fake
def _codesigned_probe_score_fake(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    b = query.shape[0]
    device = query.device
    ids = torch.empty((b, k), dtype=torch.long, device=device)
    scores = torch.empty((b, k), dtype=torch.float32, device=device)
    return ids, scores


@custom_op("retrieve::codesigned_probe_score_bloom", mutates_args=())
def codesigned_probe_score_bloom(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    query_bits: Tensor,
    bloom_sigs: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Int8 ANN scoring fused with the paper's bloom subset test.

    Sibling of ``codesigned_probe_score`` for the bloom-filtered path. Split
    into a separate ``custom_op`` (rather than one op with an
    ``Optional[Tensor]`` / ``has_bloom`` flag + dummies) so the layer just
    routes to the right op — no dummy buffers, no constexpr-flag plumbing.
    Mirrors the ``oporp_1bit_match_topk_full`` / ``_indirect`` split in linr.
    """
    return _codesigned_probe_score_impl(
        query, flat_probed_items, item_codes, global_scale, k,
        query_bits=query_bits, bloom_sigs=bloom_sigs, config=None,
    )


@codesigned_probe_score_bloom.register_fake
def _codesigned_probe_score_bloom_fake(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    query_bits: Tensor,
    bloom_sigs: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    b = query.shape[0]
    device = query.device
    ids = torch.empty((b, k), dtype=torch.long, device=device)
    scores = torch.empty((b, k), dtype=torch.float32, device=device)
    return ids, scores
