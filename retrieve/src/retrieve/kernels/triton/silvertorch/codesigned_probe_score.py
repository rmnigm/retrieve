from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# Cached singleton placeholders for absent qb/sigs/mask. The kernel never
# dereferences these pointers (HAS_QB / HAS_MASK gate every load), so one 1x1
# tensor per (device, dtype) is enough — avoids 2-3 fresh allocs per call.
_DUMMIES: dict[tuple[torch.device, torch.dtype], Tensor] = {}


def _dummy(device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (device, dtype)
    t = _DUMMIES.get(key)
    if t is None:
        t = torch.empty(1, 1, dtype=dtype, device=device)
        _DUMMIES[key] = t
    return t


@triton.jit
def _or_combine(a, b):
    return a | b


def _autotune_configs() -> list[triton.Config]:
    configs = []
    for block_p in (32, 64, 128, 256):
        for num_warps in (4, 8):
            configs.append(
                triton.Config(
                    {"BLOCK_P": block_p},
                    num_warps=num_warps,
                    num_stages=3,
                )
            )
    return configs


@triton.autotune(configs=_autotune_configs(), key=["P", "D", "W", "HAS_QB", "HAS_MASK"])
@triton.jit
def _codesigned_probe_score_kernel(
    query_ptr,
    qb_ptr,
    flat_items_ptr,
    item_codes_ptr,
    item_scales_ptr,
    bloom_sigs_ptr,
    mask_ptr,
    out_scores_ptr,
    P: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    stride_qb,
    stride_qd,
    stride_qbb,
    stride_qbw,
    stride_fb,
    stride_fp,
    stride_cn,
    stride_cd,
    stride_sn,
    stride_bn,
    stride_bw,
    stride_mb,
    stride_mn,
    stride_ob,
    stride_op,
    HAS_QB: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # tile on axis-0 (CUDA grid_x ≤ 2^31), batch on axis-1 (grid_y ≤ 65535):
    # n_probe × max_cluster_size can be millions, which would overflow grid_y.
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)

    p_off = tile_id * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p_off < P

    d_off = tl.arange(0, D)
    q = tl.load(query_ptr + bid * stride_qb + d_off * stride_qd)

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

    if HAS_MASK:
        mvals = tl.load(
            mask_ptr + bid * stride_mb + safe_ids * stride_mn,
            mask=valid,
            other=False,
        )
        keep = keep & mvals

    codes = tl.load(
        item_codes_ptr + safe_ids[:, None] * stride_cn + d_off[None, :] * stride_cd,
        mask=keep[:, None],
        other=0,
    ).to(tl.float32)

    scales = tl.load(
        item_scales_ptr + safe_ids * stride_sn,
        mask=keep,
        other=0.0,
    )

    dots = tl.sum(codes * q[None, :], axis=1) * scales
    dots = tl.where(keep, dots, float("-inf"))

    tl.store(
        out_scores_ptr + bid * stride_ob + p_off * stride_op,
        dots,
        mask=p_valid,
    )


@triton.autotune(configs=_autotune_configs(), key=["P", "D", "W", "HAS_QB", "HAS_MASK"])
@triton.jit
def _codesigned_probe_score_fp32_kernel(
    query_ptr,
    qb_ptr,
    flat_items_ptr,
    item_embs_ptr,
    bloom_sigs_ptr,
    mask_ptr,
    out_scores_ptr,
    P: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    stride_qb,
    stride_qd,
    stride_qbb,
    stride_qbw,
    stride_fb,
    stride_fp,
    stride_en,
    stride_ed,
    stride_bn,
    stride_bw,
    stride_mb,
    stride_mn,
    stride_ob,
    stride_op,
    HAS_QB: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # tile on axis-0 (CUDA grid_x ≤ 2^31), batch on axis-1 (grid_y ≤ 65535):
    # n_probe × max_cluster_size can be millions, which would overflow grid_y.
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)

    p_off = tile_id * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p_off < P

    d_off = tl.arange(0, D)
    q = tl.load(query_ptr + bid * stride_qb + d_off * stride_qd)

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

    if HAS_MASK:
        mvals = tl.load(
            mask_ptr + bid * stride_mb + safe_ids * stride_mn,
            mask=valid,
            other=False,
        )
        keep = keep & mvals

    embs = tl.load(
        item_embs_ptr + safe_ids[:, None] * stride_en + d_off[None, :] * stride_ed,
        mask=keep[:, None],
        other=0.0,
    )

    dots = tl.sum(embs * q[None, :], axis=1)
    dots = tl.where(keep, dots, float("-inf"))

    tl.store(
        out_scores_ptr + bid * stride_ob + p_off * stride_op,
        dots,
        mask=p_valid,
    )


def codesigned_probe_score(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    item_scales: Tensor,
    k: int,
    *,
    query_bits: Tensor | None = None,
    bloom_sigs: Tensor | None = None,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused phase-2+3 of SilverTorch's co-designed ANN+bloom (Algorithm 1).

    For each (query, probed-item) cell: optionally evaluate the bloom subset
    test ``(qb & sig) == qb`` against the item's bloom signature, optionally
    AND with an external mask, and (if both passes) score
    ``(item_codes[id].float() @ q) * item_scales[id]``. Items failing either
    filter or with id == -1 (cluster padding) get score ``-inf``.

    The bloom intermediate (``[B, P, W]`` sigs / bool match) and the int8→fp32
    code cast (``[B, P, D]``) never touch HBM — they live in registers/SRAM.

    Inputs:
        query:             [B, D]  fp32
        flat_probed_items: [B, P]  int64 (-1 padding for empty cluster slots)
        item_codes:        [N, D]  int8
        item_scales:       [N]     fp32
        query_bits:        [B, W]  int64, optional (skip bloom filter when None)
        bloom_sigs:        [N, W]  int64, required iff query_bits is not None
        mask:              [B, N]  bool, optional external mask

    Returns ``(ids[B, K], scores[B, K])``; pads with ``-1`` / ``-inf`` when
    fewer than K candidates pass.
    """
    if query.dim() != 2 or flat_probed_items.dim() != 2:
        raise ValueError("query must be [B, D] and flat_probed_items [B, P]")
    b, d = query.shape
    p = flat_probed_items.shape[1]

    if p == 0:
        return (
            torch.full((b, k), -1, dtype=torch.long, device=query.device),
            torch.full((b, k), float("-inf"), dtype=torch.float32, device=query.device),
        )

    has_qb = query_bits is not None
    if has_qb and bloom_sigs is None:
        raise ValueError("bloom_sigs is required when query_bits is provided")
    has_mask = mask is not None

    query = query.contiguous()
    flat_probed_items = flat_probed_items.contiguous()
    item_codes = item_codes.contiguous()
    item_scales = item_scales.contiguous()
    if has_qb:
        query_bits = query_bits.contiguous()
        bloom_sigs = bloom_sigs.contiguous()
        w = query_bits.shape[1]
    else:
        # Cached singleton dummies — pointer never dereferenced (HAS_QB guards
        # the load), so one 1x1 tensor per (device, dtype) is reused.
        query_bits = _dummy(query.device, torch.int64)
        bloom_sigs = _dummy(query.device, torch.int64)
        w = 1
    if has_mask:
        mask = mask.contiguous()
    else:
        mask = _dummy(query.device, torch.bool)

    # `torch.empty` is safe: the kernel writes every slot in [0, P) — either a
    # real dot product or -inf for filtered/padding lanes — so downstream topk
    # sees deterministic values.
    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)

    # Tile axis on grid_x (≤ 2^31) since num_tiles can exceed grid_y/grid_z's
    # 65535 limit at large n_probe × max_cluster_size.
    grid = lambda meta: (triton.cdiv(p, meta["BLOCK_P"]), b)

    _codesigned_probe_score_kernel[grid](
        query,
        query_bits,
        flat_probed_items,
        item_codes,
        item_scales,
        bloom_sigs,
        mask,
        all_scores,
        P=p,
        D=d,
        W=w,
        stride_qb=query.stride(0),
        stride_qd=query.stride(1),
        stride_qbb=query_bits.stride(0),
        stride_qbw=query_bits.stride(1),
        stride_fb=flat_probed_items.stride(0),
        stride_fp=flat_probed_items.stride(1),
        stride_cn=item_codes.stride(0),
        stride_cd=item_codes.stride(1),
        stride_sn=item_scales.stride(0),
        stride_bn=bloom_sigs.stride(0),
        stride_bw=bloom_sigs.stride(1),
        stride_mb=mask.stride(0),
        stride_mn=mask.stride(1),
        stride_ob=all_scores.stride(0),
        stride_op=all_scores.stride(1),
        HAS_QB=has_qb,
        HAS_MASK=has_mask,
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


def codesigned_probe_score_fp32(
    query: Tensor,
    flat_probed_items: Tensor,
    item_embs: Tensor,
    k: int,
    *,
    query_bits: Tensor | None = None,
    bloom_sigs: Tensor | None = None,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused phase-2+3 of SilverTorch's co-designed ANN+bloom — fp32 variant.

    Identical to ``codesigned_probe_score`` but scores fp32 item embeddings
    directly: no int8 codes, no per-item scales, no dequant cast.
    """
    if query.dim() != 2 or flat_probed_items.dim() != 2:
        raise ValueError("query must be [B, D] and flat_probed_items [B, P]")
    b, d = query.shape
    p = flat_probed_items.shape[1]

    if p == 0:
        return (
            torch.full((b, k), -1, dtype=torch.long, device=query.device),
            torch.full((b, k), float("-inf"), dtype=torch.float32, device=query.device),
        )

    has_qb = query_bits is not None
    if has_qb and bloom_sigs is None:
        raise ValueError("bloom_sigs is required when query_bits is provided")
    has_mask = mask is not None

    query = query.contiguous()
    flat_probed_items = flat_probed_items.contiguous()
    item_embs = item_embs.contiguous()
    if has_qb:
        query_bits = query_bits.contiguous()
        bloom_sigs = bloom_sigs.contiguous()
        w = query_bits.shape[1]
    else:
        query_bits = _dummy(query.device, torch.int64)
        bloom_sigs = _dummy(query.device, torch.int64)
        w = 1
    if has_mask:
        mask = mask.contiguous()
    else:
        mask = _dummy(query.device, torch.bool)

    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)

    # Tile axis on grid_x (≤ 2^31) since num_tiles can exceed grid_y/grid_z's
    # 65535 limit at large n_probe × max_cluster_size.
    grid = lambda meta: (triton.cdiv(p, meta["BLOCK_P"]), b)

    _codesigned_probe_score_fp32_kernel[grid](
        query,
        query_bits,
        flat_probed_items,
        item_embs,
        bloom_sigs,
        mask,
        all_scores,
        P=p,
        D=d,
        W=w,
        stride_qb=query.stride(0),
        stride_qd=query.stride(1),
        stride_qbb=query_bits.stride(0),
        stride_qbw=query_bits.stride(1),
        stride_fb=flat_probed_items.stride(0),
        stride_fp=flat_probed_items.stride(1),
        stride_en=item_embs.stride(0),
        stride_ed=item_embs.stride(1),
        stride_bn=bloom_sigs.stride(0),
        stride_bw=bloom_sigs.stride(1),
        stride_mb=mask.stride(0),
        stride_mn=mask.stride(1),
        stride_ob=all_scores.stride(0),
        stride_op=all_scores.stride(1),
        HAS_QB=has_qb,
        HAS_MASK=has_mask,
    )

    actual_k = min(k, p)
    topk_scores, topk_local = torch.topk(all_scores, actual_k, dim=1)
    topk_ids = flat_probed_items.gather(1, topk_local)

    if actual_k == k:
        return topk_ids, topk_scores

    out_ids = torch.full((b, k), -1, dtype=torch.long, device=query.device)
    out_scores = torch.full((b, k), float("-inf"), dtype=torch.float32, device=query.device)
    out_ids[:, :actual_k] = topk_ids
    out_scores[:, :actual_k] = topk_scores
    return out_ids, out_scores
