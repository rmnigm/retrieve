from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


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
    bid = tl.program_id(0)
    tile_id = tl.program_id(1)

    p_off = tile_id * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p_off < P

    d_off = tl.arange(0, D)
    q = tl.load(query_ptr + bid * stride_qb + d_off * stride_qd)

    item_ids = tl.load(
        flat_items_ptr + bid * stride_fb + p_off * stride_fp,
        mask=p_valid,
        other=-1,
    ).to(tl.int64)
    valid = p_valid & (item_ids >= 0)
    safe_ids = tl.where(valid, item_ids, 0)

    keep = valid

    if HAS_QB:
        w_off = tl.arange(0, W)
        qb = tl.load(qb_ptr + bid * stride_qbb + w_off * stride_qbw)
        sigs = tl.load(
            bloom_sigs_ptr
            + safe_ids[:, None] * stride_bn
            + w_off[None, :] * stride_bw,
            mask=valid[:, None],
            other=0,
        )
        masked = qb[None, :] & sigs
        eq_per_word = (masked == qb[None, :]).to(tl.int32)
        all_eq = tl.min(eq_per_word, axis=1)
        bloom_pass = all_eq != 0
        keep = keep & bloom_pass

    if HAS_MASK:
        mvals = tl.load(
            mask_ptr + bid * stride_mb + safe_ids * stride_mn,
            mask=valid,
            other=False,
        )
        keep = keep & mvals

    codes = tl.load(
        item_codes_ptr
        + safe_ids[:, None] * stride_cn
        + d_off[None, :] * stride_cd,
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
        # Dummy 1x1 tensors — pointer never dereferenced (HAS_QB guards the load).
        query_bits = torch.empty(1, 1, dtype=torch.int64, device=query.device)
        bloom_sigs = torch.empty(1, 1, dtype=torch.int64, device=query.device)
        w = 1
    if has_mask:
        mask = mask.contiguous()
    else:
        mask = torch.empty(1, 1, dtype=torch.bool, device=query.device)

    all_scores = torch.full(
        (b, p), float("-inf"), dtype=torch.float32, device=query.device
    )

    grid = lambda meta: (b, triton.cdiv(p, meta["BLOCK_P"]))

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

    if actual_k < k:
        pad = k - actual_k
        topk_ids = torch.cat(
            [
                topk_ids,
                torch.full((b, pad), -1, dtype=torch.long, device=query.device),
            ],
            dim=1,
        )
        topk_scores = torch.cat(
            [
                topk_scores,
                torch.full(
                    (b, pad), float("-inf"), dtype=torch.float32, device=query.device
                ),
            ],
            dim=1,
        )

    return topk_ids, topk_scores
