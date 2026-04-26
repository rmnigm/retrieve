from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _int8_ann_fused_kernel(
    query_ptr,
    codes_ptr,
    scales_ptr,
    pos_indices_ptr,
    counts_ptr,
    out_scores_ptr,
    P: tl.constexpr,
    D: tl.constexpr,
    stride_qb,
    stride_qd,
    stride_cn,
    stride_cd,
    stride_sn,
    stride_pb,
    stride_pp,
    stride_ob,
    stride_op,
    BLOCK_P: tl.constexpr,
):
    bid = tl.program_id(0)
    count = tl.load(counts_ptr + bid)

    d_off = tl.arange(0, D)
    q = tl.load(query_ptr + bid * stride_qb + d_off * stride_qd)

    for block_start in range(0, P, BLOCK_P):
        p_off = block_start + tl.arange(0, BLOCK_P)
        valid = p_off < count

        item_ids = tl.load(
            pos_indices_ptr + bid * stride_pb + p_off * stride_pp,
            mask=valid,
            other=0,
        ).to(tl.int64)

        codes = tl.load(
            codes_ptr + item_ids[:, None] * stride_cn + d_off[None, :] * stride_cd,
            mask=valid[:, None],
            other=0,
        ).to(tl.float32)

        scales = tl.load(
            scales_ptr + item_ids * stride_sn,
            mask=valid,
            other=0.0,
        )

        dots = tl.sum(codes * q[None, :], axis=1) * scales
        dots = tl.where(valid, dots, float("-inf"))

        out_valid = p_off < P
        tl.store(
            out_scores_ptr + bid * stride_ob + p_off * stride_op,
            dots,
            mask=out_valid,
        )


def int8_ann_fused(
    query: Tensor,
    item_codes: Tensor,
    item_scales: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Fused int8 dequant-dot + mask + top-K over a sparse candidate set.

    Inputs:
        query: [B, D] fp32
        item_codes: [N, D] int8
        item_scales: [N] fp32
        positive_indices: [B, P] int64 (-1 padded; counts gives per-row valid length)
        counts: [B] int64 — number of valid candidates in each row of positive_indices.

    Returns (ids[B, K], scores[B, K]) — K is min(k, P) padded back to k with -1/-inf.
    """
    b, d = query.shape
    p = positive_indices.shape[1]
    if p == 0:
        return (
            torch.full((b, k), -1, dtype=torch.long, device=query.device),
            torch.full((b, k), float("-inf"), dtype=torch.float32, device=query.device),
        )

    all_scores = torch.full((b, p), float("-inf"), dtype=torch.float32, device=query.device)
    block_p = 128 if p >= 128 else triton.next_power_of_2(p)

    _int8_ann_fused_kernel[(b,)](
        query,
        item_codes,
        item_scales,
        positive_indices,
        counts,
        all_scores,
        P=p,
        D=d,
        stride_qb=query.stride(0),
        stride_qd=query.stride(1),
        stride_cn=item_codes.stride(0),
        stride_cd=item_codes.stride(1),
        stride_sn=item_scales.stride(0),
        stride_pb=positive_indices.stride(0),
        stride_pp=positive_indices.stride(1),
        stride_ob=all_scores.stride(0),
        stride_op=all_scores.stride(1),
        BLOCK_P=block_p,
    )

    actual_k = min(k, p)
    topk_scores, topk_local = torch.topk(all_scores, actual_k, dim=1)
    topk_ids = positive_indices.gather(1, topk_local)

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
                    (b, pad),
                    float("-inf"),
                    dtype=torch.float32,
                    device=query.device,
                ),
            ],
            dim=1,
        )
    return topk_ids, topk_scores
