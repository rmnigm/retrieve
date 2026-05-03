from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


def _autotune_configs() -> list[triton.Config]:
    configs = []
    for block_n in (32, 64, 128, 256):
        for num_warps in (4, 8):
            configs.append(
                triton.Config(
                    {"BLOCK_N": block_n},
                    num_warps=num_warps,
                    num_stages=3,
                )
            )
    return configs


@triton.autotune(configs=_autotune_configs(), key=["P", "D"])
@triton.jit
def _fused_masked_knn_topk_kernel(
    query_ptr,
    item_embs_ptr,
    pos_indices_ptr,
    counts_ptr,
    out_scores_ptr,
    P: tl.constexpr,
    D: tl.constexpr,
    stride_qb,
    stride_qd,
    stride_in,
    stride_id,
    stride_pb,
    stride_pp,
    stride_sb,
    stride_sp,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(1)

    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, D)

    p_valid = n_offsets < P

    count = tl.load(counts_ptr + bid)
    in_count = n_offsets < count

    q = tl.load(query_ptr + bid * stride_qb + d_offsets * stride_qd)

    item_ids = tl.load(
        pos_indices_ptr + bid * stride_pb + n_offsets * stride_pp,
        mask=p_valid,
        other=0,
    ).to(tl.int64)

    emb_rows = tl.load(
        item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
        mask=in_count[:, None],
        other=0.0,
    )

    dots = tl.sum(emb_rows * q[None, :], axis=1)
    dots = tl.where(in_count, dots, float("-inf"))

    tl.store(
        out_scores_ptr + bid * stride_sb + n_offsets * stride_sp,
        dots,
        mask=p_valid,
    )


def fused_masked_knn_topk(
    query: Tensor,
    item_embs: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Fused masked gather + dot + topk. Returns ``(ids[B, K], scores[B, K])``.

    2-D launch over ``(B, cdiv(P, BLOCK_N))``. Each program holds one query
    in registers and gathers a ``BLOCK_N``-wide slab of candidate item ids
    via indirect load. Items past ``counts[b]`` get score ``-inf``. The
    post-kernel ``torch.topk`` selects K from the ``[B, P]`` score buffer.

    Per-cell scoring is elementwise (``tl.sum``) rather than ``tl.dot``: the
    gathered rows differ per (B, p) cell so a true GEMM would re-load rows
    per query column. For the dense (high pass rate) path with no
    pre-filter, callers should fall back to ``query @ item_embs.T`` +
    ``torch.topk`` — there's no fused-kernel equivalent because cuBLAS +
    CUB already cover that case.
    """
    if query.dim() != 2 or item_embs.dim() != 2:
        raise ValueError("query must be [B, D] and item_embs [N, D]")
    b, d = query.shape
    p = positive_indices.shape[1]

    if p == 0:
        return (
            torch.full((b, k), -1, dtype=torch.long, device=query.device),
            torch.full((b, k), float("-inf"), dtype=torch.float32, device=query.device),
        )

    query = query.contiguous()
    item_embs = item_embs.contiguous()
    positive_indices = positive_indices.contiguous()
    counts = counts.contiguous()

    all_scores = torch.full((b, p), float("-inf"), dtype=torch.float32, device=query.device)

    grid = lambda meta: (b, triton.cdiv(p, meta["BLOCK_N"]))

    _fused_masked_knn_topk_kernel[grid](
        query,
        item_embs,
        positive_indices,
        counts,
        all_scores,
        P=p,
        D=d,
        stride_qb=query.stride(0),
        stride_qd=query.stride(1),
        stride_in=item_embs.stride(0),
        stride_id=item_embs.stride(1),
        stride_pb=positive_indices.stride(0),
        stride_pp=positive_indices.stride(1),
        stride_sb=all_scores.stride(0),
        stride_sp=all_scores.stride(1),
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
                torch.full((b, pad), float("-inf"), dtype=torch.float32, device=query.device),
            ],
            dim=1,
        )

    return topk_ids, topk_scores
