from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


def _autotune_configs() -> list[triton.Config]:
    # BLOCK_M kept >= 16 so tl.dot lands on tensor cores even when B is small
    # (the kernel masks out padded rows). BLOCK_N spans a small range that
    # covers the bench cells (N from 4k to 262k).
    configs = []
    for block_m in (16, 32):
        for block_n in (64, 128, 256):
            for num_warps in (4, 8):
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n},
                        num_warps=num_warps,
                        num_stages=3,
                    )
                )
    return configs


@triton.autotune(configs=_autotune_configs(), key=["B", "N", "D"])
@triton.jit
def _fused_matmul_topk_kernel(
    query_ptr,
    item_embs_ptr,
    mask_ptr,
    out_scores_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    stride_qb,
    stride_qd,
    stride_in,
    stride_id,
    stride_mb,
    stride_mn,
    stride_sb,
    stride_sn,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, D)

    m_valid = m_offsets < B
    n_valid = n_offsets < N

    q_tile = tl.load(
        query_ptr + m_offsets[:, None] * stride_qb + d_offsets[None, :] * stride_qd,
        mask=m_valid[:, None],
        other=0.0,
    )
    item_tile = tl.load(
        item_embs_ptr + n_offsets[:, None] * stride_in + d_offsets[None, :] * stride_id,
        mask=n_valid[:, None],
        other=0.0,
    )

    scores = tl.dot(q_tile, tl.trans(item_tile))

    if HAS_MASK:
        m = tl.load(
            mask_ptr + m_offsets[:, None] * stride_mb + n_offsets[None, :] * stride_mn,
            mask=(m_valid[:, None] & n_valid[None, :]),
            other=False,
        )
        scores = tl.where(m, scores, float("-inf"))

    out_valid = m_valid[:, None] & n_valid[None, :]
    scores = tl.where(out_valid, scores, float("-inf"))

    tl.store(
        out_scores_ptr + m_offsets[:, None] * stride_sb + n_offsets[None, :] * stride_sn,
        scores,
        mask=out_valid,
    )


def fused_matmul_topk(
    query: Tensor,
    item_embs: Tensor,
    k: int,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused matmul + optional mask + topk. Returns ``(ids[B, K], scores[B, K])``.

    Tile-parallel 2-D launch over (B, N), tensor-core matmul via ``tl.dot``,
    autotuned over ``(BLOCK_M, BLOCK_N, num_warps, num_stages)``. The
    post-kernel ``torch.topk`` selects K from the ``[B, N]`` score buffer.
    """
    if query.dim() != 2 or item_embs.dim() != 2:
        raise ValueError("query must be [B, D] and item_embs [N, D]")
    if query.shape[1] != item_embs.shape[1]:
        raise ValueError(
            f"query D={query.shape[1]} != item_embs D={item_embs.shape[1]}"
        )
    b, d = query.shape
    n = item_embs.shape[0]

    if d & (d - 1) != 0:
        raise ValueError(f"D must be a power of 2 for tl.dot, got {d}")

    query = query.contiguous()
    item_embs = item_embs.contiguous()
    if mask is not None:
        mask = mask.contiguous()

    all_scores = torch.empty(b, n, dtype=torch.float32, device=query.device)

    has_mask = mask is not None
    if not has_mask:
        # Dummy tensor — pointer never dereferenced because HAS_MASK guards the load.
        mask = torch.empty(1, 1, dtype=torch.bool, device=query.device)

    grid = lambda meta: (
        triton.cdiv(b, meta["BLOCK_M"]),
        triton.cdiv(n, meta["BLOCK_N"]),
    )

    _fused_matmul_topk_kernel[grid](
        query,
        item_embs,
        mask,
        all_scores,
        B=b,
        N=n,
        D=d,
        stride_qb=query.stride(0),
        stride_qd=query.stride(1),
        stride_in=item_embs.stride(0),
        stride_id=item_embs.stride(1),
        stride_mb=mask.stride(0),
        stride_mn=mask.stride(1),
        stride_sb=all_scores.stride(0),
        stride_sn=all_scores.stride(1),
        HAS_MASK=has_mask,
    )

    topk_scores, topk_ids = torch.topk(all_scores, k, dim=1)
    return topk_ids, topk_scores
