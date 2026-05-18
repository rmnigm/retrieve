from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


_P_BUCKETS = (256, 2048, 16384, 131072, 1048576)


def _bucket_p(p: int) -> int:
    """Round P up to the nearest static bucket so autotune compiles once
    across many sweeps (instead of once per distinct ``counts.max()``)."""
    for b in _P_BUCKETS:
        if p <= b:
            return b
    # Catalogs larger than 1M items: round up to next power of 2.
    return 1 << (p - 1).bit_length()


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


@triton.autotune(configs=_autotune_configs(), key=["P_BUCKET", "D"])
@triton.jit(do_not_specialize=["P_REAL"])
def _fused_masked_knn_topk_kernel(
    query_ptr,
    item_embs_ptr,
    pos_indices_ptr,
    counts_ptr,
    out_scores_ptr,
    P_REAL,                    # runtime int — actual width of pos_indices / all_scores
    P_BUCKET: tl.constexpr,    # cache-key stabilizer; not referenced in the body
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

    p_valid = n_offsets < P_REAL

    count = tl.load(counts_ptr + bid)
    in_count = n_offsets < count

    q = tl.load(query_ptr + bid * stride_qb + d_offsets * stride_qd)

    item_ids = tl.load(
        pos_indices_ptr + bid * stride_pb + n_offsets * stride_pp,
        mask=in_count,
        other=0,
    )

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


@torch._dynamo.disable
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

    # Bucket P only as the autotune cache key so the kernel compiles once
    # per (bucket, D) instead of once per distinct `counts.max()`. The
    # `positive_indices` buffer and the `all_scores` buffer stay at the
    # caller-provided width — the kernel uses `P_REAL` (runtime, with
    # `do_not_specialize`) for its store mask, and `mask=in_count` keeps the
    # indices load in-bounds without materializing a padded copy.
    p_bucket = _bucket_p(p)

    # `torch.empty` is safe: the kernel writes every slot in [0, P_REAL) —
    # either a real dot product or `-inf` for lanes past `counts[bid]` — so
    # the post-topk `where(isfinite(scores), …)` mask sees deterministic
    # values.
    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)

    grid = lambda meta: (b, triton.cdiv(p, meta["BLOCK_N"]))

    _fused_masked_knn_topk_kernel[grid](
        query,
        item_embs,
        positive_indices,
        counts,
        all_scores,
        p,                        # P_REAL (runtime)
        P_BUCKET=p_bucket,
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
    # When counts[b] < actual_k, the bottom slots tie at -inf and topk picks
    # padding positions whose ids in `positive_indices` are uninitialized
    # memory (clause_compact / bloom_compact allocate via torch.empty). Force
    # those slots to -1 so they match the oracle's -1 sentinel and don't leak
    # garbage into top-K.
    topk_ids = torch.where(
        torch.isfinite(topk_scores),
        topk_ids,
        topk_ids.new_full((), -1),
    )

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
