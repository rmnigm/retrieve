from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import custom_op


_P_BUCKETS = (256, 2048, 16384, 131072, 1048576)


def _bucket_p(p: int) -> int:
    """Round P up to the nearest static bucket so the kernel JIT cache
    compiles once across many sweeps (instead of once per distinct
    ``counts.max()``). Same role the value used to play as ``P_BUCKET``
    in the autotune key; now it sizes ``P`` directly as a ``tl.constexpr``.
    """
    for b in _P_BUCKETS:
        if p <= b:
            return b
    # Catalogs larger than 1M items: round up to next power of 2.
    return 1 << (p - 1).bit_length()


@dataclass(frozen=True)
class FusedMaskedKnnTopkConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Single default the library ships with. Re-tune on a new arch by
# running ``evaluation/scripts/tune_kernels.py`` and pasting the
# resulting line in. Callers who want a different tile config pass
# ``config=`` through to the wrapper.
# Tuned on A100 (sm_80): block_n=32 wins all but the 1M-bucket; num_warps=8 wins everywhere.
DEFAULT_CONFIG = FusedMaskedKnnTopkConfig(block_n=32, num_warps=8)


@triton.jit
def _fused_masked_knn_topk_kernel(
    query_ptr,
    item_embs_ptr,
    pos_indices_ptr,
    counts_ptr,
    out_scores_ptr,
    P: tl.constexpr,           # bucketed width (constexpr); see _bucket_p
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
    # tile on grid_x (<=2^31), batch on grid_y (<=65535): cdiv(P, BLOCK_N)
    # can overflow grid_y at large P with small BLOCK_N.
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)

    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, D)

    p_valid = n_offsets < P

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


def _fused_masked_knn_topk_impl(
    query: Tensor,
    item_embs: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    k: int,
    config: FusedMaskedKnnTopkConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused masked gather + dot + topk. Returns ``(ids[B, K], scores[B, K])``.

    2-D launch over ``(B, cdiv(P_BUCKET, BLOCK_N))``. Each program holds one
    query in registers and gathers a ``BLOCK_N``-wide slab of candidate
    item ids via indirect load. Items past ``counts[b]`` get score
    ``-inf``. The post-kernel ``torch.topk`` selects K from the
    ``[B, P_BUCKET]`` score buffer.

    Per-cell scoring is elementwise (``tl.sum``) rather than ``tl.dot``: the
    gathered rows differ per (B, p) cell so a true GEMM would re-load rows
    per query column. For the dense (high pass rate) path with no
    pre-filter, callers should fall back to ``query @ item_embs.T`` +
    ``torch.topk`` — there's no fused-kernel equivalent because cuBLAS +
    CUB already cover that case.

    The kernel runs over a bucketed width ``P_BUCKET = _bucket_p(P)`` so
    the JIT cache compiles once per bucket × D (the role formerly played
    by ``@triton.autotune``'s cache key). ``positive_indices`` stays at
    its caller-supplied width; the ``in_count`` indirect-load mask keeps
    the tail in-bounds.
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

    cfg = config if config is not None else DEFAULT_CONFIG
    p_bucket = _bucket_p(p)

    # Allocate at bucketed width so the kernel's ``P: tl.constexpr`` is
    # stable across sweeps. Lanes past ``counts[bid]`` get ``-inf``
    # (via the ``in_count`` branch); lanes in ``[p, p_bucket)`` also
    # fail ``in_count`` because ``counts[bid] <= p <= p_bucket``, so
    # the padding tail is correctly ``-inf`` without a pre-fill.
    all_scores = torch.empty((b, p_bucket), dtype=torch.float32, device=query.device)

    # Tile axis on grid_x, batch on grid_y — see kernel comment.
    grid = (triton.cdiv(p_bucket, cfg.block_n), b)

    _fused_masked_knn_topk_kernel[grid](
        query,
        item_embs,
        positive_indices,
        counts,
        all_scores,
        P=p_bucket,
        D=d,
        stride_qb=query.stride(0),
        stride_qd=query.stride(1),
        stride_in=item_embs.stride(0),
        stride_id=item_embs.stride(1),
        stride_pb=positive_indices.stride(0),
        stride_pp=positive_indices.stride(1),
        stride_sb=all_scores.stride(0),
        stride_sp=all_scores.stride(1),
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    actual_k = min(k, p)
    topk_scores, topk_local = torch.topk(all_scores, actual_k, dim=1)
    # ``topk_local`` indexes into [0, p_bucket). When counts[b] < actual_k
    # the bottom slots tie at -inf and torch.topk can pick positions in
    # [p, p_bucket) — out-of-bounds for ``positive_indices`` (width p).
    # Clamp before gather; the where() below masks those slots to -1
    # regardless of the gathered value, so the clamp value is irrelevant
    # for correctness (it only avoids the OOB read).
    safe_local = topk_local.clamp_max(p - 1)
    topk_ids = positive_indices.gather(1, safe_local)
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


@custom_op("retrieve::fused_masked_knn_topk", mutates_args=())
def fused_masked_knn_topk(
    query: Tensor,
    item_embs: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Production wrapper for ``_fused_masked_knn_topk_impl`` with the default
    tile config. Registered as an opaque ``custom_op`` so dynamo can stitch
    the algo forward into a single cudagraph (no graph break per call). The
    ``config=`` keyword is dropped here because ``custom_op``'s schema
    inference doesn't accept dataclass args; tune scripts and parity tests
    that need a non-default config call ``_fused_masked_knn_topk_impl``
    directly.
    """
    return _fused_masked_knn_topk_impl(
        query, item_embs, positive_indices, counts, k, config=None
    )


@fused_masked_knn_topk.register_fake
def _fused_masked_knn_topk_fake(
    query: Tensor,
    item_embs: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    b = query.shape[0]
    device = query.device
    ids = torch.empty((b, k), dtype=torch.long, device=device)
    scores = torch.empty((b, k), dtype=torch.float32, device=device)
    return ids, scores
