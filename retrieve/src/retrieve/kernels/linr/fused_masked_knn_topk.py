"""Fused masked gather + dot + top-K over precompacted candidates (linr prefilter path).

Bucketing policy — the shared note for this file and ``oporp_1bit_match_topk.py``:
``_fused_masked_knn_topk_impl`` buckets P via ``_bucket_p`` while the public
``fused_masked_knn_topk`` op does not, and ``oporp_1bit_match_topk_indirect``
buckets in both entry points. Both are correct today: candidate widths are
static per deployment (the compact kernel family returns full-width ``[B, N]``
buffers; linr_v3's stage-2 width is the fixed ``candidate_pool``), so the
public ops compile once either way. The eager ``_impl``s (tune sweeps, parity
tests) see many distinct widths per process and bucket to keep the JIT cache
small; oporp's bucket additionally doubles as the >= k-lanes guarantee for
``topk(k)`` (see ``oporp_1bit_match_topk_indirect``). Written down once, here,
so the next reader doesn't re-derive it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

_P_BUCKETS = (256, 2048, 16384, 131072, 1048576)

# Dtypes the kernel is exercised with today: parity tests feed fp32, PrefilterKNN feeds fp16.
# Scores are always fp32 (the output buffer's dtype). Documents reality per kernels.md → I/O.
_SUPPORTED_DTYPES = (torch.float16, torch.float32)


def _bucket_p(p: int) -> int:
    """Round P up to the nearest static bucket so the kernel's ``P`` constexpr (hence its JIT cache)
    compiles once across many sweeps instead of once per distinct ``counts.max()``."""
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


# Default tile config (tuned on A100/sm_80); pass config= to the wrapper to override.
DEFAULT_CONFIG = FusedMaskedKnnTopkConfig(block_n=32, num_warps=8)


@triton.jit
def _fused_masked_knn_topk_kernel(
    query_ptr,
    item_embs_ptr,
    pos_indices_ptr,
    counts_ptr,
    out_scores_ptr,
    P: tl.constexpr,  # bucketed width (constexpr); see _bucket_p
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
    # Tile on grid_x (≤ 2³¹), batch on grid_y (≤ 65535): cdiv(P, BLOCK_N) can overflow grid_y at
    # large P.
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


@dataclass(frozen=True)
class _FmktLaunch:
    grid: tuple[int, int]
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    all_scores: Tensor
    positive_indices: Tensor  # post-contiguous, for the epilogue gather
    p: int  # true (unbucketed) candidate width
    b: int


def _fmkt_prep(
    query: Tensor,
    item_embs: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    cfg: FusedMaskedKnnTopkConfig,
    *,
    bucket: bool,
) -> _FmktLaunch:
    """Validation + contiguity + score buffer + launch-arg dict + grid dims.
    THE single place input checking happens — shared by ``_impl`` and the public op.

    ``bucket=True`` (``_impl``: tune sweeps / parity tests see many widths per process) runs the
    kernel at ``P = _bucket_p(p)`` so the JIT cache compiles once per bucket × D; ``bucket=False``
    (public op: catalog width is fixed per deployment) runs at ``P = p`` directly. See the module
    docstring for the policy rationale. The bucketed caller early-returns before prep when
    ``p == 0``."""
    if query.dim() != 2 or item_embs.dim() != 2:
        raise ValueError("query must be [B, D] and item_embs [N, D]")
    if query.dtype not in _SUPPORTED_DTYPES or item_embs.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "query and item_embs must be float16 or float32, "
            f"got {query.dtype} and {item_embs.dtype}"
        )
    b, d = query.shape
    p = positive_indices.shape[1]

    query = query.contiguous()
    item_embs = item_embs.contiguous()
    positive_indices = positive_indices.contiguous()
    counts = counts.contiguous()

    p_kernel = _bucket_p(p) if bucket else p

    # Allocate at kernel width so P (constexpr) is stable across sweeps. Lanes past counts[bid] —
    # including [p, p_kernel) when bucketed, since counts[bid] <= p — fail in_count and get -inf,
    # so the tail is correct without a pre-fill.
    all_scores = torch.empty((b, p_kernel), dtype=torch.float32, device=query.device)

    kwargs = dict(
        query_ptr=query,
        item_embs_ptr=item_embs,
        pos_indices_ptr=positive_indices,
        counts_ptr=counts,
        out_scores_ptr=all_scores,
        P=p_kernel,
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
    # Tile axis on grid_x, batch on grid_y — see kernel comment.
    grid = (triton.cdiv(p_kernel, cfg.block_n), b)
    return _FmktLaunch(grid, kwargs, all_scores, positive_indices, p, b)


def _fmkt_finish(launch: _FmktLaunch, k: int, *, pad_to_k: bool) -> tuple[Tensor, Tensor]:
    """topk → gather → -1-sentinel epilogue, shared by ``_impl`` and the public op.

    ``pad_to_k=True`` (``_impl``): takes top-``min(k, P)`` and pads short rows back out to ``k``
    columns with -1/-inf. ``pad_to_k=False`` (public op): ``p >= k > 0`` is guaranteed by
    ``PrefilterKNN``, so ``topk(k)`` runs directly with no pad tail."""
    p = launch.p
    actual_k = min(k, p) if pad_to_k else k
    topk_scores, topk_local = torch.topk(launch.all_scores, actual_k, dim=1)
    # topk_local indexes [0, p_kernel); when counts[b] < actual_k, ties at -inf can pick [p,
    # p_kernel) — OOB for positive_indices. Clamp before gather (the where() below masks these to
    # -1, so the clamp value only matters for avoiding the OOB read). With the unbucketed public
    # op p_kernel == p and the clamp is an identity.
    safe_local = topk_local.clamp_max(p - 1)
    topk_ids = launch.positive_indices.gather(1, safe_local)
    # When counts[b] < actual_k, ties at -inf can pick padding positions whose ids are
    # uninitialised (compact kernels use torch.empty); force those to -1 to match the oracle's
    # sentinel.
    topk_ids = torch.where(
        torch.isfinite(topk_scores),
        topk_ids,
        topk_ids.new_full((), -1),
    )

    if pad_to_k and actual_k < k:
        pad = k - actual_k
        device = launch.all_scores.device
        topk_ids = torch.cat(
            [
                topk_ids,
                torch.full((launch.b, pad), -1, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        topk_scores = torch.cat(
            [
                topk_scores,
                torch.full((launch.b, pad), float("-inf"), dtype=torch.float32, device=device),
            ],
            dim=1,
        )

    return topk_ids, topk_scores


def _fused_masked_knn_topk_impl(
    query: Tensor,
    item_embs: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    k: int,
    config: FusedMaskedKnnTopkConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused masked gather + dot + topk → (ids [B, K], scores [B, K]). 2-D launch over ``(B,
    cdiv(P_BUCKET, BLOCK_N))``; each program holds one query and gathers a ``BLOCK_N`` slab of
    candidate ids by indirect load, items past ``counts[b]`` scoring ``-inf``. Scoring is
    elementwise (``tl.sum``, not ``tl.dot``) since rows differ per cell — the dense no-filter
    path should use ``query @ item_embs.T`` instead. Runs over a bucketed width ``P_BUCKET =
    _bucket_p(P)`` so the JIT cache compiles once per bucket × D."""
    b = query.shape[0]
    p = positive_indices.shape[1]

    if p == 0:
        return (
            torch.full((b, k), -1, dtype=torch.long, device=query.device),
            torch.full((b, k), float("-inf"), dtype=torch.float32, device=query.device),
        )

    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _fmkt_prep(query, item_embs, positive_indices, counts, cfg, bucket=True)
    _fused_masked_knn_topk_kernel[launch.grid](**launch.kwargs)
    return _fmkt_finish(launch, k, pad_to_k=True)


@triton_op("retrieve::fused_masked_knn_topk", mutates_args=())
def fused_masked_knn_topk(
    query: Tensor,
    item_embs: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Production wrapper for ``_fused_masked_knn_topk_impl``, registered as a ``triton_op`` for
    ``torch.compile`` capture. Differs from ``_impl``: hard-codes ``DEFAULT_CONFIG``, skips
    bucketing (catalog size is fixed per deployment; see the module docstring), and assumes
    ``p >= k > 0`` (guaranteed by ``PrefilterKNN``) — no ``p == 0`` early-return, no pad tail."""
    launch = _fmkt_prep(query, item_embs, positive_indices, counts, DEFAULT_CONFIG, bucket=False)
    wrap_triton(_fused_masked_knn_topk_kernel)[launch.grid](**launch.kwargs)
    return _fmkt_finish(launch, k, pad_to_k=False)
