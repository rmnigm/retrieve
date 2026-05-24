from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

# N-bucket ladder for the HAS_INDICES path. Clamps runtime candidate
# width onto a small set of constexpr values so the JIT cache compiles
# once per bucket × W. The HAS_INDICES=False path uses
# ``item_bits.shape[0]`` directly (registered index size — fixed per
# process) so no bucketing is needed there.
_N_BUCKETS = (4096, 65536, 1048576, 16777216)


def _bucket_n(n: int) -> int:
    for b in _N_BUCKETS:
        if n <= b:
            return b
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class Oporp1BitMatchTopkConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Single default the library ships with. Re-tune on a new arch by
# running ``uv run tune-kernels oporp-1bit-match-topk`` and pasting the
# resulting line in. Callers who want a different tile config pass
# ``config=`` through to the wrapper.
# Tuned on A100 (sm_80): block_n=512 dominates at N >= 65k (5/8 regimes);
# small-N regimes are within ~10% noise across configs.
DEFAULT_CONFIG = Oporp1BitMatchTopkConfig(block_n=512, num_warps=4)


@triton.jit
def _popcount_int64(x):
    """Bit-twiddle popcount over int64 lanes (no libdevice dependency)."""
    M1 = 0x5555555555555555
    M2 = 0x3333333333333333
    M4 = 0x0F0F0F0F0F0F0F0F
    H01 = 0x0101010101010101
    x = x - ((x >> 1) & M1)
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    return ((x * H01) >> 56).to(tl.int32)


@triton.jit
def _oporp_1bit_match_topk_kernel(
    qb_ptr,
    item_bits_ptr,
    pos_indices_ptr,
    counts_ptr,
    out_scores_ptr,
    N: tl.constexpr,
    W: tl.constexpr,
    D_TOTAL: tl.constexpr,
    stride_qb_b,
    stride_qb_w,
    stride_ib_n,
    stride_ib_w,
    stride_pb,
    stride_pp,
    stride_sb,
    stride_sn,
    HAS_INDICES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # tile on grid_x (<=2^31), batch on grid_y (<=65535): cdiv(N, BLOCK_N)
    # can overflow grid_y at large N (e.g. 16M / BLOCK_N=64 = 262144 tiles).
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)

    n_off = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    w_off = tl.arange(0, W)
    n_valid = n_off < N

    qb = tl.load(qb_ptr + bid * stride_qb_b + w_off * stride_qb_w)

    if HAS_INDICES:
        count = tl.load(counts_ptr + bid)
        in_count = n_off < count
        # Gate pos_indices load by in_count (not n_valid): with bucketed N,
        # n_valid spans [0, n_bucket) but pos_indices physically has only
        # ``n_loop <= n_bucket`` columns. ``in_count`` is the tight mask
        # (count[bid] <= n_loop <= n_bucket) so it never reads OOB.
        item_ids = tl.load(
            pos_indices_ptr + bid * stride_pb + n_off * stride_pp,
            mask=in_count,
            other=0,
        ).to(tl.int64)
        item_rows = tl.load(
            item_bits_ptr + item_ids[:, None] * stride_ib_n + w_off[None, :] * stride_ib_w,
            mask=in_count[:, None],
            other=0,
        )
        valid_score = in_count
    else:
        item_rows = tl.load(
            item_bits_ptr + n_off[:, None] * stride_ib_n + w_off[None, :] * stride_ib_w,
            mask=n_valid[:, None],
            other=0,
        )
        valid_score = n_valid

    xor_words = qb[None, :] ^ item_rows
    pop_words = _popcount_int64(xor_words)
    hamming = tl.sum(pop_words, axis=1)

    scores = (D_TOTAL - 2 * hamming).to(tl.float32)
    scores = tl.where(valid_score, scores, float("-inf"))

    tl.store(
        out_scores_ptr + bid * stride_sb + n_off * stride_sn,
        scores,
        mask=n_valid,
    )


def _oporp_1bit_match_topk_impl(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
    positive_indices: Tensor | None,
    counts: Tensor | None,
    config: Oporp1BitMatchTopkConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused Sign-OPORP 1-bit Hamming similarity + top-K.

    Score = ``D - 2 * popcount(query_bits ^ item_bits)`` with ``D = 64 * W``.

    Args:
        query_bits: ``[B, W]`` int64 packed query sign bits.
        item_bits: ``[N, W]`` int64 packed item sign bits.
        k: top-K to return.
        positive_indices: ``[B, P]`` int64 or ``None``. When given, scores
            only those candidate ids per row; ``counts[b]`` bounds valid
            columns. When ``None``, runs full-scan over ``item_bits``.
        counts: ``[B]`` int64 valid counts; required iff ``positive_indices``
            is given.
        config: optional ``Oporp1BitMatchTopkConfig`` override; default
            is ``DEFAULT_CONFIG``.

    Returns ``(ids[B, K], scores[B, K])``. ``ids`` are global item ids; rows
    that ran short are padded with ``-1`` / ``-inf``.

    The HAS_INDICES path runs over a bucketed width ``n_kernel =
    max(_bucket_n(positive_indices.shape[1]), _bucket_n(k))`` so the JIT
    cache compiles once per bucket × W (the role formerly played by
    ``@triton.autotune``'s cache key) and the buffer always has >= k
    lanes for ``torch.topk(scores, k)``. The full-scan path uses
    ``N=item_bits.shape[0]`` directly — the registered index size is
    fixed per process and the layer asserts ``k <= n_items_total``.

    Eager entry point for tune scripts and parity tests. The compiled
    path goes through the two ``@triton_op`` wrappers
    (``oporp_1bit_match_topk_full`` / ``oporp_1bit_match_topk_indirect``)
    which mirror this body inline so that ``wrap_triton`` is textually
    in the decorated function's source — torch.export's kernel registry
    requires that.
    """
    if query_bits.dim() != 2 or item_bits.dim() != 2:
        raise ValueError("query_bits must be [B, W] and item_bits [N, W]")
    if query_bits.dtype != torch.int64 or item_bits.dtype != torch.int64:
        raise TypeError("query_bits and item_bits must be int64")
    if query_bits.shape[1] != item_bits.shape[1]:
        raise ValueError(
            f"W mismatch: query_bits W={query_bits.shape[1]} vs item_bits W={item_bits.shape[1]}"
        )

    has_indices = positive_indices is not None
    if has_indices != (counts is not None):
        raise ValueError("positive_indices and counts must be provided together")

    b, w = query_bits.shape
    d_total = 64 * w
    n_items_total = item_bits.shape[0]

    query_bits = query_bits.contiguous()
    item_bits = item_bits.contiguous()
    cfg = config if config is not None else DEFAULT_CONFIG

    if has_indices:
        positive_indices = positive_indices.contiguous()
        counts = counts.contiguous()
        n_loop = positive_indices.shape[1]
        # Widen to max(bucket(n_loop), bucket(k)) so the topk(k) below
        # always has >= k lanes to draw from. Lanes in [n_loop, n_kernel)
        # get -inf via the kernel's `in_count = n_off < count[bid]` gate,
        # so the extra width carries -inf scores that `where(isfinite,
        # ..., -1)` masks to the per-row "ran short" sentinel.
        n_kernel = max(_bucket_n(n_loop), _bucket_n(k))
        all_scores = torch.empty((b, n_kernel), dtype=torch.float32, device=query_bits.device)
        pos_arg = positive_indices
        counts_arg = counts
        stride_pb = positive_indices.stride(0)
        stride_pp = positive_indices.stride(1)
    else:
        # No bucketing for full-scan: ``n_items_total`` is fixed per
        # registered index. Pass N=n_items_total as constexpr; JIT
        # compiles once per registered corpus size.
        n_loop = n_items_total
        n_kernel = n_loop
        all_scores = torch.empty(b, n_kernel, dtype=torch.float32, device=query_bits.device)
        # Dummies: HAS_INDICES=False gates the load so these are never
        # dereferenced. Allocated locally so the public surface stays clean.
        pos_arg = torch.empty(1, 1, dtype=torch.int64, device=query_bits.device)
        counts_arg = torch.empty(1, dtype=torch.int64, device=query_bits.device)
        stride_pb = pos_arg.stride(0)
        stride_pp = pos_arg.stride(1)

    # Tile axis on grid_x, batch on grid_y — see kernel comment.
    grid = (triton.cdiv(n_kernel, cfg.block_n), b)

    _oporp_1bit_match_topk_kernel[grid](
        query_bits,
        item_bits,
        pos_arg,
        counts_arg,
        all_scores,
        N=n_kernel,
        W=w,
        D_TOTAL=d_total,
        stride_qb_b=query_bits.stride(0),
        stride_qb_w=query_bits.stride(1),
        stride_ib_n=item_bits.stride(0),
        stride_ib_w=item_bits.stride(1),
        stride_pb=stride_pb,
        stride_pp=stride_pp,
        stride_sb=all_scores.stride(0),
        stride_sn=all_scores.stride(1),
        HAS_INDICES=has_indices,
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    # n_kernel >= k by construction (indirect: widened above; full:
    # corpus size >> k by layer-construction assert). topk(k) works
    # directly; no min/pad path needed.
    topk_scores, topk_local = torch.topk(all_scores, k, dim=1)
    if has_indices:
        # ``topk_local`` may index into [n_loop, n_kernel) when
        # counts[b] < k (bottom slots tie at -inf). Clamp before gather;
        # the where() below masks those slots to -1 regardless.
        safe_local = topk_local.clamp_max(n_loop - 1)
        topk_ids = positive_indices.gather(1, safe_local)
        topk_ids = torch.where(torch.isfinite(topk_scores), topk_ids, topk_ids.new_full((), -1))
    else:
        topk_ids = topk_local.to(torch.long)

    return topk_ids, topk_scores


@triton_op("retrieve::oporp_1bit_match_topk_full", mutates_args=())
def oporp_1bit_match_topk_full(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Full-scan Sign-OPORP 1-bit Hamming top-K.

    Mirrors ``_oporp_1bit_match_topk_impl(has_indices=False)`` inline so
    ``wrap_triton`` appears textually in the decorated source — required
    by torch.export's kernel registry. Uses ``DEFAULT_CONFIG``; for tune
    grids call ``_oporp_1bit_match_topk_impl`` directly.

    Layer-construction asserts ``k <= n_items_total`` so the score
    buffer is always >= k wide and ``torch.topk(all_scores, k)`` works
    without a pad tail.
    """
    b, w = query_bits.shape
    d_total = 64 * w
    n_kernel = item_bits.shape[0]

    query_bits = query_bits.contiguous()
    item_bits = item_bits.contiguous()
    all_scores = torch.empty((b, n_kernel), dtype=torch.float32, device=query_bits.device)
    # Dummies for the HAS_INDICES=False kernel path: gated by the
    # constexpr so never dereferenced. Allocated locally so the public
    # surface stays clean (and the @triton_op signature stays Optional-free).
    pos_arg = torch.empty(1, 1, dtype=torch.int64, device=query_bits.device)
    counts_arg = torch.empty(1, dtype=torch.int64, device=query_bits.device)

    def grid(meta):
        return (triton.cdiv(n_kernel, meta["BLOCK_N"]), b)

    wrap_triton(_oporp_1bit_match_topk_kernel)[grid](
        query_bits,
        item_bits,
        pos_arg,
        counts_arg,
        all_scores,
        N=n_kernel,
        W=w,
        D_TOTAL=d_total,
        stride_qb_b=query_bits.stride(0),
        stride_qb_w=query_bits.stride(1),
        stride_ib_n=item_bits.stride(0),
        stride_ib_w=item_bits.stride(1),
        stride_pb=pos_arg.stride(0),
        stride_pp=pos_arg.stride(1),
        stride_sb=all_scores.stride(0),
        stride_sn=all_scores.stride(1),
        HAS_INDICES=False,
        BLOCK_N=DEFAULT_CONFIG.block_n,
        num_warps=DEFAULT_CONFIG.num_warps,
        num_stages=DEFAULT_CONFIG.num_stages,
    )

    topk_scores, topk_local = torch.topk(all_scores, k, dim=1)
    topk_ids = topk_local.to(torch.long)
    return topk_ids, topk_scores


@triton_op("retrieve::oporp_1bit_match_topk_indirect", mutates_args=())
def oporp_1bit_match_topk_indirect(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
    positive_indices: Tensor,
    counts: Tensor,
) -> tuple[Tensor, Tensor]:
    """Indirect-load Sign-OPORP 1-bit Hamming top-K.

    Scores only the ids in ``positive_indices[b, :counts[b]]`` per row.
    Mirrors ``_oporp_1bit_match_topk_impl(has_indices=True)`` inline for
    export-discovery reasons (see the full wrapper docstring).

    Buffer width is ``max(_bucket_n(positive_indices.shape[1]),
    _bucket_n(k))`` so ``torch.topk(all_scores, k)`` always has >= k
    lanes. Lanes in ``[n_loop, n_kernel)`` carry ``-inf``; the
    ``where(isfinite, ..., -1)`` tail masks them to the per-row "ran
    short" sentinel.
    """
    b, w = query_bits.shape
    d_total = 64 * w

    query_bits = query_bits.contiguous()
    item_bits = item_bits.contiguous()
    positive_indices = positive_indices.contiguous()
    counts = counts.contiguous()

    n_loop = positive_indices.shape[1]
    n_kernel = max(_bucket_n(n_loop), _bucket_n(k))
    all_scores = torch.empty((b, n_kernel), dtype=torch.float32, device=query_bits.device)

    def grid(meta):
        return (triton.cdiv(n_kernel, meta["BLOCK_N"]), b)

    wrap_triton(_oporp_1bit_match_topk_kernel)[grid](
        query_bits,
        item_bits,
        positive_indices,
        counts,
        all_scores,
        N=n_kernel,
        W=w,
        D_TOTAL=d_total,
        stride_qb_b=query_bits.stride(0),
        stride_qb_w=query_bits.stride(1),
        stride_ib_n=item_bits.stride(0),
        stride_ib_w=item_bits.stride(1),
        stride_pb=positive_indices.stride(0),
        stride_pp=positive_indices.stride(1),
        stride_sb=all_scores.stride(0),
        stride_sn=all_scores.stride(1),
        HAS_INDICES=True,
        BLOCK_N=DEFAULT_CONFIG.block_n,
        num_warps=DEFAULT_CONFIG.num_warps,
        num_stages=DEFAULT_CONFIG.num_stages,
    )

    topk_scores, topk_local = torch.topk(all_scores, k, dim=1)
    safe_local = topk_local.clamp_max(n_loop - 1)
    topk_ids = positive_indices.gather(1, safe_local)
    topk_ids = torch.where(torch.isfinite(topk_scores), topk_ids, topk_ids.new_full((), -1))
    return topk_ids, topk_scores
