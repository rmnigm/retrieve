"""Fused Sign-OPORP 1-bit Hamming similarity + top-K (full-scan and indirect ops).

The indirect path buckets its candidate width (docs/system/kernels.md § Bucketing) and returns
``min(k, P)`` columns, the no-pad convention of the torch twin.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

from retrieve.ops.triton._host import check_contiguous, check_pow2, wide
from retrieve.ops.triton.common import popcount_int64, row_base, tile_rows

# N-bucket ladder for the HAS_INDICES path: clamps candidate width to constexpr values so the JIT
# cache compiles once per bucket × W. Full-scan uses item_bits.shape[0] directly (fixed per
# process).
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


# Default tile config (tuned on A100/sm_80); pass config= to the wrapper to override.
DEFAULT_CONFIG = Oporp1BitMatchTopkConfig(block_n=512, num_warps=4)


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
    WIDE: tl.constexpr,
):
    # Tile on grid_x (≤ 2³¹), batch on grid_y (≤ 65535): cdiv(N, BLOCK_N) can overflow grid_y at
    # large N.
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)

    row0 = tile_id * BLOCK_N
    n_off = row0 + tl.arange(0, BLOCK_N)
    w_off = tl.arange(0, W)
    n_valid = n_off < N

    qb = tl.load(qb_ptr + bid * stride_qb_b + w_off * stride_qb_w)

    if HAS_INDICES:
        count = tl.load(counts_ptr + bid)
        in_count = n_off < count
        # Gate pos_indices by in_count (not n_valid): with bucketed N, pos_indices has only n_loop
        # <= n_bucket columns, and in_count (count[bid] <= n_loop) never reads OOB.
        item_ids = tl.load(
            row_base(pos_indices_ptr, bid, stride_pb, WIDE) + n_off * stride_pp,
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
        bits_base, ids = tile_rows(item_bits_ptr, row0, tl.arange(0, BLOCK_N), stride_ib_n, WIDE)
        item_rows = tl.load(
            bits_base + ids[:, None] * stride_ib_n + w_off[None, :] * stride_ib_w,
            mask=n_valid[:, None],
            other=0,
        )
        valid_score = n_valid

    xor_words = qb[None, :] ^ item_rows
    pop_words = popcount_int64(xor_words)
    hamming = tl.sum(pop_words, axis=1)

    scores = (D_TOTAL - 2 * hamming).to(tl.float32)
    scores = tl.where(valid_score, scores, float("-inf"))

    tl.store(
        row_base(out_scores_ptr, bid, stride_sb, WIDE) + n_off * stride_sn,
        scores,
        mask=n_valid,
    )


@dataclass(frozen=True)
class _OporpLaunch:
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    all_scores: Tensor
    positive_indices: Tensor | None  # post-contiguous, for the epilogue gather (indirect only)
    has_indices: bool
    n_loop: int  # true candidate width (indirect) / corpus size (full-scan)
    n_kernel: int  # score-buffer + topk lane width; >= k on the indirect path by construction
    b: int


def _oporp_prep(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
    positive_indices: Tensor | None,
    counts: Tensor | None,
    cfg: Oporp1BitMatchTopkConfig,
) -> _OporpLaunch:
    """Validation + contiguity + buffers (incl. the HAS_INDICES dummy-tensor branch) + launch-arg
    dict + grid dims. THE single place input checking happens — shared by ``_impl`` and both
    ops."""
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

    check_contiguous(item_bits=item_bits)
    check_pow2(W=w)
    query_bits = query_bits.contiguous()

    if has_indices:
        check_contiguous(positive_indices=positive_indices)
        counts = counts.contiguous()
        n_loop = positive_indices.shape[1]
        # Lanes in [n_loop, n_kernel) score -inf (via in_count); topk takes min(k, n_loop).
        n_kernel = _bucket_n(n_loop)
        pos_arg = positive_indices
        counts_arg = counts
    else:
        # No bucketing for full-scan: n_items_total is fixed per registered index, so N as constexpr
        # compiles once per corpus size.
        n_loop = n_items_total
        n_kernel = n_loop
        # HAS_INDICES=False gates the loads through these pointers, so any int64 tensor stands
        # in (no per-call allocation).
        pos_arg = query_bits
        counts_arg = query_bits[0]

    all_scores = torch.empty((b, n_kernel), dtype=torch.float32, device=query_bits.device)

    kwargs = {
        "qb_ptr": query_bits,
        "item_bits_ptr": item_bits,
        "pos_indices_ptr": pos_arg,
        "counts_ptr": counts_arg,
        "out_scores_ptr": all_scores,
        "N": n_kernel,
        "W": w,
        "D_TOTAL": d_total,
        "stride_qb_b": query_bits.stride(0),
        "stride_qb_w": query_bits.stride(1),
        "stride_ib_n": item_bits.stride(0),
        "stride_ib_w": item_bits.stride(1),
        "stride_pb": pos_arg.stride(0),
        "stride_pp": pos_arg.stride(1),
        "stride_sb": all_scores.stride(0),
        "stride_sn": all_scores.stride(1),
        "HAS_INDICES": has_indices,
        "BLOCK_N": cfg.block_n,
        "WIDE": wide(item_bits, pos_arg, all_scores),
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
    return _OporpLaunch(kwargs, all_scores, positive_indices, has_indices, n_loop, n_kernel, b)


def _oporp_finish(launch: _OporpLaunch, k: int) -> tuple[Tensor, Tensor]:
    """topk epilogue; the indirect path adds the ``clamp_max(n_loop - 1)`` +
    ``where(isfinite, ..., -1)`` tail."""
    if launch.has_indices:
        # min(k, P) columns, no pad: the candidate-path convention of the torch twin.
        topk_scores, topk_local = torch.topk(
            launch.all_scores, torch.sym_min(k, launch.n_loop), dim=1
        )
        # topk_local may index [n_loop, n_kernel) when counts[b] < k (ties at -inf); clamp before
        # gather (the where() below masks these to -1).
        safe_local = topk_local.clamp_max(launch.n_loop - 1)
        topk_ids = launch.positive_indices.gather(1, safe_local)
        topk_ids = torch.where(torch.isfinite(topk_scores), topk_ids, topk_ids.new_full((), -1))
    else:
        # Full scan: the layer asserts k <= N.
        topk_scores, topk_local = torch.topk(launch.all_scores, k, dim=1)
        topk_ids = topk_local.to(torch.long)
    return topk_ids, topk_scores


def _oporp_1bit_match_topk_impl(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
    positive_indices: Tensor | None,
    counts: Tensor | None,
    config: Oporp1BitMatchTopkConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused Sign-OPORP 1-bit Hamming similarity + top-K: score = ``D - 2*popcount(q ^ item)``, D =
    64·W. With ``positive_indices [B, P]`` (+ ``counts [B]``) it scores only those candidates per
    row; with both ``None`` it full-scans ``item_bits``.

    Inputs query_bits [B, W] int64, item_bits [N, W] int64. Returns (ids, scores): ``[B, K]`` on
    the full scan, ``[B, min(K, P)]`` on the candidate path, rows short of candidates ending in
    -1/-inf. The HAS_INDICES path runs over a bucketed width ``_bucket_n(P)`` so the JIT cache
    compiles once per bucket × W; full-scan uses ``N = item_bits.shape[0]`` directly. Eager entry
    point for tune scripts / parity tests; the compiled path goes through the ``@triton_op``
    wrappers."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _oporp_prep(query_bits, item_bits, k, positive_indices, counts, cfg)

    # Tile axis on grid_x, batch on grid_y — see kernel comment.
    grid = (triton.cdiv(launch.n_kernel, cfg.block_n), launch.b)

    _oporp_1bit_match_topk_kernel[grid](**launch.kwargs)
    return _oporp_finish(launch, k)


@triton_op("retrieve::oporp_1bit_match_topk_full", mutates_args=())
def oporp_1bit_match_topk_full(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Full-scan Sign-OPORP 1-bit Hamming top-K; shares ``_oporp_prep``/``_oporp_finish`` with
    ``_oporp_1bit_match_topk_impl`` — only the launch line lives here (``wrap_triton`` must
    appear textually in the decorated source for torch.export). Layer asserts ``k <=
    n_items_total`` so topk(k) needs no pad tail."""
    launch = _oporp_prep(query_bits, item_bits, k, None, None, DEFAULT_CONFIG)
    n_kernel, b = launch.n_kernel, launch.b

    def grid(meta):
        return (triton.cdiv(n_kernel, meta["BLOCK_N"]), b)

    wrap_triton(_oporp_1bit_match_topk_kernel)[grid](**launch.kwargs)
    return _oporp_finish(launch, k)


@triton_op("retrieve::oporp_1bit_match_topk_indirect", mutates_args=())
def oporp_1bit_match_topk_indirect(
    query_bits: Tensor,
    item_bits: Tensor,
    k: int,
    positive_indices: Tensor,
    counts: Tensor,
) -> tuple[Tensor, Tensor]:
    """Indirect-load Sign-OPORP 1-bit Hamming top-K — scores only ``positive_indices[b,
    :counts[b]]`` per row; shares ``_oporp_prep``/``_oporp_finish`` with
    ``_oporp_1bit_match_topk_impl`` (see the full wrapper). Buffer width is ``_bucket_n(P)``;
    lanes in ``[P, n_kernel)`` carry -inf. Returns ``min(k, P)`` columns."""
    launch = _oporp_prep(query_bits, item_bits, k, positive_indices, counts, DEFAULT_CONFIG)
    n_kernel, b = launch.n_kernel, launch.b

    def grid(meta):
        return (triton.cdiv(n_kernel, meta["BLOCK_N"]), b)

    wrap_triton(_oporp_1bit_match_topk_kernel)[grid](**launch.kwargs)
    return _oporp_finish(launch, k)
