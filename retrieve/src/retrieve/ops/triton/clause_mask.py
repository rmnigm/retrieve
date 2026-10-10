"""Fused clause evaluation emitting ``[B, N]`` bool directly, avoiding the ``[B, N, C, A_max]``
intermediate of the pure-torch path; same inner loop as ``clause_compact`` minus the compaction
epilogue. ``clause_mask_scores`` is the same pass writing ``scores`` or ``-inf`` instead of the
bool (LiNR V1's mask folded into its score write, kernels.md § clause_mask)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton

# Imported by name, not as `common.<fn>`: torch.compile re-compiles a
# @triton_op's kernel through inductor, which rebuilds the kernel's global
# namespace from the JITFunction's globals and captures @triton.jit callees
# by name. A module object is not captured, so `clause_pass(...)`
# raises NameError('common is not defined') at ast_to_ttir time — eager
# Triton resolves the attribute and never sees it (tests/compile catches it).
from retrieve.ops.triton._host import check_contiguous, grid_batch_tiles, wide
from retrieve.ops.triton.common import clause_pass, row_base, tile_rows


@dataclass(frozen=True)
class ClauseMaskConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Default tile config (tuned on A100/sm_80 against real-eval shapes); pass config= to
# _clause_mask_impl to override.
DEFAULT_CONFIG = ClauseMaskConfig(block_n=512, num_warps=2)
# clause_mask_scores streams fp32 scores in and out: its own tile (kernels.md § clause_mask).
SCORES_CONFIG = ClauseMaskConfig(block_n=128, num_warps=4)
# B > 1 masked scores: one program per item tile and BLOCK_B rows (kernels.md § clause_mask,
# "Batched rows").
BATCHED_CONFIG = ClauseMaskConfig(block_n=128, num_warps=4)
BLOCK_B = 16


@triton.jit
def _clause_mask_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] bool (stored as int8 in torch)
    query_attrs_ptr,  # [B, C] int64
    scores_ptr,  # [B, N] fp32, read only when HAS_SCORES
    out_ptr,  # [B, N] bool, or fp32 when HAS_SCORES
    N,
    tiles_y,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,
    stride_ic,
    stride_ia,
    stride_qb,
    stride_qc,
    stride_sb,
    stride_sn,
    stride_ob,
    stride_on,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
    HAS_SCORES: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(2) * tiles_y + tl.program_id(1)
    row0 = tile_id * BLOCK_N
    lane = tl.arange(0, BLOCK_N)
    n_offsets = row0 + lane
    n_valid = n_offsets < N

    # Result is already ANDed with n_valid inside the helper (keep seeds from load_mask).
    attrs_base, ids = tile_rows(item_attrs_ptr, row0, lane, stride_in, WIDE)
    pass_mask = clause_pass(
        attrs_base,
        is_reverse_ptr,
        query_attrs_ptr,
        ids,
        n_valid,
        bid,
        stride_in,
        stride_ic,
        stride_ia,
        stride_qb,
        stride_qc,
        C=C,
        A_MAX=A_MAX,
    )

    out = row_base(out_ptr, bid, stride_ob, WIDE) + n_offsets * stride_on
    if HAS_SCORES:
        s = tl.load(
            row_base(scores_ptr, bid, stride_sb, WIDE) + n_offsets * stride_sn, mask=n_valid
        )
        tl.store(out, tl.where(pass_mask, s, float("-inf")), mask=n_valid)
    else:
        tl.store(out, pass_mask, mask=n_valid)


@triton.jit
def _clause_mask_scores_batched_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] int8
    query_attrs_ptr,  # [B, C] int64
    scores_ptr,  # [B, N] fp32
    out_ptr,  # [B, N] fp32
    B,
    N,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,
    stride_ic,
    stride_ia,
    stride_qb,
    stride_qc,
    stride_sb,
    stride_ob,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One item tile for BLOCK_B rows: each clause's attrs are loaded once and matched against
    # every row's value, where the per-row kernel reloads them per row.
    n = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
    n_valid = n < N
    b_valid = b < B
    keep = b_valid[:, None] & n_valid[None, :]
    for c in tl.static_range(C):
        q_c = tl.load(query_attrs_ptr + b * stride_qb + c * stride_qc, mask=b_valid, other=-1)
        active = q_c != -1
        # Skip a clause no row of the block queries (an inactive clause passes, as clause_pass).
        if tl.max(active.to(tl.int32), axis=0) > 0:
            rev_c = tl.load(is_reverse_ptr + c).to(tl.int1)
            clause_match = tl.zeros([BLOCK_B, BLOCK_N], tl.int1)
            for a in tl.static_range(A_MAX):
                ia = tl.load(
                    item_attrs_ptr + n * stride_in + c * stride_ic + a * stride_ia,
                    mask=n_valid,
                    other=-1,
                )
                clause_match = clause_match | (ia[None, :] == q_c[:, None])
            keep = keep & ((clause_match ^ rev_c) | ~active[:, None])
    both = b_valid[:, None] & n_valid[None, :]
    s = tl.load(scores_ptr + b[:, None].to(tl.int64) * stride_sb + n[None, :], mask=both)
    tl.store(
        out_ptr + b[:, None].to(tl.int64) * stride_ob + n[None, :],
        tl.where(keep, s, float("-inf")),
        mask=both,
    )


@dataclass(frozen=True)
class _ClauseMaskLaunch:
    grid: tuple[int, int, int]
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    out: Tensor
    batched: bool = False  # the launch is _clause_mask_scores_batched_kernel's


def _clause_mask_prep(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
    *,
    cfg: ClauseMaskConfig,
    scores: Tensor | None = None,  # [B, N] fp32: emit masked scores instead of the bool
) -> _ClauseMaskLaunch:
    """Validation + contiguity + output buffer + the full launch-arg dict. The one place inputs
    are checked — shared by ``_clause_mask_impl`` and the public ops."""
    if item_clause_attrs.dim() != 3:
        raise ValueError("item_clause_attrs must be [N, C, A_max]")
    if query_clause_attrs.dim() != 2:
        raise ValueError("query_clause_attrs must be [B, C]")

    n, c, a_max = item_clause_attrs.shape
    b, c_q = query_clause_attrs.shape
    if c != c_q:
        raise ValueError(f"clause-count mismatch: items C={c}, query C={c_q}")

    check_contiguous(item_clause_attrs=item_clause_attrs)
    clause_is_reverse = clause_is_reverse.contiguous().to(torch.int8)
    query_clause_attrs = query_clause_attrs.contiguous()

    if scores is not None:
        if scores.shape != (b, n) or scores.dtype != torch.float32:
            raise ValueError(
                f"scores must be [B, N] = {[b, n]} float32, got {scores.shape} {scores.dtype}"
            )
        scores = scores.contiguous()
    dtype = torch.bool if scores is None else torch.float32
    out = torch.empty((b, n), dtype=dtype, device=query_clause_attrs.device)
    src = out if scores is None else scores  # an unread pointer when not HAS_SCORES

    if scores is not None and b > 1:
        # Batched rows: one program per item tile and BLOCK_B rows.
        kwargs_b = {
            "item_attrs_ptr": item_clause_attrs,
            "is_reverse_ptr": clause_is_reverse,
            "query_attrs_ptr": query_clause_attrs,
            "scores_ptr": scores,
            "out_ptr": out,
            "B": b,
            "N": n,
            "C": c,
            "A_MAX": a_max,
            "stride_in": item_clause_attrs.stride(0),
            "stride_ic": item_clause_attrs.stride(1),
            "stride_ia": item_clause_attrs.stride(2),
            "stride_qb": query_clause_attrs.stride(0),
            "stride_qc": query_clause_attrs.stride(1),
            "stride_sb": scores.stride(0),
            "stride_ob": out.stride(0),
            "BLOCK_B": BLOCK_B,
            "BLOCK_N": BATCHED_CONFIG.block_n,
            "num_warps": BATCHED_CONFIG.num_warps,
            "num_stages": BATCHED_CONFIG.num_stages,
        }
        grid_b = (triton.cdiv(n, BATCHED_CONFIG.block_n), triton.cdiv(b, BLOCK_B), 1)
        return _ClauseMaskLaunch(grid_b, kwargs_b, out, batched=True)
    grid, tiles_y = grid_batch_tiles(b, n, cfg.block_n)

    kwargs = {
        "item_attrs_ptr": item_clause_attrs,
        "is_reverse_ptr": clause_is_reverse,
        "query_attrs_ptr": query_clause_attrs,
        "scores_ptr": src,
        "out_ptr": out,
        "N": n,
        "tiles_y": tiles_y,
        "C": c,
        "A_MAX": a_max,
        "stride_in": item_clause_attrs.stride(0),
        "stride_ic": item_clause_attrs.stride(1),
        "stride_ia": item_clause_attrs.stride(2),
        "stride_qb": query_clause_attrs.stride(0),
        "stride_qc": query_clause_attrs.stride(1),
        "stride_sb": src.stride(0),
        "stride_sn": src.stride(1),
        "stride_ob": out.stride(0),
        "stride_on": out.stride(1),
        "BLOCK_N": cfg.block_n,
        "WIDE": wide(item_clause_attrs, out),
        "HAS_SCORES": scores is not None,
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
    return _ClauseMaskLaunch(grid, kwargs, out)


def _clause_mask_impl(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
    *,
    config: ClauseMaskConfig | None = None,
    scores: Tensor | None = None,
) -> Tensor:
    """Direct-launch body used by the offline tuner and unit tests. Takes an optional ``config=`` so
    tile parameters can be swept, and ``scores=`` for the ``clause_mask_scores`` form; the public
    ops use ``DEFAULT_CONFIG`` / ``SCORES_CONFIG``."""
    cfg = config if config is not None else DEFAULT_CONFIG if scores is None else SCORES_CONFIG
    launch = _clause_mask_prep(
        item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=cfg, scores=scores
    )
    if launch.batched:
        _clause_mask_scores_batched_kernel[launch.grid](**launch.kwargs)
    else:
        _clause_mask_kernel[launch.grid](**launch.kwargs)
    return launch.out


@triton_op("retrieve::clause_mask", mutates_args=())
def clause_mask(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """Fused clause evaluation → ``[B, N]`` bool, no intermediate. Registered as a ``triton_op`` so
    the launch is captured for ``torch.compile``; mirrors ``_clause_mask_impl`` with
    ``DEFAULT_CONFIG``."""
    launch = _clause_mask_prep(
        item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=DEFAULT_CONFIG
    )
    wrap_triton(_clause_mask_kernel)[launch.grid](**launch.kwargs)  # keep inline (export)
    return launch.out


@triton_op("retrieve::clause_mask_scores", mutates_args=())
def clause_mask_scores(
    scores: Tensor,  # [B, N] fp32
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """``where(clause_mask(...), scores, -inf)`` in one pass: the predicate never leaves the
    kernel as a ``[B, N]`` bool. Functional (a new buffer), so inductor has no in-place mask to
    copy."""
    launch = _clause_mask_prep(
        item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=SCORES_CONFIG, scores=scores
    )
    if launch.batched:
        wrap_triton(_clause_mask_scores_batched_kernel)[launch.grid](**launch.kwargs)
    else:
        wrap_triton(_clause_mask_kernel)[launch.grid](**launch.kwargs)  # keep inline (export)
    return launch.out


@triton.jit
def _clause_mask_packed_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] bool (stored as int8 in torch)
    query_attrs_ptr,  # [B, C] int64
    out_ptr,  # [B, ceil(N / 64)] int64
    N,
    n_words,
    tiles_y,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,
    stride_ic,
    stride_ia,
    stride_qb,
    stride_qc,
    stride_ob,
    BLOCK_N: tl.constexpr,
    WIDE: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(2) * tiles_y + tl.program_id(1)
    row0 = tile_id * BLOCK_N
    lane = tl.arange(0, BLOCK_N)
    n_valid = row0 + lane < N
    attrs_base, ids = tile_rows(item_attrs_ptr, row0, lane, stride_in, WIDE)
    pass_mask = clause_pass(
        attrs_base,
        is_reverse_ptr,
        query_attrs_ptr,
        ids,
        n_valid,
        bid,
        stride_in,
        stride_ic,
        stride_ia,
        stride_qb,
        stride_qc,
        C=C,
        A_MAX=A_MAX,
    )
    # Doc d at bit 63 - d % 64 of word d // 64 (high-first); disjoint bits, so the sum is the OR.
    bits = pass_mask.to(tl.int64) << (63 - (lane % 64)).to(tl.int64)
    words = tl.sum(tl.reshape(bits, (BLOCK_N // 64, 64)), axis=1)
    w_offsets = row0 // 64 + tl.arange(0, BLOCK_N // 64)
    tl.store(row_base(out_ptr, bid, stride_ob, WIDE) + w_offsets, words, mask=w_offsets < n_words)


@triton_op("retrieve::clause_mask_packed", mutates_args=())
def clause_mask_packed(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """The clause test over every item, packed: ``[B, ceil(N / 64)]`` int64 with doc ``d`` at bit
    ``63 - d % 64`` of word ``d // 64`` (the official scorer's ``filtering_bit_mask`` order), no
    ``[B, N]`` bool on the way (kernels.md § clause_mask)."""
    launch = _clause_mask_prep(
        item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=DEFAULT_CONFIG
    )
    b, n = launch.out.shape
    n_words = triton.cdiv(n, 64)
    out = torch.empty((b, n_words), dtype=torch.int64, device=query_clause_attrs.device)
    kw = launch.kwargs
    wrap_triton(_clause_mask_packed_kernel)[launch.grid](
        item_attrs_ptr=kw["item_attrs_ptr"],
        is_reverse_ptr=kw["is_reverse_ptr"],
        query_attrs_ptr=kw["query_attrs_ptr"],
        out_ptr=out,
        N=n,
        n_words=n_words,
        tiles_y=kw["tiles_y"],
        C=kw["C"],
        A_MAX=kw["A_MAX"],
        stride_in=kw["stride_in"],
        stride_ic=kw["stride_ic"],
        stride_ia=kw["stride_ia"],
        stride_qb=kw["stride_qb"],
        stride_qc=kw["stride_qc"],
        stride_ob=out.stride(0),
        BLOCK_N=DEFAULT_CONFIG.block_n,
        WIDE=wide(item_clause_attrs, out),
        num_warps=DEFAULT_CONFIG.num_warps,
        num_stages=DEFAULT_CONFIG.num_stages,
    )  # keep inline (export)
    return out
