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


@dataclass(frozen=True)
class _ClauseMaskLaunch:
    grid: tuple[int, int, int]
    kwargs: dict[str, object]  # every kernel arg: tensors, strides, constexprs, cfg
    out: Tensor


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
) -> Tensor:
    """Direct-launch body used by the offline tuner and unit tests. Takes an optional ``config=`` so
    tile parameters can be swept; the public ``@triton_op``-wrapped ``clause_mask`` always uses
    ``DEFAULT_CONFIG``."""
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _clause_mask_prep(item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=cfg)
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
        item_clause_attrs, clause_is_reverse, query_clause_attrs, cfg=DEFAULT_CONFIG, scores=scores
    )
    wrap_triton(_clause_mask_kernel)[launch.grid](**launch.kwargs)  # keep inline (export)
    return launch.out
