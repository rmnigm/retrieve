"""Fused clause evaluation emitting ``[B, N]`` bool directly.

Avoids the ``[B, N, C, A_max]`` intermediate that
``ExactAttributeFilter.evaluate_mask``'s pure-torch broadcast materializes.
Same inner loop as ``clause_compact`` minus the cumsum + atomic_add epilogue.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import triton_op, wrap_triton


@dataclass(frozen=True)
class ClauseMaskConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Single default the library ships with. Re-tune on a new arch by running
# ``evaluation/scripts/tune_kernels.py --kernel clause_mask`` and pasting
# the resulting line in. Callers who want a different tile pass ``config=``
# to ``_clause_mask_impl`` (the public ``@triton_op`` wrapper has a fixed
# schema and always uses the default).
# Tuned on A100 (sm_80) against real-eval shapes (Goodreads N=797K /
# arXiv N=3M / arXiv-synth N=15M, C∈{4,5}, A_MAX=4, B∈{1, 16}):
# block_n=512, num_warps=2 wins both batched regimes at N≥3M and stays
# within ~10% of the per-regime winner at small-N / B=1.
DEFAULT_CONFIG = ClauseMaskConfig(block_n=512, num_warps=2)


@triton.jit
def _clause_mask_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] bool (stored as int8 in torch)
    query_attrs_ptr,  # [B, C] int64
    out_ptr,  # [B, N] bool
    N,
    tiles_y,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,
    stride_ic,
    stride_ia,
    stride_qb,
    stride_qc,
    stride_ob,
    stride_on,
    BLOCK_N: tl.constexpr,
):
    # 3D grid: batch on grid_x (small, restores L2 reuse on item_attrs
    # because adjacent dispatched programs share the same tile), tiles
    # split across grid_y × grid_z to dodge the 65535 cap on a single
    # axis. `tile_id = tile_x * tiles_y + tile_y` keeps tiles contiguous.
    bid = tl.program_id(0)
    tile_y = tl.program_id(1)
    tile_x = tl.program_id(2)
    tile_id = tile_x * tiles_y + tile_y

    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    n_valid = n_offsets < N

    pass_mask = tl.full([BLOCK_N], 1, tl.int1)

    for c in tl.static_range(C):
        q_c = tl.load(query_attrs_ptr + bid * stride_qb + c * stride_qc)  # scalar
        rev_c = tl.load(is_reverse_ptr + c).to(tl.int1)  # scalar

        clause_match = tl.full([BLOCK_N], 0, tl.int1)
        for a in tl.static_range(A_MAX):
            ia = tl.load(
                item_attrs_ptr + n_offsets * stride_in + c * stride_ic + a * stride_ia,
                mask=n_valid,
                other=-1,
            )
            clause_match = clause_match | (ia == q_c)

        clause_match = clause_match ^ rev_c

        inactive = q_c == -1
        clause_match = clause_match | inactive

        pass_mask = pass_mask & clause_match

    pass_mask = pass_mask & n_valid

    tl.store(
        out_ptr + bid * stride_ob + n_offsets * stride_on,
        pass_mask,
        mask=n_valid,
    )


def _clause_mask_impl(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
    *,
    config: ClauseMaskConfig | None = None,
) -> Tensor:
    """Direct-launch body used by the offline tuner and unit tests. Takes an
    optional ``config=`` so tile parameters can be swept; the public
    ``@triton_op``-wrapped ``clause_mask`` always uses ``DEFAULT_CONFIG``."""
    if item_clause_attrs.dim() != 3:
        raise ValueError("item_clause_attrs must be [N, C, A_max]")
    if query_clause_attrs.dim() != 2:
        raise ValueError("query_clause_attrs must be [B, C]")

    n, c, a_max = item_clause_attrs.shape
    b, c_q = query_clause_attrs.shape
    if c != c_q:
        raise ValueError(f"clause-count mismatch: items C={c}, query C={c_q}")

    device = query_clause_attrs.device
    item_clause_attrs = item_clause_attrs.contiguous()
    clause_is_reverse = clause_is_reverse.contiguous().to(torch.int8)
    query_clause_attrs = query_clause_attrs.contiguous()

    out = torch.empty((b, n), dtype=torch.bool, device=device)

    cfg = config if config is not None else DEFAULT_CONFIG
    # 3D grid (batch, tiles_y, tiles_x) — keeps batch on grid_x for L2
    # reuse on item_attrs while letting tiles overflow into grid_z when
    # they don't fit a single 65535-cap axis. See kernel comment.
    tiles = triton.cdiv(n, cfg.block_n)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    grid = (b, tiles_y, tiles_x)

    _clause_mask_kernel[grid](
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        out,
        N=n,
        tiles_y=tiles_y,
        C=c,
        A_MAX=a_max,
        stride_in=item_clause_attrs.stride(0),
        stride_ic=item_clause_attrs.stride(1),
        stride_ia=item_clause_attrs.stride(2),
        stride_qb=query_clause_attrs.stride(0),
        stride_qc=query_clause_attrs.stride(1),
        stride_ob=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    return out


@triton_op("retrieve::clause_mask", mutates_args=())
def clause_mask(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """Fused clause evaluation → ``[B, N]`` bool. No intermediate.

    Registered as ``triton_op`` so the kernel launch is captured as a HOP
    that ``torch.compile`` can stitch into surrounding cudagraphs; the
    ``torch.empty`` allocation inside the body acts as the fake/meta kernel.
    Mirrors ``_clause_mask_impl`` but routes the launch through
    ``wrap_triton`` for HOP capture and hard-codes ``DEFAULT_CONFIG``."""
    cfg = DEFAULT_CONFIG

    n, c, a_max = item_clause_attrs.shape
    b, _ = query_clause_attrs.shape

    item_clause_attrs = item_clause_attrs.contiguous()
    clause_is_reverse = clause_is_reverse.contiguous().to(torch.int8)
    query_clause_attrs = query_clause_attrs.contiguous()

    out = torch.empty((b, n), dtype=torch.bool, device=query_clause_attrs.device)

    tiles = triton.cdiv(n, cfg.block_n)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    grid = (b, tiles_y, tiles_x)

    wrap_triton(_clause_mask_kernel)[grid](
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        out,
        N=n,
        tiles_y=tiles_y,
        C=c,
        A_MAX=a_max,
        stride_in=item_clause_attrs.stride(0),
        stride_ic=item_clause_attrs.stride(1),
        stride_ia=item_clause_attrs.stride(2),
        stride_qb=query_clause_attrs.stride(0),
        stride_qc=query_clause_attrs.stride(1),
        stride_ob=out.stride(0),
        stride_on=out.stride(1),
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    return out
