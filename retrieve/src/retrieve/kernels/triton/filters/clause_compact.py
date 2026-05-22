"""Fused clause evaluation + stream compaction.

Avoids materializing the dense ``[B, N]`` bool that
``ExactAttributeFilter.evaluate_mask`` would otherwise produce. One kernel launch:

  per program (b, tile):
      evaluate clauses for BLOCK_N items against query_clause_attrs[b]
      tl.cumsum within tile to get intra-tile write offsets
      tl.atomic_add into counts[b] to get the row's base offset
      tl.store passing item ids at positive_indices[b, base + intra]

Output ordering within a row is unspecified (atomics across tiles). V2's
``fused_masked_knn_topk`` only consumes the *set*, not the order — callers that
care must sort.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.library import custom_op


@dataclass(frozen=True)
class ClauseCompactConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3


# Single default the library ships with. Re-tune on a new arch by running
# ``evaluation/scripts/tune_kernels.py --kernel clause_compact`` and pasting
# the resulting line in.
#
# Sweeping tiles in-kernel via ``@triton.autotune`` is unsafe here: the
# ``tl.atomic_add(counts_ptr + bid, ...)`` accumulates across trials, and
# ``out_indices`` is written in-place. The offline tuner sidesteps both
# because the host wrapper allocates fresh ``out_indices`` (full of -1)
# and ``counts`` (zeros) on every call.
# Tuned on A100 (sm_80) against real-eval shapes (Goodreads N=797K /
# arXiv N=3M / arXiv-synth N=15M, C∈{4,5}, A_MAX=4, B∈{1, 16}):
# block_n=512, num_warps=2 wins 3/5 regimes (all batched at N≥797K) and
# stays within ~5% at the others. Low num_warps avoids the cliff seen
# at num_warps=8 on B=1.
DEFAULT_CONFIG = ClauseCompactConfig(block_n=512, num_warps=2)


@triton.jit
def _clause_compact_kernel(
    item_attrs_ptr,  # [N, C, A_max] int64
    is_reverse_ptr,  # [C] bool (stored as int8 in torch)
    query_attrs_ptr,  # [B, C] int64
    out_indices_ptr,  # [B, N] int64 (worst-case scratch)
    counts_ptr,  # [B] int64 (init 0)
    N,
    tiles_y,
    C: tl.constexpr,
    A_MAX: tl.constexpr,
    stride_in,  # item_attrs.stride(0)
    stride_ic,  # item_attrs.stride(1)
    stride_ia,  # item_attrs.stride(2)
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

    # pass_mask starts as all True; AND in each clause's verdict.
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

        # reverse: flip clause_match if rev_c is True.
        # XOR with broadcast scalar bool flips per element.
        clause_match = clause_match ^ rev_c

        # inactive (q_c == -1) → clause always passes. Overrides reverse.
        inactive = q_c == -1
        clause_match = clause_match | inactive

        pass_mask = pass_mask & clause_match

    pass_mask = pass_mask & n_valid

    # Stream compaction: cumsum gives intra-tile offsets; atomic_add gives base.
    pass_int = tl.where(pass_mask, 1, 0).to(tl.int32)
    intra = tl.cumsum(pass_int, axis=0) - 1  # 0-indexed inclusive position
    tile_sum = tl.sum(pass_int)

    base = tl.atomic_add(counts_ptr + bid, tile_sum.to(tl.int64))
    write_pos = base + intra.to(tl.int64)

    tl.store(
        out_indices_ptr + bid * stride_ob + write_pos * stride_on,
        n_offsets.to(tl.int64),
        mask=pass_mask,
    )


def _clause_compact_impl(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
    *,
    config: ClauseCompactConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Real body for ``clause_compact``. Takes an optional ``config=`` so
    the offline tuner and unit tests can sweep tile parameters. The public
    ``@custom_op``-wrapped ``clause_compact`` always passes ``config=None``."""
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

    # Initialise to -1 sentinel: the kernel only writes at positions
    # `[base, base + tile_sum)` for passing items, so positions beyond
    # `counts[bid]` remain at their initial value. With `torch.empty` those
    # positions held uninitialised memory, which leaks into V2's gather when
    # `counts[b] < k` and `torch.topk` falls back to padding slots tied at
    # -inf. -1 is the canonical "no item" sentinel (unmatched against any
    # real id by `_hits_mask`).
    out_indices = torch.full((b, n), -1, dtype=torch.int64, device=device)
    counts = torch.zeros((b,), dtype=torch.int64, device=device)

    cfg = config if config is not None else DEFAULT_CONFIG
    # 3D grid (batch, tiles_y, tiles_x) — see kernel comment.
    tiles = triton.cdiv(n, cfg.block_n)
    tiles_x = triton.cdiv(tiles, 65535)
    tiles_y = triton.cdiv(tiles, tiles_x)
    grid = (b, tiles_y, tiles_x)

    _clause_compact_kernel[grid](
        item_clause_attrs,
        clause_is_reverse,
        query_clause_attrs,
        out_indices,
        counts,
        N=n,
        tiles_y=tiles_y,
        C=c,
        A_MAX=a_max,
        stride_in=item_clause_attrs.stride(0),
        stride_ic=item_clause_attrs.stride(1),
        stride_ia=item_clause_attrs.stride(2),
        stride_qb=query_clause_attrs.stride(0),
        stride_qc=query_clause_attrs.stride(1),
        stride_ob=out_indices.stride(0),
        stride_on=out_indices.stride(1),
        BLOCK_N=cfg.block_n,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )

    return out_indices, counts


@custom_op("retrieve::clause_compact", mutates_args=())
def clause_compact(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> tuple[Tensor, Tensor]:
    """Fused clause evaluation + compaction.

    Returns ``(positive_indices [B, N] int64, counts [B] int64)``. The full
    item-width ``[B, N]`` indices buffer is returned with ``-1`` sentinels
    in the unused tail; downstream consumers (``fused_masked_knn_topk``,
    ``oporp_1bit_match_topk``) row-bound by ``counts[b]`` so the wider
    buffer never costs a re-read. Indices within a row are unordered
    (atomic-add writes).

    Registered as an opaque ``custom_op`` so the algo-level
    ``torch.compile(dynamic=True, mode="reduce-overhead")`` can stitch
    this kernel into a single cudagraph_trees graph (no per-call graph
    break, no ``.item()`` host sync). Delegates to ``_clause_compact_impl``
    with the shipped ``DEFAULT_CONFIG``.
    """
    return _clause_compact_impl(
        item_clause_attrs, clause_is_reverse, query_clause_attrs, config=None
    )


@clause_compact.register_fake
def _clause_compact_fake(
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
) -> tuple[Tensor, Tensor]:
    n = item_clause_attrs.shape[0]
    b = query_clause_attrs.shape[0]
    device = query_clause_attrs.device
    return (
        torch.empty((b, n), dtype=torch.int64, device=device),
        torch.empty((b,), dtype=torch.int64, device=device),
    )
