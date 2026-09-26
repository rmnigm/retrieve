"""Query-time torch glue that is not a kernel: the masked top-K epilogue, mask compaction and
composition, the post-filter, the torch popcount and the two subset predicates shared by the
filter modules and ``retrieve.ops.reference``.

Everything here is pure tensor flow (no ``.item()``, no data-dependent Python branch beyond
``min(k, P)`` over static shapes) and traces cleanly under ``torch.compile(dynamic=True,
mode="reduce-overhead")`` — except ``combine_indices``, which syncs per cascade stage and says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from retrieve.interfaces import FilterModule


def counts_to_valid(counts: Tensor, p: int) -> Tensor:
    """[B] counts → [B, p] bool prefix mask."""
    return torch.arange(p, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)


def masked_topk(
    scores: Tensor,  # [B, P] fp
    k: int,
    *,
    valid: Tensor | None = None,  # [B, P] bool; None = all valid
    gather_ids: Tensor | None = None,  # [B, P] local→global map; None = identity
    pad_to_k: bool = True,  # rows with P < k or few survivors pad -1/-inf
) -> tuple[Tensor, Tensor]:
    """Mask invalid scores to -inf → topk → map local→global ids → replace
    non-finite winners with -1 → optionally pad to k. Returns (ids, scores).

    The -1 sentinel is applied whenever ``valid`` is given (or P < k): winners
    with a non-finite score — masked lanes *and* genuine -inf scores alike —
    get id -1. With ``valid=None`` and P >= k no sentinel pass runs, matching
    the dense call sites that had no -inf handling. ``pad_to_k=False`` with
    P < k returns min(k, P) columns."""
    b, p = scores.shape
    if valid is not None:
        # One pass; masked_fill(~valid) clones the scores and inverts the mask first.
        scores = torch.where(valid, scores, float("-inf"))
    actual_k = min(k, p)
    topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
    topk_ids = gather_ids.gather(1, topk_local) if gather_ids is not None else topk_local
    if valid is not None or actual_k < k:
        topk_ids = torch.where(torch.isfinite(topk_scores), topk_ids, topk_ids.new_full((), -1))
    if pad_to_k and actual_k < k:
        pad = k - actual_k
        topk_ids = torch.cat([topk_ids, topk_ids.new_full((b, pad), -1)], dim=1)
        topk_scores = torch.cat([topk_scores, topk_scores.new_full((b, pad), float("-inf"))], dim=1)
    return topk_ids, topk_scores


def compact_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    """Compact a [B, N] bool mask to (positive_indices [B, N], counts [B]).

    Same contract as the triton ``bloom_compact``/``clause_compact`` kernels, so torch and
    triton paths stay interchangeable: all three return full-width ``[B, N]`` indices with
    only the first ``counts[b]`` entries of each row meaningful. The tails differ — arbitrary
    ids (argsort tail) here vs ``-1`` (written by the kernels) — so consumers must bound
    reads by ``counts`` either way."""
    counts = mask.sum(dim=1)
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    return sorted_idx, counts


def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """Element-wise AND of N optional ``[B, N]`` masks.

    ``None`` inputs are ignored. Returns ``None`` if every input is ``None``."""
    out: Tensor | None = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else (out & m)
    return out


def combine_indices(
    filters: Sequence[FilterModule],
    query_clause_attrs: Sequence[Tensor],
) -> tuple[Tensor, Tensor]:
    """Sparse cascade across filters: the first produces ``(ids, counts)`` via its compact path,
    then each subsequent filter is applied to those ids via ``evaluate_subset`` and survivors
    re-compacted (no ``[B, N]`` materialized by the cascade). Order filters most-selective first.

    Each cascade stage does a host sync (``new_counts.max().item()``) to size the next id
    tensor — don't put this in a cudagraph-captured path; it's meant for offline composition."""
    if len(filters) == 0:
        raise ValueError("combine_indices requires at least one filter")
    if len(filters) != len(query_clause_attrs):
        raise ValueError(
            f"filters / queries length mismatch: {len(filters)} vs {len(query_clause_attrs)}",
        )

    f0, q0 = filters[0], query_clause_attrs[0]
    ids, counts = f0.evaluate_indices(q0)

    for f, q in zip(filters[1:], query_clause_attrs[1:], strict=True):
        b, p = ids.shape
        if p == 0:
            return ids, counts
        valid = counts_to_valid(counts, p)
        # ids past counts[b] are -1 (Triton) or argsort leftovers (torch); gather as 0, then
        # sub_mask & valid zeroes them out.
        safe_ids = torch.where(valid, ids, ids.new_zeros(()))
        sub_mask = f.evaluate_subset(q, safe_ids)  # [B, P] bool
        sub_mask = sub_mask & valid
        new_counts = sub_mask.sum(dim=1)
        new_p = int(new_counts.max().item())
        if new_p == 0:
            ids = torch.empty(b, 0, dtype=torch.long, device=ids.device)
            counts = new_counts
            continue
        sorted_idx = sub_mask.float().argsort(dim=1, descending=True, stable=True)[:, :new_p]
        ids = ids.gather(1, sorted_idx)
        counts = new_counts

    return ids, counts


def post_filter_topk(
    topk_ids: Tensor,
    post_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply a post-filter to pre-computed top-K results.

    Returns ``(ids, counts)`` where filtered positions carry ``id = -1`` and *counts* is the
    number of surviving items per query."""
    keep = post_mask.gather(1, topk_ids)
    topk_ids = topk_ids.masked_fill(~keep, -1)
    counts = keep.sum(dim=1)
    return topk_ids, counts


def popcount_int64(x: Tensor) -> Tensor:
    """Hamming weight (popcount) over an int64 tensor of any shape.

    Bit-twiddle algorithm — matches the Triton-side ``retrieve.ops.triton.common.popcount_int64``
    so torch reference and Triton kernel agree bit-exact."""
    if x.dtype != torch.int64:
        raise TypeError(f"popcount_int64 expects int64, got {x.dtype}")
    M1 = 0x5555555555555555
    M2 = 0x3333333333333333
    M4 = 0x0F0F0F0F0F0F0F0F
    H01 = 0x0101010101010101
    x = x - ((x >> 1) & M1)
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    return ((x * H01) >> 56).to(torch.int32)


def clause_subset_match(
    gathered_attrs: Tensor,
    query_attrs: Tensor,
    clause_is_reverse: Tensor,
) -> Tensor:
    """``[B, P, C, A]`` attrs vs ``[B, C]`` query → ``[B, P]`` bool. Reverse-XOR and
    ``q == -1`` inactive semantics — the single torch-side definition, shared by
    ``ExactAttributeFilter.evaluate_subset`` and the reference exact scorer."""
    q = query_attrs.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
    clause_match = (gathered_attrs == q).any(dim=-1)  # [B, P, C]
    rev = clause_is_reverse.unsqueeze(0).unsqueeze(0)  # [1, 1, C]
    clause_match = torch.where(rev, ~clause_match, clause_match)
    inactive = (query_attrs == -1).unsqueeze(1)  # [B, 1, C]
    return (clause_match | inactive).all(dim=-1)  # [B, P]


def bloom_subset_match(qb: Tensor, sigs: Tensor) -> Tensor:
    """``[B, W]`` query sigs vs ``[B, P, W]`` gathered sigs → ``[B, P]`` bool.

    Per-word subset test ``(qb & sig) == qb``, AND-reduced over words — the
    single torch-side definition, shared by ``BloomFilter.evaluate_subset`` and
    the reference bloom scorer."""
    match = (qb.unsqueeze(1) & sigs) == qb.unsqueeze(1)  # [B, P, W]
    return match.all(dim=-1)


__all__ = [
    "bloom_subset_match",
    "clause_subset_match",
    "combine_indices",
    "combine_masks",
    "compact_mask",
    "counts_to_valid",
    "masked_topk",
    "popcount_int64",
    "post_filter_topk",
]
