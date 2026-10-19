"""Query-time torch glue that is not a kernel: the masked top-K epilogue and its two-level
top-k, mask compaction and composition, the post-filter, the torch popcount and the exact
subset predicate shared by ``ExactAttributeFilter`` and ``retrieve.ops.reference``.

Everything here is pure tensor flow (no ``.item()``, no data-dependent Python branch beyond
``min(k, P)`` over static shapes) and traces under ``torch.compile(dynamic=True,
mode="reduce-overhead")``, except ``combine_indices``, which syncs per cascade stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from retrieve.interfaces import FilterModule


# The two-level top-k (docs/system/kernels.md § Top-k): from TOPK_MIN_SLOTS slots of [B, P],
# where a row has at least TOPK_MIN_RATIO whole TOPK_BLOCK-slot blocks per top-k slot, the top-k
# runs on the items of its k best blocks (and the ragged tail).
TOPK_BLOCK = 256
TOPK_MIN_SLOTS = 1 << 23
TOPK_MIN_RATIO = 8


def two_level_topk(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """``torch.topk(scores, k, dim=1)`` up to ties: each whole block's max, the row's ``k`` best
    blocks, and the top-k of their items plus the ``< TOPK_BLOCK`` tail slots. They hold the
    row's top-k: an item in it has its block's max at least as large, and were that block not
    among the ``k`` best, ``k`` other blocks would each hold an item as large. Scores
    ``torch.equal``, ids equal up to ties. Below the thresholds, plain ``torch.topk``."""
    b, p = scores.shape
    n_blocks = p // TOPK_BLOCK
    if b * p < TOPK_MIN_SLOTS or n_blocks < TOPK_MIN_RATIO * k:
        return torch.topk(scores, k, dim=1)
    full = n_blocks * TOPK_BLOCK
    block_max = scores[:, :full].view(b, n_blocks, TOPK_BLOCK).amax(dim=2)
    blocks = torch.topk(block_max, k, dim=1, sorted=False).indices
    lane = torch.arange(TOPK_BLOCK, device=scores.device)
    cand = (blocks.unsqueeze(2) * TOPK_BLOCK + lane).view(b, k * TOPK_BLOCK)
    if full < p:
        tail = torch.arange(full, p, device=scores.device).expand(b, p - full)
        cand = torch.cat([cand, tail], dim=1)
    top, at = torch.topk(scores.gather(1, cand), k, dim=1)
    return top, cand.gather(1, at)


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
    masked: bool = False,  # scores already carry -inf at invalid lanes (a fused mask)
) -> tuple[Tensor, Tensor]:
    """Mask invalid scores to -inf → topk → map local→global ids → replace non-finite winners
    with -1 → optionally pad to k. Returns (ids, scores).

    The -1 sentinel is applied whenever ``valid`` is given or ``masked`` (or P < k): winners
    with a non-finite score (masked lanes and genuine -inf scores alike) get id -1. With
    ``valid=None`` and P >= k no sentinel pass runs. ``pad_to_k=False`` with P < k returns
    min(k, P) columns."""
    b, p = scores.shape
    if valid is not None:
        scores = torch.where(valid, scores, float("-inf"))
    actual_k = min(k, p)
    topk_scores, topk_local = two_level_topk(scores, actual_k)
    topk_ids = gather_ids.gather(1, topk_local) if gather_ids is not None else topk_local
    if valid is not None or masked or actual_k < k:
        topk_ids = torch.where(torch.isfinite(topk_scores), topk_ids, topk_ids.new_full((), -1))
    if pad_to_k and actual_k < k:
        pad = k - actual_k
        topk_ids = torch.cat([topk_ids, topk_ids.new_full((b, pad), -1)], dim=1)
        topk_scores = torch.cat([topk_scores, topk_scores.new_full((b, pad), float("-inf"))], dim=1)
    return topk_ids, topk_scores


def compact_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    """A [B, N] bool mask → (positive_indices [B, N], counts [B]): the same full-width contract
    as the ``clause_compact`` kernel, so torch and triton paths stay interchangeable, with only
    the first ``counts[b]`` entries of each row meaningful. The tails differ (arbitrary ids from
    the argsort here, unwritten memory in the kernel), so consumers bound reads by ``counts``."""
    counts = mask.sum(dim=1)
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    return sorted_idx, counts


def combine_masks(*masks: Tensor | None) -> Tensor | None:
    """Element-wise AND of N optional ``[B, N]`` masks; ``None`` inputs are ignored, all-``None``
    returns ``None``."""
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
    Each stage syncs (``new_counts.max().item()``) to size the next id tensor, so this is for
    offline composition, not a captured path."""
    if len(filters) == 0:
        raise ValueError("combine_indices requires at least one filter")
    if len(filters) != len(query_clause_attrs):
        raise ValueError(
            f"filters / queries length mismatch: {len(filters)} vs {len(query_clause_attrs)}",
        )

    f0, q0 = filters[0], query_clause_attrs[0]
    ids, counts = f0.evaluate_indices(f0.prepare_queries(q0))

    for f, q in zip(filters[1:], query_clause_attrs[1:], strict=True):
        p = ids.shape[1]
        if p == 0:
            return ids, counts
        valid = counts_to_valid(counts, p)
        # ids past counts[b] are unwritten or argsort leftovers: gather as 0, then mask them out.
        safe_ids = torch.where(valid, ids, ids.new_zeros(()))
        sub_mask = f.evaluate_subset(f.prepare_queries(q), safe_ids)  # [B, P] bool
        sub_mask = sub_mask & valid
        new_counts = sub_mask.sum(dim=1)
        new_p = int(new_counts.max().item())
        sorted_idx = sub_mask.float().argsort(dim=1, descending=True, stable=True)[:, :new_p]
        ids = ids.gather(1, sorted_idx)
        counts = new_counts

    return ids, counts


def post_filter_topk(
    topk_ids: Tensor,
    post_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """A post-filter over pre-computed top-K ids: ``(ids, counts)`` with filtered positions at
    ``-1`` and the number of surviving items per query."""
    keep = post_mask.gather(1, topk_ids)
    topk_ids = topk_ids.masked_fill(~keep, -1)
    counts = keep.sum(dim=1)
    return topk_ids, counts


def popcount_int64(x: Tensor) -> Tensor:
    """Popcount over an int64 tensor of any shape, as int32. SWAR bit-twiddling (this torch has no
    ``bitwise_count``); exact, so it equals the Triton side's hardware ``POPC`` bit for bit."""
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
    """``[B, P, C, A]`` attrs vs ``[B, C]`` query → ``[B, P]`` bool, with the reverse-XOR and
    ``q == -1`` inactive semantics: the one torch-side definition of the exact predicate."""
    q = query_attrs.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
    clause_match = (gathered_attrs == q).any(dim=-1)  # [B, P, C]
    rev = clause_is_reverse.unsqueeze(0).unsqueeze(0)  # [1, 1, C]
    clause_match = torch.where(rev, ~clause_match, clause_match)
    inactive = (query_attrs == -1).unsqueeze(1)  # [B, 1, C]
    return (clause_match | inactive).all(dim=-1)  # [B, P]


__all__ = [
    "clause_subset_match",
    "combine_indices",
    "combine_masks",
    "compact_mask",
    "counts_to_valid",
    "masked_topk",
    "popcount_int64",
    "post_filter_topk",
    "two_level_topk",
]
