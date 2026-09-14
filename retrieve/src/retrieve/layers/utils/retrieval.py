from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule
from retrieve.layers.utils.topk import masked_topk


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


class FullScanKNN(RetrievalModule):
    """Exhaustive matmul + top-K.

    `mask` implements POST-filter semantics (LiNR baseline): top-K is selected
    over the full corpus first, then masked hits are tombstoned to id=-1 — they
    are NOT replaced by the next-best passing items, and their scores remain in
    the returned score tensor. Recall against a pre-filter oracle is therefore
    expected to be < 1 by design. For pre-filter semantics use PostfilterKNN
    (mask before top-K) or PrefilterKNN (candidates path).

    ``post_filter_topk`` also returns the per-row survivor count; ``forward``
    discards it — callers needing counts call ``post_filter_topk`` directly.

    ``candidate_ids: [B, P]`` re-ranks the given ids only; ``-1`` entries are
    padding (the tail every compact producer emits) — never gathered, scored
    or returned. Rows with fewer than ``min(k, P)`` real candidates carry
    ``-1`` / ``-inf`` in the tail; ``P < k`` returns ``P`` columns."""

    item_embs: Tensor

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def register_index(self, item_embs: Tensor) -> None:
        self.register_buffer("item_embs", item_embs)

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if candidate_ids is not None:
            return self._forward_candidates(query, candidate_ids)
        scores = query @ self.item_embs.t()
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        if mask is not None:
            topk_ids, _ = post_filter_topk(topk_ids, mask)
        return topk_ids, topk_scores

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        valid = candidate_ids >= 0
        cand_embs = self.item_embs[candidate_ids.clamp_min(0)]
        scores = torch.bmm(query.unsqueeze(1), cand_embs.transpose(1, 2)).squeeze(1)
        return masked_topk(scores, self.k, valid=valid, gather_ids=candidate_ids, pad_to_k=False)
