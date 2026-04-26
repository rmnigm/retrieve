from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule


class LiNR_V2(RetrievalModule):
    """LiNR V2, pure-torch reference.

    Sparse path: takes ``candidate_ids: [B, P]`` (passing item ids per query)
    and an optional ``counts: [B]`` (number of valid columns per row, defaults
    to all P). Gathers the passing rows into ``[B, P, D]`` via fancy indexing,
    scores with ``bmm``, top-K locally, gathers back to global ids.

    Without ``candidate_ids``, falls back to V1's dense full matmul.

    Decoupled from any filter — callers compute ``(candidate_ids, counts)``
    upstream (e.g. ``ClauseIndex.evaluate_indices``, ``compact_mask`` of an
    external mask, or directly from a quantized cascade like V3).
    """

    item_embs: Tensor

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def register_index(self, item_embs: Tensor) -> None:
        self.register_buffer("item_embs", item_embs)

    def forward(
        self,
        query: Tensor,
        candidate_ids: Tensor | None = None,
        counts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if candidate_ids is None:
            return self._forward_full(query)
        return self._forward_prefilter(query, candidate_ids, counts)

    def _forward_full(self, query: Tensor) -> tuple[Tensor, Tensor]:
        scores = query @ self.item_embs.t()
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        return topk_ids, topk_scores

    def _forward_prefilter(
        self,
        query: Tensor,
        candidate_ids: Tensor,
        counts: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        b, p = candidate_ids.shape
        device = query.device

        if p == 0:
            return (
                torch.full((b, self.k), -1, dtype=torch.long, device=device),
                torch.full((b, self.k), float("-inf"), device=device),
            )

        safe_ids = candidate_ids.clamp_min(0)
        reduced_embs = self.item_embs[safe_ids]
        scores = torch.bmm(query.unsqueeze(1), reduced_embs.transpose(1, 2)).squeeze(1)

        if counts is not None:
            valid = torch.arange(p, device=device).unsqueeze(0) < counts.unsqueeze(1)
            scores = scores.masked_fill(~valid, float("-inf"))

        actual_k = min(self.k, p)
        topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
        topk_ids = candidate_ids.gather(1, topk_local)

        if actual_k < self.k:
            pad = self.k - actual_k
            topk_ids = torch.cat(
                [topk_ids, torch.full((b, pad), -1, dtype=torch.long, device=device)],
                dim=1,
            )
            topk_scores = torch.cat(
                [topk_scores, torch.full((b, pad), float("-inf"), device=device)],
                dim=1,
            )
        return topk_ids, topk_scores


def build_linr_v2(item_embs: Tensor, k: int) -> LiNR_V2:
    module = LiNR_V2(k=k)
    module.register_index(item_embs)
    return module
