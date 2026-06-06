from __future__ import annotations

import torch
from torch import Tensor, nn

from retrieve.interfaces import Backend
from retrieve.kernels.linr.fused_masked_knn_topk import fused_masked_knn_topk


class PrefilterKNN(nn.Module):
    """Sparse-rescore KNN with selectable backend. Given ``candidate_ids: [B, P]`` (and optional
    per-row ``counts: [B]``) it scores only the passing rows and top-Ks them back to global ids;
    without ``candidate_ids`` it falls back to a dense full matmul. ``backend="triton"`` fuses
    the sparse path (no ``[B, P, D]`` intermediate); inputs are stored fp16 with fp32-accumulated
    dots.

    Decoupled from filtering — callers compute ``(candidate_ids, counts)`` upstream."""

    item_embs: Tensor

    def __init__(self, k: int, backend: Backend = "triton") -> None:
        super().__init__()
        self.k = k
        self.backend = backend

    def register_index(self, item_embs: Tensor) -> None:
        self.register_buffer("item_embs", item_embs.to(torch.float16))

    def forward(
        self,
        query: Tensor,
        candidate_ids: Tensor | None = None,
        counts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        query = query.to(torch.float16)
        if candidate_ids is None:
            return self._forward_full(query)
        if self.backend == "triton":
            return self._forward_prefilter_triton(query, candidate_ids, counts)
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

    def _forward_prefilter_triton(
        self,
        query: Tensor,
        candidate_ids: Tensor,
        counts: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        b, p = candidate_ids.shape
        if p == 0:
            device = query.device
            return (
                torch.full((b, self.k), -1, dtype=torch.long, device=device),
                torch.full((b, self.k), float("-inf"), device=device),
            )
        if counts is None:
            counts = torch.full((b,), p, dtype=torch.long, device=query.device)
        return fused_masked_knn_topk(query, self.item_embs, candidate_ids, counts, self.k)
