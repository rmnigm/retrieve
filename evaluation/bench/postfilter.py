"""The generic-torch baseline (docs/decisions.md § Harness; docs/system/evaluation.md
§ The postfilter baseline): what a practitioner writes without a retrieval library. Dense
scoring over the whole item table, ``torch.topk(alpha * k)``, the filter checked on those
candidates only, the first ``k`` survivors kept in rank order, short rows padded with the
``-1`` / ``-inf`` sentinel. Scoring is LiNR V1's (fp16 table, fp32 scores), so the two differ
only in where the filter runs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from retrieve.interfaces import FilterModule


class Postfilter(nn.Module):
    capturable = True

    item_embs_t: Tensor  # [D, N] fp16

    def __init__(
        self, k: int, *, filter: FilterModule | None, backend: str = "torch", alpha: int = 1
    ) -> None:
        super().__init__()
        if filter is None:
            raise ValueError("postfilter needs a filter")
        if backend != "torch":
            raise ValueError(f"postfilter runs on the torch backend only, got {backend!r}")
        self.k = k
        self.filter = filter
        self.backend = backend
        self.set_query_params(alpha=alpha)

    def register_index(self, item_embs: Tensor) -> None:
        self.register_buffer("item_embs_t", item_embs.to(torch.float16).t().contiguous())

    def set_query_params(self, *, alpha: int) -> None:
        if not isinstance(alpha, int) or alpha < 1:
            raise ValueError(f"alpha must be a positive int, got {alpha!r}")
        self.alpha = alpha

    def forward(self, query: Tensor, query_clause_attrs: Tensor) -> tuple[Tensor, Tensor]:
        query = query.to(torch.float16)
        if query.is_cuda:
            scores = torch.mm(query, self.item_embs_t, out_dtype=torch.float32)
        else:  # aten::mm.dtype has no CPU kernel
            scores = torch.mm(query.float(), self.item_embs_t.float())
        p = min(self.alpha * self.k, scores.shape[1])
        scores, ids = torch.topk(scores, p, dim=1)
        keep = self.filter.evaluate_subset(query_clause_attrs, ids)
        order = torch.sort(keep.to(torch.int8), dim=1, descending=True, stable=True).indices
        order = order[:, : self.k]
        keep = keep.gather(1, order)
        ids = ids.gather(1, order).masked_fill(~keep, -1)
        scores = scores.gather(1, order).masked_fill(~keep, float("-inf"))
        if p < self.k:
            pad = self.k - p
            ids = torch.cat([ids, ids.new_full((ids.shape[0], pad), -1)], dim=1)
            scores = torch.cat([scores, scores.new_full((ids.shape[0], pad), float("-inf"))], 1)
        return ids, scores
