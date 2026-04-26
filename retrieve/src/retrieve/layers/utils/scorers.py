from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import ScorerModule


class DotProductScorer(ScorerModule):
    """Score candidates via dot product between query and item embeddings."""

    item_embs: Tensor

    def __init__(self) -> None:
        super().__init__()

    def register_index(self, item_embs: Tensor) -> None:
        self.register_buffer("item_embs", item_embs)

    def forward(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        cand_embs = self.item_embs[candidate_ids]
        return torch.bmm(cand_embs, query.unsqueeze(-1)).squeeze(-1)
