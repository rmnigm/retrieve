from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule
from retrieve.layers.utils.quantize import (
    popcount_int64,
    project_oporp_1bit_query,
    quantize_oporp_1bit,
)


class OneBitKNN(RetrievalModule):
    """1-bit Sign-OPORP scoring (Hamming similarity).

    Item embeddings are projected via a deterministic Sign-OPORP transform
    (cheap O(D) sign vector + permutation) and sign-quantized to 1 bit per
    dim. Scoring is ``D - 2 * popcount(query_bits ^ item_bits)`` — purely
    bitwise, 16× memory reduction vs fp16. See docs/system/architecture.md.

    Decoupled from any filter — callers compute ``mask`` or ``candidate_ids``
    upstream.
    """

    item_bits: Tensor
    oporp_signs: Tensor
    oporp_perm: Tensor

    def __init__(self, k: int, seed: int = 0) -> None:
        super().__init__()
        self.k = k
        self.seed = seed

    def register_index(self, item_embs: Tensor) -> None:
        bits, signs, perm = quantize_oporp_1bit(item_embs, seed=self.seed)
        self.register_buffer("item_bits", bits)
        self.register_buffer("oporp_signs", signs)
        self.register_buffer("oporp_perm", perm)

    @property
    def d_total(self) -> int:
        return 64 * self.item_bits.shape[1]

    def _project_query(self, query: Tensor) -> Tensor:
        return project_oporp_1bit_query(query, self.oporp_signs, self.oporp_perm)

    def _score_full(self, query_bits: Tensor) -> Tensor:
        # [B, 1, W] xor [N, W] -> [B, N, W]; popcount + sum over W; convert.
        xor = query_bits.unsqueeze(1) ^ self.item_bits.unsqueeze(0)
        hamming = popcount_int64(xor).sum(dim=-1)
        return self.d_total - 2 * hamming.to(torch.float32)

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if candidate_ids is not None:
            return self._forward_candidates(query, candidate_ids)
        return self._forward_full(query, mask)

    def _forward_full(
        self,
        query: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        query_bits = self._project_query(query)
        scores = self._score_full(query_bits)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        return topk_ids, topk_scores

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        query_bits = self._project_query(query)
        cand_bits = self.item_bits[candidate_ids]  # [B, P, W]
        xor = query_bits.unsqueeze(1) ^ cand_bits
        hamming = popcount_int64(xor).sum(dim=-1)
        scores = (self.d_total - 2 * hamming).to(torch.float32)

        actual_k = min(self.k, scores.shape[1])
        topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
        topk_ids = candidate_ids.gather(1, topk_local)
        return topk_ids, topk_scores
