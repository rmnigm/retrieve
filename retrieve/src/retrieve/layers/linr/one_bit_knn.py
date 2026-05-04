from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import RetrievalModule
from retrieve.layers.utils.quantize import (
    popcount_int64,
    project_oporp_1bit_query,
    quantize_oporp_1bit,
)


def _score_full_oporp_eager(
    query_bits: Tensor,
    item_bits: Tensor,
) -> Tensor:
    """Loop-free body of ``OneBitKNN._score_full``.

    Free-function form so ``torch.compile`` traces a single graph reused
    across instances. ``d_total`` is derived from ``item_bits.shape[1]``
    inside the body so it stays symbolic under ``dynamic=True``.
    """
    d_total = 64 * item_bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
    hamming = popcount_int64(xor).sum(dim=-1)
    return d_total - 2 * hamming.to(torch.float32)


# Compile the xor + popcount + reduce chain. The win here is *fusion*: eager
# materializes the [B, N, W] int64 xor once, then re-reads it through six
# popcount bit-twiddle ops; Inductor fuses the popcount chain into one
# elementwise triton kernel that streams the xor tensor in a single pass.
# ``dynamic=True`` keeps B/N/W symbolic so a single graph handles every
# OneBitKNN instance and query batch size.
_score_full_oporp_compiled = torch.compile(
    _score_full_oporp_eager, dynamic=True, mode="reduce-overhead"
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
        if query_bits.is_cuda:
            return _score_full_oporp_compiled(query_bits, self.item_bits)
        return _score_full_oporp_eager(query_bits, self.item_bits)

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
        if mask is not None:
            topk_ids = torch.where(
                torch.isfinite(topk_scores),
                topk_ids,
                topk_ids.new_full((), -1),
            )
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
