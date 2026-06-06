from __future__ import annotations

import torch
from torch import Tensor, nn

from retrieve.interfaces import Backend
from retrieve.kernels.linr.oporp_1bit_match_topk import (
    oporp_1bit_match_topk_full,
    oporp_1bit_match_topk_indirect,
)
from retrieve.layers.utils.quantize import (
    popcount_int64,
    project_oporp_1bit_query,
    quantize_oporp_1bit,
)


def _score_full_oporp_eager(
    query_bits: Tensor,
    item_bits: Tensor,
) -> Tensor:
    """Loop-free xor + popcount + reduce, shared by both backends; ``d_total`` is computed inside
    the body so it stays symbolic under a ``dynamic=True`` parent compile."""
    d_total = 64 * item_bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
    hamming = popcount_int64(xor).sum(dim=-1)
    return d_total - 2 * hamming.to(torch.float32)


class OneBitKNN(nn.Module):
    """1-bit Sign-OPORP scoring with selectable backend: Hamming similarity ``D - 2*popcount(q ^
    item)``, 16× smaller than fp16. Decoupled from filtering — callers pass ``candidate_ids``
    (and optional per-row ``counts``) computed upstream.

    ``backend="triton"`` (default) routes both the full-scan and ``candidate_ids`` paths through
    one fused kernel; ``backend="torch"`` runs the same op chain eager."""

    item_bits: Tensor
    oporp_signs: Tensor
    oporp_perm: Tensor

    def __init__(
        self,
        k: int,
        seed: int = 0,
        backend: Backend = "triton",
        k_bits: int = 0,
    ) -> None:
        super().__init__()
        self.k = k
        self.seed = seed
        self.backend = backend
        # k_bits=0 sentinel ("use D from register_index"); a plain int (never None) so Dynamo sees
        # no Optional attr.
        self.k_bits = k_bits

    def register_index(self, item_embs: Tensor) -> None:
        n = item_embs.shape[0]
        # Full-scan topk runs over [B, N] with no pad tail, so the corpus must hold >= k items
        # (indirect path is safe at any width).
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds corpus size N={n}")
        bits, signs, perm = quantize_oporp_1bit(item_embs, seed=self.seed, k_bits=self.k_bits)
        # Resolve the 0 sentinel to a concrete int so _project_query sees a stable Python int under
        # Dynamo.
        if self.k_bits == 0:
            self.k_bits = item_embs.shape[1]
        self.register_buffer("item_bits", bits)
        self.register_buffer("oporp_signs", signs)
        self.register_buffer("oporp_perm", perm)

    @property
    def d_total(self) -> int:
        return 64 * self.item_bits.shape[1]

    def _project_query(self, query: Tensor) -> Tensor:
        """Pure tensor-flow query projection — shared by both backends."""
        return project_oporp_1bit_query(query, self.oporp_signs, self.oporp_perm, self.k_bits)

    def forward(
        self,
        query: Tensor,
        candidate_ids: Tensor | None = None,
        counts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.backend == "triton":
            return self._forward_triton(query, candidate_ids, counts)
        return self._forward_torch_eager(query, candidate_ids, counts)

    def _forward_torch_eager(
        self,
        query: Tensor,
        candidate_ids: Tensor | None,
        counts: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        query_bits = project_oporp_1bit_query(
            query, self.oporp_signs, self.oporp_perm, self.k_bits
        )
        if candidate_ids is not None:
            cand_bits = self.item_bits[candidate_ids]  # [B, P, W]
            xor = query_bits.unsqueeze(1) ^ cand_bits
            hamming = popcount_int64(xor).sum(dim=-1)
            scores = (self.d_total - 2 * hamming).to(torch.float32)
            if counts is not None:
                p = candidate_ids.shape[1]
                valid = torch.arange(p, device=candidate_ids.device).unsqueeze(
                    0
                ) < counts.unsqueeze(1)
                scores = scores.masked_fill(~valid, float("-inf"))
            actual_k = min(self.k, scores.shape[1])
            topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
            topk_ids = candidate_ids.gather(1, topk_local)
            if counts is not None:
                topk_ids = torch.where(
                    torch.isfinite(topk_scores),
                    topk_ids,
                    topk_ids.new_full((), -1),
                )
            return topk_ids, topk_scores

        scores = _score_full_oporp_eager(query_bits, self.item_bits)
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        return topk_ids, topk_scores

    def _forward_triton(
        self,
        query: Tensor,
        candidate_ids: Tensor | None,
        counts: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Fused-kernel path: full-scan with contiguous word loads, or indirect loads through
        ``candidate_ids`` gated by per-row ``counts``. Same XOR + popcount + ``D - 2*hamming``
        either way."""
        query_bits = self._project_query(query)

        if candidate_ids is None:
            return oporp_1bit_match_topk_full(query_bits, self.item_bits, self.k)

        if counts is None:
            counts = torch.full(
                (candidate_ids.shape[0],),
                candidate_ids.shape[1],
                dtype=torch.long,
                device=query.device,
            )
        return oporp_1bit_match_topk_indirect(
            query_bits, self.item_bits, self.k, candidate_ids, counts
        )
