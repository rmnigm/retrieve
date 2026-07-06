from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import Backend, RetrievalModule
from retrieve.kernels.linr.oporp_1bit_match_topk import (
    oporp_1bit_match_topk_full,
    oporp_1bit_match_topk_indirect,
)
from retrieve.layers.utils.quantize import popcount_int64
from retrieve.layers.utils.topk import counts_to_valid, masked_topk


def _score_full_bits_eager(
    query_bits: Tensor,
    item_bits: Tensor,
) -> Tensor:
    """Loop-free xor + popcount + reduce, shared by both backends; ``d_total`` is computed inside
    the body so it stays symbolic under a ``dynamic=True`` parent compile."""
    d_total = 64 * item_bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
    hamming = popcount_int64(xor).sum(dim=-1)
    return d_total - 2 * hamming.to(torch.float32)


class _PackedBitsKNN(RetrievalModule):
    """Hamming-similarity KNN over packed int64 sign bits: score = ``64*W - 2*popcount(q ^
    item)``. Subclasses own bit production (``_quantize_index`` / ``_project_query``); scoring,
    backend dispatch, and the candidates/full paths are common.

    Both backends produce byte-identical bits (the projection is shared), so torch and Triton
    paths score identically — the parity property the cross-backend tests assert."""

    item_bits: Tensor

    def __init__(self, k: int, backend: Backend = "triton") -> None:
        super().__init__()
        self.k = k
        self.backend = backend

    # -- subclass hooks -----------------------------------------------------

    def _quantize_index(self, item_embs: Tensor) -> None:
        """Build + register ``item_bits`` and the projection buffers."""
        raise NotImplementedError

    def _project_query(self, query: Tensor) -> Tensor:
        """[B, D] → [B, W] int64, same bit space as ``item_bits``."""
        raise NotImplementedError

    # -- common ---------------------------------------------------------------

    def register_index(self, item_embs: Tensor) -> None:
        n = item_embs.shape[0]
        # Full-scan topk runs over [B, N] with no pad tail, so the corpus must hold >= k items
        # (indirect path is safe at any width).
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds corpus size N={n}")
        self._quantize_index(item_embs)

    @property
    def d_total(self) -> int:
        return 64 * self.item_bits.shape[1]

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
        query_bits = self._project_query(query)
        if candidate_ids is not None:
            cand_bits = self.item_bits[candidate_ids]  # [B, P, W]
            xor = query_bits.unsqueeze(1) ^ cand_bits
            hamming = popcount_int64(xor).sum(dim=-1)
            scores = (self.d_total - 2 * hamming).to(torch.float32)
            valid = (
                counts_to_valid(counts, candidate_ids.shape[1]) if counts is not None else None
            )
            # pad_to_k=False: the candidates path returns min(k, P) columns (no -1/-inf tail) —
            # frozen behavior; callers bound short rows by counts.
            return masked_topk(
                scores, self.k, valid=valid, gather_ids=candidate_ids, pad_to_k=False
            )

        scores = _score_full_bits_eager(query_bits, self.item_bits)
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        return topk_ids, topk_scores

    def _forward_triton(
        self,
        query: Tensor,
        candidate_ids: Tensor | None,
        counts: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Fused-kernel path: full-scan with contiguous word loads, or indirect loads through
        ``candidate_ids`` gated by per-row ``counts`` — same XOR + popcount + ``D - 2*hamming``
        either way. Kernel input is ``[N, W]`` int64; the projection that produced the bits is
        opaque."""
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
