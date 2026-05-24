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
    project_simhash_1bit_query,
    quantize_simhash_1bit,
)


def _score_full_simhash_eager(
    query_bits: Tensor,
    item_bits: Tensor,
) -> Tensor:
    """Loop-free xor + popcount + reduce body, shared by both backends.

    ``d_total`` is derived from ``item_bits.shape[1]`` inside the body so it
    stays symbolic when this function is called inside a ``dynamic=True``
    parent compile.
    """
    d_total = 64 * item_bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
    hamming = popcount_int64(xor).sum(dim=-1)
    return d_total - 2 * hamming.to(torch.float32)


class SimHashKNN(nn.Module):
    """SimHash 1-bit Hamming scoring (Charikar 2002 / Manku 2007), selectable backend.

    Item embeddings are projected via a fixed Gaussian ``R ∈ R^{k_bits × D}``
    and sign-quantized to 1 bit per output dim. Scoring is
    ``k_bits - 2 * popcount(query_bits ^ item_bits)`` — purely bitwise, the
    same kernel as ``OneBitKNN``.

    Unlike Sign-OPORP, ``k_bits`` can exceed ``D`` for a recall-vs-memory
    trade — each output bit mixes all input coordinates. Cost is one fp32
    matmul (``[N, D] @ [D, k_bits]``) at register time plus a per-query
    matmul + bit-pack in forward.

    Reuses the OPORP Triton kernel — the kernel reads only
    ``W = item_bits.shape[1]``; the algorithm that produced the bits is opaque.
    SimHash at k_bits=512 → W=8; OPORP at k_bits=64 with D=128 → W=1; same
    kernel both cases.
    """

    item_bits: Tensor
    simhash_R: Tensor

    def __init__(
        self,
        k: int,
        k_bits: int,
        seed: int = 0,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        self.k = k
        self.k_bits = k_bits
        self.seed = seed
        self.backend = backend

    def register_index(self, item_embs: Tensor) -> None:
        n = item_embs.shape[0]
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds corpus size N={n}")
        bits, r = quantize_simhash_1bit(item_embs, self.k_bits, self.seed)
        self.register_buffer("item_bits", bits)
        self.register_buffer("simhash_R", r)

    @property
    def d_total(self) -> int:
        return 64 * self.item_bits.shape[1]

    def _project_query(self, query: Tensor) -> Tensor:
        """Pure tensor-flow query projection — shared by both backends."""
        return project_simhash_1bit_query(query, self.simhash_R)

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
        query_bits = project_simhash_1bit_query(query, self.simhash_R)
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

        scores = _score_full_simhash_eager(query_bits, self.item_bits)
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        return topk_ids, topk_scores

    def _forward_triton(
        self,
        query: Tensor,
        candidate_ids: Tensor | None,
        counts: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Fused-kernel path for full-scan and indirect-load.

        Same kernel as ``OneBitKNN`` — XOR + popcount + ``D - 2 * hamming``.
        Kernel input is ``[N, W]`` int64; the projection algorithm that
        produced the bits is opaque to it.
        """
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
