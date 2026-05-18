from __future__ import annotations

import torch
from torch import Tensor

from retrieve.interfaces import Backend, RetrievalModule
from retrieve.kernels.triton.linr.oporp_1bit_match_topk import oporp_1bit_match_topk
from retrieve.layers.utils.compact import compact_mask
from retrieve.layers.utils.quantize import (
    _project_oporp_1bit_query_eager,
    popcount_int64,
    project_oporp_1bit_query,
    quantize_oporp_1bit,
)


def _score_full_oporp_eager(
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


class OneBitKNN(RetrievalModule):
    """1-bit Sign-OPORP scoring (Hamming similarity), selectable backend.

    Item embeddings are projected via a deterministic Sign-OPORP transform
    (cheap O(D) sign vector + permutation) and sign-quantized to 1 bit per
    dim. Scoring is ``D - 2 * popcount(query_bits ^ item_bits)`` — purely
    bitwise, 16× memory reduction vs fp16. See docs/system/architecture.md.

    With ``backend="triton"`` (default), all three paths route through the
    fused ``oporp_1bit_match_topk`` kernel: full-scan and indirect-load
    versions share one kernel via the ``HAS_INDICES`` constexpr.

    With ``backend="torch"``, the same op chain runs eager — callers that
    want Inductor fusion + cudagraph capture should wrap the module with
    ``torch.compile`` themselves. Decoupled from any filter — callers
    compute ``mask`` or ``candidate_ids`` upstream.
    """

    item_bits: Tensor
    oporp_signs: Tensor
    oporp_perm: Tensor

    def __init__(self, k: int, seed: int = 0, backend: Backend = "triton") -> None:
        super().__init__()
        self.k = k
        self.seed = seed
        self.backend = backend

    def register_index(self, item_embs: Tensor) -> None:
        bits, signs, perm = quantize_oporp_1bit(item_embs, seed=self.seed)
        self.register_buffer("item_bits", bits)
        self.register_buffer("oporp_signs", signs)
        self.register_buffer("oporp_perm", perm)

    @property
    def d_total(self) -> int:
        return 64 * self.item_bits.shape[1]

    def _project_query(self, query: Tensor) -> Tensor:
        """Triton-backend query projection — routes through the standalone
        ``project_oporp_1bit_query`` (compiled on CUDA, eager on CPU). The
        torch backend traces ``_project_oporp_1bit_query_eager`` directly
        inside ``_forward_torch_eager`` to avoid a compile chain.
        """
        return project_oporp_1bit_query(query, self.oporp_signs, self.oporp_perm)

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.backend == "triton":
            return self._forward_triton(query, mask, candidate_ids)
        return self._forward_torch_eager(query, mask, candidate_ids)

    def _forward_torch_eager(
        self,
        query: Tensor,
        mask: Tensor | None,
        candidate_ids: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        query_bits = _project_oporp_1bit_query_eager(
            query, self.oporp_signs, self.oporp_perm
        )
        if candidate_ids is not None:
            cand_bits = self.item_bits[candidate_ids]  # [B, P, W]
            xor = query_bits.unsqueeze(1) ^ cand_bits
            hamming = popcount_int64(xor).sum(dim=-1)
            scores = (self.d_total - 2 * hamming).to(torch.float32)
            actual_k = min(self.k, scores.shape[1])
            topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
            topk_ids = candidate_ids.gather(1, topk_local)
            return topk_ids, topk_scores

        scores = _score_full_oporp_eager(query_bits, self.item_bits)
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

    def _forward_triton(
        self,
        query: Tensor,
        mask: Tensor | None,
        candidate_ids: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Fused-kernel path for all three sub-cases.

        Full path scans every item with contiguous int64-word loads; masked /
        candidate paths use indirect loads through a positive-indices buffer.
        Same operation either way — XOR + popcount + ``D - 2 * hamming`` — so
        there's no separate dequant or fp32 dot. Popcount is cheap enough
        that the gather penalty never crosses the dense-fallback break-even
        point; a single sparse path covers all cases.
        """
        query_bits = self._project_query(query)

        if candidate_ids is not None:
            counts = torch.full(
                (candidate_ids.shape[0],),
                candidate_ids.shape[1],
                dtype=torch.long,
                device=query.device,
            )
            return oporp_1bit_match_topk(
                query_bits,
                self.item_bits,
                self.k,
                positive_indices=candidate_ids,
                counts=counts,
            )

        if mask is None:
            return oporp_1bit_match_topk(query_bits, self.item_bits, self.k)

        positive_indices, counts = compact_mask(mask)
        if int(counts.max().item()) == 0:
            b = query.shape[0]
            device = query.device
            return (
                torch.full((b, self.k), -1, dtype=torch.long, device=device),
                torch.full((b, self.k), float("-inf"), device=device),
            )
        return oporp_1bit_match_topk(
            query_bits,
            self.item_bits,
            self.k,
            positive_indices=positive_indices,
            counts=counts,
        )
