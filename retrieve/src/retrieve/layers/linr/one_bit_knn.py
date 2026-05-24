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
    """Loop-free xor + popcount + reduce body, shared by both backends.

    ``d_total`` is derived from ``item_bits.shape[1]`` inside the body so it
    stays symbolic when this function is called inside a ``dynamic=True``
    parent compile.
    """
    d_total = 64 * item_bits.shape[1]
    xor = query_bits.unsqueeze(1) ^ item_bits.unsqueeze(0)
    hamming = popcount_int64(xor).sum(dim=-1)
    return d_total - 2 * hamming.to(torch.float32)


class OneBitKNN(nn.Module):
    """1-bit Sign-OPORP scoring (Hamming similarity), selectable backend.

    Item embeddings are projected via a deterministic Sign-OPORP transform
    (cheap O(D) sign vector + permutation) and sign-quantized to 1 bit per
    dim. Scoring is ``D - 2 * popcount(query_bits ^ item_bits)`` — purely
    bitwise, 16× memory reduction vs fp16. See docs/system/architecture.md.

    **Precision.** ``item_embs`` and ``query`` may be fp32 or fp16; the
    Sign-OPORP projection is sign-stable across float dtypes (sign of a
    non-zero fp32 value equals sign of its fp16 round) and produces
    identical packed-bit buffers either way. No internal cast is needed —
    input precision is discarded at bit-pack time.

    With ``backend="triton"`` (default), both paths route through fused
    custom_ops — ``oporp_1bit_match_topk_full`` for the dense case,
    ``oporp_1bit_match_topk_indirect`` when ``candidate_ids`` is given.
    Both delegate to the same underlying Triton kernel (shared via the
    ``HAS_INDICES`` constexpr).

    With ``backend="torch"``, the same op chain runs eager — callers that
    want Inductor fusion + cudagraph capture should wrap the module with
    ``torch.compile`` themselves. Decoupled from any filter — callers
    compute ``candidate_ids`` (and optionally per-row ``counts``) upstream.
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
        n = item_embs.shape[0]
        # The full-scan ``@triton_op`` wrapper runs ``torch.topk(all_scores,
        # self.k)`` over an ``[B, n_items_total]`` buffer with no pad tail,
        # so the corpus must hold at least k items. The indirect path is
        # safe at any candidate width — its score buffer is widened to
        # ``max(_bucket_n(n_loop), _bucket_n(k))`` inside the kernel
        # wrapper.
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds corpus size N={n}")
        bits, signs, perm = quantize_oporp_1bit(item_embs, seed=self.seed)
        self.register_buffer("item_bits", bits)
        self.register_buffer("oporp_signs", signs)
        self.register_buffer("oporp_perm", perm)

    @property
    def d_total(self) -> int:
        return 64 * self.item_bits.shape[1]

    def _project_query(self, query: Tensor) -> Tensor:
        """Pure tensor-flow query projection — shared by both backends."""
        return project_oporp_1bit_query(query, self.oporp_signs, self.oporp_perm)

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
        query_bits = project_oporp_1bit_query(query, self.oporp_signs, self.oporp_perm)
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
        """Fused-kernel path for full-scan and indirect-load.

        Full path scans every item with contiguous int64-word loads; the
        indirect path uses indirect loads through ``candidate_ids`` gated
        by per-row ``counts``. Same operation either way — XOR + popcount
        + ``D - 2 * hamming`` — so there's no separate dequant or fp32
        dot. Popcount is cheap enough that the gather penalty never
        crosses the dense-fallback break-even point; a single sparse path
        covers all cases.
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
