"""Dense and sparse KNN primitives over fp16 / int8 / fp32 item tables.

Precision contract: ``PostfilterKNN`` and ``PrefilterKNN`` store fp16 and accumulate dots in fp32
— cuBLAS (``PostfilterKNN``, ``PrefilterKNN(backend="torch")``) rounds the score to fp16 on output,
the fused Triton kernel (``PrefilterKNN(backend="triton")``) writes it as fp32.
``PostfilterKNNInt8`` is int32 end to end; ``FullScanKNN`` keeps the input dtype."""

from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import masked_topk, post_filter_topk
from retrieve.indexing.quantize import quantize_int8_global, quantize_int8_global_codes
from retrieve.interfaces import LinrBackend, RetrievalModule, check_backend, ops_for


class PostfilterKNN(RetrievalModule):
    """Pure-torch dense scoring (``query @ item_embs.T``) + optional boolean mask + top-K; inputs
    are cast to fp16 (storage fp16, fp32 accumulate). The ``backend=`` flag is accepted for API
    symmetry but has no effect — cuBLAS + CUB already match a fused kernel here."""

    item_embs_t: Tensor

    def __init__(self, k: int, backend: LinrBackend = "triton") -> None:
        super().__init__()
        check_backend(backend, LinrBackend)
        self.k = k
        self.backend = backend

    def register_index(self, item_embs: Tensor) -> None:
        # Pre-transpose to a contiguous D×N buffer; a .t() view at call time dispatches a different
        # kernel whose accumulator order can flip K-th-place tiebreaks at the noise floor.
        self.register_buffer("item_embs_t", item_embs.to(torch.float16).t().contiguous())

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        scores = query.to(torch.float16) @ self.item_embs_t
        if mask is not None:
            return masked_topk(scores, self.k, valid=mask)
        return masked_topk(scores, self.k)


class PostfilterKNNInt8(RetrievalModule):
    """Single-stage int8 dense scoring + optional mask + top-K, int32 end-to-end (SilverTorch §3.2):
    ``torch._int_mm`` (int8×int8 → int32) feeds ``torch.topk`` directly with no fp32
    intermediate, and the global scales make ``dot_int`` a monotonic transform of cosine, so
    topk ordering is exact up to ties introduced by the ``>>5`` range compression (boundary
    ties are quality-equivalent). Storage is one ``[D, N]`` int8 buffer, half of
    ``PostfilterKNN``; ``backend=`` has no effect."""

    # cuBLAS LtGemm's int8 kernel requires M >= 17 (see docs/system/kernels.md →
    # PostfilterKNNInt8); smaller batches are zero-padded to this M and sliced back.
    _PAD_M = 17

    item_codes_t: Tensor  # [D, N_padded] int8
    n_items: Tensor  # 0-d int64: N before padding (not recoverable from the padded table)

    def __init__(self, k: int, backend: LinrBackend = "triton") -> None:
        super().__init__()
        check_backend(backend, LinrBackend)
        self.k = k
        self.backend = backend
        self._n_real = 0
        self.register_load_state_dict_post_hook(_rederive_n_real)

    def register_index(self, item_embs: Tensor) -> None:
        codes, _ = quantize_int8_global(item_embs)  # [N, D] int8, chunked at build
        # _int_mm needs N (after transpose) a multiple of 8; pad with zero items (sliced off in
        # forward), quantize before padding so the global scale is unaffected.
        n = codes.shape[0]
        self._n_real = n
        pad_n = (-n) % 8
        if pad_n:
            pad = codes.new_zeros((pad_n, codes.shape[1]))
            codes = torch.cat([codes, pad], dim=0)
        self.register_buffer("item_codes_t", codes.t().contiguous())
        self.register_buffer("n_items", torch.tensor(n, dtype=torch.int64, device=codes.device))

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        q_codes = quantize_int8_global_codes(query)
        b = q_codes.shape[0]

        if b < self._PAD_M:
            pad_rows = self._PAD_M - b
            pad = q_codes.new_zeros((pad_rows, q_codes.shape[1]))
            q_codes_pad = torch.cat([q_codes, pad], dim=0)
            dots = torch._int_mm(q_codes_pad, self.item_codes_t)[:b]
        else:
            dots = torch._int_mm(q_codes, self.item_codes_t)
        dots = dots[:, : self._n_real]

        # int32 → fp16 for topk: >>5 brings worst-case |dot| ≈ D·127² (~2²¹) under fp16's ~2¹⁶ range
        # while preserving order; fp16 also halves CUB radix-select passes (2 vs 4).
        scores = (dots >> 5).to(torch.float16)
        if mask is not None:
            return masked_topk(scores, self.k, valid=mask)
        return masked_topk(scores, self.k)


def _rederive_n_real(module: PostfilterKNNInt8, incompatible_keys) -> None:
    """``load_state_dict`` post-hook: the forward slices the padded ``_int_mm`` output at the
    Python int ``_n_real`` (no per-call sync); a load replaces the buffer it came from."""
    if hasattr(module, "n_items"):
        module._n_real = int(module.n_items.item())


class PrefilterKNN(RetrievalModule):
    """Sparse-rescore KNN with selectable backend. Given ``candidate_ids: [B, P]`` (and optional
    per-row ``counts: [B]``) it scores only the passing rows and top-Ks them back to global ids;
    without ``candidate_ids`` it falls back to a dense full matmul. ``backend="triton"`` fuses
    the sparse path (no ``[B, P, D]`` intermediate); inputs are stored fp16 with fp32-accumulated
    dots on both backends (module docstring).

    Decoupled from filtering — callers compute ``(candidate_ids, counts)`` upstream."""

    item_embs: Tensor

    def __init__(self, k: int, backend: LinrBackend = "triton") -> None:
        super().__init__()
        check_backend(backend, LinrBackend)
        self.k = k
        self.backend = backend
        ops_for(backend)

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
            scores = query @ self.item_embs.t()
            topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
            return topk_ids, topk_scores
        b, p = candidate_ids.shape
        if counts is None:
            counts = torch.full((b,), p, dtype=torch.long, device=query.device)
        if p < self.k:
            # The op needs >= k columns; lanes past counts are never read, so -1 pads them.
            candidate_ids = torch.nn.functional.pad(candidate_ids, (0, self.k - p), value=-1)
        return ops_for(self.backend).fused_masked_knn_topk(
            query, self.item_embs, candidate_ids, counts, self.k
        )


class FullScanKNN(RetrievalModule):
    """Exhaustive matmul + top-K.

    `mask` implements POST-filter semantics (LiNR baseline): top-K is selected
    over the full corpus first, then masked hits are tombstoned to id=-1 — they
    are NOT replaced by the next-best passing items, and their scores remain in
    the returned score tensor. Recall against a pre-filter oracle is therefore
    expected to be < 1 by design. For pre-filter semantics use PostfilterKNN
    (mask before top-K) or PrefilterKNN (candidates path).

    ``post_filter_topk`` also returns the per-row survivor count; ``forward``
    discards it — callers needing counts call ``post_filter_topk`` directly.

    ``candidate_ids: [B, P]`` re-ranks the given ids only; ``-1`` entries are
    padding (the tail every compact producer emits) — never gathered, scored
    or returned. Rows with fewer than ``min(k, P)`` real candidates carry
    ``-1`` / ``-inf`` in the tail; ``P < k`` returns ``P`` columns."""

    item_embs: Tensor

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def register_index(self, item_embs: Tensor) -> None:
        self.register_buffer("item_embs", item_embs)

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
        candidate_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if candidate_ids is not None:
            return self._forward_candidates(query, candidate_ids)
        scores = query @ self.item_embs.t()
        topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
        if mask is not None:
            topk_ids, _ = post_filter_topk(topk_ids, mask)
        return topk_ids, topk_scores

    def _forward_candidates(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        valid = candidate_ids >= 0
        cand_embs = self.item_embs[candidate_ids.clamp_min(0)]
        scores = torch.bmm(query.unsqueeze(1), cand_embs.transpose(1, 2)).squeeze(1)
        return masked_topk(scores, self.k, valid=valid, gather_ids=candidate_ids, pad_to_k=False)
