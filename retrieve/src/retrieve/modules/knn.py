"""Dense and sparse KNN primitives over fp16 / int8 / fp32 item tables.

Precision contract: ``PostfilterKNN`` and ``PrefilterKNN`` store items fp16 (as the LiNR paper
does), cast the query to fp16, accumulate in fp32 and return fp32 scores on every backend — an
fp16 score would round near-tied items together (docs/system/kernels.md § Score conventions).
``FullScanKNN`` keeps the input dtype."""

from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import masked_topk, post_filter_topk
from retrieve.interfaces import LinrBackend, RetrievalModule, check_backend, ops_for


def calibration_rows(n: int, device: torch.device) -> Tensor:
    """8 item rows spread over ``[0, n)``, first and last included, in int64 arithmetic: a float32
    ``linspace`` rounds ``n - 1`` up to ``n`` past 2^24 items."""
    return torch.arange(8, device=device) * (n - 1) // 7


class PostfilterKNN(RetrievalModule):
    """Dense scoring (``query @ item_embs.T``) + optional boolean mask + top-K; fp16 inputs, fp32
    scores (module docstring). cuBLAS scores every batch, except a single query on the triton
    backend where ``register_index`` found ``ops.gemv_scores`` equal to cuBLAS bit for bit on this
    table (``gemv_exact``; kernels.md § gemv_scores)."""

    item_embs_t: Tensor
    gemv_exact: bool = False  # set by register_index; False on a module loaded from a state dict

    def __init__(self, k: int, backend: LinrBackend = "triton") -> None:
        super().__init__()
        check_backend(backend, LinrBackend)
        self.k = k
        self.backend = backend

    def register_index(self, item_embs: Tensor) -> None:
        if item_embs.is_cuda:
            # Create cuBLAS's handle while memory is free: it allocates outside the caching
            # allocator, and the check below would otherwise make the first cuBLAS call at the
            # build's peak.
            torch.mm(item_embs[:1, :1].half(), item_embs[:1, :1].half().t())
        # Pre-transpose to a contiguous D×N buffer; a .t() view at call time dispatches a different
        # kernel whose accumulator order can flip K-th-place tiebreaks at the noise floor.
        self.register_buffer("item_embs_t", item_embs.to(torch.float16).t().contiguous())
        self.gemv_exact = self.backend == "triton" and item_embs.is_cuda and self._gemv_matches()

    def _gemv_matches(self) -> bool:
        """Whether the triton GEMV reproduces cuBLAS's single-query scores on this table: cuBLAS
        picks its kernel by shape, and only its sequential-FMA ``gemv2N`` has the GEMV's order.
        Checked on 8 item rows taken as queries (a kernel's order does not depend on the data)."""
        rows = calibration_rows(self.item_embs_t.shape[1], self.item_embs_t.device)
        return all(
            torch.equal(
                ops_for("triton").gemv_scores(q, self.item_embs_t),
                torch.mm(q, self.item_embs_t, out_dtype=torch.float32),
            )
            for q in self.item_embs_t[:, rows].t().contiguous().split(1)
        )

    def score(self, query: Tensor) -> Tensor:
        """``[B, N]`` fp32 scores, every item."""
        query = query.to(torch.float16)
        if query.is_cuda and query.shape[0] == 1 and self.gemv_exact:
            return ops_for("triton").gemv_scores(query, self.item_embs_t)
        if query.is_cuda:
            return torch.mm(query, self.item_embs_t, out_dtype=torch.float32)
        return torch.mm(query.float(), self.item_embs_t.float())  # aten::mm.dtype: no CPU kernel

    def forward(
        self,
        query: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return masked_topk(self.score(query), self.k, valid=mask)


class PrefilterKNN(RetrievalModule):
    """Sparse-rescore KNN with selectable backend. Given ``candidate_ids: [B, P]`` (and optional
    per-row ``counts: [B]``) it scores only the passing rows and top-Ks them back to global ids;
    without ``candidate_ids`` it falls back to a dense full matmul. ``backend="triton"`` fuses
    the sparse path (no ``[B, P, D]`` intermediate); fp16 inputs, fp32 scores on both backends
    (module docstring).

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
            if query.is_cuda:
                scores = torch.mm(query, self.item_embs.t(), out_dtype=torch.float32)
            else:  # aten::mm.dtype has no CPU kernel
                scores = torch.mm(query.float(), self.item_embs.t().float())
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
    padding — never gathered, scored or returned. There is no ``counts``: mask a
    compaction's output to ``-1`` past its counts first (its tail is unwritten).
    Rows with fewer than ``min(k, P)`` real candidates carry ``-1`` / ``-inf`` in
    the tail; ``P < k`` returns ``P`` columns."""

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
