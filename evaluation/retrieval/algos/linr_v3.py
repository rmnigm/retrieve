"""LiNR V3 → V2 cascade: 1-bit Hamming pre-filter, fp32 rerank.

Stage 1 (LiNR §4.3, Fig knn-v3): ``OneBitKNN`` produces a
top-``candidate_pool`` list at 1-bit precision. Stage 2:
``PrefilterKNN`` rescores those candidates at full precision. Both
stages share the same ``backend`` (default ``"triton"``).

When filtered, the cascade calls ``filter_mod.evaluate_indices``
(single fused compact kernel — ``clause_compact`` / ``bloom_compact``
on triton) and passes the ``(candidate_ids, counts)`` directly into
stage 1's indirect-load path. No ``[B, N]`` mask materialized; the
filter is consulted in exactly one place. Stage 1 itself no longer
takes a ``mask=`` arg. The whole algo forward (filter compact +
stage 1 + ``(cand_ids >= 0).sum`` stitch + stage 2) is wrapped with
``torch.compile(
dynamic=True, mode="reduce-overhead")`` in ``__init__`` regardless of
backend. Inductor fuses popcount + reduce in stage 1 and gather +
matmul + topk in stage 2; one cudagraph captures both stages, so
the inter-stage ``cand_ids`` tensor stays inside a single graph
(compiling each stage separately trips ``RuntimeError: accessing
tensor output of CUDAGraphs that has been overwritten by a subsequent
run``).

The dense ``PostfilterKNN`` path is intentionally absent as stage-2 —
its full matmul does the same work as ``triton_knn`` alone, so a 1-bit
prefilter into a dense rescore is strictly slower than the unfiltered
baseline (measured ~1.22 ms vs 0.89 ms at 500M scale).
"""

from __future__ import annotations

from torch import Tensor, nn

from retrieve import OneBitKNN, PrefilterKNN
from retrieve.interfaces import Backend, FilterModule

from ._helpers import collect_modules


class LinrV3Algo(nn.Module):
    is_cpu = False

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        candidate_pool: int = 5000,
        v3_seed: int = 0,
        filter_mod: FilterModule | None = None,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        device = item_embs.device
        # Stage 1 (OneBitKNN) is dtype-agnostic — sign-OPORP discards float
        # precision at bit-pack time. Stage 2 (PrefilterKNN) casts to fp16
        # internally for paper-faithful storage. See retrieve.layers.linr
        # package docstring for the full precision contract.
        self.stage1 = OneBitKNN(k=candidate_pool, seed=v3_seed, backend=backend).to(device)
        self.stage1.register_index(item_embs)
        self.stage2 = PrefilterKNN(k=k, backend=backend).to(device)
        self.stage2.register_index(item_embs)
        self.filter_mod = filter_mod
        self.algo_modules = collect_modules(self.stage1, self.stage2, filter_mod=filter_mod)
        self.compile(dynamic=True, mode="reduce-overhead")

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        filtered = self.filter_mod is not None and qa_narrow is not None
        if filtered:
            # Single fused compact straight from the filter — no [B, N] bool
            # intermediate. On backend="triton" this is the clause_compact /
            # bloom_compact custom_op; on backend="torch" it falls back to
            # compact_mask(evaluate_mask(...)). Either way: one filter call,
            # one place. The retrieve library is 0-indexed over real items
            # (loaders.py drops the training-side padding row), so no item-0
            # fixup is needed.
            pos_idx, pcounts = self.filter_mod.evaluate_indices(qa_narrow)
            cand_ids, _ = self.stage1(q, candidate_ids=pos_idx, counts=pcounts)
        else:
            cand_ids, _ = self.stage1(q)
        # When the filter admits < candidate_pool items stage 1's trailing
        # slots are -1; the stage-2 fused_masked_knn_topk does indirect loads
        # via raw pointer arithmetic, so item_embs_ptr + (-1)*stride walks
        # off the buffer. Pass per-row counts when filtered so stage 2 only
        # scores the valid prefix.
        if filtered:
            counts = (cand_ids >= 0).sum(dim=1)
            return self.stage2(q, candidate_ids=cand_ids, counts=counts)
        return self.stage2(q, candidate_ids=cand_ids)
