"""LiNR V1 — dense matmul + optional boolean mask, then topk.

Covers two registered names: ``linr_v1_filter_mask`` (canonical for
filter cells) and ``triton_knn`` (historic alias used by yambda
configs). Same implementation; the alias exists so both YAML lineages
keep working without a rename. Backed by ``SimilarityMasking`` with
``backend="triton"`` (default) or ``"torch"``.

The whole algo forward (filter mask build + index call) is wrapped
with ``torch.compile(dynamic=True, mode="reduce-overhead")`` in
``__init__`` regardless of backend — Inductor fuses the matmul +
optional mask + topk body and cudagraph_trees replay collapses launch
overhead.
"""

from __future__ import annotations

from torch import Tensor, nn

from retrieve import SimilarityMasking
from retrieve.interfaces import Backend, FilterModule

from ._helpers import collect_modules
from .filter import make_mask


class LinrV1Algo(nn.Module):
    is_cpu = False

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_mod: FilterModule | None = None,
        backend: Backend = "triton",
    ) -> None:
        super().__init__()
        self.idx = SimilarityMasking(k=k, backend=backend).to(item_embs.device)
        self.idx.register_index(item_embs)
        self.filter_mod = filter_mod
        self.algo_modules = collect_modules(self.idx, filter_mod=filter_mod)
        self.compile(dynamic=True, mode="reduce-overhead")

    def forward(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.idx(q, mask=make_mask(self.filter_mod, qa_narrow))
