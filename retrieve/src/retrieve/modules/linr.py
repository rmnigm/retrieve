"""The LiNR paper's four variants as modules (plan L D5): each composes the primitives of
``modules.knn`` / ``modules.bit_knn`` with an optional ``FilterModule`` held as ``self.filter``
(so ``buffers()`` covers index and filter) and exposes ``forward(query, query_clause_attrs=None)
-> (ids [B, k], scores [B, k])``. ``k`` forwards to the primitive that owns the final top-k, so
it is settable after ``register_index``; ``capturable`` is a class attribute (every LiNR backend
captures). The bodies are the harness wrappers' bodies, moved verbatim."""

from __future__ import annotations

from torch import Tensor

from retrieve.interfaces import FilterModule, LinrBackend, RetrievalModule, load_prebuilt
from retrieve.modules.bit_knn import OneBitKNN
from retrieve.modules.knn import PostfilterKNN, PostfilterKNNInt8, PrefilterKNN

__all__ = ["LiNRBuilder", "LiNRV1", "LiNRV2", "LiNRV3", "LiNRV4"]


def _k_of(attr: str) -> property:
    """``module.k`` forwards to the inner primitive that owns the final top-k."""
    return property(
        lambda self: getattr(self, attr).k,
        lambda self, k: setattr(getattr(self, attr), "k", int(k)),
    )


def _mask(filter_mod: FilterModule | None, qa: Tensor | None) -> Tensor | None:
    if qa is None:
        return None
    assert filter_mod is not None, "query attrs given but the module has no filter"
    return filter_mod.evaluate_mask(qa)


class LiNRV1(RetrievalModule):
    """LiNR V1 — dense fp16 matmul (cuBLAS), optional ``[B, N]`` bool mask from the filter,
    top-k."""

    capturable = True
    k = _k_of("idx")

    def __init__(
        self, k: int, *, filter: FilterModule | None = None, backend: LinrBackend = "triton"
    ) -> None:
        super().__init__()
        self.idx = PostfilterKNN(k=k, backend=backend)
        self.filter = filter
        self.backend = backend

    def register_index(
        self,
        item_embs: Tensor,
        item_clause_attrs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        self.idx.register_index(item_embs)
        if item_clause_attrs is not None:
            self.filter.register_index(item_clause_attrs, clause_is_reverse=clause_is_reverse)

    def forward(self, query: Tensor, query_clause_attrs: Tensor | None = None):
        return self.idx(query, mask=_mask(self.filter, query_clause_attrs))


class LiNRV2(RetrievalModule):
    """LiNR V2 — the filter's compact candidate list rescored exactly (``PrefilterKNN``). The
    candidate source *is* the filter, so ``query_clause_attrs`` is required."""

    capturable = True
    k = _k_of("idx")

    def __init__(self, k: int, *, filter: FilterModule, backend: LinrBackend = "triton") -> None:
        super().__init__()
        self.idx = PrefilterKNN(k=k, backend=backend)
        self.filter = filter
        self.backend = backend

    def register_index(
        self,
        item_embs: Tensor,
        item_clause_attrs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        self.idx.register_index(item_embs)
        if item_clause_attrs is not None:
            self.filter.register_index(item_clause_attrs, clause_is_reverse=clause_is_reverse)

    def forward(self, query: Tensor, query_clause_attrs: Tensor):
        cand, counts = self.filter.evaluate_indices(query_clause_attrs)
        return self.idx(query, candidate_ids=cand, counts=counts)


class LiNRV3(RetrievalModule):
    """LiNR V3 → V2 cascade: 1-bit OPORP top-``candidate_pool``, then exact rescoring.
    ``candidate_pool`` is a query-time parameter (``set_query_params``)."""

    capturable = True
    k = _k_of("stage2")

    def __init__(
        self,
        k: int,
        *,
        candidate_pool: int = 5000,
        seed: int = 0,
        filter: FilterModule | None = None,
        backend: LinrBackend = "triton",
    ) -> None:
        super().__init__()
        self.stage1 = OneBitKNN(k=candidate_pool, seed=seed, backend=backend)
        self.stage2 = PrefilterKNN(k=k, backend=backend)
        self.filter = filter
        self.backend = backend

    def register_index(
        self,
        item_embs: Tensor,
        item_clause_attrs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        self.stage1.register_index(item_embs)
        self.stage2.register_index(item_embs)
        if item_clause_attrs is not None:
            self.filter.register_index(item_clause_attrs, clause_is_reverse=clause_is_reverse)

    def set_query_params(self, *, candidate_pool: int) -> None:
        if candidate_pool > self.stage1.item_bits.shape[0]:
            raise ValueError(f"candidate_pool={candidate_pool} exceeds N")
        self.stage1.k = int(candidate_pool)

    def forward(self, query: Tensor, query_clause_attrs: Tensor | None = None):
        if query_clause_attrs is None:
            cand, _ = self.stage1(query)
            return self.stage2(query, candidate_ids=cand)
        pos, pcounts = self.filter.evaluate_indices(query_clause_attrs)
        cand, _ = self.stage1(query, candidate_ids=pos, counts=pcounts)
        # Rows with fewer survivors than candidate_pool carry -1 tails; bound stage 2 by counts.
        return self.stage2(query, candidate_ids=cand, counts=(cand >= 0).sum(dim=1))


class LiNRV4(RetrievalModule):
    """LiNR V4 — int8 dense matmul (cuBLAS ``_int_mm``), optional mask from the filter, top-k."""

    capturable = True
    k = _k_of("idx")

    def __init__(
        self, k: int, *, filter: FilterModule | None = None, backend: LinrBackend = "triton"
    ) -> None:
        super().__init__()
        self.idx = PostfilterKNNInt8(k=k, backend=backend)
        self.filter = filter
        self.backend = backend

    def register_index(
        self,
        item_embs: Tensor,
        item_clause_attrs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
    ) -> None:
        self.idx.register_index(item_embs)
        if item_clause_attrs is not None:
            self.filter.register_index(item_clause_attrs, clause_is_reverse=clause_is_reverse)

    def forward(self, query: Tensor, query_clause_attrs: Tensor | None = None):
        return self.idx(query, mask=_mask(self.filter, query_clause_attrs))


VARIANTS = {"v1": LiNRV1, "v2": LiNRV2, "v3": LiNRV3, "v4": LiNRV4}


class LiNRBuilder:
    """Fluent construction of a registered LiNR variant: ``LiNRBuilder("v3", k=100,
    candidate_pool=8000)`` takes the variant's constructor keywords, then
    ``set_item_embeddings`` (+ ``set_filter(filter, item_attrs, clause_is_reverse)``) to build,
    or ``set_state_dict`` (+ ``set_filter(filter)`` for the filter submodule's buffers) to load
    a prebuilt index. ``build()`` is construct → register (or load) → ``.to(device)``."""

    def __init__(self, variant: str, **kwargs) -> None:
        self._cls = VARIANTS[variant]
        self._kwargs = kwargs
        self._embs: Tensor | None = None
        self._attrs: Tensor | None = None
        self._reverse: Tensor | None = None
        self._state_dict: dict[str, Tensor] | None = None
        self._device = None

    def set_item_embeddings(self, item_embs: Tensor) -> LiNRBuilder:
        self._embs = item_embs
        return self

    def set_filter(
        self,
        filter: FilterModule,
        item_clause_attrs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
    ) -> LiNRBuilder:
        self._kwargs["filter"] = filter
        self._attrs = item_clause_attrs
        self._reverse = clause_is_reverse
        return self

    def set_backend(self, backend: LinrBackend) -> LiNRBuilder:
        self._kwargs["backend"] = backend
        return self

    def set_device(self, device) -> LiNRBuilder:
        self._device = device
        return self

    def set_state_dict(self, state_dict: dict[str, Tensor]) -> LiNRBuilder:
        self._state_dict = state_dict
        return self

    def build(self) -> RetrievalModule:
        if (self._embs is None) == (self._state_dict is None):
            raise ValueError("set exactly one of set_item_embeddings / set_state_dict")
        module = self._cls(**self._kwargs)
        if self._state_dict is not None:
            load_prebuilt(module, self._state_dict)
        else:
            module.register_index(self._embs, self._attrs, self._reverse)
        return module if self._device is None else module.to(self._device)
