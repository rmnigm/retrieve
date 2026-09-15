from __future__ import annotations

import abc
import importlib
from types import ModuleType
from typing import Literal, get_args

from torch import Tensor, nn

# Two backend vocabularies, one per family. Every LiNR layer and every standalone filter
# has exactly a Triton path and a torch path; only ``SilverTorch`` also routes to Meta's
# official ops. Each constructor validates its own literal via ``check_backend`` so a
# typo — or a SilverTorch-only value handed to a LiNR module — raises instead of
# silently running the torch path.
LinrBackend = Literal["torch", "triton"]
SilverTorchBackend = Literal["torch", "triton", "official"]


def check_backend(backend: str, allowed: object) -> None:
    """Raise ``ValueError`` unless ``backend`` is one of the ``Literal`` values ``allowed``."""
    if backend not in get_args(allowed):
        expected = ", ".join(map(repr, get_args(allowed)))
        raise ValueError(f"unknown backend {backend!r}; expected one of {expected}")


_OPS_NAMESPACE = {
    "triton": "retrieve.ops.triton",
    "torch": "retrieve.ops.reference",
    "official": "retrieve.ops.official",
}
_OPS_LOADED: dict[str, ModuleType] = {}


def ops_for(backend: str) -> ModuleType:
    """The op namespace a backend scores with (``retrieve.ops.triton`` / ``.reference`` /
    ``.official``), imported on first use so ``import retrieve`` registers no kernels. Modules
    call it once in ``__init__`` (the import, and its errors, happen at construction) and again
    in ``forward`` (a dict hit; not stored on the instance, which must stay deep-copyable)."""
    if backend in _OPS_LOADED:
        return _OPS_LOADED[backend]
    _OPS_LOADED[backend] = importlib.import_module(_OPS_NAMESPACE[backend])
    return _OPS_LOADED[backend]


class RetrievalModule(nn.Module, abc.ABC):
    """A top-K retriever over a registered item index.

    Contract: construct with k (+ knobs) → register_index(item_embs, ...)
    exactly once → forward(query, ...) → (ids [B, k] int64, scores [B, k]).
    -1 / -inf are the "no item" sentinels. Re-registration is unsupported.

    This ABC constrains only the lifecycle. Forward signatures deliberately vary
    by family (mask vs candidates vs fused-filter).
    """

    k: int

    @abc.abstractmethod
    def register_index(self, item_embs: Tensor, **kwargs) -> None: ...


class FilterModule(nn.Module, abc.ABC):
    """Boolean predicate over a registered item index; ``forward`` aliases ``evaluate_mask``.

    Three eval paths: ``evaluate_mask(q) -> [B, N] bool`` (dense), ``evaluate_indices(q) -> ([B,
    P] int64, [B] int64)`` (compact candidates), ``evaluate_subset(q, candidate_ids) -> [B, P]
    bool`` (check only the given ids)."""

    @abc.abstractmethod
    def register_index(
        self,
        item_clause_attrs: Tensor,
        *,
        clause_is_reverse: Tensor | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor: ...

    def evaluate_indices(
        self,
        query_clause_attrs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from retrieve.functional import compact_mask

        return compact_mask(self.evaluate_mask(query_clause_attrs))

    def evaluate_subset(
        self,
        query_clause_attrs: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor:
        mask = self.evaluate_mask(query_clause_attrs)
        return mask.gather(1, candidate_ids)

    def forward(self, query_clause_attrs: Tensor) -> Tensor:
        return self.evaluate_mask(query_clause_attrs)
