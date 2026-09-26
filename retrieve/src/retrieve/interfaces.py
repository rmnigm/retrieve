from __future__ import annotations

import abc
import importlib
from types import ModuleType
from typing import Literal, get_args

import torch
from torch import Tensor, nn

from retrieve.functional import compact_mask

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


# The code path each backend runs per module: ``"cublas"`` where the flag is a
# no-op, ``None`` where the constructor raises. Keyed by class name so the table is plain data
# with no import of ``retrieve.modules``; the harness derives its ``PATHS`` from it
# (docs/system/architecture.md § Backend dispatch).
DISPATCH: dict[str, dict[str, str | None]] = {
    "SilverTorch": {"triton": "triton", "torch": "torch", "official": "official"},
    "LiNRV1": {"triton": "cublas", "torch": "cublas", "official": None},
    "LiNRV2": {"triton": "triton", "torch": "torch", "official": None},
    "LiNRV3": {"triton": "triton", "torch": "torch", "official": None},
    "LiNRV4": {"triton": "cublas", "torch": "cublas", "official": None},
    "PostfilterKNN": {"triton": "cublas", "torch": "cublas", "official": None},
    "PostfilterKNNInt8": {"triton": "cublas", "torch": "cublas", "official": None},
    "PrefilterKNN": {"triton": "triton", "torch": "torch", "official": None},
    "OneBitKNN": {"triton": "triton", "torch": "torch", "official": None},
    "SimHashKNN": {"triton": "triton", "torch": "torch", "official": None},
    "ExactAttributeFilter": {"triton": "triton", "torch": "torch", "official": None},
    "BloomFilter": {"triton": "triton", "torch": "torch", "official": None},
}


def load_prebuilt(module: nn.Module, state_dict: dict[str, Tensor]) -> None:
    """Give a freshly constructed module (no ``register_index``) the buffers of a saved one:
    one buffer per state-dict key, shaped and placed like the saved tensor, then
    ``load_state_dict`` so every load hook re-derives its cached scalars. The builders'
    ``set_state_dict`` path."""
    for name, t in state_dict.items():
        owner, _, leaf = name.rpartition(".")
        module.get_submodule(owner).register_buffer(leaf, torch.empty_like(t))
    module.load_state_dict(state_dict)


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
