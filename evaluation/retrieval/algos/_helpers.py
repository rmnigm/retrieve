"""Shared algo-construction helpers.

Lives in its own module (rather than ``algos/__init__.py``) so individual algo
files can import it without triggering a circular import through
``algos/__init__.py``'s class re-exports.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch.nn as nn
from torch import Tensor

from retrieve.interfaces import FilterModule


@runtime_checkable
class RetrievalAlgo(Protocol):
    """Structural type of one benchmark algo instance.

    The sweep driver only ever needs two things from an algo: the
    ``algo_modules`` cleanup list (iterated by ``_release_algo`` for
    explicit per-cell GPU-memory release) and the compiled forward
    ``algo(q, qa_narrow=None) -> (ids, scores)``. ``runtime_checkable``
    lets ``build_algorithm`` assert ``isinstance(obj, RetrievalAlgo)``
    in one place.
    """

    algo_modules: list[nn.Module]

    def __call__(
        self, q: Tensor, qa_narrow: Tensor | None = None
    ) -> tuple[Tensor, Tensor]: ...


class AlgoBase(nn.Module):
    """Shared tail for eval algo wrappers.

    Subclasses build their retrieve-layer modules in ``__init__`` then call
    ``_finalize(...)`` exactly once, as the LAST statement. It wires the
    ``algo_modules`` cleanup list and applies **the** ``torch.compile`` call for the
    whole harness: ``dynamic=True, mode="reduce-overhead"``, which captures one
    cudagraph over the entire forward (filter + index + cascade).

    This is the single compile site on purpose — every algo gets identical compile
    treatment, so cross-algo latency rows stay comparable."""

    filter_mod: FilterModule | None
    algo_modules: list[nn.Module]

    def _finalize(self, *modules: nn.Module, filter_mod: FilterModule | None) -> None:
        self.filter_mod = filter_mod
        mods = list(modules)
        if filter_mod is not None:
            mods.append(filter_mod)
        self.algo_modules = mods
        self.compile(dynamic=True, mode="reduce-overhead")
