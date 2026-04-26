from __future__ import annotations

import abc

from torch import Tensor, nn


class FilterModule(nn.Module, abc.ABC):
    """Produces a ``[B, N]`` boolean mask over the item pool."""

    @abc.abstractmethod
    def register_index(
        self,
        item_attrs: Tensor,
        item_embs: Tensor | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def forward(self, query: Tensor) -> Tensor: ...


class RetrievalModule(nn.Module, abc.ABC):
    """Produces ``(ids[B, K], scores[B, K])`` from the full item pool.

    Concrete subclasses extend ``forward`` with whichever filter representation
    they consume (e.g. ``mask`` for V1's dense path, ``candidate_ids`` for V2,
    both for V3). The base contract is just ``forward(query)``.
    """

    @abc.abstractmethod
    def register_index(self, item_embs: Tensor) -> None: ...

    @abc.abstractmethod
    def forward(self, query: Tensor) -> tuple[Tensor, Tensor]: ...


class ScorerModule(nn.Module, abc.ABC):
    """Re-scores a set of candidate items: returns ``scores[B, K]``."""

    @abc.abstractmethod
    def register_index(self, item_embs: Tensor) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        query: Tensor,
        candidate_ids: Tensor,
    ) -> Tensor: ...
