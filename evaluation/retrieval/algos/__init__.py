"""Algorithm registry for the retrieval benchmark.

One ``AlgoBase`` subclass per algo, exposing::

    algo.algo_modules: list[nn.Module]          # for per-cell memory cleanup
    algo(q, qa_narrow=None) -> (ids, scores)    # via Module.__call__

Three things must stay in sync when adding an algo: ``ALGORITHMS``,
``SUPPORTED_FILTER_KINDS``, and the ``build_algorithm`` branch. A test asserts the
first two match. See docs/system/evaluation.md § Algorithms.

``SUPPORTED_FILTER_KINDS`` is backend-blind, and now it does not need to be:
``silvertorch`` supports every filter kind on every backend, ``cuda`` included
(the cuda backend gained an exact-clause kernel, so clause sweeps build there).
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from retrieval.config import FilterKind
from retrieve.interfaces import Backend, FilterModule

from ._helpers import RetrievalAlgo
from .filter import build_filter, make_mask
from .linr_v1 import LinrV1Algo
from .linr_v2 import LinrV2Algo
from .linr_v3 import LinrV3Algo
from .linr_v4 import LinrV4Algo
from .silvertorch import SilvertorchAlgo

ALGORITHMS = (
    "triton_knn",
    "linr_v1_filter_mask",
    "linr_v3",
    "linr_v4",
    "linr_v2",
    "silvertorch",
)

# Declarative cell eligibility: which filter_kinds each algo can run.
# The sweep driver consults this via ``supports`` before descending into
# a cell; ``build_algorithm``'s raises below are defensive, not routing.
SUPPORTED_FILTER_KINDS: dict[str, frozenset[FilterKind]] = {
    "triton_knn":          frozenset({"none", "clause", "bloom"}),
    "linr_v1_filter_mask": frozenset({"none", "clause", "bloom"}),
    "linr_v2":             frozenset({"clause", "bloom"}),  # candidate source IS the filter
    "linr_v3":             frozenset({"none", "clause", "bloom"}),
    "linr_v4":             frozenset({"none", "clause", "bloom"}),
    "silvertorch":         frozenset({"none", "clause", "bloom"}),
}


def supports(algo: str, filter_kind: FilterKind) -> bool:
    """True when ``algo`` can run cells of this ``filter_kind``."""
    return filter_kind in SUPPORTED_FILTER_KINDS[algo]


def is_valid_combo(algo: str, params: dict[str, Any]) -> bool:
    """Skip combos the underlying algo would assert on."""
    if algo == "silvertorch":
        n_lists = params.get("n_lists")
        n_probe = params.get("n_probe")
        if n_lists is not None and n_probe is not None and n_probe > n_lists:
            return False
    return True


def build_algorithm(
    name: str,
    item_embs: Tensor,
    k: int,
    *,
    filter_kind: FilterKind = "none",
    filter_mod: FilterModule | None = None,
    item_attrs_narrow: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
    params: dict[str, Any] | None = None,
    backend: Backend = "triton",
) -> RetrievalAlgo:
    """Return an algo instance for one ``(name, filter_kind, backend)`` cell.

    Unpacks ``params`` into the chosen class and forwards the filter inputs so each
    class opts in to what it consumes. Raises here are genuine construction errors,
    not routing — eligibility was already checked by ``supports``.
    """
    p = params or {}

    algo: object
    if name in ("triton_knn", "linr_v1_filter_mask"):
        algo = LinrV1Algo(item_embs, k, filter_mod=filter_mod, backend=backend)
    elif name == "linr_v3":
        algo = LinrV3Algo(
            item_embs,
            k,
            candidate_pool=int(p.get("candidate_pool", 5000)),
            v3_seed=int(p.get("v3_seed", 0)),
            filter_mod=filter_mod,
            backend=backend,
        )
    elif name == "linr_v4":
        algo = LinrV4Algo(item_embs, k, filter_mod=filter_mod, backend=backend)
    elif name == "linr_v2":
        if filter_mod is None:
            raise ValueError("linr_v2 requires a filter (clause or bloom)")
        algo = LinrV2Algo(item_embs, k, filter_mod=filter_mod, backend=backend)
    elif name == "silvertorch":
        algo = SilvertorchAlgo(
            item_embs,
            k,
            filter_kind=filter_kind,
            item_attrs_narrow=item_attrs_narrow,
            clause_is_reverse=clause_is_reverse,
            n_lists=int(p.get("n_lists", 1024)),
            n_probe=int(p.get("n_probe", 24)),
            n_iter=int(p.get("n_iter", 10)),
            m_bits=int(p.get("m_bits", 1024)),
            k_hash=int(p.get("k_hash", 5)),
            seed=int(p.get("seed", 0)),
            backend=backend,
        )
    else:
        raise ValueError(f"unknown algorithm: {name!r}")

    assert isinstance(algo, RetrievalAlgo)
    return algo


__all__ = [
    "ALGORITHMS",
    "SUPPORTED_FILTER_KINDS",
    "RetrievalAlgo",
    "build_algorithm",
    "build_filter",
    "is_valid_combo",
    "make_mask",
    "supports",
]
