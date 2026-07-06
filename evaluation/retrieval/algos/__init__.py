"""Algorithm registry for the retrieval benchmark.

One ``AlgoBase`` subclass per algo, with::

    algo.algo_modules: list[nn.Module]          # for memory cleanup
    algo(q, qa_narrow=None) -> (ids, scores)    # via Module.__call__

Each algo's ``__init__`` ends with ``self._finalize(...)`` (see
``_helpers.AlgoBase``), which applies the one canonical
``torch.compile(dynamic=True, mode="reduce-overhead")`` call so the
whole forward (filter + index + cascade) becomes one cudagraph
capture. ``algo_modules`` is a separately maintained list (distinct
from ``nn.Module.modules()``) that the sweep driver iterates for
explicit per-cell GPU-memory cleanup.

``build_algorithm`` is a thin factory: it picks the class for ``name``,
unpacks ``params`` into its constructor, and forwards
``filter_kind``/``filter_mod``/``item_attrs_narrow``/``clause_is_reverse``
so each class can opt in to whichever inputs it actually consumes.

Cell eligibility is declared in ``SUPPORTED_FILTER_KINDS``; the sweep
driver checks ``supports(algo, filter_kind)`` before building a cell,
so construction-time raises are genuine errors, never control flow.
Silvertorch with ``filter_kind='clause'`` accepts ``clause_is_reverse``
and supports reverse-clause sweeps end-to-end via the fused codesigned
exact-clause kernel.
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

    Eligibility is decided up front by ``supports``; any ``ValueError``
    raised here (e.g. ``linr_v2`` without a filter) is a genuine
    construction error and propagates to kill the run loudly. ``backend``
    is threaded into the underlying retrieval module by every algo.
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
