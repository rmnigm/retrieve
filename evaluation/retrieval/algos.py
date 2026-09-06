"""Benchmark algorithms for harness v2 (H §3.1 ``algos.py``).

Five ``nn.Module`` wrappers, one per paper variant, with the filter module registered as a
submodule (so ``module.buffers()`` — and ``bench.index_bytes`` — covers index *and* filter,
the thesis's memory definition) and ``k`` settable after ``register_index`` (perf sets
``module.k = k`` per variant instead of rebuilding; ``tests/test_algos.py`` checks that no
layer bakes ``k`` into a buffer). ``forward(q, qa=None) -> (ids [B, k], scores [B, k])``.

Tables: ``ALGOS`` (name → class), ``FILTER_KINDS``, ``BACKENDS`` and ``PATHS`` — for every
``(algo, filter_kind, backend)`` the code path that actually runs, or ``None`` when there is
no such cell. Backends collapse where they run the same code (``linr_v1``/``linr_v4`` on
``none`` are cuBLAS whatever the flag says); on filter cells the path names both halves
(``cublas+triton``: cuBLAS scoring, Triton ``clause_mask``). ``official`` is Meta's
SilverTorch reference backend and exists for ``silvertorch`` only — the library's other
layers reject it at construction (``LinrBackend``, architecture.md § Backend dispatch), so
those cells are ``None`` rather than a build error. Standalone filter
modules for ``official`` cells are Triton (O §6.2): ``FILTER_BACKEND``.
"""

from __future__ import annotations

from typing import Any, get_args

from torch import Tensor, nn

from retrieve import (
    BloomFilter,
    ExactAttributeFilter,
    OneBitKNN,
    PostfilterKNN,
    PostfilterKNNInt8,
    PrefilterKNN,
    SilverTorch,
)
from retrieve.interfaces import FilterModule, LinrBackend, SilverTorchBackend

FILTER_KINDS = ("none", "clause", "bloom")
BACKENDS = ("triton", "torch", "official")
# Backend of the standalone FilterModule (and the oracle's exact filter) for a cell's backend.
FILTER_BACKEND = {"triton": "triton", "torch": "torch", "official": "triton"}
# H amendment / O D7: the official ops cannot be CUDA-graph captured → no ``graph`` variant.
CAPTURABLE = {"triton": True, "torch": True, "official": False}


def _k_of(attr: str) -> property:
    """``module.k`` forwards to the inner layer that owns the final top-k."""
    return property(
        lambda self: getattr(self, attr).k,
        lambda self, k: setattr(getattr(self, attr), "k", int(k)),
    )


def _mask(filter_mod: FilterModule | None, qa: Tensor | None) -> Tensor | None:
    if qa is None:
        return None
    assert filter_mod is not None, "query attrs given but the algo has no filter"
    return filter_mod.evaluate_mask(qa)


class LinrV1(nn.Module):
    """LiNR V1 — dense fp16 matmul (cuBLAS), optional ``[B, N]`` bool mask, top-k."""

    k = _k_of("idx")

    def __init__(
        self, item_embs: Tensor, k: int, *, filter_mod=None, backend: LinrBackend = "triton"
    ):
        super().__init__()
        self.idx = PostfilterKNN(k=k, backend=backend)
        self.idx.register_index(item_embs)
        self.filter = filter_mod

    def forward(self, q: Tensor, qa: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.idx(q, mask=_mask(self.filter, qa))


class LinrV2(nn.Module):
    """LiNR V2 — the filter's compact candidate list rescored exactly (``PrefilterKNN``).
    The candidate source *is* the filter, so ``qa`` is required."""

    k = _k_of("idx")

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_mod: FilterModule,
        backend: LinrBackend = "triton",
    ):
        super().__init__()
        self.idx = PrefilterKNN(k=k, backend=backend)
        self.idx.register_index(item_embs)
        self.filter = filter_mod

    def forward(self, q: Tensor, qa: Tensor) -> tuple[Tensor, Tensor]:
        cand, counts = self.filter.evaluate_indices(qa)
        return self.idx(q, candidate_ids=cand, counts=counts)


class LinrV3(nn.Module):
    """LiNR V3 → V2 cascade: 1-bit OPORP top-``candidate_pool``, then exact rescoring.
    ``candidate_pool`` is a query-time parameter (``set_query_params``)."""

    k = _k_of("stage2")

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        candidate_pool: int = 5000,
        seed: int = 0,
        filter_mod=None,
        backend: LinrBackend = "triton",
    ):
        super().__init__()
        self.stage1 = OneBitKNN(k=candidate_pool, seed=seed, backend=backend)
        self.stage1.register_index(item_embs)
        self.stage2 = PrefilterKNN(k=k, backend=backend)
        self.stage2.register_index(item_embs)
        self.filter = filter_mod

    def set_query_params(self, *, candidate_pool: int) -> None:
        if candidate_pool > self.stage1.item_bits.shape[0]:
            raise ValueError(f"candidate_pool={candidate_pool} exceeds N")
        self.stage1.k = int(candidate_pool)

    def forward(self, q: Tensor, qa: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if qa is None:
            cand, _ = self.stage1(q)
            return self.stage2(q, candidate_ids=cand)
        pos, pcounts = self.filter.evaluate_indices(qa)
        cand, _ = self.stage1(q, candidate_ids=pos, counts=pcounts)
        # Rows with fewer survivors than candidate_pool carry -1 tails; bound stage 2 by counts.
        return self.stage2(q, candidate_ids=cand, counts=(cand >= 0).sum(dim=1))


class LinrV4(nn.Module):
    """LiNR V4 — int8 dense matmul (cuBLAS ``_int_mm``), optional mask, top-k."""

    k = _k_of("idx")

    def __init__(
        self, item_embs: Tensor, k: int, *, filter_mod=None, backend: LinrBackend = "triton"
    ):
        super().__init__()
        self.idx = PostfilterKNNInt8(k=k, backend=backend)
        self.idx.register_index(item_embs)
        self.filter = filter_mod

    def forward(self, q: Tensor, qa: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.idx(q, mask=_mask(self.filter, qa))


class Silvertorch(nn.Module):
    """SilverTorch Algorithm 1: IVF + INT8 with the predicate fused into the probe
    (``filter_kind`` → ``filter_mode`` none / exact / bloom). No filter submodule — the
    attribute buffers live inside ``SilverTorch`` and count toward ``index_bytes``.
    ``n_probe`` is a query-time parameter (``set_query_params``, H §8.2 A)."""

    def __init__(
        self,
        item_embs: Tensor,
        k: int,
        *,
        filter_kind: str = "none",
        item_attrs: Tensor | None = None,
        clause_is_reverse: Tensor | None = None,
        n_lists: int = 1024,
        n_probe: int = 24,
        n_iter: int = 10,
        m_bits: int = 1024,
        k_hash: int = 5,
        seed: int = 0,
        backend: SilverTorchBackend = "triton",
    ):
        super().__init__()
        self.filter_kind = filter_kind
        self.filter = None
        mode = {"none": "none", "clause": "exact", "bloom": "bloom"}[filter_kind]
        bloom = {"m_bits": m_bits, "k_hash": k_hash} if mode == "bloom" else {}
        self.idx = SilverTorch(
            k=k,
            n_lists=n_lists,
            n_probe=n_probe,
            filter_mode=mode,
            n_iter=n_iter,
            seed=seed,
            backend=backend,
            **bloom,
        )
        if mode == "none":
            self.idx.register_index(item_embs)
        else:
            assert item_attrs is not None, f"silvertorch/{filter_kind} needs item_attrs"
            rev = clause_is_reverse if mode == "exact" else None
            self.idx.register_index(item_embs, item_clause_attrs=item_attrs, clause_is_reverse=rev)

    @property
    def k(self) -> int:
        return self.idx.k

    @k.setter
    def k(self, k: int) -> None:
        self._check_probe_pool(self.idx.n_probe, int(k))
        self.idx.k = int(k)

    def set_query_params(self, *, n_probe: int) -> None:
        """The two ``register_index`` validations, re-run on mutation (main.py:168-169, 187-192)."""
        if n_probe > self.idx.n_lists:
            raise ValueError(f"n_probe ({n_probe}) cannot exceed n_lists ({self.idx.n_lists}).")
        self._check_probe_pool(int(n_probe), self.idx.k)
        self.idx.n_probe = int(n_probe)

    def _check_probe_pool(self, n_probe: int, k: int) -> None:
        max_size = self.idx.padded_cluster_items.shape[1]
        if n_probe * max_size < k:
            raise ValueError(
                f"k={k} exceeds probe pool n_probe * max_cluster_size = {n_probe} * {max_size}"
            )

    def forward(self, q: Tensor, qa: Tensor | None = None) -> tuple[Tensor, Tensor]:
        assert (qa is None) == (self.filter_kind == "none"), "qa must match filter_kind"
        return self.idx(q, query_clause_attrs=qa)


ALGOS: dict[str, type[nn.Module]] = {
    "linr_v1_filter_mask": LinrV1,
    "linr_v2": LinrV2,
    "linr_v3": LinrV3,
    "linr_v4": LinrV4,
    "silvertorch": Silvertorch,
}


def _path(algo: str, filter_kind: str, backend: str) -> str | None:
    if backend == "official":
        return "official" if algo == "silvertorch" else None
    if algo == "linr_v2" and filter_kind == "none":
        return None  # the candidate source is the filter
    if algo in ("linr_v1_filter_mask", "linr_v4"):
        return "cublas" if filter_kind == "none" else f"cublas+{backend}"
    return backend


PATHS: dict[tuple[str, str, str], str | None] = {
    (a, f, b): _path(a, f, b) for a in ALGOS for f in FILTER_KINDS for b in BACKENDS
}


def is_valid_combo(algo: str, params: dict[str, Any]) -> bool:
    """``n_probe <= n_lists`` — the one combo the library would reject at build."""
    p = params
    return not (algo == "silvertorch" and p.get("n_probe", 0) > p.get("n_lists", 1 << 30))


def build_filter(
    filter_kind: str,
    item_attrs: Tensor | None,
    *,
    clause_is_reverse: Tensor | None = None,
    backend: str = "triton",
    m_bits: int = 1024,
    k_hash: int = 5,
) -> FilterModule | None:
    """One standalone FilterModule per (filter_kind, filter backend); ``None`` for ``none``.
    Bloom is paper-strict forward-only, so it never sees ``clause_is_reverse``."""
    if filter_kind == "none":
        return None
    assert item_attrs is not None, f"filter_kind={filter_kind} needs item_attrs"
    if filter_kind == "clause":
        f: FilterModule = ExactAttributeFilter(backend=backend)
        f.register_index(item_attrs, clause_is_reverse=clause_is_reverse)
    elif filter_kind == "bloom":
        f = BloomFilter(m_bits=m_bits, k_hash=k_hash, backend=backend)
        f.register_index(item_attrs)
    else:
        raise ValueError(f"unknown filter_kind {filter_kind!r}")
    return f


def build(
    algo: str,
    item_embs: Tensor,
    *,
    k: int,
    backend: str,
    filter_kind: str = "none",
    filter_mod: FilterModule | None = None,
    item_attrs: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
    params: dict[str, Any] | None = None,
    seed: int = 0,
) -> nn.Module:
    """Construct one cell's module. ``params`` = build + query params merged (``n_lists``,
    ``n_probe``, ``n_iter``, ``m_bits``, ``k_hash``, ``candidate_pool``); ``seed`` drives the
    k-means and the OPORP projection. Raises on cells ``PATHS`` marks ``None``."""
    p = dict(params or {})
    if PATHS.get((algo, filter_kind, backend), None) is None:
        raise ValueError(f"no code path for ({algo}, {filter_kind}, {backend})")
    if not is_valid_combo(algo, p):
        raise ValueError(f"invalid params for {algo}: {p}")
    if backend == "official" and "official" not in get_args(SilverTorchBackend):
        raise NotImplementedError("backend='official' is not in retrieve yet (roadmap B1)")
    module: nn.Module
    if algo == "silvertorch":
        module = Silvertorch(
            item_embs,
            k,
            filter_kind=filter_kind,
            item_attrs=item_attrs,
            clause_is_reverse=clause_is_reverse,
            seed=seed,
            backend=backend,
            **p,
        )
    elif algo == "linr_v3":
        module = LinrV3(item_embs, k, seed=seed, filter_mod=filter_mod, backend=backend, **p)
    else:
        module = ALGOS[algo](item_embs, k, filter_mod=filter_mod, backend=backend, **p)
    module.backend = backend
    module.capturable = CAPTURABLE[backend]
    return module


__all__ = [
    "ALGOS",
    "BACKENDS",
    "CAPTURABLE",
    "FILTER_BACKEND",
    "FILTER_KINDS",
    "PATHS",
    "build",
    "build_filter",
    "is_valid_combo",
]
