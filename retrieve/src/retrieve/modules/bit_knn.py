"""1-bit Hamming KNNs over packed int64 sign bits: ``OneBitKNN`` (Sign-OPORP) and ``SimHashKNN``
(SimHash) share the ``_PackedBitsKNN`` base, which owns scoring and backend dispatch; the
subclasses own bit production. Both are dtype-agnostic since the 1-bit projection is
sign-stable."""

from __future__ import annotations

import abc

import torch
from torch import Tensor

from retrieve.indexing.quantize import (
    project_oporp_1bit_query,
    project_simhash_1bit_query,
    quantize_oporp_1bit,
    quantize_simhash_1bit,
)
from retrieve.interfaces import LinrBackend, RetrievalModule, check_backend, ops_for


class _PackedBitsKNN(RetrievalModule):
    """Hamming-similarity KNN over packed int64 sign bits: score = ``64*W - 2*popcount(q ^
    item)``. Subclasses own bit production (``_quantize_index`` / ``_project_query``); scoring,
    backend dispatch, and the candidates/full paths are common.

    Both backends produce byte-identical bits (the projection is shared), so torch and Triton
    paths score identically — the parity property the cross-backend tests assert."""

    item_bits: Tensor

    def __init__(self, k: int, backend: LinrBackend = "triton") -> None:
        super().__init__()
        check_backend(backend, LinrBackend)
        self.k = k
        self.backend = backend
        ops_for(backend)

    @abc.abstractmethod
    def _quantize_index(self, item_embs: Tensor) -> None:
        """Build + register ``item_bits`` and the projection buffers."""

    @abc.abstractmethod
    def _project_query(self, query: Tensor) -> Tensor:
        """[B, D] → [B, W] int64, same bit space as ``item_bits``."""

    def register_index(self, item_embs: Tensor) -> None:
        n = item_embs.shape[0]
        # Full-scan topk runs over [B, N] with no pad tail, so the corpus must hold >= k items
        # (indirect path is safe at any width).
        if self.k > n:
            raise ValueError(f"k={self.k} exceeds corpus size N={n}")
        self._quantize_index(item_embs)

    @property
    def d_total(self) -> int:
        return 64 * self.item_bits.shape[1]

    def forward(
        self,
        query: Tensor,
        candidate_ids: Tensor | None = None,
        counts: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Full-scan with contiguous word loads, or indirect loads through ``candidate_ids``
        gated by per-row ``counts`` — same XOR + popcount + ``D - 2*hamming`` either way. Op
        input is ``[N, W]`` int64; the projection that produced the bits is opaque."""
        ops = ops_for(self.backend)
        query_bits = self._project_query(query)
        if candidate_ids is None:
            return ops.oporp_1bit_match_topk_full(query_bits, self.item_bits, self.k)
        if counts is None:
            counts = torch.full(
                (candidate_ids.shape[0],),
                candidate_ids.shape[1],
                dtype=torch.long,
                device=query.device,
            )
        return ops.oporp_1bit_match_topk_indirect(
            query_bits, self.item_bits, self.k, candidate_ids, counts
        )


class OneBitKNN(_PackedBitsKNN):
    """1-bit Sign-OPORP scoring (Li et al.) with selectable backend: Hamming similarity ``D -
    2*popcount(q ^ item)``, 16× smaller than fp16. Decoupled from filtering — callers pass
    ``candidate_ids`` (and optional per-row ``counts``) computed upstream.

    ``backend="triton"`` (default) routes both the full-scan and ``candidate_ids`` paths through
    one fused kernel; ``backend="torch"`` runs the same op chain eager."""

    item_bits: Tensor
    oporp_signs: Tensor
    oporp_perm: Tensor

    def __init__(
        self,
        k: int,
        seed: int = 0,
        backend: LinrBackend = "triton",
        k_bits: int = 0,
    ) -> None:
        super().__init__(k, backend)
        self.seed = seed
        # k_bits=0 sentinel ("use D from register_index"); a plain int (never None) so Dynamo sees
        # no Optional attr.
        self.k_bits = k_bits
        # Pristine constructor arg: the sentinel is re-resolved from it on every register_index,
        # so re-registering with a different-dim corpus can't silently keep the first D.
        self._k_bits_arg = k_bits
        # A state-dict load into a module that never registered (the builders' prebuilt path)
        # must resolve the sentinel from the loaded projection instead.
        self.register_load_state_dict_post_hook(_rederive_k_bits)

    def _quantize_index(self, item_embs: Tensor) -> None:
        # Resolve the 0 sentinel to a concrete int so _project_query sees a stable Python int under
        # Dynamo.
        self.k_bits = self._k_bits_arg if self._k_bits_arg != 0 else item_embs.shape[1]
        bits, signs, perm = quantize_oporp_1bit(item_embs, seed=self.seed, k_bits=self.k_bits)
        self.register_buffer("item_bits", bits)
        self.register_buffer("oporp_signs", signs)
        self.register_buffer("oporp_perm", perm)

    def _project_query(self, query: Tensor) -> Tensor:
        """Pure tensor-flow query projection — shared by both backends."""
        return project_oporp_1bit_query(query, self.oporp_signs, self.oporp_perm, self.k_bits)


def _rederive_k_bits(module: OneBitKNN, incompatible_keys) -> None:
    if module._k_bits_arg == 0 and hasattr(module, "oporp_signs"):
        module.k_bits = int(module.oporp_signs.shape[0])


class SimHashKNN(_PackedBitsKNN):
    """SimHash 1-bit Hamming scoring (Charikar 2002 / Manku 2007) with selectable backend: fixed
    Gaussian projection ``R ∈ R^{k_bits × D}`` then sign-quantize, scored as ``k_bits -
    2*popcount(q ^ item)`` by the same kernel as ``OneBitKNN``. Unlike Sign-OPORP, ``k_bits`` may
    exceed ``D`` for a recall-vs-memory trade since every bit mixes all coordinates."""

    item_bits: Tensor
    simhash_R: Tensor

    def __init__(
        self,
        k: int,
        k_bits: int,
        seed: int = 0,
        backend: LinrBackend = "triton",
    ) -> None:
        super().__init__(k, backend)
        self.k_bits = k_bits
        self.seed = seed

    def _quantize_index(self, item_embs: Tensor) -> None:
        bits, r = quantize_simhash_1bit(item_embs, self.k_bits, self.seed)
        self.register_buffer("item_bits", bits)
        self.register_buffer("simhash_R", r)

    def _project_query(self, query: Tensor) -> Tensor:
        """Pure tensor-flow query projection — shared by both backends."""
        return project_simhash_1bit_query(query, self.simhash_R)
