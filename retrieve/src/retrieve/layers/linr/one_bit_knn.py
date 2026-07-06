from __future__ import annotations

from torch import Tensor

from retrieve.interfaces import Backend
from retrieve.layers.linr._bit_knn import _PackedBitsKNN
from retrieve.layers.utils.quantize import (
    project_oporp_1bit_query,
    quantize_oporp_1bit,
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
        backend: Backend = "triton",
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
