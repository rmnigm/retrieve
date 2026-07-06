from __future__ import annotations

from torch import Tensor

from retrieve.interfaces import Backend
from retrieve.layers.linr._bit_knn import _PackedBitsKNN
from retrieve.layers.utils.quantize import (
    project_simhash_1bit_query,
    quantize_simhash_1bit,
)


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
        backend: Backend = "triton",
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
