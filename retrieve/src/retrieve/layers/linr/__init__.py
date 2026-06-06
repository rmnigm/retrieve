"""LiNR retrieval layers.

Precision contract: all layers accept fp32 or fp16 ``item_embs``/``query``. ``PostfilterKNN`` and
``PrefilterKNN`` store fp16 and accumulate dots in fp32 (matmul/bmm and the fused Triton kernel);
``OneBitKNN``/``SimHashKNN`` are dtype-agnostic since the 1-bit projection is sign-stable."""

from retrieve.layers.linr.one_bit_knn import OneBitKNN
from retrieve.layers.linr.postfilter_knn import PostfilterKNN
from retrieve.layers.linr.postfilter_knn_int8 import PostfilterKNNInt8
from retrieve.layers.linr.prefilter_knn import PrefilterKNN
from retrieve.layers.linr.simhash_knn import SimHashKNN

__all__ = [
    "OneBitKNN",
    "PostfilterKNN",
    "PostfilterKNNInt8",
    "PrefilterKNN",
    "SimHashKNN",
]
