"""The ``nn.Module``s a user instantiates: ``SilverTorch`` (Algorithm 1, ``triton`` / ``torch`` /
``official``), the LiNR paper variants ``LiNRV1``–``LiNRV4``, the primitives they compose (dense,
sparse and 1-bit KNNs) and the two standalone filters, plus the two builders. Every module follows
``construct → register_index → forward``; ``retrieve.modules.official`` re-exports Meta's own."""

from retrieve.modules.bit_knn import OneBitKNN, SimHashKNN
from retrieve.modules.filters import BloomFilter, ExactAttributeFilter
from retrieve.modules.knn import FullScanKNN, PostfilterKNN, PostfilterKNNInt8, PrefilterKNN
from retrieve.modules.linr import LiNRBuilder, LiNRV1, LiNRV2, LiNRV3, LiNRV4
from retrieve.modules.silvertorch import OfficialConfig, SilverTorch, SilverTorchBuilder

__all__ = [
    "BloomFilter",
    "ExactAttributeFilter",
    "FullScanKNN",
    "LiNRBuilder",
    "LiNRV1",
    "LiNRV2",
    "LiNRV3",
    "LiNRV4",
    "OfficialConfig",
    "OneBitKNN",
    "PostfilterKNN",
    "PostfilterKNNInt8",
    "PrefilterKNN",
    "SilverTorch",
    "SilverTorchBuilder",
    "SimHashKNN",
]
