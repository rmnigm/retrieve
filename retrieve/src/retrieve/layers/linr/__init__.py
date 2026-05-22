"""LiNR retrieval layers.

**Precision contract.** All three layers accept ``item_embs`` and ``query`` as
either ``torch.float32`` or ``torch.float16``. ``SimilarityMasking`` and
``PrefilterKNN`` cast inputs to fp16 internally — storage is fp16 (paper-
faithful, §6.1 of the LiNR paper: *"embedding dimension is 128, stored as
fp16"*) and PyTorch matmul / bmm + the fused Triton kernel both accumulate the
dot product in fp32 regardless. ``OneBitKNN`` is dtype-agnostic by design: the
1-bit Sign-OPORP projection is sign-stable across float dtypes, so input
precision is discarded at bit-pack time.

Callers that want strict-fp32 numerics for a parity comparison should run the
oracle directly (``evaluation/retrieval/oracle.py``) — it explicitly upcasts.
"""

from retrieve.layers.linr.int8_similarity_masking import Int8SimilarityMasking
from retrieve.layers.linr.one_bit_knn import OneBitKNN
from retrieve.layers.linr.prefilter_knn import PrefilterKNN
from retrieve.layers.linr.similarity_masking import SimilarityMasking

__all__ = [
    "Int8SimilarityMasking",
    "OneBitKNN",
    "PrefilterKNN",
    "SimilarityMasking",
]
