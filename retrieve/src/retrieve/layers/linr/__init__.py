"""LiNR retrieval layers.

**Precision contract.** All three layers accept ``item_embs`` and ``query`` as
either ``torch.float32`` or ``torch.float16``. ``PostfilterKNN`` and
``PrefilterKNN`` cast inputs to fp16 internally — storage is fp16 (paper-
faithful, §6.1 of the LiNR paper: *"embedding dimension is 128, stored as
fp16"*) and PyTorch matmul / bmm + the fused Triton kernel both accumulate the
dot product in fp32 regardless. ``OneBitKNN`` is dtype-agnostic by design: the
1-bit Sign-OPORP projection is sign-stable across float dtypes, so input
precision is discarded at bit-pack time.

Callers that want strict-fp32 numerics for a parity comparison should run the
oracle directly (``evaluation/retrieval/oracle.py``) — it explicitly upcasts.
"""

from retrieve.layers.linr.one_bit_knn import OneBitKNN
from retrieve.layers.linr.postfilter_knn import PostfilterKNN
from retrieve.layers.linr.postfilter_knn_int8 import PostfilterKNNInt8
from retrieve.layers.linr.prefilter_knn import PrefilterKNN

__all__ = [
    "OneBitKNN",
    "PostfilterKNN",
    "PostfilterKNNInt8",
    "PrefilterKNN",
]
