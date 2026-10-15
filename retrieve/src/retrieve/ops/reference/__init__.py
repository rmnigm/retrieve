"""The ``"torch"`` backend: every ``retrieve.ops.triton`` op with the same signature in pure
torch, the eager code the LiNR modules and the filter run on ``backend="torch"`` and the oracle the
parity suite scores the kernels against. Plain functions, no registration; ``torch.compile``
traces through them."""

from retrieve.ops.reference.clause_compact import clause_compact
from retrieve.ops.reference.clause_mask import clause_mask, clause_mask_packed, clause_mask_scores
from retrieve.ops.reference.fused_masked_knn_topk import fused_masked_knn_topk
from retrieve.ops.reference.gemv_scores import gemv_scores
from retrieve.ops.reference.oporp_1bit_match_topk import (
    oporp_1bit_match_topk_full,
    oporp_1bit_match_topk_indirect,
)

__all__ = [
    "clause_compact",
    "clause_mask",
    "clause_mask_packed",
    "clause_mask_scores",
    "fused_masked_knn_topk",
    "gemv_scores",
    "oporp_1bit_match_topk_full",
    "oporp_1bit_match_topk_indirect",
]
