"""The Triton kernels as ``torch.ops.retrieve.*`` ops. Importing this package imports every kernel
file, which registers the eight ``retrieve::*`` ops (seven ``@triton_op``, one opaque
``@custom_op``): the one side-effecting import, as Meta's ``silvertorch.ops._load_ops``. The
package attribute ``retrieve.ops.triton.<op>`` is the op; each kernel file's private ``_impl`` /
``Config`` is reached through the file (``from retrieve.ops.triton.<op> import ...``)."""

from retrieve.ops.triton.clause_compact import clause_compact
from retrieve.ops.triton.clause_mask import clause_mask, clause_mask_packed, clause_mask_scores
from retrieve.ops.triton.fused_masked_knn_topk import fused_masked_knn_topk
from retrieve.ops.triton.gemv_scores import gemv_scores
from retrieve.ops.triton.oporp_1bit_match_topk import (
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
