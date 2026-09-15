"""Importing this module imports every kernel file, which registers the ten ``retrieve::*`` ops
(eight ``@triton_op``, two opaque ``@custom_op``) — the one side-effecting import, as Meta's
``silvertorch.ops._load_ops``. ``retrieve.ops.triton`` re-exports the ops from here; each kernel
file's private ``_impl`` / ``Config`` is reached through the file
(``from retrieve.ops.triton.<kernel> import ...``)."""

from retrieve.ops.triton.bloom_compact import bloom_compact
from retrieve.ops.triton.bloom_match import bloom_match
from retrieve.ops.triton.clause_compact import clause_compact
from retrieve.ops.triton.clause_mask import clause_mask
from retrieve.ops.triton.codesigned_probe_score import (
    codesigned_probe_score,
    codesigned_probe_score_bloom,
)
from retrieve.ops.triton.codesigned_probe_score_exact import codesigned_probe_score_exact
from retrieve.ops.triton.fused_masked_knn_topk import fused_masked_knn_topk
from retrieve.ops.triton.oporp_1bit_match_topk import (
    oporp_1bit_match_topk_full,
    oporp_1bit_match_topk_indirect,
)

__all__ = [
    "bloom_compact",
    "bloom_match",
    "clause_compact",
    "clause_mask",
    "codesigned_probe_score",
    "codesigned_probe_score_bloom",
    "codesigned_probe_score_exact",
    "fused_masked_knn_topk",
    "oporp_1bit_match_topk_full",
    "oporp_1bit_match_topk_indirect",
]
