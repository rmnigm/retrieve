"""The ``"torch"`` backend: every ``retrieve.ops.triton`` op with the same signature in pure
torch — the eager code the modules run on ``backend="torch"`` and the oracle the parity suite
scores the kernels against. Plain functions, no registration; ``torch.compile`` traces through
them."""

from retrieve.ops.reference.bloom_compact import bloom_compact
from retrieve.ops.reference.bloom_full_mask import bloom_full_mask
from retrieve.ops.reference.bloom_match import bloom_match, bloom_match_scores
from retrieve.ops.reference.clause_compact import clause_compact
from retrieve.ops.reference.clause_mask import clause_mask, clause_mask_scores
from retrieve.ops.reference.codesigned_probe_score import (
    codesigned_probe_score,
    codesigned_probe_score_bloom,
)
from retrieve.ops.reference.codesigned_probe_score_exact import codesigned_probe_score_exact
from retrieve.ops.reference.fused_masked_knn_topk import fused_masked_knn_topk
from retrieve.ops.reference.oporp_1bit_match_topk import (
    oporp_1bit_match_topk_full,
    oporp_1bit_match_topk_indirect,
)

__all__ = [
    "bloom_compact",
    "bloom_full_mask",
    "bloom_match",
    "bloom_match_scores",
    "clause_compact",
    "clause_mask",
    "clause_mask_scores",
    "codesigned_probe_score",
    "codesigned_probe_score_bloom",
    "codesigned_probe_score_exact",
    "fused_masked_knn_topk",
    "oporp_1bit_match_topk_full",
    "oporp_1bit_match_topk_indirect",
]
