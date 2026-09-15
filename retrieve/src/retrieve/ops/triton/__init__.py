"""The Triton kernels as ``torch.ops.retrieve.*`` ops; importing this package registers them
(``_load``). The package attribute ``retrieve.ops.triton.<op>`` is the op — the kernel file of
the same name is reached with ``from retrieve.ops.triton.<op> import ...``."""

from retrieve.ops.triton._load import (
    bloom_compact,
    bloom_match,
    clause_compact,
    clause_mask,
    codesigned_probe_score,
    codesigned_probe_score_bloom,
    codesigned_probe_score_exact,
    fused_masked_knn_topk,
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
