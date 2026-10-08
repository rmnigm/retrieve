from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import counts_to_valid, masked_topk


def fused_masked_knn_topk(
    query: Tensor,
    item_embs: Tensor,
    positive_indices: Tensor,
    counts: Tensor,
    k: int,
) -> tuple[Tensor, Tensor]:
    """Gather + ``bmm`` + ``masked_topk`` over the caller's candidates — the ``[B, P, D]``
    intermediate the Triton kernel avoids. Scores are fp32, as the kernel writes them; ids past
    ``counts[b]`` are never gathered (a compaction's tail is unwritten memory)."""
    p = positive_indices.shape[1]
    if p < k:
        raise ValueError(f"positive_indices has {p} columns, fewer than k={k}")
    valid = counts_to_valid(counts, p)
    reduced_embs = item_embs[torch.where(valid, positive_indices, 0)]
    q, items_t = query.unsqueeze(1), reduced_embs.transpose(1, 2)
    if query.is_cuda:
        scores = torch.bmm(q, items_t, out_dtype=torch.float32).squeeze(1)
    else:  # aten::bmm.dtype has no CPU kernel
        scores = torch.bmm(q.float(), items_t.float()).squeeze(1)
    return masked_topk(scores, k, valid=valid, gather_ids=positive_indices)
