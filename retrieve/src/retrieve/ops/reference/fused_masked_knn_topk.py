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
    intermediate the Triton kernel avoids. Scores keep ``query``'s dtype (fp16 from
    ``PrefilterKNN``); the kernel writes fp32."""
    p = positive_indices.shape[1]
    if p < k:
        raise ValueError(f"positive_indices has {p} columns, fewer than k={k}")
    reduced_embs = item_embs[positive_indices.clamp_min(0)]
    scores = torch.bmm(query.unsqueeze(1), reduced_embs.transpose(1, 2)).squeeze(1)
    return masked_topk(scores, k, valid=counts_to_valid(counts, p), gather_ids=positive_indices)
