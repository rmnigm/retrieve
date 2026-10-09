from __future__ import annotations

import torch
from torch import Tensor

from retrieve.functional import clause_subset_match


def clause_mask(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """Exact clause evaluation → ``[B, N]`` bool: the shared ``clause_subset_match`` over every
    item, materializing the ``[B, N, C, A_max]`` intermediate the Triton kernel exists to avoid."""
    b = query_clause_attrs.shape[0]
    every_item = item_clause_attrs.unsqueeze(0).expand(b, *item_clause_attrs.shape)
    return clause_subset_match(every_item, query_clause_attrs, clause_is_reverse)


def clause_mask_scores(
    scores: Tensor,  # [B, N] fp32
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """``scores`` where the clause test passes, ``-inf`` elsewhere."""
    passed = clause_mask(item_clause_attrs, clause_is_reverse, query_clause_attrs)
    return torch.where(passed, scores, float("-inf"))


def clause_mask_packed(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """``clause_mask`` packed into ``[B, ceil(N / 64)]`` int64, doc ``d`` at bit ``63 - d % 64`` of
    word ``d // 64``."""
    passed = clause_mask(item_clause_attrs, clause_is_reverse, query_clause_attrs)
    b, n = passed.shape
    w = (n + 63) // 64
    bits = torch.zeros(b, w * 64, dtype=torch.int64, device=passed.device)
    bits[:, :n] = passed
    shifts = 63 - torch.arange(64, dtype=torch.int64, device=passed.device)
    return (bits.view(b, w, 64) << shifts).sum(dim=-1)
