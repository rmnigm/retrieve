from __future__ import annotations

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
