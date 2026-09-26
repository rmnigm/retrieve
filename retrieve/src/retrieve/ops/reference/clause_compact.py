from __future__ import annotations

from torch import Tensor

from retrieve.functional import compact_mask
from retrieve.ops.reference.clause_mask import clause_mask


def clause_compact(
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
) -> tuple[Tensor, Tensor]:
    """``compact_mask(clause_mask(...))`` → (positive_indices [B, N] int64, counts [B] int64);
    the tail past ``counts[b]`` is the argsort's (see ``compact_mask``)."""
    return compact_mask(clause_mask(item_clause_attrs, clause_is_reverse, query_clause_attrs))
