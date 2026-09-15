from __future__ import annotations

import torch
from torch import Tensor


def clause_mask(
    item_clause_attrs: Tensor,  # [N, C, A_max] int64
    clause_is_reverse: Tensor,  # [C] bool
    query_clause_attrs: Tensor,  # [B, C] int64
) -> Tensor:
    """Exact clause evaluation → ``[B, N]`` bool through a ``[B, N, C, A_max]`` broadcast — the
    intermediate the Triton kernel exists to avoid."""
    q = query_clause_attrs.unsqueeze(1).unsqueeze(-1)
    ic = item_clause_attrs.unsqueeze(0)
    match = q == ic
    clause_pass = match.any(dim=-1)
    rev = clause_is_reverse.unsqueeze(0).unsqueeze(0)
    clause_pass = torch.where(rev, ~clause_pass, clause_pass)
    inactive = (query_clause_attrs == -1).unsqueeze(1)
    clause_pass = clause_pass | inactive
    return clause_pass.all(dim=-1)
