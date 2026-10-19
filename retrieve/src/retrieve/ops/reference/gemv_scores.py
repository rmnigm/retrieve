"""The ``gemv_scores`` twin: cuBLAS's ``torch.mm`` with fp32 output."""

from __future__ import annotations

import torch
from torch import Tensor


def gemv_scores(query: Tensor, item_embs_t: Tensor) -> Tensor:
    """``query [1, D]`` fp16 @ ``item_embs_t [D, N]`` fp16 → ``[1, N]`` fp32."""
    return torch.mm(query, item_embs_t, out_dtype=torch.float32)
