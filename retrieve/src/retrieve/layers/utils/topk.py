"""Shared masked top-K epilogue for the torch-side layer paths.

Pure tensor-flow (no .item(), no data-dependent Python branches beyond
min(k, P) over static shapes) — traces cleanly under the algo-level
torch.compile(dynamic=True, mode='reduce-overhead') exactly like the
previously-inlined originals."""

from __future__ import annotations

import torch
from torch import Tensor


def counts_to_valid(counts: Tensor, p: int) -> Tensor:
    """[B] counts → [B, p] bool prefix mask."""
    return torch.arange(p, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)


def masked_topk(
    scores: Tensor,  # [B, P] fp
    k: int,
    *,
    valid: Tensor | None = None,  # [B, P] bool; None = all valid
    gather_ids: Tensor | None = None,  # [B, P] local→global map; None = identity
    pad_to_k: bool = True,  # rows with P < k or few survivors pad -1/-inf
) -> tuple[Tensor, Tensor]:
    """Mask invalid scores to -inf → topk → map local→global ids → replace
    non-finite winners with -1 → optionally pad to k. Returns (ids, scores).

    The -1 sentinel is applied whenever ``valid`` is given (or P < k): winners
    with a non-finite score — masked lanes *and* genuine -inf scores alike —
    get id -1. With ``valid=None`` and P >= k no sentinel pass runs, matching
    the dense call sites that had no -inf handling. ``pad_to_k=False`` with
    P < k returns min(k, P) columns."""
    b, p = scores.shape
    if valid is not None:
        scores = scores.masked_fill(~valid, float("-inf"))
    actual_k = min(k, p)
    topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
    topk_ids = gather_ids.gather(1, topk_local) if gather_ids is not None else topk_local
    if valid is not None or actual_k < k:
        topk_ids = torch.where(torch.isfinite(topk_scores), topk_ids, topk_ids.new_full((), -1))
    if pad_to_k and actual_k < k:
        pad = k - actual_k
        topk_ids = torch.cat([topk_ids, topk_ids.new_full((b, pad), -1)], dim=1)
        topk_scores = torch.cat([topk_scores, topk_scores.new_full((b, pad), float("-inf"))], dim=1)
    return topk_ids, topk_scores
