"""Shared helpers for kernel parity tests."""

from __future__ import annotations

import torch


def assert_topk_matches(
    out_ids: torch.Tensor,
    out_scores: torch.Tensor,
    ref_ids: torch.Tensor,
    ref_scores: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Assert two top-K implementations agree on finite-id sets and sorted scores.

    Tie-breaking on the id permutation can differ between backends, so we compare
    *sets* of finite-score ids per row and the *sorted* descending scores
    (with -inf replaced by 0 so ``allclose`` still works on padded rows).
    """
    b, k = out_ids.shape
    for bi in range(b):
        out_set = {out_ids[bi, j].item() for j in range(k) if torch.isfinite(out_scores[bi, j])}
        ref_set = {ref_ids[bi, j].item() for j in range(k) if torch.isfinite(ref_scores[bi, j])}
        assert out_set == ref_set, f"row {bi}: id sets differ"

    out_sorted, _ = out_scores.sort(dim=1, descending=True)
    ref_sorted, _ = ref_scores.sort(dim=1, descending=True)
    out_finite = torch.where(torch.isfinite(out_sorted), out_sorted, torch.zeros_like(out_sorted))
    ref_finite = torch.where(torch.isfinite(ref_sorted), ref_sorted, torch.zeros_like(ref_sorted))
    assert torch.allclose(out_finite, ref_finite, atol=atol, rtol=rtol)
