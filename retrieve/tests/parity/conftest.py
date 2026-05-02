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
        out_pairs = [
            (out_ids[bi, j].item(), out_scores[bi, j].item())
            for j in range(k)
            if torch.isfinite(out_scores[bi, j])
        ]
        ref_pairs = [
            (ref_ids[bi, j].item(), ref_scores[bi, j].item())
            for j in range(k)
            if torch.isfinite(ref_scores[bi, j])
        ]
        out_set = {p[0] for p in out_pairs}
        ref_set = {p[0] for p in ref_pairs}
        if out_set == ref_set:
            continue
        # Tensor-core matmul (`tl.dot`) and torch `@` differ in accumulator
        # order — score-tied items can swap at the K-th boundary. Allow that
        # provided each side's unique ids lie within `atol` of its own min.
        out_min = min(s for _, s in out_pairs) if out_pairs else float("-inf")
        ref_min = min(s for _, s in ref_pairs) if ref_pairs else float("-inf")
        for i in ref_set - out_set:
            s = next(sc for idx, sc in ref_pairs if idx == i)
            assert s <= ref_min + atol + rtol * abs(ref_min), (
                f"row {bi}: ref-only id {i} score={s:.6f} not at boundary {ref_min:.6f}"
            )
        for i in out_set - ref_set:
            s = next(sc for idx, sc in out_pairs if idx == i)
            assert s <= out_min + atol + rtol * abs(out_min), (
                f"row {bi}: out-only id {i} score={s:.6f} not at boundary {out_min:.6f}"
            )

    out_sorted, _ = out_scores.sort(dim=1, descending=True)
    ref_sorted, _ = ref_scores.sort(dim=1, descending=True)
    out_finite = torch.where(torch.isfinite(out_sorted), out_sorted, torch.zeros_like(out_sorted))
    ref_finite = torch.where(torch.isfinite(ref_sorted), ref_sorted, torch.zeros_like(ref_sorted))
    assert torch.allclose(out_finite, ref_finite, atol=atol, rtol=rtol)
