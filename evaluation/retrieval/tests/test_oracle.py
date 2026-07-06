"""CPU-only tests for the filtered-oracle padding semantics.

``compute_filtered_oracle`` brute-forces ``q @ E_t`` under an exact filter
mask. When the filter admits fewer than K_GT items, the losing top-k slots
tie at -inf and torch.topk would emit lowest-indexed junk (0, 1, 2, ...);
the oracle must rewrite those slots to -1 so they never score as real
ground truth against the algos' -1 padding. Skipped rows stay all -1.

Runs on CPU: the oracle only needs plain torch ops.
"""

from __future__ import annotations

import torch

from retrieval.oracle import compute_filtered_oracle

CPU = torch.device("cpu")


class _FixedMaskFilter:
    """Minimal FilterModule stand-in: admits a fixed per-item mask for
    every query (``evaluate_mask`` is all the oracle calls)."""

    def __init__(self, item_mask: torch.Tensor):
        self.item_mask = item_mask  # [N] bool

    def evaluate_mask(self, qa_narrow: torch.Tensor) -> torch.Tensor:
        return self.item_mask.unsqueeze(0).expand(qa_narrow.shape[0], -1)


def test_oracle_pads_short_rows_with_minus_one():
    # 3 items, filter admits only item 1 → with K_GT=2 every row is
    # [1, -1], not [1, 0] (index-0 junk from the -inf tie).
    item_embs = torch.eye(3)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    qa = torch.zeros(2, 1, dtype=torch.long)
    filt = _FixedMaskFilter(torch.tensor([False, True, False]))

    out = compute_filtered_oracle(
        item_embs, queries, qa, skip_mask=None, filter_mod=filt, K_GT=2, device=CPU
    )
    assert out.dtype == torch.long
    assert out.tolist() == [[1, -1], [1, -1]]


def test_oracle_skip_mask_rows_stay_minus_one():
    item_embs = torch.eye(3)
    queries = torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 1.0]])
    qa = torch.zeros(2, 1, dtype=torch.long)
    filt = _FixedMaskFilter(torch.tensor([True, True, True]))
    skip_mask = torch.tensor([True, False])

    out = compute_filtered_oracle(
        item_embs, queries, qa, skip_mask=skip_mask, filter_mod=filt, K_GT=2, device=CPU
    )
    # Skipped row untouched; kept row ranked by score (item 1 > item 2).
    assert out[0].tolist() == [-1, -1]
    assert out[1].tolist() == [1, 2]


def test_oracle_k_gt_beyond_catalog_pads_tail():
    # K_GT=5 over a 3-item catalog: K_eff=3 real ids, tail stays -1.
    item_embs = torch.eye(3)
    queries = torch.tensor([[3.0, 2.0, 1.0]])  # distinct scores → order 0,1,2

    out = compute_filtered_oracle(
        item_embs, queries, None, skip_mask=None, filter_mod=None, K_GT=5, device=CPU
    )
    assert out.shape == (1, 5)
    assert out[0].tolist() == [0, 1, 2, -1, -1]


# E6 adds fingerprint tests here
