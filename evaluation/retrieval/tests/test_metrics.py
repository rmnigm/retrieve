"""CPU-only unit tests for retrieval quality metrics.

Lock the three semantic contracts downstream rows depend on:

- ``-1`` padding (on either candidate or target side) never counts as a hit
  (``metrics.py::_hits_mask``);
- NDCG's IDCG clamps the ideal-hit count to ``min(num_targets, k)``;
- recall's denominator is ``num_targets``, not ``k`` — the rationale behind
  ``sweep.py::_run_quality``'s per-row ``nt_k`` for tight filters.

Hand-built tensors only; no GPU, no data files.
"""

from __future__ import annotations

import math

import pytest
import torch

from retrieval.metrics import ndcg_at_k, recall_at_k


def test_hits_mask_ignores_padding():
    # Both sides carry -1 padding; -1 == -1 must NOT register as a hit.
    cand = torch.tensor([[5, -1, 3]])
    tgt = torch.tensor([[-1, 3]])
    nt = torch.tensor([1])

    # k=3 reaches the real hit (3); the -1s on both sides contribute nothing.
    assert recall_at_k(cand, tgt, nt, k=3).item() == pytest.approx(1.0)
    # k=2 sees only [5, -1]; candidate -1 vs target -1 is padding, not a hit.
    assert recall_at_k(cand, tgt, nt, k=2).item() == pytest.approx(0.0)


def test_ndcg_idcg_clamps_to_k():
    # 5 relevant targets but k=2: IDCG must be built from min(num_targets, k)=2
    # ideal hits, so a perfect top-2 scores exactly 1.0 (not < 1).
    cand = torch.tensor([[1, 2]])
    tgt = torch.tensor([[1, 2, 3, 4, 5]])
    nt = torch.tensor([5])
    assert ndcg_at_k(cand, tgt, nt, k=2).item() == pytest.approx(1.0)

    # Partial case: single hit at rank 2, one relevant target, k=2.
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1 (one ideal hit at rank 1).
    cand = torch.tensor([[9, 1]])
    tgt = torch.tensor([[1]])
    nt = torch.tensor([1])
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(cand, tgt, nt, k=2).item() == pytest.approx(expected)


def test_recall_denominator_is_num_targets_not_k():
    # One relevant target retrieved within k=3 → recall is 1/1, not 1/3.
    cand = torch.tensor([[7, 8, 9]])
    tgt = torch.tensor([[7, -1]])
    nt = torch.tensor([1])
    assert recall_at_k(cand, tgt, nt, k=3).item() == pytest.approx(1.0)

    # Two relevant targets, one retrieved → 1/2, again independent of k.
    tgt = torch.tensor([[7, 5]])
    nt = torch.tensor([2])
    assert recall_at_k(cand, tgt, nt, k=3).item() == pytest.approx(0.5)

    # Zero-target row: clamp(min=1) keeps the denominator finite → 0.0, not NaN.
    tgt = torch.tensor([[-1, -1]])
    nt = torch.tensor([0])
    assert recall_at_k(cand, tgt, nt, k=3).item() == pytest.approx(0.0)
