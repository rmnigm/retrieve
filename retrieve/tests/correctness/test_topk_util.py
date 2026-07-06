"""``masked_topk`` / ``counts_to_valid`` — direct correctness for the shared
masked top-K epilogue (K4.1).

Pins the per-call-site behaviors the six inlined originals had: -1/-inf
sentinels for masked or short rows, the isfinite boundary (genuine -inf
scores are indistinguishable from masked lanes), gather_ids local→global
mapping, and pad_to_k both ways. Device-agnostic math — CPU tensors suffice
(the suite-wide CUDA gate in conftest still applies to collection).
"""

from __future__ import annotations

import torch

from retrieve.layers.utils.topk import counts_to_valid, masked_topk

NEG_INF = float("-inf")


def test_counts_to_valid_prefix_mask():
    counts = torch.tensor([0, 2, 5], dtype=torch.int64)
    valid = counts_to_valid(counts, 4)
    expected = torch.tensor(
        [
            [False, False, False, False],
            [True, True, False, False],
            [True, True, True, True],  # counts > p saturates
        ]
    )
    assert valid.dtype == torch.bool
    assert torch.equal(valid, expected)


def test_empty_row_counts_zero():
    scores = torch.tensor(
        [
            [10.0, 30.0, 20.0, 5.0, 1.0, 0.0],
            [10.0, 30.0, 20.0, 5.0, 1.0, 0.0],
        ]
    )
    counts = torch.tensor([0, 6], dtype=torch.int64)
    ids, out_scores = masked_topk(scores, 4, valid=counts_to_valid(counts, 6))

    assert ids.shape == (2, 4)
    assert out_scores.shape == (2, 4)
    # Empty row: every winner is a sentinel.
    assert ids[0].tolist() == [-1, -1, -1, -1]
    assert out_scores[0].tolist() == [NEG_INF] * 4
    # Full row unaffected.
    assert ids[1].tolist() == [1, 2, 0, 3]
    assert out_scores[1].tolist() == [30.0, 20.0, 10.0, 5.0]


def test_counts_less_than_k():
    scores = torch.tensor([[10.0, 30.0, 20.0, 5.0, 1.0, 0.0]])
    counts = torch.tensor([3], dtype=torch.int64)
    ids, out_scores = masked_topk(scores, 4, valid=counts_to_valid(counts, 6))

    # Three survivors sorted by score, fourth slot is the sentinel pair.
    assert ids[0].tolist() == [1, 2, 0, -1]
    assert out_scores[0].tolist() == [30.0, 20.0, 10.0, NEG_INF]


def test_all_masked_row():
    scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    valid = torch.tensor([[False, False, False], [True, True, True]])
    ids, out_scores = masked_topk(scores, 3, valid=valid)

    assert ids[0].tolist() == [-1, -1, -1]
    assert out_scores[0].tolist() == [NEG_INF] * 3
    assert ids[1].tolist() == [2, 1, 0]
    assert out_scores[1].tolist() == [6.0, 5.0, 4.0]


def test_tie_at_neg_inf_boundary_with_valid():
    """A *valid* lane whose score is genuinely -inf is indistinguishable from a
    masked lane: the isfinite sentinel pass turns its id into -1."""
    scores = torch.tensor([[NEG_INF, 1.0, 2.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    ids, out_scores = masked_topk(scores, 3, valid=valid)

    assert ids[0].tolist() == [2, 1, -1]
    assert out_scores[0].tolist() == [2.0, 1.0, NEG_INF]


def test_no_sentinel_without_valid():
    """valid=None and P >= k: no isfinite pass runs — a raw -inf score keeps
    its real id (pins the dense SilverTorch._forward_candidates behavior)."""
    scores = torch.tensor([[NEG_INF, 1.0, 2.0]])
    ids, out_scores = masked_topk(scores, 3)

    assert ids[0].tolist() == [2, 1, 0]  # id 0 NOT replaced by -1
    assert out_scores[0].tolist() == [2.0, 1.0, NEG_INF]


def test_gather_ids_mapping():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.7]])
    gather_ids = torch.tensor([[100, 200, 300, 400]], dtype=torch.int64)
    ids, out_scores = masked_topk(scores, 2, gather_ids=gather_ids)

    assert ids[0].tolist() == [200, 400]
    assert out_scores[0].tolist() == [0.9, 0.7]


def test_gather_ids_with_mask_sentinel():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.7]])
    gather_ids = torch.tensor([[100, 200, 300, 400]], dtype=torch.int64)
    counts = torch.tensor([1], dtype=torch.int64)
    ids, out_scores = masked_topk(
        scores, 2, valid=counts_to_valid(counts, 4), gather_ids=gather_ids
    )

    # Only lane 0 valid: its global id wins, the rest are sentinels.
    assert ids[0].tolist() == [100, -1]
    assert out_scores[0].tolist() == [0.1, NEG_INF]


def test_pad_to_k_true_p_less_than_k():
    scores = torch.tensor([[3.0, 1.0, 2.0]])
    ids, out_scores = masked_topk(scores, 5, pad_to_k=True)

    assert ids.shape == (1, 5)
    assert out_scores.shape == (1, 5)
    assert ids[0].tolist() == [0, 2, 1, -1, -1]
    assert out_scores[0].tolist() == [3.0, 2.0, 1.0, NEG_INF, NEG_INF]


def test_pad_to_k_false_p_less_than_k():
    scores = torch.tensor([[3.0, 1.0, 2.0]])
    ids, out_scores = masked_topk(scores, 5, pad_to_k=False)

    # min(k, P) columns, no pad.
    assert ids.shape == (1, 3)
    assert out_scores.shape == (1, 3)
    assert ids[0].tolist() == [0, 2, 1]
    assert out_scores[0].tolist() == [3.0, 2.0, 1.0]


def test_pad_and_mask_interact():
    """P < k with a partial mask: masked winners get -1 via isfinite, then the
    pad extends with more -1/-inf to exactly k columns."""
    scores = torch.tensor([[3.0, 1.0, 2.0]])
    counts = torch.tensor([1], dtype=torch.int64)
    ids, out_scores = masked_topk(scores, 5, valid=counts_to_valid(counts, 3))

    assert ids[0].tolist() == [0, -1, -1, -1, -1]
    assert out_scores[0].tolist() == [3.0, NEG_INF, NEG_INF, NEG_INF, NEG_INF]


def test_output_dtypes():
    scores = torch.tensor([[1.0, 2.0]])
    ids, out_scores = masked_topk(scores, 2)
    assert ids.dtype == torch.int64  # topk indices
    assert out_scores.dtype == scores.dtype

    gather_ids = torch.tensor([[7, 9]], dtype=torch.int64)
    ids, _ = masked_topk(scores, 2, gather_ids=gather_ids)
    assert ids.dtype == torch.int64
