"""``combine_masks`` / ``combine_indices``: composition correctness over two exact filters on two
attribute tables of the same items (one on each backend)."""

from __future__ import annotations

import pytest
import torch

from retrieve.functional import combine_indices, combine_masks, compact_mask
from retrieve.modules import ExactAttributeFilter
from tests.conftest import make_attrs, make_query_attrs


def _mk_filters(n=512, c=2, a_max=3, n_vocab=80, pad_rate=0.2, seed=0):
    f1 = ExactAttributeFilter(backend="triton").to("cuda")
    f1.register_index(
        make_attrs(n, c=c, a_max=a_max, n_vocab=n_vocab, pad_rate=pad_rate, seed=seed)
    )
    f2 = ExactAttributeFilter(backend="torch").to("cuda")
    f2.register_index(
        make_attrs(n, c=c, a_max=a_max, n_vocab=n_vocab, pad_rate=pad_rate, seed=seed + 100)
    )
    return f1, f2


def test_combine_masks_none_tolerance():
    a = torch.tensor([[True, False, True]], device="cuda")
    b = torch.tensor([[True, True, False]], device="cuda")

    assert combine_masks(None) is None
    assert combine_masks(None, None) is None
    out_one = combine_masks(a, None)
    assert out_one is a
    out_one_swapped = combine_masks(None, a, None)
    assert out_one_swapped is a

    out_and = combine_masks(a, b)
    assert torch.equal(out_and, torch.tensor([[True, False, False]], device="cuda"))


def test_combine_masks_three_inputs():
    a = torch.tensor([[True, True, True, True]], device="cuda")
    b = torch.tensor([[True, True, False, True]], device="cuda")
    c = torch.tensor([[True, False, True, True]], device="cuda")
    out = combine_masks(a, b, c)
    assert torch.equal(out, torch.tensor([[True, False, False, True]], device="cuda"))


def test_combine_indices_matches_compact_of_combined_masks():
    f1, f2 = _mk_filters(n=1024, c=2, a_max=3, seed=3)
    q1 = make_query_attrs(b=8, c=2, n_vocab=80, inactive_rate=0.2, seed=4)
    q2 = make_query_attrs(b=8, c=2, n_vocab=80, inactive_rate=0.5, seed=5)

    expected_mask = combine_masks(
        f1.evaluate_mask(f1.prepare_queries(q1)), f2.evaluate_mask(f2.prepare_queries(q2))
    )
    exp_ids, exp_counts = compact_mask(expected_mask)
    got_ids, got_counts = combine_indices([f1, f2], [q1, q2])

    assert torch.equal(got_counts, exp_counts), (
        f"counts mismatch {got_counts.tolist()} vs {exp_counts.tolist()}"
    )
    for r in range(q1.shape[0]):
        c = int(exp_counts[r].item())
        assert set(got_ids[r, :c].tolist()) == set(exp_ids[r, :c].tolist()), (
            f"row {r}: id-set mismatch"
        )


def test_combine_indices_filter_order_independent_for_set():
    f1, f2 = _mk_filters(n=1024, c=2, a_max=3, seed=7)
    q = make_query_attrs(b=8, c=2, n_vocab=80, inactive_rate=0.2, seed=8)

    ab_ids, ab_counts = combine_indices([f1, f2], [q, q])
    ba_ids, ba_counts = combine_indices([f2, f1], [q, q])

    assert torch.equal(ab_counts, ba_counts)
    for r in range(q.shape[0]):
        c = int(ab_counts[r].item())
        assert set(ab_ids[r, :c].tolist()) == set(ba_ids[r, :c].tolist())


def _constant_filters(n=256, c=2):
    """Two exact filters over a table where every item holds value 7 in every clause."""
    attrs = torch.full((n, c, 1), 7, dtype=torch.long, device="cuda")
    f1 = ExactAttributeFilter().to("cuda")
    f1.register_index(attrs)
    f2 = ExactAttributeFilter(backend="torch").to("cuda")
    f2.register_index(attrs)
    return f1, f2


def test_combine_indices_empty_intermediate():
    """If the first filter passes nothing for any row, the cascade short-circuits."""
    f1, f2 = _constant_filters()
    q = torch.full((4, 2), 9999, dtype=torch.long, device="cuda")  # a value no item carries
    ids, counts = combine_indices([f1, f2], [q, q])
    assert ids.shape == (4, 0)
    assert torch.equal(counts, torch.zeros(4, dtype=torch.int64, device="cuda"))


def test_combine_indices_single_filter():
    f1, _ = _mk_filters(n=512, c=2, a_max=3, seed=11)
    q = make_query_attrs(b=4, c=2, n_vocab=80, inactive_rate=0.2, seed=12)
    ids, counts = combine_indices([f1], [q])
    ref_ids, ref_counts = f1.evaluate_indices(f1.prepare_queries(q))
    assert torch.equal(counts, ref_counts)
    for r in range(q.shape[0]):
        c = int(ref_counts[r].item())
        assert set(ids[r, :c].tolist()) == set(ref_ids[r, :c].tolist())


def test_combine_masks_all_none():
    """All-None inputs collapse to a None result (caller short-circuits)."""
    assert combine_masks() is None
    assert combine_masks(None) is None
    assert combine_masks(None, None, None) is None


def test_combine_indices_no_filters_raises():
    with pytest.raises(ValueError, match="at least one filter"):
        combine_indices([], [])


def test_combine_indices_length_mismatch_raises():
    f1, f2 = _mk_filters(n=128, seed=21)
    q = make_query_attrs(b=2, c=2, n_vocab=80, seed=22)
    with pytest.raises(ValueError, match="length mismatch"):
        combine_indices([f1, f2], [q])


def test_combine_indices_subsequent_filter_drains():
    """First filter passes some items; second filter rejects them all → empty out."""
    f1, f2 = _constant_filters()
    q1 = torch.full((4, 2), -1, dtype=torch.long, device="cuda")  # inactive: passes everything
    q2 = torch.full((4, 2), 9999, dtype=torch.long, device="cuda")
    ids, counts = combine_indices([f1, f2], [q1, q2])
    assert ids.shape == (4, 0)
    assert torch.equal(counts, torch.zeros(4, dtype=torch.int64, device="cuda"))
