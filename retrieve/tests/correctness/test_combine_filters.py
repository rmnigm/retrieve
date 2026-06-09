"""``combine_masks`` / ``combine_indices`` — composition correctness."""

from __future__ import annotations

import torch

from retrieve.layers.filters import (
    BloomFilter,
    ExactAttributeFilter,
    combine_indices,
    combine_masks,
)
from retrieve.layers.utils.compact import compact_mask
from tests.conftest import make_attrs, make_query_attrs


def _mk_filters(n=512, c=2, a_max=3, n_vocab=80, pad_rate=0.2, seed=0):
    attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=n_vocab, pad_rate=pad_rate, seed=seed)
    ci = ExactAttributeFilter().to("cuda")
    ci.register_index(attrs)
    bf = BloomFilter(m_bits=2048, k_hash=7).to("cuda")
    bf.register_index(attrs)
    return ci, bf


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
    ci, bf = _mk_filters(n=1024, c=2, a_max=3, seed=3)
    qc = make_query_attrs(b=8, c=2, n_vocab=80, inactive_rate=0.2, seed=4)
    qb = qc.clone()  # both filters built from the same attrs

    expected_mask = combine_masks(ci.evaluate_mask(qc), bf.evaluate_mask(qb))
    exp_ids, exp_counts = compact_mask(expected_mask)
    got_ids, got_counts = combine_indices([ci, bf], [qc, qb])

    assert torch.equal(got_counts, exp_counts), (
        f"counts mismatch {got_counts.tolist()} vs {exp_counts.tolist()}"
    )
    for r in range(qc.shape[0]):
        c = int(exp_counts[r].item())
        assert set(got_ids[r, :c].tolist()) == set(exp_ids[r, :c].tolist()), (
            f"row {r}: id-set mismatch"
        )


def test_combine_indices_filter_order_independent_for_set():
    ci, bf = _mk_filters(n=1024, c=2, a_max=3, seed=7)
    q = make_query_attrs(b=8, c=2, n_vocab=80, inactive_rate=0.2, seed=8)

    cb_ids, cb_counts = combine_indices([ci, bf], [q, q])
    bc_ids, bc_counts = combine_indices([bf, ci], [q, q])

    assert torch.equal(cb_counts, bc_counts)
    for r in range(q.shape[0]):
        c = int(cb_counts[r].item())
        assert set(cb_ids[r, :c].tolist()) == set(bc_ids[r, :c].tolist())


def test_combine_indices_empty_intermediate():
    """If the first filter passes nothing for any row, the cascade short-circuits."""
    n, c = 256, 2
    attrs = torch.full((n, c, 1), 7, dtype=torch.long, device="cuda")
    ci = ExactAttributeFilter().to("cuda")
    ci.register_index(attrs)
    bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
    bf.register_index(attrs)

    # Query asking for an attribute that no item carries → all rows are empty.
    q = torch.full((4, c), 9999, dtype=torch.long, device="cuda")
    ids, counts = combine_indices([ci, bf], [q, q])
    assert ids.shape == (4, 0)
    assert torch.equal(counts, torch.zeros(4, dtype=torch.int64, device="cuda"))


def test_combine_indices_single_filter():
    ci, _ = _mk_filters(n=512, c=2, a_max=3, seed=11)
    q = make_query_attrs(b=4, c=2, n_vocab=80, inactive_rate=0.2, seed=12)
    ids, counts = combine_indices([ci], [q])
    ref_ids, ref_counts = ci.evaluate_indices(q)
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
    with torch.no_grad():
        try:
            combine_indices([], [])
        except ValueError as e:
            assert "at least one filter" in str(e)
            return
    raise AssertionError("expected ValueError for empty filter list")


def test_combine_indices_length_mismatch_raises():
    ci, bf = _mk_filters(n=128, seed=21)
    q = make_query_attrs(b=2, c=2, n_vocab=80, seed=22)
    try:
        combine_indices([ci, bf], [q])
    except ValueError as e:
        assert "length mismatch" in str(e)
        return
    raise AssertionError("expected ValueError for filter/query length mismatch")


def test_combine_indices_subsequent_filter_drains():
    """First filter passes some items; second filter rejects them all → empty out."""
    n, c = 256, 2
    attrs = torch.full((n, c, 1), 7, dtype=torch.long, device="cuda")
    ci = ExactAttributeFilter().to("cuda")
    ci.register_index(attrs)
    bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
    bf.register_index(attrs)

    # ci (first) passes everything for q1=[-1,-1]; bf (second) rejects on q2=[9999, 9999].
    q1 = torch.full((4, c), -1, dtype=torch.long, device="cuda")
    q2 = torch.full((4, c), 9999, dtype=torch.long, device="cuda")
    ids, counts = combine_indices([ci, bf], [q1, q2])
    assert ids.shape == (4, 0)
    assert torch.equal(counts, torch.zeros(4, dtype=torch.int64, device="cuda"))
