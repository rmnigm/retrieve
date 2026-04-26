"""ClauseIndex correctness — evaluate_mask + evaluate_indices."""

from __future__ import annotations

import torch

from retrieve.layers.utils.compact import compact_mask
from retrieve.layers.utils.filters import ClauseIndex
from tests.conftest import make_attrs, make_query_attrs


def test_clause_index_and_or_semantics():
    """Per-clause OR over attributes; AND across clauses."""
    n = 200
    attrs = torch.full((n, 2, 2), -1, dtype=torch.long, device="cuda")
    attrs[:100, 0, 0] = 1
    attrs[100:, 0, 0] = 2
    attrs[:, 1, 0] = 50
    # Some items in [50, 70) also carry clause-0 attr=3 → multi-value clause.
    attrs[50:70, 0, 1] = 3

    ci = ClauseIndex()
    ci.register_index(attrs)

    q = torch.tensor(
        [
            [1, 50],   # items 0..99 ∩ all-have-50 → 0..99
            [2, 50],   # items 100..199
            [3, -1],   # items in [50, 70) (multi-value clause), any 2nd-clause
            [-1, -1],  # both inactive → all
        ],
        dtype=torch.long,
        device="cuda",
    )
    mask = ci.evaluate_mask(q)
    assert mask.shape == (4, n)
    assert mask[0, :100].all() and not mask[0, 100:].any()
    assert mask[1, 100:].all() and not mask[1, :100].any()
    assert mask[2, 50:70].all()
    assert not mask[2, :50].any() and not mask[2, 70:].any()
    assert mask[3].all()


def test_evaluate_indices_matches_compact_evaluate_mask():
    """The fused kernel path must produce the same passing-set as the dense path."""
    n = 4096
    attrs = make_attrs(n, c=3, a_max=4, n_vocab=20, pad_rate=0.3, seed=99)
    ci = ClauseIndex()
    ci.register_index(attrs)
    q = make_query_attrs(b=8, c=3, n_vocab=20, inactive_rate=0.2, seed=100)

    expected_mask = ci.evaluate_mask(q)
    expected_ids, expected_counts = compact_mask(expected_mask)

    got_ids, got_counts = ci.evaluate_indices(q)

    # counts must match exactly.
    assert torch.equal(got_counts, expected_counts), (
        f"counts mismatch: got {got_counts.tolist()} vs {expected_counts.tolist()}"
    )

    # ids per row must match as sets (kernel order is unspecified).
    for b in range(q.shape[0]):
        c = int(expected_counts[b].item())
        exp_set = set(expected_ids[b, :c].tolist())
        got_set = set(got_ids[b, :c].tolist())
        assert exp_set == got_set, f"row {b} set mismatch (n_passing={c})"


def test_evaluate_indices_with_reverse_clauses():
    """Reverse clauses + inactive query slots both honored on the kernel path."""
    n = 1024
    attrs = make_attrs(n, c=2, a_max=3, n_vocab=10, pad_rate=0.2, seed=7)
    is_reverse = torch.tensor([True, False], device="cuda")
    ci = ClauseIndex()
    ci.register_index(attrs, clause_is_reverse=is_reverse)
    q = make_query_attrs(b=4, c=2, n_vocab=10, inactive_rate=0.5, seed=8)

    expected_mask = ci.evaluate_mask(q)
    expected_ids, expected_counts = compact_mask(expected_mask)
    got_ids, got_counts = ci.evaluate_indices(q)

    assert torch.equal(got_counts, expected_counts)
    for b in range(q.shape[0]):
        c = int(expected_counts[b].item())
        assert set(got_ids[b, :c].tolist()) == set(expected_ids[b, :c].tolist())


def test_evaluate_indices_all_inactive_passes_all():
    n = 256
    attrs = make_attrs(n, c=2, a_max=2, n_vocab=8, pad_rate=0.0, seed=1)
    ci = ClauseIndex()
    ci.register_index(attrs)
    q = torch.full((3, 2), -1, dtype=torch.long, device="cuda")
    cand, counts = ci.evaluate_indices(q)
    assert torch.equal(counts, torch.full((3,), n, dtype=torch.int64, device="cuda"))
    for b in range(3):
        assert set(cand[b, :n].tolist()) == set(range(n))
