"""Triton ``clause_compact`` vs the pure-torch ``ClauseIndex.evaluate_mask`` baseline.

``ClauseIndex.evaluate_indices`` already routes to ``clause_compact`` on CUDA,
so we call the kernel directly to keep this a true kernel-vs-pure-torch parity
check. Output id ordering is unspecified per the kernel doc — we compare row
*sets* of returned ids, not positions.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.filters.clause_compact import clause_compact
from retrieve.layers.filters import ClauseIndex
from retrieve.layers.utils.compact import compact_mask
from tests.conftest import make_attrs, make_query_attrs


def _ref(attrs: torch.Tensor, is_reverse: torch.Tensor, q: torch.Tensor):
    ci = ClauseIndex().to("cuda")
    ci.register_index(attrs, clause_is_reverse=is_reverse)
    ref_mask = ci.evaluate_mask(q)
    ref_ids, ref_counts = compact_mask(ref_mask)
    return ref_ids, ref_counts


def _set_match(out_ids, out_counts, ref_ids, ref_counts) -> None:
    assert torch.equal(
        out_counts, ref_counts
    ), f"counts: {out_counts.tolist()} vs {ref_counts.tolist()}"
    b = ref_counts.shape[0]
    for r in range(b):
        c = int(ref_counts[r].item())
        assert set(out_ids[r, :c].tolist()) == set(
            ref_ids[r, :c].tolist()
        ), f"row {r}: id-set mismatch"


@pytest.mark.parametrize("n", [256, 4096])
@pytest.mark.parametrize("c", [1, 3])
@pytest.mark.parametrize("a_max", [1, 4])
@pytest.mark.parametrize("pad_rate", [0.0, 0.3])
def test_clause_compact_matches_pure_torch(n, c, a_max, pad_rate):
    attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=40, pad_rate=pad_rate, seed=n + c)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=8, c=c, n_vocab=40, inactive_rate=0.2, seed=a_max + 1)

    out_ids, out_counts = clause_compact(attrs, is_reverse, q)
    ref_ids, ref_counts = _ref(attrs, is_reverse, q)
    _set_match(out_ids, out_counts, ref_ids, ref_counts)


def test_clause_compact_with_reverse_clauses():
    n, c = 1024, 3
    attrs = make_attrs(n, c=c, a_max=3, n_vocab=20, pad_rate=0.2, seed=42)
    is_reverse = torch.tensor([True, False, True], device="cuda")
    q = make_query_attrs(b=4, c=c, n_vocab=20, inactive_rate=0.0, seed=43)

    out_ids, out_counts = clause_compact(attrs, is_reverse, q)
    ref_ids, ref_counts = _ref(attrs, is_reverse, q)
    _set_match(out_ids, out_counts, ref_ids, ref_counts)


def test_clause_compact_inactive_query_passes_all():
    n, c = 512, 2
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=10, pad_rate=0.0, seed=7)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = torch.full((4, c), -1, dtype=torch.long, device="cuda")

    out_ids, out_counts = clause_compact(attrs, is_reverse, q)

    expected = torch.full((4,), n, dtype=torch.int64, device="cuda")
    assert torch.equal(out_counts, expected)
    for r in range(4):
        assert set(out_ids[r, :n].tolist()) == set(range(n))


def test_clause_compact_no_passing_items():
    """Query attribute that no item carries → counts all zero."""
    n, c = 256, 2
    attrs = torch.full((n, c, 1), 7, dtype=torch.long, device="cuda")
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = torch.full((4, c), 9999, dtype=torch.long, device="cuda")

    _, out_counts = clause_compact(attrs, is_reverse, q)
    assert torch.equal(out_counts, torch.zeros(4, dtype=torch.int64, device="cuda"))


def test_clause_compact_b_one():
    """Single-query batch — kernel grid axis 0 is degenerate but must work."""
    n, c = 512, 2
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=20, pad_rate=0.1, seed=99)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=1, c=c, n_vocab=20, inactive_rate=0.2, seed=100)

    out_ids, out_counts = clause_compact(attrs, is_reverse, q)
    ref_ids, ref_counts = _ref(attrs, is_reverse, q)
    _set_match(out_ids, out_counts, ref_ids, ref_counts)
