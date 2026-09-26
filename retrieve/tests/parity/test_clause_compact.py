"""Triton ``clause_compact`` vs the pure-torch ``ExactAttributeFilter.evaluate_mask`` baseline.

``ExactAttributeFilter.evaluate_indices`` already routes to ``clause_compact`` on CUDA,
so we call the kernel directly to keep this a true kernel-vs-pure-torch parity
check. Rows are compared in order: the kernel's contract is ascending item order (plan L3),
the same order ``compact_mask`` emits.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.functional import compact_mask
from retrieve.modules import ExactAttributeFilter
from retrieve.ops.triton.clause_compact import (
    DEFAULT_CONFIG,
    ClauseCompactConfig,
    _clause_compact_impl,
    clause_compact,
)
from retrieve.ops.triton.clause_mask import clause_mask
from tests.conftest import make_attrs, make_query_attrs
from tests.parity.conftest import poison_empty


def _ref(attrs: torch.Tensor, is_reverse: torch.Tensor, q: torch.Tensor):
    ci = ExactAttributeFilter().to("cuda")
    ci.register_index(attrs, clause_is_reverse=is_reverse)
    ref_mask = ci.evaluate_mask(q)
    ref_ids, ref_counts = compact_mask(ref_mask)
    return ref_ids, ref_counts


def _rows_equal(out_ids, out_counts, ref_ids, ref_counts) -> None:
    assert torch.equal(out_counts, ref_counts), (
        f"counts: {out_counts.tolist()} vs {ref_counts.tolist()}"
    )
    b = ref_counts.shape[0]
    for r in range(b):
        c = int(ref_counts[r].item())
        assert torch.equal(out_ids[r, :c], ref_ids[r, :c]), f"row {r}: ids differ"


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
    _rows_equal(out_ids, out_counts, ref_ids, ref_counts)


def test_clause_compact_with_reverse_clauses():
    n, c = 1024, 3
    attrs = make_attrs(n, c=c, a_max=3, n_vocab=20, pad_rate=0.2, seed=42)
    is_reverse = torch.tensor([True, False, True], device="cuda")
    q = make_query_attrs(b=4, c=c, n_vocab=20, inactive_rate=0.0, seed=43)

    out_ids, out_counts = clause_compact(attrs, is_reverse, q)
    ref_ids, ref_counts = _ref(attrs, is_reverse, q)
    _rows_equal(out_ids, out_counts, ref_ids, ref_counts)


def test_clause_compact_inactive_query_passes_all():
    n, c = 512, 2
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=10, pad_rate=0.0, seed=7)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = torch.full((4, c), -1, dtype=torch.long, device="cuda")

    out_ids, out_counts = clause_compact(attrs, is_reverse, q)

    expected = torch.full((4,), n, dtype=torch.int64, device="cuda")
    assert torch.equal(out_counts, expected)
    for r in range(4):
        assert torch.equal(out_ids[r, :n], torch.arange(n, device="cuda"))


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
    _rows_equal(out_ids, out_counts, ref_ids, ref_counts)


@pytest.mark.parametrize("block_n, num_warps", [(128, 2), (512, 8), (1024, 4)])
def test_clause_compact_config_override(block_n, num_warps):
    """Non-default ``ClauseCompactConfig`` produces the same rows —
    proves the ``config=`` kwarg plumbs through ``_clause_compact_impl``
    to the kernel launch and the two-phase compaction stays correct
    under non-default tiles."""
    n, c = 4096, 3
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=30, pad_rate=0.2, seed=121)
    is_reverse = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=8, c=c, n_vocab=30, inactive_rate=0.2, seed=122)

    cfg = ClauseCompactConfig(block_n=block_n, num_warps=num_warps)
    out_ids, out_counts = _clause_compact_impl(attrs, is_reverse, q, config=cfg)
    ref_ids, ref_counts = _ref(attrs, is_reverse, q)
    _rows_equal(out_ids, out_counts, ref_ids, ref_counts)


@pytest.mark.parametrize("r", [0, 1])
def test_clause_compact_equals_compacted_clause_mask(r):
    """``clause_compact`` is ``compact_mask(clause_mask(...))`` — the two Triton kernels agree
    bit for bit on counts and the ascending prefix ``[:counts]`` (the only defined part of a
    row); ``N % DEFAULT_CONFIG.block_n`` in {0, 1} (a full and a one-lane last tile), and one
    row passes nothing."""
    n, c = 4 * DEFAULT_CONFIG.block_n + r, 2
    assert n % DEFAULT_CONFIG.block_n == r
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=10, pad_rate=0.2, seed=5)
    is_reverse = torch.tensor([True, False], device="cuda")
    q = make_query_attrs(b=4, c=c, n_vocab=10, inactive_rate=0.2, seed=6)
    q[3] = torch.tensor([9999, 9999])
    is_reverse_none = torch.zeros(c, dtype=torch.bool, device="cuda")
    for rev in (is_reverse, is_reverse_none):
        ids, counts = clause_compact(attrs, rev, q)
        m_ids, m_counts = compact_mask(clause_mask(attrs, rev, q))
        assert torch.equal(counts, m_counts)
        width = torch.arange(n, device="cuda")[None, :] < counts[:, None]
        assert torch.equal(torch.where(width, ids, -1), torch.where(width, m_ids, -1))
    assert counts[3].item() == 0


@pytest.mark.parametrize("r", [0, 1])
def test_clause_compact_writes_nothing_past_counts(monkeypatch, r):
    """The output is ``torch.empty`` and the scatter kernel writes the runs only. Poisoned with
    an id no row can hold, every slot in ``[:counts]`` comes back as the reference's id and every
    slot past it still holds the poison (a full and a one-lane last tile; one row passes
    nothing)."""
    n, c, b = 4 * DEFAULT_CONFIG.block_n + r, 2, 4
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=10, pad_rate=0.2, seed=5)
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    q = make_query_attrs(b=b, c=c, n_vocab=10, inactive_rate=0.2, seed=6)
    q[3] = torch.tensor([9999, 9999])
    ref_ids, ref_counts = _ref(attrs, rev, q)
    hits = poison_empty(monkeypatch, (b, n), torch.int64, 123_456_789)
    ids, counts = clause_compact(attrs, rev, q)
    assert hits
    width = torch.arange(n, device="cuda")[None, :] < ref_counts[:, None]
    assert torch.equal(counts, ref_counts)
    assert torch.equal(ids, torch.where(width, ref_ids, 123_456_789))
