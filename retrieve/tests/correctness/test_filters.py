"""ExactAttributeFilter correctness — evaluate_mask + evaluate_indices."""

from __future__ import annotations

import torch

from retrieve.layers.filters import BloomFilter, ExactAttributeFilter, combine_masks
from retrieve.layers.linr.one_bit_knn import OneBitKNN
from retrieve.layers.utils.compact import compact_mask
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs


def test_clause_index_and_or_semantics():
    """Per-clause OR over attributes; AND across clauses."""
    n = 200
    attrs = torch.full((n, 2, 2), -1, dtype=torch.long, device="cuda")
    attrs[:100, 0, 0] = 1
    attrs[100:, 0, 0] = 2
    attrs[:, 1, 0] = 50
    # Some items in [50, 70) also carry clause-0 attr=3 → multi-value clause.
    attrs[50:70, 0, 1] = 3

    ci = ExactAttributeFilter()
    ci.register_index(attrs)

    q = torch.tensor(
        [
            [1, 50],  # items 0..99 ∩ all-have-50 → 0..99
            [2, 50],  # items 100..199
            [3, -1],  # items in [50, 70) (multi-value clause), any 2nd-clause
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
    ci = ExactAttributeFilter()
    ci.register_index(attrs)
    q = make_query_attrs(b=8, c=3, n_vocab=20, inactive_rate=0.2, seed=100)

    expected_mask = ci.evaluate_mask(q)
    expected_ids, expected_counts = compact_mask(expected_mask)

    got_ids, got_counts = ci.evaluate_indices(q)

    # counts must match exactly.
    assert torch.equal(
        got_counts, expected_counts
    ), f"counts mismatch: got {got_counts.tolist()} vs {expected_counts.tolist()}"

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
    ci = ExactAttributeFilter()
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
    ci = ExactAttributeFilter()
    ci.register_index(attrs)
    q = torch.full((3, 2), -1, dtype=torch.long, device="cuda")
    cand, counts = ci.evaluate_indices(q)
    assert torch.equal(counts, torch.full((3,), n, dtype=torch.int64, device="cuda"))
    for b in range(3):
        assert set(cand[b, :n].tolist()) == set(range(n))


def test_clause_index_all_reverse():
    """Every clause reverse → an item passes iff *no* clause attribute matches the query."""
    n, c = 256, 2
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=20, pad_rate=0.0, seed=51)
    is_reverse = torch.ones(c, dtype=torch.bool, device="cuda")
    ci = ExactAttributeFilter().to("cuda")
    ci.register_index(attrs, clause_is_reverse=is_reverse)

    q = make_query_attrs(b=4, c=c, n_vocab=20, inactive_rate=0.0, seed=52)
    mask = ci.evaluate_mask(q)

    # Reference: a_match[r, b, n, c] := q[b, c] in attrs[n, c, :]
    item = attrs.unsqueeze(0)  # [1, N, C, A]
    qb = q.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, 1]
    matched = (item == qb).any(dim=-1)  # [B, N, C]
    expected = (~matched).all(dim=-1)  # AND over reversed-clauses
    assert torch.equal(mask, expected)


def test_clause_index_single_item_index():
    """N=1 — broadcast and the fused kernel both must handle the corner cleanly."""
    attrs = torch.tensor([[[5, -1], [10, -1]]], dtype=torch.long, device="cuda")
    ci = ExactAttributeFilter().to("cuda")
    ci.register_index(attrs)
    q = torch.tensor([[5, 10], [5, 11], [-1, -1]], dtype=torch.long, device="cuda")
    mask = ci.evaluate_mask(q)
    assert mask.shape == (3, 1)
    assert mask[0, 0].item() is True or bool(mask[0, 0].item())  # both clauses match
    assert not bool(mask[1, 0].item())  # second clause fails
    assert bool(mask[2, 0].item())  # both inactive

    ids, counts = ci.evaluate_indices(q)
    assert counts.tolist() == [1, 0, 1]


def test_clause_index_evaluate_subset_p_zero():
    n = 128
    attrs = make_attrs(n, c=2, a_max=2, n_vocab=10, pad_rate=0.0, seed=61)
    ci = ExactAttributeFilter().to("cuda")
    ci.register_index(attrs)
    q = make_query_attrs(b=4, c=2, n_vocab=10, inactive_rate=0.0, seed=62)
    empty = torch.empty(4, 0, dtype=torch.long, device="cuda")
    out = ci.evaluate_subset(q, empty)
    assert out.shape == (4, 0)
    assert out.dtype == torch.bool


def test_one_bit_knn_with_combined_filter_mask():
    """Cross-compat smoke: BloomFilter & ExactAttributeFilter AND'd together feed OneBitKNN."""
    n, d, b, k = 4096, 128, 8, 32
    embs = make_index(n, d)
    query = make_query(b, d)
    attrs = make_attrs(n, c=3, a_max=3, n_vocab=40, pad_rate=0.2, seed=21)
    q_attrs = make_query_attrs(b, c=3, n_vocab=40, inactive_rate=0.2, seed=22)

    ci = ExactAttributeFilter().to("cuda")
    ci.register_index(attrs)
    bf = BloomFilter(m_bits=1024, k_hash=5).to("cuda")
    bf.register_index(attrs)

    clause_mask = ci.evaluate_mask(q_attrs)
    bloom_mask = bf.evaluate_mask(q_attrs)
    mask = combine_masks(clause_mask, bloom_mask)
    # Combined mask must be the clause mask exactly (bloom is a superset).
    assert torch.equal(mask, clause_mask)

    knn = OneBitKNN(k=k, backend="triton").to("cuda")
    knn.register_index(embs)
    ids, _ = knn(query, mask=mask)
    assert ids.shape == (b, k)

    # Every returned id must satisfy the clause conjunction (where the row had any).
    for r in range(b):
        if int(clause_mask[r].sum().item()) == 0:
            continue
        for j in range(k):
            i = int(ids[r, j].item())
            if i < 0:
                continue
            assert bool(clause_mask[r, i].item()), f"row {r}: returned id {i} fails clause filter"
