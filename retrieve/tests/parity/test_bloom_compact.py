"""Triton ``bloom_compact`` vs ``compact_mask(bloom_match(.))`` baseline.

``BloomFilter.evaluate_indices`` already routes to ``bloom_compact`` on CUDA,
so we build the query signature by hand and call the kernel directly to keep
this a true kernel-vs-pure-torch parity check. Output id ordering is
unspecified per the kernel doc — we compare row *sets*, not positions.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.filters.bloom_compact import (
    BloomCompactConfig,
    _bloom_compact_impl,
    bloom_compact,
)
from retrieve.kernels.silvertorch.bloom_match import bloom_match
from retrieve.layers.filters import BloomFilter
from retrieve.layers.filters.bloom import _build_signatures
from retrieve.layers.utils.compact import compact_mask
from tests.conftest import make_attrs, make_query_attrs


def _build_qb(bf: BloomFilter, q: torch.Tensor) -> torch.Tensor:
    return _build_signatures(
        q.long().unsqueeze(-1),
        bf.hash_seeds,
        bf.m_bits,
        bf.k_hash,
        bf.word_count,
    )


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


@pytest.mark.parametrize("n", [512, 4096])
@pytest.mark.parametrize("m_bits", [256, 1024])
@pytest.mark.parametrize("k_hash", [3, 7])
def test_bloom_compact_matches_pure_torch(n, m_bits, k_hash):
    attrs = make_attrs(n, c=2, a_max=3, n_vocab=200, pad_rate=0.1, seed=n)
    bf = BloomFilter(m_bits=m_bits, k_hash=k_hash).to("cuda")
    bf.register_index(attrs)

    q = make_query_attrs(b=8, c=2, n_vocab=200, inactive_rate=0.2, seed=m_bits + k_hash)
    qb = _build_qb(bf, q)

    out_ids, out_counts = bloom_compact(qb, bf.bloom_sigs)
    ref_mask = bloom_match(qb, bf.bloom_sigs)
    ref_ids, ref_counts = compact_mask(ref_mask)
    _set_match(out_ids, out_counts, ref_ids, ref_counts)


def test_bloom_compact_inactive_query_passes_all():
    """All-inactive query → qb is all-zero → subset test passes for every item."""
    n = 512
    attrs = make_attrs(n, c=2, a_max=2, n_vocab=50, pad_rate=0.1, seed=11)
    bf = BloomFilter(m_bits=512, k_hash=4).to("cuda")
    bf.register_index(attrs)

    q = torch.full((4, 2), -1, dtype=torch.long, device="cuda")
    qb = _build_qb(bf, q)

    out_ids, out_counts = bloom_compact(qb, bf.bloom_sigs)

    expected = torch.full((4,), n, dtype=torch.int64, device="cuda")
    assert torch.equal(out_counts, expected)
    for r in range(4):
        assert set(out_ids[r, :n].tolist()) == set(range(n))


def test_bloom_compact_b_one():
    """Single-query batch — degenerate grid axis 0."""
    n = 1024
    attrs = make_attrs(n, c=2, a_max=2, n_vocab=50, pad_rate=0.1, seed=99)
    bf = BloomFilter(m_bits=512, k_hash=4).to("cuda")
    bf.register_index(attrs)

    q = make_query_attrs(b=1, c=2, n_vocab=50, inactive_rate=0.0, seed=100)
    qb = _build_qb(bf, q)

    out_ids, out_counts = bloom_compact(qb, bf.bloom_sigs)
    ref_ids, ref_counts = compact_mask(bloom_match(qb, bf.bloom_sigs))
    _set_match(out_ids, out_counts, ref_ids, ref_counts)


def test_bloom_compact_n_smaller_than_block():
    """N below the fixed BLOCK_N=256 — single-tile path."""
    n = 64
    attrs = make_attrs(n, c=2, a_max=2, n_vocab=30, pad_rate=0.1, seed=5)
    bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
    bf.register_index(attrs)

    q = make_query_attrs(b=4, c=2, n_vocab=30, inactive_rate=0.0, seed=6)
    qb = _build_qb(bf, q)

    out_ids, out_counts = bloom_compact(qb, bf.bloom_sigs)
    ref_ids, ref_counts = compact_mask(bloom_match(qb, bf.bloom_sigs))
    _set_match(out_ids, out_counts, ref_ids, ref_counts)


def test_bloom_compact_routed_via_layer():
    """``BloomFilter.evaluate_indices`` on CUDA must use the fused kernel."""
    n = 1024
    attrs = make_attrs(n, c=2, a_max=2, n_vocab=80, pad_rate=0.2, seed=21)
    bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
    bf.register_index(attrs)

    q = make_query_attrs(b=4, c=2, n_vocab=80, inactive_rate=0.1, seed=22)

    got_ids, got_counts = bf.evaluate_indices(q)
    ref_ids, ref_counts = compact_mask(bf.evaluate_mask(q))
    _set_match(got_ids, got_counts, ref_ids, ref_counts)


@pytest.mark.parametrize("block_n, num_warps", [(128, 2), (512, 8), (1024, 4)])
def test_bloom_compact_config_override(block_n, num_warps):
    """Non-default ``BloomCompactConfig`` produces the same row-id sets —
    proves the ``config=`` kwarg plumbs through ``_bloom_compact_impl``
    to the kernel launch and the atomic_add compaction stays correct
    under non-default tiles."""
    n = 4096
    attrs = make_attrs(n, c=2, a_max=3, n_vocab=200, pad_rate=0.1, seed=141)
    bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
    bf.register_index(attrs)

    q = make_query_attrs(b=8, c=2, n_vocab=200, inactive_rate=0.2, seed=142)
    qb = _build_qb(bf, q)

    cfg = BloomCompactConfig(block_n=block_n, num_warps=num_warps)
    out_ids, out_counts = _bloom_compact_impl(qb, bf.bloom_sigs, config=cfg)
    ref_mask = bloom_match(qb, bf.bloom_sigs)
    ref_ids, ref_counts = compact_mask(ref_mask)
    _set_match(out_ids, out_counts, ref_ids, ref_counts)
