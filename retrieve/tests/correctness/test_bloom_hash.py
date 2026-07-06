"""bloom_hash builders — chunked vs loop-free equivalence, seed determinism.

The hash math is pinned (persisted ``bloom_sigs`` buffers must stay bit-valid
across refactors), so every assertion here is exact int64 equality — never a
tolerance."""

from __future__ import annotations

import torch

from retrieve.layers.filters import bloom_hash
from retrieve.layers.filters.bloom_hash import (
    build_query_signatures,
    build_signatures,
    generate_seeds,
)
from tests.conftest import make_attrs

M_BITS, K_HASH = 512, 5
WORD_COUNT = M_BITS // 64


def test_index_and_query_builders_agree_rowwise():
    """``build_signatures(attrs)[i] == build_query_signatures(attrs[i:i+1])[0]`` —
    the chunked (index-side) and loop-free (query-side) paths share one core and
    must agree exactly; this property was assumed but never asserted before."""
    attrs = make_attrs(n=257, c=3, a_max=4, n_vocab=100, pad_rate=0.3, seed=90)
    seeds = generate_seeds(K_HASH, device=attrs.device)
    sigs = build_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT)
    for i in (0, 1, 128, 256):
        row = build_query_signatures(attrs[i : i + 1], seeds, M_BITS, K_HASH, WORD_COUNT)
        assert torch.equal(sigs[i], row[0]), f"row {i}: chunked != loop-free"


def test_chunked_build_crosses_batch_boundary(monkeypatch):
    """Force several chunks through the index-side loop (incl. a ragged tail);
    output must be bit-identical to the loop-free build of the same slab."""
    monkeypatch.setattr(bloom_hash, "_BUILD_SIGS_BATCH", 7)
    attrs = make_attrs(n=100, c=2, a_max=3, n_vocab=50, pad_rate=0.2, seed=91)
    seeds = generate_seeds(K_HASH, device=attrs.device)
    chunked = build_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT)
    loop_free = build_query_signatures(attrs, seeds, M_BITS, K_HASH, WORD_COUNT)
    assert torch.equal(chunked, loop_free)


def test_generate_seeds_deterministic_and_odd():
    """Fixed CPU generator → identical seeds on every call; multipliers are odd."""
    a = generate_seeds(7, device=torch.device("cuda"))
    b = generate_seeds(7, device=torch.device("cuda"))
    assert torch.equal(a, b)
    assert a.shape == (7, 2)
    assert bool((a & 1).eq(1).all())
