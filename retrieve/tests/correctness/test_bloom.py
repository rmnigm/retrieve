"""BloomIndex correctness — shape, no-FN invariant, FPR vs theory."""

from __future__ import annotations

import math

import pytest
import torch

from retrieve.layers.silvertorch.bloom import BloomIndex
from retrieve.layers.utils.filters import ClauseIndex
from tests.conftest import make_attrs, make_query_attrs


@pytest.fixture(scope="module")
def attrs():
    return make_attrs(n=2048, c=2, a_max=3, n_vocab=100, pad_rate=0.1)


@pytest.fixture(scope="module")
def query():
    return make_query_attrs(b=64, c=2, n_vocab=100, inactive_rate=0.0)


class TestShapeAndDtype:
    def test_register_index_buffer_shapes(self, attrs):
        bi = BloomIndex()
        bi.register_index(attrs, m_bits=512, k_hash=5)
        assert bi.bloom_sigs.shape == (attrs.shape[0], 512 // 64)
        assert bi.bloom_sigs.dtype == torch.int64
        assert bi.hash_seeds.shape == (5, 2)

    def test_evaluate_shape_and_dtype(self, attrs, query):
        bi = BloomIndex()
        bi.register_index(attrs, m_bits=512, k_hash=5)
        mask = bi.evaluate(query)
        assert mask.shape == (query.shape[0], attrs.shape[0])
        assert mask.dtype == torch.bool

    def test_invalid_m_bits_rejected(self, attrs):
        bi = BloomIndex()
        with pytest.raises(ValueError, match="power of 2"):
            bi.register_index(attrs, m_bits=500, k_hash=3)
        with pytest.raises(ValueError, match="multiple of 64"):
            bi.register_index(attrs, m_bits=32, k_hash=3)

    def test_invalid_k_hash_rejected(self, attrs):
        bi = BloomIndex()
        with pytest.raises(ValueError, match="k_hash must be positive"):
            bi.register_index(attrs, m_bits=512, k_hash=0)


class TestNoFalseNegatives:
    """Every clause-passing item must also pass the bloom — that's the invariant."""

    def test_random_inputs(self):
        attrs = make_attrs(n=512, c=3, a_max=4, n_vocab=80, pad_rate=0.3, seed=10)
        q = make_query_attrs(b=32, c=3, n_vocab=80, inactive_rate=0.2, seed=11)

        bi = BloomIndex()
        bi.register_index(attrs, m_bits=2048, k_hash=7)
        bloom_mask = bi.evaluate(q)

        ci = ClauseIndex()
        ci.register_index(attrs)
        clause_mask = ci.evaluate_mask(q)

        assert (clause_mask <= bloom_mask).all()

    def test_inactive_clauses_pass_all(self, attrs):
        bi = BloomIndex()
        bi.register_index(attrs, m_bits=512, k_hash=5)
        q = torch.tensor([[-1, -1]], dtype=torch.long, device="cuda")
        mask = bi.evaluate(q)
        assert mask.all()


class TestFalsePositiveRate:
    """Empirical FPR should be in the same ballpark as the analytic bound.

    For a Bloom filter holding ``a`` items with ``m`` bits and ``k`` hashes,
    the per-bit fill probability is ``1 - (1 - 1/m) ** (k * a)``, and the
    probability that ``k`` independent hashes all hit set bits is roughly
    ``(1 - exp(-k * a / m)) ** k``. The Bloom-per-clause structure here has
    the same shape with ``a`` = number of attribute IDs hashed per item.
    """

    def test_fpr_in_expected_range(self):
        # Single-clause scheme so the analytic formula applies cleanly:
        # each item contributes exactly k hash bits to its row's signature.
        n = 8192
        c, a_max = 1, 1
        m_bits, k_hash = 1024, 5
        attrs = make_attrs(n, c, a_max, n_vocab=10_000, pad_rate=0.0, seed=20)

        bi = BloomIndex()
        bi.register_index(attrs, m_bits=m_bits, k_hash=k_hash)

        # Queries drawn from a *disjoint* vocab so true matches are vanishingly
        # rare — every "True" in the bloom mask is a false positive.
        q = make_query_attrs(b=128, c=c, n_vocab=10_000, inactive_rate=0.0, seed=21)
        # Shift query vocab away from item vocab.
        q = q + 100_000

        ci = ClauseIndex()
        ci.register_index(attrs)
        true_match_rate = ci.evaluate_mask(q).float().mean().item()
        assert true_match_rate < 1e-3, (
            "Expected near-zero true match rate; FPR estimate would be biased."
        )

        bloom_match_rate = bi.evaluate(q).float().mean().item()
        # Analytic FPR for a single attribute hashed into k bits per signature:
        #   fpr ≈ (1 - exp(-k / m)) ** k    per item
        # This is the *single-item* fill estimate; it is a tight lower bound
        # because items' signatures are independent draws.
        per_item_fill = 1.0 - math.exp(-k_hash / m_bits)
        analytic_fpr = per_item_fill**k_hash
        # Allow a generous 4× / 1/4 band — analytic FPR is approximate.
        assert bloom_match_rate <= max(analytic_fpr * 4, 1e-6), (
            f"observed FPR={bloom_match_rate:.4g}, analytic≤{analytic_fpr:.4g}"
        )
