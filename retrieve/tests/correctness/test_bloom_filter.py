"""BloomFilter correctness — shape, no-FN invariant, FPR, subset / indices paths."""

from __future__ import annotations

import math

import pytest
import torch

from retrieve.layers.filters import BloomFilter, ExactAttributeFilter
from tests.conftest import make_attrs, make_query_attrs


@pytest.fixture(scope="module")
def attrs():
    return make_attrs(n=2048, c=2, a_max=3, n_vocab=100, pad_rate=0.1)


@pytest.fixture(scope="module")
def query():
    return make_query_attrs(b=64, c=2, n_vocab=100, inactive_rate=0.0)


class TestShapeAndDtype:
    def test_register_index_buffer_shapes(self, attrs):
        bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
        bf.register_index(attrs)
        assert bf.bloom_sigs.shape == (attrs.shape[0], 512 // 64)
        assert bf.bloom_sigs.dtype == torch.int64
        assert bf.hash_seeds.shape == (5, 2)

    def test_evaluate_mask_shape_and_dtype(self, attrs, query):
        bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
        bf.register_index(attrs)
        mask = bf.evaluate_mask(query)
        assert mask.shape == (query.shape[0], attrs.shape[0])
        assert mask.dtype == torch.bool

    def test_invalid_m_bits_rejected(self):
        with pytest.raises(ValueError, match="power of 2"):
            BloomFilter(m_bits=500, k_hash=3)
        with pytest.raises(ValueError, match="multiple of 64"):
            BloomFilter(m_bits=32, k_hash=3)

    def test_invalid_k_hash_rejected(self):
        with pytest.raises(ValueError, match="k_hash must be positive"):
            BloomFilter(m_bits=512, k_hash=0)


class TestNoFalseNegatives:
    """Every clause-passing item must also pass the bloom — that's the invariant."""

    def test_random_inputs(self):
        attrs = make_attrs(n=512, c=3, a_max=4, n_vocab=80, pad_rate=0.3, seed=10)
        q = make_query_attrs(b=32, c=3, n_vocab=80, inactive_rate=0.2, seed=11)

        bf = BloomFilter(m_bits=2048, k_hash=7).to("cuda")
        bf.register_index(attrs)
        bloom_mask = bf.evaluate_mask(q)

        ci = ExactAttributeFilter().to("cuda")
        ci.register_index(attrs)
        clause_mask = ci.evaluate_mask(q)

        assert (clause_mask <= bloom_mask).all()

    def test_inactive_clauses_pass_all(self, attrs):
        bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
        bf.register_index(attrs)
        q = torch.tensor([[-1, -1]], dtype=torch.long, device="cuda")
        mask = bf.evaluate_mask(q)
        assert mask.all()


class TestFalsePositiveRate:
    """Empirical FPR should be in the same ballpark as the analytic bound.

    For a Bloom filter holding ``a`` items with ``m`` bits and ``k`` hashes,
    the per-bit fill probability is ``1 - exp(-k * a / m)``, and the
    probability that ``k`` independent hashes all hit set bits is roughly
    ``(1 - exp(-k * a / m)) ** k``. With one item per row's signature, the
    same formula applies with ``a = 1``.
    """

    def test_fpr_in_expected_range(self):
        # Single-clause, one attribute per item — analytic formula applies cleanly.
        n = 8192
        c, a_max = 1, 1
        m_bits, k_hash = 1024, 5
        attrs = make_attrs(n, c, a_max, n_vocab=10_000, pad_rate=0.0, seed=20)

        bf = BloomFilter(m_bits=m_bits, k_hash=k_hash).to("cuda")
        bf.register_index(attrs)

        q = make_query_attrs(b=128, c=c, n_vocab=10_000, inactive_rate=0.0, seed=21)
        # Shift query vocab away from item vocab — every "True" is a false positive.
        q = q + 100_000

        ci = ExactAttributeFilter().to("cuda")
        ci.register_index(attrs)
        true_match_rate = ci.evaluate_mask(q).float().mean().item()
        assert true_match_rate < 1e-3, "Expected near-zero true match rate"

        bloom_match_rate = bf.evaluate_mask(q).float().mean().item()
        per_item_fill = 1.0 - math.exp(-k_hash / m_bits)
        analytic_fpr = per_item_fill**k_hash
        assert bloom_match_rate <= max(
            analytic_fpr * 4, 1e-6
        ), f"observed FPR={bloom_match_rate:.4g}, analytic≤{analytic_fpr:.4g}"


class TestEvaluateSubset:
    def test_subset_parity_with_mask_gather(self, attrs, query):
        """``evaluate_subset(q, ids)`` must equal ``evaluate_mask(q).gather(1, ids)``."""
        bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
        bf.register_index(attrs)

        b = query.shape[0]
        n = attrs.shape[0]
        # Pick a random subset of P ids per query row.
        p = 128
        g = torch.Generator(device="cuda").manual_seed(42)
        ids = torch.randint(0, n, (b, p), generator=g, dtype=torch.long, device="cuda")

        mask_full = bf.evaluate_mask(query)
        expected = mask_full.gather(1, ids)
        got = bf.evaluate_subset(query, ids)
        assert torch.equal(got, expected)


class TestEdgeCases:
    def test_evaluate_subset_p_zero(self, attrs, query):
        bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
        bf.register_index(attrs)
        empty = torch.empty(query.shape[0], 0, dtype=torch.long, device="cuda")
        out = bf.evaluate_subset(query, empty)
        assert out.shape == (query.shape[0], 0)
        assert out.dtype == torch.bool

    def test_single_item_index(self):
        """N=1 corner — signature shape and subset test still well-formed."""
        attrs = make_attrs(n=1, c=2, a_max=2, n_vocab=10, pad_rate=0.0, seed=33)
        bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
        bf.register_index(attrs)
        q = make_query_attrs(b=4, c=2, n_vocab=10, inactive_rate=0.0, seed=34)
        mask = bf.evaluate_mask(q)
        assert mask.shape == (4, 1)
        assert mask.dtype == torch.bool

    def test_cpu_eval_mask_matches_cuda(self, attrs, query):
        """The torch-side broadcast subset path must agree with the Triton kernel."""
        bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
        bf.register_index(attrs)

        # CUDA path → bloom_match kernel.
        cuda_mask = bf.evaluate_mask(query)

        # CPU path → pure-torch broadcast subset.
        bf_cpu = BloomFilter(m_bits=512, k_hash=5)
        bf_cpu.register_index(attrs.cpu())
        cpu_mask = bf_cpu.evaluate_mask(query.cpu())

        assert torch.equal(cuda_mask.cpu(), cpu_mask)


class TestRegisterIndexReverseGuard:
    """Paper-strict: BloomFilter rejects any reverse clause at registration."""

    def test_register_raises_on_any_reverse_clause(self):
        attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=70)
        bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
        is_reverse = torch.tensor([False, True], device="cuda")
        with pytest.raises(ValueError, match="paper-strict"):
            bf.register_index(attrs, clause_is_reverse=is_reverse)

    def test_register_accepts_all_false(self):
        attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=71)
        bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
        is_reverse = torch.tensor([False, False], device="cuda")
        bf.register_index(attrs, clause_is_reverse=is_reverse)
        assert bf.bloom_sigs.shape == (64, 256 // 64)

    def test_register_none_skips_check(self):
        attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=72)
        bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
        bf.register_index(attrs, clause_is_reverse=None)
        assert bf.bloom_sigs.shape == (64, 256 // 64)
