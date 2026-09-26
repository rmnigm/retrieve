"""Triton ``bloom_match`` vs the pure-torch ``BloomFilter.evaluate_mask`` baseline.

``BloomFilter.evaluate_mask`` already routes to ``bloom_match`` on CUDA, so we
build the query signature by hand and call the kernel directly to keep this
test a true kernel-vs-pure-torch parity check.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.indexing.bloom_hash import build_signatures
from retrieve.modules import BloomFilter
from retrieve.ops.triton.bloom_match import bloom_match
from tests.conftest import make_attrs, make_query_attrs
from tests.parity.conftest import make_words


@pytest.mark.parametrize("n", [512, 4096])
@pytest.mark.parametrize("m_bits", [256, 1024])
@pytest.mark.parametrize("k_hash", [3, 7])
def test_bloom_match_matches_pure_torch(n, m_bits, k_hash):
    attrs = make_attrs(n, c=2, a_max=3, n_vocab=200, pad_rate=0.1, seed=n)
    bf = BloomFilter(m_bits=m_bits, k_hash=k_hash).to("cuda")
    bf.register_index(attrs)

    q = make_query_attrs(b=64, c=2, n_vocab=200, inactive_rate=0.0, seed=m_bits)

    # Pure-torch reference: subset test on CPU.
    qb_sigs_cpu = build_signatures(
        q.cpu().long().unsqueeze(-1),
        bf.hash_seeds.cpu(),
        bf.m_bits,
        bf.k_hash,
        bf.word_count,
    )
    sigs_cpu = bf.bloom_sigs.cpu()
    ref = (
        ((qb_sigs_cpu.unsqueeze(1) & sigs_cpu.unsqueeze(0)) == qb_sigs_cpu.unsqueeze(1))
        .all(dim=-1)
        .to("cuda")
    )

    qb_sigs = build_signatures(
        q.long().unsqueeze(-1),
        bf.hash_seeds,
        bf.m_bits,
        bf.k_hash,
        bf.word_count,
    )
    out = bloom_match(qb_sigs, bf.bloom_sigs)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("w", [3, 12])
def test_bloom_match_non_power_of_two_words(w):
    """``W`` pads to ``next_power_of_2`` in-kernel; the answer is the subset test at the true
    ``W``. ``BloomFilter`` keeps ``m_bits`` a power of two, so this is the op on its own."""
    sigs, qb = make_words(4096, w, seed=w)
    ref = ((qb.unsqueeze(1) & sigs.unsqueeze(0)) == qb.unsqueeze(1)).all(dim=-1)
    out = bloom_match(qb, sigs)
    assert torch.equal(out, ref)
    assert out.any() and not out.all()
