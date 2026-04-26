"""Triton ``bloom_match`` vs the pure-torch ``BloomIndex.evaluate`` baseline."""

from __future__ import annotations

import pytest
import torch

from retrieve.kernels.triton.silvertorch.bloom_match import bloom_match
from retrieve.layers.silvertorch.bloom import BloomIndex, _build_signatures
from tests.conftest import make_attrs, make_query_attrs


@pytest.mark.parametrize("n", [512, 4096])
@pytest.mark.parametrize("m_bits", [256, 1024])
@pytest.mark.parametrize("k_hash", [3, 7])
def test_bloom_match_matches_pure_torch(n, m_bits, k_hash):
    attrs = make_attrs(n, c=2, a_max=3, n_vocab=200, pad_rate=0.1, seed=n)
    bi = BloomIndex().to("cuda")
    bi.register_index(attrs, m_bits=m_bits, k_hash=k_hash)

    q = make_query_attrs(b=64, c=2, n_vocab=200, inactive_rate=0.0, seed=m_bits)
    ref = bi.evaluate(q)

    qb_sigs = _build_signatures(
        q.long().unsqueeze(-1),
        bi.hash_seeds,
        bi.m_bits,
        bi.k_hash,
        bi.word_count,
    )
    out = bloom_match(qb_sigs, bi.bloom_sigs)
    assert torch.equal(out, ref)
