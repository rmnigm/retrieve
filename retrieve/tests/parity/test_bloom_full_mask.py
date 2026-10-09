"""``bloom_full_mask`` (Triton) against its torch twin and the row-wise subset test, and
SilverTorch's ``bloom_path="full"`` against the default fused path: the same keep set read from
a packed full-N mask, so ids and scores are bit-exact (kernels.md § bloom_full_mask)."""

from __future__ import annotations

import pytest
import torch

from retrieve.modules.silvertorch import SilverTorch
from retrieve.ops import reference
from retrieve.ops.triton import bloom_full_mask
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs
from tests.parity.conftest import assert_topk_equal, make_bloom


@pytest.mark.parametrize("n", [500, 4096, 70_000])
def test_bloom_full_mask_matches_torch_and_the_subset_test(n):
    qpos, table, sigs, qb = make_bloom(n, 9, seed=n)
    qpos[0] = -1  # no active clause: every item passes
    out = bloom_full_mask(qpos, table)
    assert torch.equal(out, reference.bloom_full_mask(qpos, table))
    lanes = torch.arange(n, device=out.device)
    bits = ((out[:, lanes >> 6] >> (lanes & 63)) & 1).bool()
    subset = ((qb[:, None, :] & sigs[None]) == qb[:, None, :]).all(dim=2)
    assert torch.equal(bits[1:], subset[1:])
    assert bits[0].all()


def _layer(embs, attrs, backend, bloom_path):
    m = SilverTorch(k=20, n_lists=32, n_probe=8, filter_mode="bloom", m_bits=512, k_hash=5,
                    n_iter=3, backend=backend, bloom_path=bloom_path)  # fmt: skip
    m.register_index(embs, attrs)
    return m


@pytest.mark.parametrize("d", [64, 768])
@pytest.mark.parametrize("b", [1, 16])
def test_full_path_equals_the_fused_path(d, b):
    """Triton ``full`` against Triton ``partial`` and against the torch backend's ``partial``
    (the int32 path makes both bit-exact)."""
    embs = make_index(4096, d)
    attrs = make_attrs(4096, c=2, a_max=2, n_vocab=4)
    query, q_attrs = make_query(b, d), make_query_attrs(b, c=2, n_vocab=4)
    full = _layer(embs, attrs, "triton", "full")(query, q_attrs)
    assert_topk_equal(*full, *_layer(embs, attrs, "triton", "partial")(query, q_attrs))
    assert_topk_equal(*full, *_layer(embs, attrs, "torch", "partial")(query, q_attrs))
    assert_topk_equal(*full, *_layer(embs, attrs, "torch", "full")(query, q_attrs))


def test_full_path_is_bloom_on_triton_or_torch_only():
    with pytest.raises(ValueError, match="bloom_path='full' applies to"):
        SilverTorch(k=5, n_lists=8, n_probe=4, bloom_path="full")
    with pytest.raises(ValueError, match="bloom_path must be"):
        SilverTorch(k=5, n_lists=8, n_probe=4, filter_mode="bloom", m_bits=512, k_hash=4,
                    bloom_path="none")  # fmt: skip
