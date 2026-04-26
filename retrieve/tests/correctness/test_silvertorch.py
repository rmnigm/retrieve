"""SilverTorch correctness — IVF + Bloom co-design.

Fixtures shrink the SilverTorch paper's eval (D=128, K=2048, n_probe=64) to
sizes that fit well on a single GPU; the bench suite scales up.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.layers.silvertorch import SilverTorch, build_silvertorch
from retrieve.layers.silvertorch.bloom import BloomIndex
from retrieve.layers.silvertorch.ivf import IVF_INT8_ANN
from retrieve.layers.utils.retrieval import FullScanKNN
from tests.conftest import (
    assert_recall_monotone,
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
    recall_at_k,
)

N, D, B, K = 4096, 128, 16, 64
N_LISTS, N_PROBE = 64, 8
M_BITS, K_HASH = 512, 5


@pytest.fixture(scope="module")
def data():
    embs = make_index(N, D)
    query = make_query(B, D)
    attrs = make_attrs(N, c=2, a_max=2)
    q_attrs = make_query_attrs(B, c=2)
    return {"embs": embs, "query": query, "attrs": attrs, "q_attrs": q_attrs}


def _build(with_attrs: bool, data, **overrides):
    kw = {
        "k": K,
        "n_lists": N_LISTS,
        "n_probe": N_PROBE,
        "m_bits": M_BITS,
        "k_hash": K_HASH,
        "n_iter": 3,
    }
    kw.update(overrides)
    m = SilverTorch(**kw)
    m.register_index(data["embs"], data["attrs"] if with_attrs else None)
    return m


class TestShape:
    def test_with_attrs(self, data):
        m = _build(with_attrs=True, data=data)
        ids, scores = m(data["query"], data["q_attrs"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)

    def test_without_attrs(self, data):
        m = _build(with_attrs=False, data=data)
        ids, _ = m(data["query"])
        assert ids.shape == (B, K)

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError, match="power of 2"):
            SilverTorch(k=5, n_lists=8, n_probe=4, m_bits=500, k_hash=5)
        with pytest.raises(ValueError, match="k_hash"):
            SilverTorch(k=5, n_lists=8, n_probe=4, m_bits=512, k_hash=0)


class TestEquivalence:
    def test_matches_composed_ivf_plus_bloom(self, data):
        """SilverTorch ⇔ IVF_INT8_ANN(mask=BloomIndex.evaluate(...))."""
        st = _build(with_attrs=True, data=data)
        ivf = IVF_INT8_ANN(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        ivf.register_index(data["embs"])
        bi = BloomIndex()
        bi.register_index(data["attrs"], m_bits=M_BITS, k_hash=K_HASH)

        bloom_mask = bi.evaluate(data["q_attrs"])
        ref_ids, ref_scores = ivf(data["query"], mask=bloom_mask)
        st_ids, st_scores = st(data["query"], data["q_attrs"])

        for b in range(B):
            assert sorted(st_ids[b].tolist()) == sorted(ref_ids[b].tolist())
        st_sorted, _ = st_scores.sort(dim=1, descending=True)
        ref_sorted, _ = ref_scores.sort(dim=1, descending=True)
        assert torch.allclose(st_sorted, ref_sorted, atol=1e-3)


class TestRecallVsExact:
    """SilverTorch paper claim: at sufficient probes, recall ≈ exact."""

    def test_recall_at_full_probe(self, data):
        """When n_probe == n_lists, IVF candidate set equals the full index → recall ≈ 1."""
        st = _build(with_attrs=False, data=data, n_probe=N_LISTS)
        ids, _ = st(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recall = recall_at_k(ids, ex_ids)
        # int8 quantization can shuffle ties; allow modest slack.
        assert recall >= 0.90, f"recall@{K} at full-probe = {recall:.3f}"

    def test_recall_grows_with_probe(self, data):
        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recalls = []
        for nprobe in (1, 4, 16, N_LISTS):
            st = _build(with_attrs=False, data=data, n_probe=nprobe)
            ids, _ = st(data["query"])
            recalls.append(recall_at_k(ids, ex_ids))
        assert_recall_monotone(recalls)
        assert recalls[-1] > recalls[0]


class TestCandidates:
    def test_candidate_ids_overrides(self, data):
        m = _build(with_attrs=True, data=data, k=2)
        cand = torch.tensor([[10, 20, 30]] * B, dtype=torch.long, device="cuda")
        ids, _ = m(data["query"], data["q_attrs"], candidate_ids=cand)
        allowed = {10, 20, 30}
        assert all(i.item() in allowed for row in ids for i in row)


class TestBuilder:
    def test_build_helper(self, data):
        m = build_silvertorch(
            data["embs"],
            k=K,
            n_lists=N_LISTS,
            n_probe=N_PROBE,
            m_bits=M_BITS,
            k_hash=K_HASH,
            n_iter=3,
            item_clause_attrs=data["attrs"],
        )
        assert isinstance(m, SilverTorch)
