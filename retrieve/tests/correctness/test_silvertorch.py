"""SilverTorch correctness — IVF + INT8 ANN with optional Bloom co-design.

Fixtures shrink the SilverTorch paper's eval (D=128, K=2048, n_probe=64) to
sizes that fit well on a single GPU; larger sweeps live in ``evaluation/``.
Tests cover both the bloom-configured path (``m_bits``/``k_hash`` set, plus
``item_clause_attrs``/``query_clause_attrs``) and the bloom-disabled path
(both kwargs ``None``, behaving as plain IVF + INT8 ANN).
"""

from __future__ import annotations

import pytest
import torch

from retrieve import ExactAttributeFilter
from retrieve.layers.filters import BloomFilter
from retrieve.layers.silvertorch import SilverTorch, build_silvertorch
from retrieve.layers.utils.retrieval import FullScanKNN
from tests.conftest import (
    assert_recall_monotone,
    make_attrs,
    make_index,
    make_mask,
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


def _build_no_bloom(data, **overrides):
    kw = {"k": K, "n_lists": N_LISTS, "n_probe": N_PROBE, "n_iter": 3}
    kw.update(overrides)
    m = SilverTorch(**kw)
    m.register_index(data["embs"])
    return m


class TestShape:
    def test_with_attrs(self, data):
        m = _build(with_attrs=True, data=data)
        ids, scores = m(data["query"], data["q_attrs"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert ids.dtype == torch.long
        assert scores.dtype == torch.float32

    def test_without_attrs_bloom_configured(self, data):
        m = _build(with_attrs=False, data=data)
        ids, _ = m(data["query"])
        assert ids.shape == (B, K)

    def test_no_bloom(self, data):
        m = _build_no_bloom(data)
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert ids.dtype == torch.long
        assert scores.dtype == torch.float32

    def test_buffer_dtypes(self, data):
        m = _build_no_bloom(data)
        assert m.item_codes.dtype == torch.int8
        assert m.item_scales.dtype == torch.float32
        assert m.centroids.dtype == torch.float32
        assert m.padded_cluster_items.shape[0] == N_LISTS
        # No bloom config → bloom buffers must not be allocated.
        assert not hasattr(m, "bloom_sigs")
        assert not hasattr(m, "hash_seeds")


class TestParamValidation:
    def test_invalid_bloom_params_rejected(self):
        with pytest.raises(ValueError, match="power of 2"):
            SilverTorch(k=5, n_lists=8, n_probe=4, m_bits=500, k_hash=5)
        with pytest.raises(ValueError, match="k_hash"):
            SilverTorch(k=5, n_lists=8, n_probe=4, m_bits=512, k_hash=0)

    def test_partial_bloom_config_rejected(self):
        with pytest.raises(ValueError, match="m_bits and k_hash"):
            SilverTorch(k=5, n_lists=8, n_probe=4, m_bits=512)
        with pytest.raises(ValueError, match="m_bits and k_hash"):
            SilverTorch(k=5, n_lists=8, n_probe=4, k_hash=4)

    def test_invalid_ivf_params_rejected(self, data):
        m = SilverTorch(k=5, n_lists=10_000, n_probe=4)
        with pytest.raises(ValueError, match="n_lists"):
            m.register_index(data["embs"])
        m2 = SilverTorch(k=5, n_lists=8, n_probe=16)
        with pytest.raises(ValueError, match="n_probe"):
            m2.register_index(data["embs"])

    def test_query_clause_attrs_without_bloom_rejected(self, data):
        m = _build_no_bloom(data)
        qa = torch.zeros(B, 1, dtype=torch.long, device="cuda")
        with pytest.raises(ValueError, match="query_clause_attrs requires bloom"):
            m(data["query"], query_clause_attrs=qa)

    def test_item_clause_attrs_without_bloom_rejected(self, data):
        m = SilverTorch(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        attrs = make_attrs(N, c=1, a_max=1, n_vocab=10)
        with pytest.raises(ValueError, match="item_clause_attrs requires bloom"):
            m.register_index(data["embs"], item_clause_attrs=attrs)


class TestMask:
    @pytest.mark.parametrize("pass_rate", [0.05, 0.5])
    def test_mask_is_honored(self, data, pass_rate):
        m = _build_no_bloom(data, n_probe=N_LISTS)
        mask = make_mask(B, N, pass_rate=pass_rate)
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()

    def test_mask_all_false_no_bloom(self, data):
        m = _build_no_bloom(data, n_probe=N_LISTS)
        all_false = torch.zeros(B, N, dtype=torch.bool, device="cuda")
        _, scores = m(data["query"], mask=all_false)
        assert not torch.isfinite(scores).any()

    def test_mask_all_false_with_bloom(self, data):
        st = _build(with_attrs=True, data=data)
        all_false = torch.zeros(B, N, dtype=torch.bool, device="cuda")
        _, scores = st(data["query"], data["q_attrs"], mask=all_false)
        assert not torch.isfinite(scores).any()

    def test_clause_index_composed_externally(self, data):
        attrs = make_attrs(N, c=1, a_max=1, n_vocab=10)
        ci = ExactAttributeFilter()
        ci.register_index(attrs)
        m = _build_no_bloom(data, n_probe=N_LISTS)
        q_attrs = torch.full((B, 1), 1, dtype=torch.long, device="cuda")
        mask = ci.evaluate_mask(q_attrs)
        ids, _ = m(data["query"], mask=mask)
        passing = (attrs[:, 0, 0] == 1).nonzero(as_tuple=True)[0]
        passing_set = set(passing.tolist())
        for b in range(B):
            for j in range(K):
                if ids[b, j].item() >= 0:
                    assert ids[b, j].item() in passing_set


class TestEquivalence:
    def test_bloom_matches_no_bloom_plus_external_mask(self, data):
        """SilverTorch(qa) ⇔ SilverTorch (no bloom)(mask=BloomFilter.evaluate_mask(...))."""
        st = _build(with_attrs=True, data=data)
        ref = _build_no_bloom(data)
        bf = BloomFilter(m_bits=M_BITS, k_hash=K_HASH)
        bf.register_index(data["attrs"])

        bloom_mask = bf.evaluate_mask(data["q_attrs"])
        ref_ids, ref_scores = ref(data["query"], mask=bloom_mask)
        st_ids, st_scores = st(data["query"], data["q_attrs"])

        for b in range(B):
            assert sorted(st_ids[b].tolist()) == sorted(ref_ids[b].tolist())
        st_sorted, _ = st_scores.sort(dim=1, descending=True)
        ref_sorted, _ = ref_scores.sort(dim=1, descending=True)
        assert torch.allclose(st_sorted, ref_sorted, atol=1e-3)

    def test_query_clause_attrs_none_equals_no_bloom(self, data):
        """``query_clause_attrs=None`` on a bloom-configured module ≡ no-bloom module.

        Same kmeans seed → identical centroids → identical scoring; bloom branch
        is skipped at query time when qa is None regardless of has_bloom.
        """
        st = _build(with_attrs=True, data=data)
        nb = _build_no_bloom(data)

        st_ids, st_scores = st(data["query"])
        nb_ids, nb_scores = nb(data["query"])

        for b in range(B):
            assert sorted(st_ids[b].tolist()) == sorted(nb_ids[b].tolist())
        st_sorted, _ = st_scores.sort(dim=1, descending=True)
        nb_sorted, _ = nb_scores.sort(dim=1, descending=True)
        assert torch.allclose(st_sorted, nb_sorted, atol=1e-3)


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
            m = _build_no_bloom(data, n_probe=nprobe)
            ids, _ = m(data["query"])
            recalls.append(recall_at_k(ids, ex_ids))
        assert_recall_monotone(recalls)
        assert recalls[-1] >= 0.85, recalls


class TestCandidates:
    def test_candidate_ids_with_bloom(self, data):
        m = _build(with_attrs=True, data=data, k=2)
        cand = torch.tensor([[10, 20, 30]] * B, dtype=torch.long, device="cuda")
        ids, _ = m(data["query"], data["q_attrs"], candidate_ids=cand)
        allowed = {10, 20, 30}
        assert all(i.item() in allowed for row in ids for i in row)

    def test_candidate_ids_no_bloom(self, data):
        m = _build_no_bloom(data, k=2)
        g = torch.Generator(device="cuda").manual_seed(7)
        cand = torch.randint(0, N, (B, 16), generator=g, device="cuda")
        ids, _ = m(data["query"], candidate_ids=cand)
        assert ids.shape == (B, 2)
        for b in range(B):
            allowed = set(cand[b].tolist())
            for j in range(2):
                assert ids[b, j].item() in allowed

    def test_candidate_ids_p_less_than_k(self, data):
        """``candidate_ids`` smaller than K — forward returns ``actual_k = p`` columns."""
        m = _build_no_bloom(data)
        p = K // 2
        g = torch.Generator(device="cuda").manual_seed(42)
        cand = torch.randint(0, N, (B, p), generator=g, dtype=torch.long, device="cuda")
        ids, scores = m(data["query"], candidate_ids=cand)
        assert ids.shape == (B, p)
        assert scores.shape == (B, p)
        for b in range(B):
            allowed = set(cand[b].tolist())
            for j in range(p):
                assert int(ids[b, j].item()) in allowed


class TestBuilder:
    def test_build_helper_with_bloom(self, data):
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

    def test_build_helper_no_bloom(self, data):
        m = build_silvertorch(data["embs"], k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        assert isinstance(m, SilverTorch)
        ids, _ = m(data["query"])
        assert ids.shape == (B, K)


class TestEdgeCases:
    def test_n_lists_equals_n(self, data):
        """Degenerate clustering (one item per cluster); full probe → recall ≈ 1."""
        small_n = 256
        embs = data["embs"][:small_n]
        m = SilverTorch(k=K, n_lists=small_n, n_probe=small_n, n_iter=2)
        m.register_index(embs)
        ids, _ = m(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(embs)
        ex_ids, _ = exact(data["query"])
        # int8 quant + kmeans degeneracy can shuffle near-ties; allow modest slack.
        assert recall_at_k(ids, ex_ids) >= 0.85
