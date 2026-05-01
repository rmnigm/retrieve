"""IVF_INT8_ANN correctness — shape, mask, clause path, candidate path."""

from __future__ import annotations

import pytest
import torch

from retrieve import ClauseIndex
from retrieve.layers.silvertorch.ivf import IVF_INT8_ANN, build_ivf_int8
from retrieve.layers.utils.retrieval import FullScanKNN
from tests.conftest import (
    assert_recall_monotone,
    make_attrs,
    make_index,
    make_mask,
    make_query,
    recall_at_k,
)

N, D, B, K = 4096, 128, 16, 64
N_LISTS, N_PROBE = 64, 8


@pytest.fixture(scope="module")
def data():
    return {"embs": make_index(N, D), "query": make_query(B, D)}


class TestShapeAndDtype:
    def test_forward_shapes(self, data):
        m = IVF_INT8_ANN(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert ids.dtype == torch.long
        assert scores.dtype == torch.float32

    def test_buffer_dtypes(self, data):
        m = IVF_INT8_ANN(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        m.register_index(data["embs"])
        assert m.item_codes.dtype == torch.int8
        assert m.item_scales.dtype == torch.float32
        assert m.centroids.dtype == torch.float32
        assert m.padded_cluster_items.shape[0] == N_LISTS

    def test_invalid_params_rejected(self, data):
        m = IVF_INT8_ANN(k=5, n_lists=10_000, n_probe=4)
        with pytest.raises(ValueError, match="n_lists"):
            m.register_index(data["embs"])
        m2 = IVF_INT8_ANN(k=5, n_lists=8, n_probe=16)
        with pytest.raises(ValueError, match="n_probe"):
            m2.register_index(data["embs"])


class TestMask:
    @pytest.mark.parametrize("pass_rate", [0.05, 0.5])
    def test_mask_is_honored(self, data, pass_rate):
        m = IVF_INT8_ANN(k=K, n_lists=N_LISTS, n_probe=N_LISTS, n_iter=3)
        m.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()

    def test_clause_index_composed_externally(self, data):
        attrs = make_attrs(N, c=1, a_max=1, n_vocab=10)
        ci = ClauseIndex()
        ci.register_index(attrs)
        m = IVF_INT8_ANN(k=K, n_lists=N_LISTS, n_probe=N_LISTS, n_iter=3)
        m.register_index(data["embs"])
        q_attrs = torch.full((B, 1), 1, dtype=torch.long, device="cuda")
        mask = ci.evaluate_mask(q_attrs)
        ids, _ = m(data["query"], mask=mask)
        passing = (attrs[:, 0, 0] == 1).nonzero(as_tuple=True)[0]
        passing_set = set(passing.tolist())
        for b in range(B):
            for j in range(K):
                if ids[b, j].item() >= 0:
                    assert ids[b, j].item() in passing_set


class TestCandidates:
    def test_candidate_ids_returns_subset(self, data):
        m = IVF_INT8_ANN(k=2, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        m.register_index(data["embs"])
        g = torch.Generator(device="cuda").manual_seed(7)
        cand = torch.randint(0, N, (B, 16), generator=g, device="cuda")
        ids, _ = m(data["query"], candidate_ids=cand)
        assert ids.shape == (B, 2)
        for b in range(B):
            allowed = set(cand[b].tolist())
            for j in range(2):
                assert ids[b, j].item() in allowed


class TestRecallVsExact:
    def test_recall_grows_with_probe(self, data):
        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recalls = []
        for nprobe in (1, 4, 16, N_LISTS):
            m = IVF_INT8_ANN(k=K, n_lists=N_LISTS, n_probe=nprobe, n_iter=3)
            m.register_index(data["embs"])
            ids, _ = m(data["query"])
            recalls.append(recall_at_k(ids, ex_ids))
        assert_recall_monotone(recalls)
        assert recalls[-1] >= 0.85, recalls


class TestBuilder:
    def test_build_helper(self, data):
        m = build_ivf_int8(data["embs"], k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        assert isinstance(m, IVF_INT8_ANN)
        ids, _ = m(data["query"])
        assert ids.shape == (B, K)


class TestEdgeCases:
    def test_mask_all_false_returns_no_finite_scores(self, data):
        """All-False mask masks every item; every score is ``-inf``."""
        m = IVF_INT8_ANN(k=K, n_lists=N_LISTS, n_probe=N_LISTS, n_iter=3)
        m.register_index(data["embs"])
        all_false = torch.zeros(B, N, dtype=torch.bool, device="cuda")
        _, scores = m(data["query"], mask=all_false)
        assert not torch.isfinite(scores).any()

    def test_n_lists_equals_n(self, data):
        """Degenerate clustering (one item per cluster); full probe → recall ≈ 1."""
        from retrieve.layers.utils.retrieval import FullScanKNN
        from tests.conftest import recall_at_k

        # Use a smaller N so kmeans with n_lists=N stays cheap.
        small_n, d = 256, D
        embs = data["embs"][:small_n]
        m = IVF_INT8_ANN(k=K, n_lists=small_n, n_probe=small_n, n_iter=2)
        m.register_index(embs)
        ids, _ = m(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(embs)
        ex_ids, _ = exact(data["query"])
        # int8 quant + kmeans degeneracy can shuffle near-ties; allow modest slack.
        assert recall_at_k(ids, ex_ids) >= 0.85

    def test_candidate_ids_p_less_than_k(self, data):
        """``candidate_ids`` smaller than K — forward returns ``actual_k = p`` columns."""
        m = IVF_INT8_ANN(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        m.register_index(data["embs"])
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
