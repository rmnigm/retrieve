"""LiNR correctness — SimilarityMasking / PrefilterKNN / OneBitKNN in both
torch and Triton backends.

Fixture sizes mirror the LinR paper's small evaluation slice (D=128,
batch=16, K=200) at a smaller N so the suite stays interactive on a single
GPU. Latency / recall-vs-N benchmarks live in ``evaluation/``.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.layers.filters import ExactAttributeFilter
from retrieve.layers.linr.one_bit_knn import OneBitKNN
from retrieve.layers.linr.one_bit_knn_triton import OneBitKNNTriton
from retrieve.layers.linr.prefilter_knn import PrefilterKNN
from retrieve.layers.linr.prefilter_knn_triton import PrefilterKNNTriton
from retrieve.layers.linr.similarity_masking import SimilarityMasking
from retrieve.layers.linr.similarity_masking_triton import SimilarityMaskingTriton
from retrieve.layers.utils.compact import compact_mask
from retrieve.layers.utils.retrieval import FullScanKNN
from tests.conftest import (
    assert_topk_id_sets_match,
    make_attrs,
    make_index,
    make_mask,
    make_query,
    recall_at_k,
)

N, D, B, K = 2048, 128, 16, 200


@pytest.fixture(scope="module")
def data():
    embs = make_index(N, D)
    query = make_query(B, D)
    return {"embs": embs, "query": query}


# ---------------------------------------------------------------------------
# SimilarityMasking: full matmul + topk (mask-based filtering only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [SimilarityMasking, SimilarityMaskingTriton])
class TestSimilarityMasking:
    def test_no_mask_returns_topk(self, data, cls):
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert (scores[:, :-1] >= scores[:, 1:]).all()

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_external_mask(self, data, cls, pass_rate):
        m = cls(k=K)
        m.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()


# ---------------------------------------------------------------------------
# PrefilterKNN: pre-filter via candidate_ids + scoped topk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [PrefilterKNN, PrefilterKNNTriton])
class TestPrefilterKNN:
    def test_no_candidates_full_path(self, data, cls):
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert (scores[:, :-1] >= scores[:, 1:]).all()

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_candidate_ids_returns_passing(self, data, cls, pass_rate):
        m = cls(k=K)
        m.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        cand, counts = compact_mask(mask)
        ids, scores = m(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()

    def test_candidate_ids_default_counts(self, data, cls):
        """When all rows have exactly P valid candidates, counts is optional."""
        m = cls(k=8)
        m.register_index(data["embs"])
        g = torch.Generator(device="cuda").manual_seed(123)
        cand = torch.randint(0, N, (B, 32), generator=g, device="cuda")
        ids, scores = m(data["query"], candidate_ids=cand)
        for b in range(B):
            allowed = set(cand[b].tolist())
            for j in range(8):
                if ids[b, j].item() >= 0:
                    assert ids[b, j].item() in allowed


# ---------------------------------------------------------------------------
# Cross-backend agreement: torch and triton classes return the same top-K ids.
# ---------------------------------------------------------------------------


class TestCrossBackendAgreement:
    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_prefilter_torch_matches_prefilter_triton(self, data, pass_rate):
        ref = PrefilterKNN(k=K)
        ref.register_index(data["embs"])
        tri = PrefilterKNNTriton(k=K)
        tri.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        cand, counts = compact_mask(mask)
        ids_ref, sc_ref = ref(data["query"], candidate_ids=cand, counts=counts)
        ids_tri, sc_tri = tri(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            assert_topk_id_sets_match(ids_tri, sc_tri, ids_ref, sc_ref, b)

    @pytest.mark.parametrize("mask_pass_rate", [None, 0.01, 0.1, 0.8])
    def test_one_bit_torch_matches_one_bit_triton(self, data, mask_pass_rate):
        """OneBitKNN torch and Triton must produce identical bits → identical top-K."""
        ref = OneBitKNN(k=K, seed=11)
        ref.register_index(data["embs"])
        tri = OneBitKNNTriton(k=K, seed=11)
        tri.register_index(data["embs"])
        mask = None if mask_pass_rate is None else make_mask(B, N, pass_rate=mask_pass_rate)
        ids_ref, sc_ref = ref(data["query"], mask=mask)
        ids_tri, sc_tri = tri(data["query"], mask=mask)
        for b in range(B):
            assert_topk_id_sets_match(ids_tri, sc_tri, ids_ref, sc_ref, b)

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_similarity_masking_matches_prefilter_topk_set(self, data, pass_rate):
        """Mask-based path and candidate_ids path on the same passing set → same top-K."""
        sm = SimilarityMasking(k=K)
        sm.register_index(data["embs"])
        pf = PrefilterKNN(k=K)
        pf.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        cand, counts = compact_mask(mask)
        ids1, sc1 = sm(data["query"], mask=mask)
        ids2, sc2 = pf(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            assert_topk_id_sets_match(ids2, sc2, ids1, sc1, b)


# ---------------------------------------------------------------------------
# OneBitKNN: 1-bit Sign-OPORP scoring.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [OneBitKNN, OneBitKNNTriton])
class TestOneBitKNN:
    def test_full_scan_topk_recall_against_exact(self, data, cls):
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, _ = m(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recall = recall_at_k(ids, ex_ids)
        assert recall >= 0.4, f"OneBitKNN recall@{K} = {recall:.3f}"

    def test_candidate_ids_subset(self, data, cls):
        m = cls(k=8)
        m.register_index(data["embs"])
        g = torch.Generator(device="cuda").manual_seed(7)
        cand = torch.randint(0, N, (B, 32), generator=g, device="cuda")
        ids, _ = m(data["query"], candidate_ids=cand)
        assert ids.shape == (B, 8)
        for b in range(B):
            allowed = set(cand[b].tolist())
            for j in range(8):
                if ids[b, j].item() >= 0:
                    assert ids[b, j].item() in allowed


# ---------------------------------------------------------------------------
# Decoupled clause filter: caller composes ExactAttributeFilter with retriever.
# ---------------------------------------------------------------------------


class TestClauseDecoupledComposition:
    def _setup(self):
        attrs = make_attrs(N, c=2, a_max=2, n_vocab=20)
        q_attrs = torch.full((B, 2), 5, dtype=torch.long, device="cuda")
        extra_mask = make_mask(B, N, pass_rate=0.3, seed=42)
        f = ExactAttributeFilter()
        f.register_index(attrs)
        return f, q_attrs, extra_mask

    @pytest.mark.parametrize("cls", [SimilarityMasking, SimilarityMaskingTriton])
    def test_similarity_masking_with_clause_and_external_mask(self, data, cls):
        f, q_attrs, extra = self._setup()
        mask = f.evaluate_mask(q_attrs) & extra
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()

    @pytest.mark.parametrize("cls", [PrefilterKNN, PrefilterKNNTriton])
    def test_prefilter_with_clause_via_evaluate_indices(self, data, cls):
        f, q_attrs, extra = self._setup()
        # When combining with an external mask, just AND the two masks first
        # then compact (callers compose explicitly).
        combined = f.evaluate_mask(q_attrs) & extra
        cand, counts = compact_mask(combined)
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, scores = m(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert combined[b, ids[b][valid]].all()

    @pytest.mark.parametrize("cls", [PrefilterKNN, PrefilterKNNTriton])
    def test_prefilter_with_evaluate_indices_kernel_path(self, data, cls):
        """Direct fused-kernel path: ExactAttributeFilter.evaluate_indices → PrefilterKNN."""
        f, q_attrs, _ = self._setup()
        cand, counts = f.evaluate_indices(q_attrs)
        # Cross-check against the dense path.
        expected_mask = f.evaluate_mask(q_attrs)
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, scores = m(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert expected_mask[b, ids[b][valid]].all()


# ---------------------------------------------------------------------------
# Edge cases — boundary conditions every retrieval module should handle.
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize(
        "cls",
        [SimilarityMasking, SimilarityMaskingTriton, OneBitKNN, OneBitKNNTriton],
    )
    def test_mask_all_true_equals_unmasked(self, data, cls):
        """All-True mask path returns the same top-K id set as the unmasked path."""
        m = cls(k=K)
        m.register_index(data["embs"])
        ids_no_mask, sc_no_mask = m(data["query"])
        all_true = torch.ones(B, N, dtype=torch.bool, device="cuda")
        ids_masked, sc_masked = m(data["query"], mask=all_true)
        for b in range(B):
            assert_topk_id_sets_match(ids_masked, sc_masked, ids_no_mask, sc_no_mask, b)

    @pytest.mark.parametrize(
        "cls",
        [SimilarityMasking, SimilarityMaskingTriton, OneBitKNN, OneBitKNNTriton],
    )
    def test_mask_all_false_returns_no_finite_scores(self, data, cls):
        """All-False mask → every score is -inf; no valid (finite-score) result."""
        m = cls(k=K)
        m.register_index(data["embs"])
        all_false = torch.zeros(B, N, dtype=torch.bool, device="cuda")
        _, scores = m(data["query"], mask=all_false)
        assert not torch.isfinite(scores).any()

    @pytest.mark.parametrize("cls", [PrefilterKNN, PrefilterKNNTriton])
    def test_prefilter_candidate_ids_p_zero(self, data, cls):
        """``candidate_ids`` with shape [B, 0] → all-padding return."""
        m = cls(k=K)
        m.register_index(data["embs"])
        empty = torch.empty(B, 0, dtype=torch.long, device="cuda")
        ids, scores = m(data["query"], candidate_ids=empty)
        assert ids.shape == (B, K)
        assert (ids == -1).all()
        assert not torch.isfinite(scores).any()

    @pytest.mark.parametrize("cls", [SimilarityMasking, SimilarityMaskingTriton])
    def test_b_one(self, data, cls):
        """Single-query batch — Triton tile-parallel path masks padded rows."""
        m = cls(k=K)
        m.register_index(data["embs"])
        q = data["query"][:1]
        ids, scores = m(q)
        assert ids.shape == (1, K)
        assert torch.isfinite(scores).all()
        # Returned ids must match a torch reference top-K. Tensor-core matmul
        # in the Triton path differs from torch ``@`` in fp accumulator order,
        # so allow boundary-tied ids to swap (scores within atol of the K-th).
        ref_full = q @ data["embs"].t()
        ref_scores, ref_ids = torch.topk(ref_full, K, dim=1)
        assert_topk_id_sets_match(ids, scores, ref_ids, ref_scores, 0)

    @pytest.mark.parametrize("cls", [SimilarityMasking, SimilarityMaskingTriton])
    def test_k_equals_n(self, data, cls):
        """K=N — every item returned, no padding, scores still descending."""
        m = cls(k=N)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, N)
        # Each row is a permutation of [0, N).
        for b in range(B):
            assert sorted(ids[b].tolist()) == list(range(N))
        # Scores descending.
        assert (scores[:, :-1] >= scores[:, 1:]).all()
