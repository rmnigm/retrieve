"""LiNR correctness — V1, V2, V3 in both torch and Triton backends.

Fixture sizes mirror the LinR paper's small evaluation slice (D=128,
batch=16, K=200) at a smaller N so the suite stays interactive on a single
GPU. Latency / recall-vs-N benchmarks live in ``evaluation/``.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.layers.filters import ClauseIndex
from retrieve.layers.linr.builder import build_linr_index
from retrieve.layers.linr.v1 import LiNR_V1
from retrieve.layers.linr.v1_triton import LiNR_V1_Triton
from retrieve.layers.linr.v2 import LiNR_V2
from retrieve.layers.linr.v2_triton import LiNR_V2_Triton
from retrieve.layers.linr.v3 import LiNR_V3
from retrieve.layers.linr.v3_triton import LiNR_V3_Triton
from retrieve.layers.utils.compact import compact_mask
from retrieve.layers.utils.retrieval import FullScanKNN
from tests.conftest import (
    make_attrs,
    make_index,
    make_mask,
    make_query,
    recall_at_k,
    valid_id_set,
)

N, D, B, K = 2048, 128, 16, 200


@pytest.fixture(scope="module")
def data():
    embs = make_index(N, D)
    query = make_query(B, D)
    return {"embs": embs, "query": query}


# ---------------------------------------------------------------------------
# V1: full matmul + topk (mask-based filtering only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [LiNR_V1, LiNR_V1_Triton])
class TestLiNR_V1:
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
# V2: pre-filter via candidate_ids + scoped topk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [LiNR_V2, LiNR_V2_Triton])
class TestLiNR_V2:
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


def _valid_scores_sorted(scores):
    """Per-row finite scores, sorted descending; rows zero-padded to common length."""
    finite = torch.isfinite(scores)
    counts = finite.sum(dim=1)
    p = int(counts.max().item())
    masked = scores.masked_fill(~finite, float("-inf"))
    sorted_scores, _ = torch.topk(masked, p, dim=1)
    return sorted_scores, counts


class TestCrossBackendAgreement:
    @pytest.mark.parametrize("pass_rate", [None, 0.01, 0.1, 0.8])
    def test_v1_torch_matches_v1_triton(self, data, pass_rate):
        ref = LiNR_V1(k=K)
        ref.register_index(data["embs"])
        tri = LiNR_V1_Triton(k=K)
        tri.register_index(data["embs"])
        mask = None if pass_rate is None else make_mask(B, N, pass_rate=pass_rate)
        ids_ref, sc_ref = ref(data["query"], mask=mask)
        ids_tri, sc_tri = tri(data["query"], mask=mask)
        for b in range(B):
            assert valid_id_set(ids_ref, sc_ref, b) == valid_id_set(ids_tri, sc_tri, b)
        sc_ref_sorted, ref_counts = _valid_scores_sorted(sc_ref)
        sc_tri_sorted, tri_counts = _valid_scores_sorted(sc_tri)
        assert torch.equal(ref_counts, tri_counts)
        for b in range(B):
            n_valid = int(ref_counts[b].item())
            assert torch.allclose(
                sc_ref_sorted[b, :n_valid],
                sc_tri_sorted[b, :n_valid],
                atol=1e-3,
                rtol=1e-3,
            )

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_v2_torch_matches_v2_triton(self, data, pass_rate):
        ref = LiNR_V2(k=K)
        ref.register_index(data["embs"])
        tri = LiNR_V2_Triton(k=K)
        tri.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        cand, counts = compact_mask(mask)
        ids_ref, sc_ref = ref(data["query"], candidate_ids=cand, counts=counts)
        ids_tri, sc_tri = tri(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            assert valid_id_set(ids_ref, sc_ref, b) == valid_id_set(ids_tri, sc_tri, b)

    @pytest.mark.parametrize("mask_pass_rate", [None, 0.01, 0.1, 0.8])
    def test_v3_torch_matches_v3_triton(self, data, mask_pass_rate):
        """V3 torch and Triton must produce identical bits → identical top-K."""
        ref = LiNR_V3(k=K, seed=11)
        ref.register_index(data["embs"])
        tri = LiNR_V3_Triton(k=K, seed=11)
        tri.register_index(data["embs"])
        mask = None if mask_pass_rate is None else make_mask(B, N, pass_rate=mask_pass_rate)
        ids_ref, sc_ref = ref(data["query"], mask=mask)
        ids_tri, sc_tri = tri(data["query"], mask=mask)
        for b in range(B):
            assert valid_id_set(ids_ref, sc_ref, b) == valid_id_set(ids_tri, sc_tri, b)

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_v1_matches_v2_topk_set(self, data, pass_rate):
        """V1 (mask) and V2 (candidate_ids of same passing set) return the same top-K."""
        v1 = LiNR_V1(k=K)
        v1.register_index(data["embs"])
        v2 = LiNR_V2(k=K)
        v2.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        cand, counts = compact_mask(mask)
        ids1, sc1 = v1(data["query"], mask=mask)
        ids2, sc2 = v2(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            assert valid_id_set(ids1, sc1, b) == valid_id_set(ids2, sc2, b)


# ---------------------------------------------------------------------------
# V3: 1-bit Sign-OPORP scoring.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [LiNR_V3, LiNR_V3_Triton])
class TestLiNR_V3:
    def test_full_scan_topk_recall_against_exact(self, data, cls):
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, _ = m(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recall = recall_at_k(ids, ex_ids)
        assert recall >= 0.4, f"V3 recall@{K} = {recall:.3f}"

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
# Builder
# ---------------------------------------------------------------------------


class TestBuilder:
    @pytest.mark.parametrize("version", [1, 2, 3])
    @pytest.mark.parametrize("backend", ["torch", "triton"])
    def test_builds(self, data, version, backend):
        m = build_linr_index(version, data["embs"], K, backend=backend)
        ids, _ = m(data["query"])
        assert ids.shape == (B, K)

    def test_invalid_version(self, data):
        with pytest.raises(ValueError, match="Unknown LiNR version"):
            build_linr_index(4, data["embs"], K)

    def test_invalid_backend(self, data):
        with pytest.raises(ValueError, match="Unknown backend"):
            build_linr_index(1, data["embs"], K, backend="cuda")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Decoupled clause filter: caller composes ClauseIndex with retriever.
# ---------------------------------------------------------------------------


class TestClauseDecoupledComposition:
    def _setup(self):
        attrs = make_attrs(N, c=2, a_max=2, n_vocab=20)
        q_attrs = torch.full((B, 2), 5, dtype=torch.long, device="cuda")
        extra_mask = make_mask(B, N, pass_rate=0.3, seed=42)
        ci = ClauseIndex()
        ci.register_index(attrs)
        return ci, q_attrs, extra_mask

    @pytest.mark.parametrize("cls", [LiNR_V1, LiNR_V1_Triton])
    def test_v1_with_clause_and_external_mask(self, data, cls):
        ci, q_attrs, extra = self._setup()
        mask = ci.evaluate_mask(q_attrs) & extra
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()

    @pytest.mark.parametrize("cls", [LiNR_V2, LiNR_V2_Triton])
    def test_v2_with_clause_via_evaluate_indices(self, data, cls):
        ci, q_attrs, extra = self._setup()
        # When combining with an external mask, just AND the two masks first
        # then compact (callers compose explicitly).
        combined = ci.evaluate_mask(q_attrs) & extra
        cand, counts = compact_mask(combined)
        m = cls(k=K)
        m.register_index(data["embs"])
        ids, scores = m(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert combined[b, ids[b][valid]].all()

    @pytest.mark.parametrize("cls", [LiNR_V2, LiNR_V2_Triton])
    def test_v2_with_evaluate_indices_kernel_path(self, data, cls):
        """Direct fused-kernel path: ClauseIndex.evaluate_indices → V2."""
        ci, q_attrs, _ = self._setup()
        cand, counts = ci.evaluate_indices(q_attrs)
        # Cross-check against the dense path.
        expected_mask = ci.evaluate_mask(q_attrs)
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
    @pytest.mark.parametrize("cls", [LiNR_V1, LiNR_V1_Triton, LiNR_V3, LiNR_V3_Triton])
    def test_mask_all_true_equals_unmasked(self, data, cls):
        """All-True mask path returns the same top-K id set as the unmasked path."""
        m = cls(k=K)
        m.register_index(data["embs"])
        ids_no_mask, _ = m(data["query"])
        all_true = torch.ones(B, N, dtype=torch.bool, device="cuda")
        ids_masked, _ = m(data["query"], mask=all_true)
        for b in range(B):
            assert set(ids_no_mask[b].tolist()) == set(ids_masked[b].tolist())

    @pytest.mark.parametrize("cls", [LiNR_V1, LiNR_V1_Triton, LiNR_V3, LiNR_V3_Triton])
    def test_mask_all_false_returns_no_finite_scores(self, data, cls):
        """All-False mask → every score is -inf; no valid (finite-score) result."""
        m = cls(k=K)
        m.register_index(data["embs"])
        all_false = torch.zeros(B, N, dtype=torch.bool, device="cuda")
        _, scores = m(data["query"], mask=all_false)
        assert not torch.isfinite(scores).any()

    @pytest.mark.parametrize("cls", [LiNR_V2, LiNR_V2_Triton])
    def test_v2_candidate_ids_p_zero(self, data, cls):
        """``candidate_ids`` with shape [B, 0] → all-padding return."""
        m = cls(k=K)
        m.register_index(data["embs"])
        empty = torch.empty(B, 0, dtype=torch.long, device="cuda")
        ids, scores = m(data["query"], candidate_ids=empty)
        assert ids.shape == (B, K)
        assert (ids == -1).all()
        assert not torch.isfinite(scores).any()

    @pytest.mark.parametrize("cls", [LiNR_V1, LiNR_V1_Triton])
    def test_b_one(self, data, cls):
        """Single-query batch — Triton tile-parallel path masks padded rows."""
        m = cls(k=K)
        m.register_index(data["embs"])
        q = data["query"][:1]
        ids, scores = m(q)
        assert ids.shape == (1, K)
        assert torch.isfinite(scores).all()
        # Returned ids must match a torch reference top-K (set equality, fp32 ties).
        ref_scores = q @ data["embs"].t()
        _, ref_ids = torch.topk(ref_scores, K, dim=1)
        assert set(ids[0].tolist()) == set(ref_ids[0].tolist())

    @pytest.mark.parametrize("cls", [LiNR_V1, LiNR_V1_Triton])
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
