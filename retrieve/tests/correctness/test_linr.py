"""LiNR correctness — PostfilterKNN / PrefilterKNN / OneBitKNN in both
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
from retrieve.layers.linr.postfilter_knn import PostfilterKNN
from retrieve.layers.linr.postfilter_knn_int8 import PostfilterKNNInt8
from retrieve.layers.linr.prefilter_knn import PrefilterKNN
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

BACKENDS = ["torch", "triton"]


@pytest.fixture(scope="module")
def data():
    embs = make_index(N, D)
    query = make_query(B, D)
    return {"embs": embs, "query": query}


# ---------------------------------------------------------------------------
# PostfilterKNN: full matmul + topk (mask-based filtering only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
class TestPostfilterKNN:
    def test_no_mask_returns_topk(self, data, backend):
        m = PostfilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert (scores[:, :-1] >= scores[:, 1:]).all()

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_external_mask(self, data, backend, pass_rate):
        m = PostfilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()


# ---------------------------------------------------------------------------
# PrefilterKNN: pre-filter via candidate_ids + scoped topk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
class TestPrefilterKNN:
    def test_no_candidates_full_path(self, data, backend):
        m = PrefilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert (scores[:, :-1] >= scores[:, 1:]).all()

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_candidate_ids_returns_passing(self, data, backend, pass_rate):
        m = PrefilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        cand, counts = compact_mask(mask)
        ids, scores = m(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()

    def test_candidate_ids_default_counts(self, data, backend):
        """When all rows have exactly P valid candidates, counts is optional."""
        m = PrefilterKNN(k=8, backend=backend)
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
# Cross-backend agreement: torch and triton backends return the same top-K ids.
# ---------------------------------------------------------------------------


class TestCrossBackendAgreement:
    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_prefilter_torch_matches_prefilter_triton(self, data, pass_rate):
        ref = PrefilterKNN(k=K, backend="torch")
        ref.register_index(data["embs"])
        tri = PrefilterKNN(k=K, backend="triton")
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
        ref = OneBitKNN(k=K, seed=11, backend="torch")
        ref.register_index(data["embs"])
        tri = OneBitKNN(k=K, seed=11, backend="triton")
        tri.register_index(data["embs"])
        if mask_pass_rate is None:
            ids_ref, sc_ref = ref(data["query"])
            ids_tri, sc_tri = tri(data["query"])
        else:
            mask = make_mask(B, N, pass_rate=mask_pass_rate)
            cand, counts = compact_mask(mask)
            ids_ref, sc_ref = ref(data["query"], candidate_ids=cand, counts=counts)
            ids_tri, sc_tri = tri(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            assert_topk_id_sets_match(ids_tri, sc_tri, ids_ref, sc_ref, b)

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_postfilter_knn_matches_prefilter_topk_set(self, data, pass_rate):
        """Mask-based path and candidate_ids path on the same passing set → same top-K."""
        sm = PostfilterKNN(k=K)
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


@pytest.mark.parametrize("backend", BACKENDS)
class TestOneBitKNN:
    def test_full_scan_topk_recall_against_exact(self, data, backend):
        m = OneBitKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, _ = m(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recall = recall_at_k(ids, ex_ids)
        assert recall >= 0.4, f"OneBitKNN recall@{K} = {recall:.3f}"

    def test_candidate_ids_subset(self, data, backend):
        m = OneBitKNN(k=8, backend=backend)
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

    def test_torch_compile_fullgraph_no_break(self, data, backend):
        """``torch.compile(fullgraph=True)`` must not graph-break on either
        path. The triton path goes through the
        ``retrieve::oporp_1bit_match_topk_full`` / ``_indirect`` custom_ops
        (opaque to Dynamo); the torch path is all native ops."""
        m = OneBitKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        compiled = torch.compile(m, fullgraph=True, dynamic=True)

        # Full-scan path.
        ids_eager, sc_eager = m(data["query"])
        ids_comp, sc_comp = compiled(data["query"])
        torch.testing.assert_close(ids_eager, ids_comp)
        torch.testing.assert_close(sc_eager, sc_comp)

        # Candidates path (mirrors the LinrV3 cascade call shape).
        g = torch.Generator(device="cuda").manual_seed(11)
        cand = torch.randint(0, N, (B, 64), generator=g, device="cuda")
        counts = torch.full((B,), 64, dtype=torch.long, device="cuda")
        ids_eager, sc_eager = m(data["query"], candidate_ids=cand, counts=counts)
        ids_comp, sc_comp = compiled(data["query"], candidate_ids=cand, counts=counts)
        torch.testing.assert_close(ids_eager, ids_comp)
        torch.testing.assert_close(sc_eager, sc_comp)


# ---------------------------------------------------------------------------
# PostfilterKNNInt8: per-item symmetric int8 + per-query symmetric int8
# + cuBLAS int8 GEMM (paper-faithful). Single-stage analog of
# PostfilterKNN — full-scan dense scoring + optional mask + topk.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
class TestPostfilterKNNInt8:
    def test_no_mask_returns_topk(self, data, backend):
        m = PostfilterKNNInt8(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert (scores[:, :-1] >= scores[:, 1:]).all()

    def test_full_scan_topk_recall_against_exact(self, data, backend):
        m = PostfilterKNNInt8(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, _ = m(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recall = recall_at_k(ids, ex_ids)
        # Dual int8 (query + items) on unit-norm D=128 data: ≥0.95 typical
        # (paper notes the dual-int8 path "cannot reach 0.95 recall" at
        # production scale, but on random data the noise floor is lower).
        assert recall >= 0.95, f"PostfilterKNNInt8 recall@{K} = {recall:.3f}"

    @pytest.mark.parametrize("pass_rate", [0.01, 0.1, 0.8])
    def test_external_mask(self, data, backend, pass_rate):
        m = PostfilterKNNInt8(k=K, backend=backend)
        m.register_index(data["embs"])
        mask = make_mask(B, N, pass_rate=pass_rate)
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()


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

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_postfilter_knn_with_clause_and_external_mask(self, data, backend):
        f, q_attrs, extra = self._setup()
        mask = f.evaluate_mask(q_attrs) & extra
        m = PostfilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, scores = m(data["query"], mask=mask)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert mask[b, ids[b][valid]].all()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_prefilter_with_clause_via_evaluate_indices(self, data, backend):
        f, q_attrs, extra = self._setup()
        # When combining with an external mask, just AND the two masks first
        # then compact (callers compose explicitly).
        combined = f.evaluate_mask(q_attrs) & extra
        cand, counts = compact_mask(combined)
        m = PrefilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, scores = m(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert combined[b, ids[b][valid]].all()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_prefilter_with_evaluate_indices_kernel_path(self, data, backend):
        """Direct fused-kernel path: ExactAttributeFilter.evaluate_indices → PrefilterKNN."""
        f, q_attrs, _ = self._setup()
        cand, counts = f.evaluate_indices(q_attrs)
        # Cross-check against the dense path.
        expected_mask = f.evaluate_mask(q_attrs)
        m = PrefilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, scores = m(data["query"], candidate_ids=cand, counts=counts)
        for b in range(B):
            valid = torch.isfinite(scores[b]) & (ids[b] >= 0)
            assert expected_mask[b, ids[b][valid]].all()


# ---------------------------------------------------------------------------
# Edge cases — boundary conditions every retrieval module should handle.
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("cls", [PostfilterKNN, PostfilterKNNInt8])
    def test_mask_all_true_equals_unmasked(self, data, cls, backend):
        """All-True mask path returns the same top-K id set as the unmasked path."""
        m = cls(k=K, backend=backend)
        m.register_index(data["embs"])
        ids_no_mask, sc_no_mask = m(data["query"])
        all_true = torch.ones(B, N, dtype=torch.bool, device="cuda")
        ids_masked, sc_masked = m(data["query"], mask=all_true)
        for b in range(B):
            assert_topk_id_sets_match(ids_masked, sc_masked, ids_no_mask, sc_no_mask, b)

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("cls", [PostfilterKNN, PostfilterKNNInt8])
    def test_mask_all_false_returns_no_finite_scores(self, data, cls, backend):
        """All-False mask → every slot is padded (``id == -1``)."""
        m = cls(k=K, backend=backend)
        m.register_index(data["embs"])
        all_false = torch.zeros(B, N, dtype=torch.bool, device="cuda")
        ids, _ = m(data["query"], mask=all_false)
        # Sentinel is ``id == -1``; ``PostfilterKNNInt8`` returns int32
        # scores so the previous ``torch.isfinite(scores)`` check would be
        # vacuously True for it.
        assert (ids == -1).all()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_one_bit_knn_all_candidates_equals_unmasked(self, data, backend):
        """OneBitKNN with candidate_ids = arange(N) per row matches full-scan."""
        m = OneBitKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        ids_full, sc_full = m(data["query"])
        all_cand = torch.arange(N, device="cuda").unsqueeze(0).expand(B, N).contiguous()
        ids_cand, sc_cand = m(data["query"], candidate_ids=all_cand)
        for b in range(B):
            assert_topk_id_sets_match(ids_cand, sc_cand, ids_full, sc_full, b)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_one_bit_knn_zero_counts_returns_sentinels(self, data, backend):
        """OneBitKNN with counts=0 per row → every slot is padded (``id == -1``)."""
        m = OneBitKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        cand = torch.zeros(B, 8, dtype=torch.long, device="cuda")
        counts = torch.zeros(B, dtype=torch.long, device="cuda")
        ids, _ = m(data["query"], candidate_ids=cand, counts=counts)
        assert (ids == -1).all()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_prefilter_candidate_ids_p_zero(self, data, backend):
        """``candidate_ids`` with shape [B, 0] → all-padding return."""
        m = PrefilterKNN(k=K, backend=backend)
        m.register_index(data["embs"])
        empty = torch.empty(B, 0, dtype=torch.long, device="cuda")
        ids, scores = m(data["query"], candidate_ids=empty)
        assert ids.shape == (B, K)
        assert (ids == -1).all()
        assert not torch.isfinite(scores).any()

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("small_b", [1, 8, 16])
    def test_postfilter_knn_int8_small_batch_padding(self, data, backend, small_b):
        """``torch._int_mm`` requires M >= 17; the layer pads small batches
        with zero rows and slices back. Verify the padded path returns
        sensible top-K (high recall vs the exact fp32 baseline) — can't
        compare against another ``PostfilterKNNInt8`` call because
        the global query scale depends on batch contents and would
        change between calls."""
        m = PostfilterKNNInt8(k=K, backend=backend)
        m.register_index(data["embs"])
        ids, _ = m(data["query"][:small_b])
        assert ids.shape == (small_b, K)
        assert (ids >= 0).all()

        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"][:small_b])
        recall = recall_at_k(ids, ex_ids)
        assert recall >= 0.95, f"small-batch padding recall@{K} = {recall:.3f}"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_b_one(self, data, backend):
        """Single-query batch — Triton tile-parallel path masks padded rows."""
        m = PostfilterKNN(k=K, backend=backend)
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

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_k_equals_n(self, data, backend):
        """K=N — every item returned, no padding, scores still descending."""
        m = PostfilterKNN(k=N, backend=backend)
        m.register_index(data["embs"])
        ids, scores = m(data["query"])
        assert ids.shape == (B, N)
        # Each row is a permutation of [0, N).
        for b in range(B):
            assert sorted(ids[b].tolist()) == list(range(N))
        # Scores descending.
        assert (scores[:, :-1] >= scores[:, 1:]).all()
