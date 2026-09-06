"""SilverTorch correctness — IVF + INT8 ANN with optional Bloom / exact-clause filter.

Fixtures shrink the SilverTorch paper's eval (D=128, K=2048, n_probe=64) to
sizes that fit well on a single GPU; larger sweeps live in ``evaluation/``.
Tests cover three filter modes — ``"none"`` (plain IVF), ``"bloom"`` (paper's
bloom subset test), ``"exact"`` (clause-attribute predicate fused into the
codesigned kernel). All forward paths are exercised with ``backend="torch"``,
``backend="triton"``, ``backend="cuda"``, ``backend="cute"`` and
``backend="official"`` across all three filter modes; cuda cells skip only when the
C++ extension can't build here, cute cells only when the CuTe DSL (the ``cute``
extra) is not installed, official cells only when ``meta-recsys/silvertorch`` (the
``official`` extra) is not installed. On ``"official"`` the bloom mode is Meta's own
bloom index (a different hash), so the bloom cross-backend checks stay
triton-vs-torch; the official parity gates live in ``tests/parity/test_official.py``.
"""

from __future__ import annotations

import copy

import pytest
import torch

from retrieve.layers.silvertorch import SilverTorch, build_silvertorch
from retrieve.layers.utils.retrieval import FullScanKNN
from tests.conftest import (
    assert_recall_monotone,
    assert_topk_id_sets_match,
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
    recall_at_k,
    require_cps_cuda,
    require_cps_cute,
    require_official,
)
from tests.parity.conftest import assert_ids_equal_up_to_ties

N, D, B, K = 4096, 128, 16, 64
N_LISTS, N_PROBE = 64, 8
M_BITS, K_HASH = 512, 5
C, A_MAX = 2, 2

BACKENDS = ["torch", "triton", "cuda", "cute", "official"]


def _require_backend(backend: str) -> None:
    """Skip-or-fail gate for the three optional backends; a no-op for torch / triton."""
    if backend == "cuda":
        require_cps_cuda()
    elif backend == "cute":
        require_cps_cute()
    elif backend == "official":
        require_official()


@pytest.fixture(scope="module")
def data():
    embs = make_index(N, D)
    query = make_query(B, D)
    attrs = make_attrs(N, c=C, a_max=A_MAX)
    q_attrs = make_query_attrs(B, c=C)
    return {"embs": embs, "query": query, "attrs": attrs, "q_attrs": q_attrs}


def _build(with_attrs: bool, data, backend: str = "triton", **overrides):
    _require_backend(backend)
    kw = {
        "k": K,
        "n_lists": N_LISTS,
        "n_probe": N_PROBE,
        "filter_mode": "bloom",
        "m_bits": M_BITS,
        "k_hash": K_HASH,
        "n_iter": 3,
        "backend": backend,
    }
    kw.update(overrides)
    m = SilverTorch(**kw)
    m.register_index(data["embs"], data["attrs"] if with_attrs else None)
    return m


def _build_no_bloom(data, backend: str = "triton", **overrides):
    _require_backend(backend)
    kw = {"k": K, "n_lists": N_LISTS, "n_probe": N_PROBE, "n_iter": 3, "backend": backend}
    kw.update(overrides)
    m = SilverTorch(**kw)
    m.register_index(data["embs"])
    return m


def _build_exact(data, backend: str = "triton", clause_is_reverse=None, **overrides):
    _require_backend(backend)
    kw = {
        "k": K,
        "n_lists": N_LISTS,
        "n_probe": N_PROBE,
        "filter_mode": "exact",
        "n_iter": 3,
        "backend": backend,
    }
    kw.update(overrides)
    m = SilverTorch(**kw)
    m.register_index(data["embs"], data["attrs"], clause_is_reverse=clause_is_reverse)
    return m


@pytest.mark.parametrize("backend", BACKENDS)
class TestShape:
    def test_with_attrs(self, data, backend):
        m = _build(with_attrs=True, data=data, backend=backend)
        ids, scores = m(data["query"], data["q_attrs"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert ids.dtype == torch.long
        assert scores.dtype == torch.float32

    def test_without_attrs_bloom_configured(self, data, backend):
        m = _build(with_attrs=False, data=data, backend=backend)
        ids, _ = m(data["query"])
        assert ids.shape == (B, K)

    def test_no_bloom(self, data, backend):
        m = _build_no_bloom(data, backend=backend)
        ids, scores = m(data["query"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert ids.dtype == torch.long
        assert scores.dtype == torch.float32

    def test_exact_with_attrs(self, data, backend):
        m = _build_exact(data, backend=backend)
        ids, scores = m(data["query"], data["q_attrs"])
        assert ids.shape == (B, K)
        assert scores.shape == (B, K)
        assert ids.dtype == torch.long
        assert scores.dtype == torch.float32

    def test_exact_without_attrs(self, data, backend):
        """``filter_mode='exact'`` with ``query_clause_attrs=None`` skips the
        filter branch and returns plain IVF results."""
        m = _build_exact(data, backend=backend)
        ids, _ = m(data["query"])
        assert ids.shape == (B, K)

    def test_buffer_dtypes(self, data, backend):
        m = _build_no_bloom(data, backend=backend)
        assert m.item_codes.dtype == torch.int8
        # Paper's per-tensor scale — single fp32 scalar, not a [N] gather.
        assert m.global_scale.dtype == torch.float32
        assert m.global_scale.shape == ()
        assert m.centroids.dtype == torch.float32
        if backend == "official":
            # Cluster-sorted CSR layout instead of the padded one (plan D4).
            assert not hasattr(m, "padded_cluster_items")
            assert m.cluster_offsets.shape == (N_LISTS + 1,)
            assert m.cluster_offsets.dtype == torch.int64
            assert m.sort_perm.shape == (N,) and m.inv_perm.shape == (N,)
            assert torch.equal(m.inv_perm[m.sort_perm], torch.arange(N, device="cuda"))
            assert int(m.cluster_offsets[-1].item()) == N
        else:
            assert m.padded_cluster_items.shape[0] == N_LISTS
        assert m.cluster_sizes.shape == (N_LISTS,)
        # No filter → no filter buffers.
        assert not hasattr(m, "bloom_sigs")
        assert not hasattr(m, "hash_seeds")
        assert not hasattr(m, "bloom_index")
        assert not hasattr(m, "item_clause_attrs")
        assert not hasattr(m, "clause_is_reverse")

    def test_buffer_dtypes_exact(self, data, backend):
        m = _build_exact(data, backend=backend)
        assert hasattr(m, "item_clause_attrs")
        assert hasattr(m, "clause_is_reverse")
        assert m.item_clause_attrs.dtype == torch.long
        assert m.item_clause_attrs.shape == (N, C, A_MAX)
        assert m.clause_is_reverse.dtype == torch.bool
        assert m.clause_is_reverse.shape == (C,)
        # Exact mode must not allocate bloom buffers.
        assert not hasattr(m, "bloom_sigs")
        assert not hasattr(m, "hash_seeds")
        assert not hasattr(m, "bloom_index")
        if backend == "official":
            # Attrs live in the cluster-sorted doc space on this backend.
            assert torch.equal(m.item_clause_attrs, data["attrs"][m.sort_perm])


class TestParamValidation:
    def test_invalid_bloom_params_rejected(self):
        with pytest.raises(ValueError, match="power of 2"):
            SilverTorch(k=5, n_lists=8, n_probe=4, filter_mode="bloom", m_bits=500, k_hash=5)
        with pytest.raises(ValueError, match="k_hash"):
            SilverTorch(k=5, n_lists=8, n_probe=4, filter_mode="bloom", m_bits=512, k_hash=0)

    def test_partial_bloom_config_rejected(self):
        with pytest.raises(ValueError, match="m_bits and k_hash"):
            SilverTorch(k=5, n_lists=8, n_probe=4, filter_mode="bloom", m_bits=512)
        with pytest.raises(ValueError, match="m_bits and k_hash"):
            SilverTorch(k=5, n_lists=8, n_probe=4, filter_mode="bloom", k_hash=4)

    def test_bloom_params_outside_bloom_filter_rejected(self):
        with pytest.raises(ValueError, match="only apply to filter_mode='bloom'"):
            SilverTorch(k=5, n_lists=8, n_probe=4, m_bits=512, k_hash=5)
        with pytest.raises(ValueError, match="only apply to filter_mode='bloom'"):
            SilverTorch(k=5, n_lists=8, n_probe=4, filter_mode="exact", m_bits=512, k_hash=5)

    def test_unknown_filter_rejected(self):
        with pytest.raises(ValueError, match="filter_mode must be"):
            SilverTorch(k=5, n_lists=8, n_probe=4, filter_mode="invalid")  # type: ignore[arg-type]

    def test_invalid_ivf_params_rejected(self, data):
        m = SilverTorch(k=5, n_lists=10_000, n_probe=4)
        with pytest.raises(ValueError, match="n_lists"):
            m.register_index(data["embs"])
        m2 = SilverTorch(k=5, n_lists=8, n_probe=16)
        with pytest.raises(ValueError, match="n_probe"):
            m2.register_index(data["embs"])

    def test_query_clause_attrs_without_filter_rejected(self, data):
        m = _build_no_bloom(data)
        qa = torch.zeros(B, 1, dtype=torch.long, device="cuda")
        with pytest.raises(ValueError, match="query_clause_attrs requires"):
            m(data["query"], query_clause_attrs=qa)

    def test_item_clause_attrs_without_filter_rejected(self, data):
        m = SilverTorch(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        attrs = make_attrs(N, c=1, a_max=1, n_vocab=10)
        with pytest.raises(ValueError, match="item_clause_attrs requires"):
            m.register_index(data["embs"], item_clause_attrs=attrs)

    def test_exact_requires_item_clause_attrs(self, data):
        m = SilverTorch(k=K, n_lists=N_LISTS, n_probe=N_PROBE, filter_mode="exact", n_iter=3)
        with pytest.raises(ValueError, match="requires item_clause_attrs"):
            m.register_index(data["embs"])

    def test_clause_is_reverse_rejected_outside_exact(self, data):
        rev = torch.zeros(C, dtype=torch.bool, device="cuda")
        m_bloom = SilverTorch(
            k=K,
            n_lists=N_LISTS,
            n_probe=N_PROBE,
            filter_mode="bloom",
            m_bits=M_BITS,
            k_hash=K_HASH,
            n_iter=3,
        )
        with pytest.raises(ValueError, match="clause_is_reverse"):
            m_bloom.register_index(data["embs"], data["attrs"], clause_is_reverse=rev)
        m_none = SilverTorch(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3)
        with pytest.raises(ValueError, match="clause_is_reverse"):
            m_none.register_index(data["embs"], clause_is_reverse=rev)


@pytest.mark.parametrize("backend", BACKENDS)
class TestEquivalence:
    def test_query_clause_attrs_none_equals_no_bloom(self, data, backend):
        """``query_clause_attrs=None`` on a bloom-configured module ≡ no-filter module.

        Same kmeans seed → identical centroids → identical scoring; filter
        branch is skipped at query time when qa is None regardless of mode.
        """
        st = _build(with_attrs=True, data=data, backend=backend)
        nb = _build_no_bloom(data, backend=backend)

        st_ids, st_scores = st(data["query"])
        nb_ids, nb_scores = nb(data["query"])

        for b in range(B):
            assert sorted(st_ids[b].tolist()) == sorted(nb_ids[b].tolist())
        st_sorted, _ = st_scores.sort(dim=1, descending=True)
        nb_sorted, _ = nb_scores.sort(dim=1, descending=True)
        assert torch.allclose(st_sorted, nb_sorted, atol=1e-3)

    def test_exact_query_clause_attrs_none_equals_no_filter(self, data, backend):
        """``query_clause_attrs=None`` on an exact-configured module ≡ no-filter."""
        ex = _build_exact(data, backend=backend)
        nb = _build_no_bloom(data, backend=backend)

        ex_ids, ex_scores = ex(data["query"])
        nb_ids, nb_scores = nb(data["query"])

        for b in range(B):
            assert sorted(ex_ids[b].tolist()) == sorted(nb_ids[b].tolist())
        ex_sorted, _ = ex_scores.sort(dim=1, descending=True)
        nb_sorted, _ = nb_scores.sort(dim=1, descending=True)
        assert torch.allclose(ex_sorted, nb_sorted, atol=1e-3)

    def test_exact_all_inactive_query_equals_no_filter(self, data, backend):
        """``query_clause_attrs == -1`` everywhere → clause predicate identically True
        → exact-mode results match the no-filter baseline."""
        ex = _build_exact(data, backend=backend)
        nb = _build_no_bloom(data, backend=backend)
        qa = torch.full((B, C), -1, dtype=torch.long, device="cuda")

        ex_ids, ex_scores = ex(data["query"], qa)
        nb_ids, nb_scores = nb(data["query"])

        for b in range(B):
            assert sorted(ex_ids[b].tolist()) == sorted(nb_ids[b].tolist())
        ex_sorted, _ = ex_scores.sort(dim=1, descending=True)
        nb_sorted, _ = nb_scores.sort(dim=1, descending=True)
        assert torch.allclose(ex_sorted, nb_sorted, atol=1e-3)


class TestCrossBackend:
    """torch and Triton SilverTorch agree on id sets (accumulator order may flip ties);
    cuda and Triton, and cute and Triton, agree on id sets through the whole layer (the
    kernel-level parity suites additionally prove the scores bit-identical on a shared
    probe family). cute and cuda register the same buffers and run the same two-kernel
    design, so through the whole layer they are compared bit for bit."""

    def test_with_bloom(self, data):
        tri = _build(with_attrs=True, data=data, backend="triton")
        trc = _build(with_attrs=True, data=data, backend="torch")
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_trc, sc_trc = trc(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_trc, sc_trc, ids_tri, sc_tri, b)

    def test_no_bloom(self, data):
        tri = _build_no_bloom(data, backend="triton")
        trc = _build_no_bloom(data, backend="torch")
        ids_tri, sc_tri = tri(data["query"])
        ids_trc, sc_trc = trc(data["query"])
        for b in range(B):
            assert_topk_id_sets_match(ids_trc, sc_trc, ids_tri, sc_tri, b)

    def test_cuda_with_bloom(self, data):
        tri = _build(with_attrs=True, data=data, backend="triton")
        cud = _build(with_attrs=True, data=data, backend="cuda")
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_cud, sc_cud = cud(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_cud, sc_cud, ids_tri, sc_tri, b)

    def test_cuda_no_bloom(self, data):
        tri = _build_no_bloom(data, backend="triton")
        cud = _build_no_bloom(data, backend="cuda")
        ids_tri, sc_tri = tri(data["query"])
        ids_cud, sc_cud = cud(data["query"])
        for b in range(B):
            assert_topk_id_sets_match(ids_cud, sc_cud, ids_tri, sc_tri, b)

    def test_with_exact(self, data):
        tri = _build_exact(data, backend="triton")
        trc = _build_exact(data, backend="torch")
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_trc, sc_trc = trc(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_trc, sc_trc, ids_tri, sc_tri, b)

    def test_with_exact_reverse(self, data):
        rev = torch.tensor([True, False], dtype=torch.bool, device="cuda")
        tri = _build_exact(data, backend="triton", clause_is_reverse=rev)
        trc = _build_exact(data, backend="torch", clause_is_reverse=rev)
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_trc, sc_trc = trc(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_trc, sc_trc, ids_tri, sc_tri, b)

    def test_cuda_with_exact(self, data):
        tri = _build_exact(data, backend="triton")
        cud = _build_exact(data, backend="cuda")
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_cud, sc_cud = cud(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_cud, sc_cud, ids_tri, sc_tri, b)

    def test_cuda_with_exact_reverse(self, data):
        rev = torch.tensor([True, False], dtype=torch.bool, device="cuda")
        tri = _build_exact(data, backend="triton", clause_is_reverse=rev)
        cud = _build_exact(data, backend="cuda", clause_is_reverse=rev)
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_cud, sc_cud = cud(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_cud, sc_cud, ids_tri, sc_tri, b)

    def test_cute_with_bloom(self, data):
        tri = _build(with_attrs=True, data=data, backend="triton")
        cut = _build(with_attrs=True, data=data, backend="cute")
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_cut, sc_cut = cut(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_cut, sc_cut, ids_tri, sc_tri, b)

    def test_cute_no_bloom(self, data):
        tri = _build_no_bloom(data, backend="triton")
        cut = _build_no_bloom(data, backend="cute")
        ids_tri, sc_tri = tri(data["query"])
        ids_cut, sc_cut = cut(data["query"])
        for b in range(B):
            assert_topk_id_sets_match(ids_cut, sc_cut, ids_tri, sc_tri, b)

    def test_cute_with_exact(self, data):
        tri = _build_exact(data, backend="triton")
        cut = _build_exact(data, backend="cute")
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_cut, sc_cut = cut(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_cut, sc_cut, ids_tri, sc_tri, b)

    def test_cute_with_exact_reverse(self, data):
        rev = torch.tensor([True, False], dtype=torch.bool, device="cuda")
        tri = _build_exact(data, backend="triton", clause_is_reverse=rev)
        cut = _build_exact(data, backend="cute", clause_is_reverse=rev)
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_cut, sc_cut = cut(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_cut, sc_cut, ids_tri, sc_tri, b)

    def test_official_no_bloom(self, data):
        tri = _build_no_bloom(data, backend="triton")
        off = _build_no_bloom(data, backend="official")
        ids_tri, sc_tri = tri(data["query"])
        ids_off, sc_off = off(data["query"])
        for b in range(B):
            assert_topk_id_sets_match(ids_off, sc_off, ids_tri, sc_tri, b)

    def test_official_with_exact(self, data):
        tri = _build_exact(data, backend="triton")
        off = _build_exact(data, backend="official")
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_off, sc_off = off(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_off, sc_off, ids_tri, sc_tri, b)

    def test_official_with_exact_reverse(self, data):
        rev = torch.tensor([True, False], dtype=torch.bool, device="cuda")
        tri = _build_exact(data, backend="triton", clause_is_reverse=rev)
        off = _build_exact(data, backend="official", clause_is_reverse=rev)
        ids_tri, sc_tri = tri(data["query"], data["q_attrs"])
        ids_off, sc_off = off(data["query"], data["q_attrs"])
        for b in range(B):
            assert_topk_id_sets_match(ids_off, sc_off, ids_tri, sc_tri, b)

    @pytest.mark.parametrize("filter_mode", ["none", "bloom", "exact"])
    def test_cute_equals_cuda_bitexact(self, data, filter_mode):
        """cute is a port of cuda with the same buffers (a cute checkpoint is a cuda
        checkpoint) and the same kernels, so on the *same index* the whole layer must
        agree bit for bit, not just on id sets. Two kmeans runs are not bit-deterministic
        on the GPU (this is why the other cross-backend tests compare id sets), so the
        cute module is handed the cuda module's index verbatim rather than rebuilding
        its own."""
        builders = {
            "none": lambda backend: _build_no_bloom(data, backend=backend),
            "bloom": lambda backend: _build(with_attrs=True, data=data, backend=backend),
            "exact": lambda backend: _build_exact(data, backend=backend),
        }
        cud = builders[filter_mode]("cuda")
        cut = builders[filter_mode]("cute")
        assert list(cud.state_dict()) == list(cut.state_dict()), "buffer sets must match"
        for name, buf in cud.named_buffers():
            setattr(cut, name, buf.clone())
        cut._max_cluster_size = cud._max_cluster_size
        cut._global_scale_f = cud._global_scale_f
        qa = data["q_attrs"] if filter_mode != "none" else None
        ids_cud, sc_cud = cud(data["query"], qa)
        ids_cut, sc_cut = cut(data["query"], qa)
        assert torch.equal(sc_cud, sc_cut)
        # topk tie order is not promised stable across launches, so ids get the usual
        # one notch of slack: equal, or permuted within a run of tied scores.
        for b in range(B):
            assert_topk_id_sets_match(ids_cut, sc_cut, ids_cud, sc_cud, b, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
class TestRecallVsExact:
    """SilverTorch paper claim: at sufficient probes, recall ≈ exact."""

    def test_recall_at_full_probe(self, data, backend):
        """When n_probe == n_lists, IVF candidate set equals the full index → recall ≈ 1."""
        st = _build(with_attrs=False, data=data, n_probe=N_LISTS, backend=backend)
        ids, _ = st(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recall = recall_at_k(ids, ex_ids)
        # int8 quantization can shuffle ties; allow modest slack.
        assert recall >= 0.90, f"recall@{K} at full-probe = {recall:.3f}"

    def test_recall_grows_with_probe(self, data, backend):
        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])

        recalls = []
        for nprobe in (1, 4, 16, N_LISTS):
            m = _build_no_bloom(data, n_probe=nprobe, backend=backend)
            ids, _ = m(data["query"])
            recalls.append(recall_at_k(ids, ex_ids))
        assert_recall_monotone(recalls)
        assert recalls[-1] >= 0.85, recalls

    def test_recall_at_full_probe_exact_inactive(self, data, backend):
        """Exact mode at n_probe == n_lists with all clauses inactive → recall ≈ 1
        (predicate always passes; only quantization noise vs FullScanKNN)."""
        m = _build_exact(data, n_probe=N_LISTS, backend=backend)
        qa = torch.full((B, C), -1, dtype=torch.long, device="cuda")
        ids, _ = m(data["query"], qa)

        exact = FullScanKNN(k=K)
        exact.register_index(data["embs"])
        ex_ids, _ = exact(data["query"])
        assert recall_at_k(ids, ex_ids) >= 0.90


@pytest.mark.parametrize("backend", BACKENDS)
class TestCandidates:
    def test_candidate_ids_with_bloom(self, data, backend):
        """Candidate-id path on a bloom-configured module — pure re-rank, no filter."""
        m = _build(with_attrs=True, data=data, k=2, backend=backend)
        cand = torch.tensor([[10, 20, 30]] * B, dtype=torch.long, device="cuda")
        ids, _ = m(data["query"], candidate_ids=cand)
        allowed = {10, 20, 30}
        assert all(i.item() in allowed for row in ids for i in row)

    def test_candidate_ids_no_bloom(self, data, backend):
        m = _build_no_bloom(data, k=2, backend=backend)
        g = torch.Generator(device="cuda").manual_seed(7)
        cand = torch.randint(0, N, (B, 16), generator=g, device="cuda")
        ids, _ = m(data["query"], candidate_ids=cand)
        assert ids.shape == (B, 2)
        for b in range(B):
            allowed = set(cand[b].tolist())
            for j in range(2):
                assert ids[b, j].item() in allowed

    def test_candidate_ids_with_exact(self, data, backend):
        """Candidate-id path on an exact-configured module — pure re-rank, no filter."""
        m = _build_exact(data, k=2, backend=backend)
        cand = torch.tensor([[10, 20, 30]] * B, dtype=torch.long, device="cuda")
        ids, _ = m(data["query"], candidate_ids=cand)
        allowed = {10, 20, 30}
        assert all(i.item() in allowed for row in ids for i in row)

    def test_candidates_with_query_attrs_raises(self, data, backend):
        """The candidate_ids path scores without the fused filter, so passing
        query_clause_attrs alongside would be a silent filter-skip — must raise."""
        cand = torch.tensor([[10, 20, 30]] * B, dtype=torch.long, device="cuda")
        for m in (
            _build(with_attrs=True, data=data, k=2, backend=backend),
            _build_exact(data, k=2, backend=backend),
        ):
            with pytest.raises(ValueError, match="not both"):
                m(data["query"], data["q_attrs"], candidate_ids=cand)

    def test_candidate_ids_pad_tail_never_scored(self, data, backend):
        """``-1``-tailed candidate ids (what every compact producer emits) are padding:
        never gathered (a raw ``-1`` would wrap to item ``N-1`` — through ``inv_perm`` on
        official), never scored, never returned. Row ``b`` has ``2 + b`` real candidates
        out of ``P = 24``; with ``k = 8`` the first six rows end in ``-1`` / ``-inf``. The
        finite part equals a pad-free re-rank of the same row (the int8 dot is exact in
        fp32, so bit-equal), and item ``N-1`` — excluded from every candidate list — never
        appears."""
        k, p = 8, 24
        m = _build_no_bloom(data, k=k, backend=backend)
        g = torch.Generator(device="cuda").manual_seed(13)
        cand = torch.randint(0, N - 1, (B, p), generator=g, dtype=torch.long, device="cuda")
        n_valid = torch.arange(B, device="cuda") + 2
        cand[torch.arange(p, device="cuda")[None, :] >= n_valid[:, None]] = -1
        ids, scores = m(data["query"], candidate_ids=cand)
        assert ids.shape == scores.shape == (B, k)
        assert not (ids == N - 1).any(), "a -1 pad wrapped to the last item"
        finite = torch.isfinite(scores)
        assert torch.equal(ids >= 0, finite), "-1 ids exactly where scores are -inf"
        for b in range(B):
            nv = int(n_valid[b].item())
            kk = min(k, nv)
            assert int(finite[b].sum().item()) == kk
            row = cand[b, :nv].unsqueeze(0)
            ref_ids, ref_scores = m(data["query"][b : b + 1], candidate_ids=row)
            assert torch.equal(scores[b, :kk], ref_scores[0, :kk])
            assert_ids_equal_up_to_ties(
                ids[b, :kk][None], ref_ids[0, :kk][None], scores[b, :kk][None]
            )
            assert set(ids[b, :kk].tolist()) <= set(row[0].tolist())

    def test_candidate_ids_p_less_than_k(self, data, backend):
        """``candidate_ids`` smaller than K — forward returns ``actual_k = p`` columns."""
        m = _build_no_bloom(data, backend=backend)
        p = K // 2
        g = torch.Generator(device="cuda").manual_seed(42)
        cand = torch.randint(0, N, (B, p), generator=g, dtype=torch.long, device="cuda")
        ids, scores = m(data["query"], candidate_ids=cand)
        assert ids.shape == (B, p)
        assert scores.shape == (B, p)
        # Current-semantics pin: min(k, P) columns, no pad tail and no sentinels —
        # every score finite, every id a real candidate.
        assert torch.isfinite(scores).all()
        assert (ids >= 0).all()
        for b in range(B):
            allowed = set(cand[b].tolist())
            for j in range(p):
                assert int(ids[b, j].item()) in allowed


@pytest.mark.parametrize("backend", BACKENDS)
class TestStateDict:
    """``load_state_dict`` re-derives the two Python-scalar caches the forwards read
    (``_global_scale_f``, ``_max_cluster_size``) from the loaded buffers, so a loaded index
    scores exactly like the one that was saved — nothing is patched by hand."""

    @pytest.mark.parametrize("filter_mode", ["none", "bloom", "exact"])
    def test_load_state_dict_rederives_cached_scalars(self, data, backend, filter_mode):
        builders = {
            "none": lambda: _build_no_bloom(data, backend=backend),
            "bloom": lambda: _build(with_attrs=True, data=data, backend=backend),
            "exact": lambda: _build_exact(data, backend=backend),
        }
        src = builders[filter_mode]()
        # The receiving module must have the saved module's buffer *shapes* (the usual
        # nn.Module rule), and a second register_index cannot promise that: GPU k-means is
        # not bit-deterministic (float index_add_ atomics), so the IVF may pad to a
        # different width. A deep copy fixes the shapes; its buffers are then zeroed and
        # its caches poisoned, so anything the load does not restore shows up.
        twin = copy.deepcopy(src)
        with torch.no_grad():
            for buf in twin.buffers():
                buf.zero_()
        twin._global_scale_f = float("nan")
        twin._max_cluster_size = -1

        twin.load_state_dict(src.state_dict())

        assert twin._global_scale_f == src._global_scale_f
        assert twin._max_cluster_size == src._max_cluster_size
        for (name, a), (_, b) in zip(src.named_buffers(), twin.named_buffers()):
            assert torch.equal(a, b), name
        qa = data["q_attrs"] if filter_mode != "none" else None
        ids_s, sc_s = src(data["query"], qa)
        ids_t, sc_t = twin(data["query"], qa)
        assert torch.equal(sc_s, sc_t)
        for b in range(B):
            assert_topk_id_sets_match(ids_t, sc_t, ids_s, sc_s, b, atol=0, rtol=0)


class TestBuilder:
    def test_build_helper_with_bloom(self, data):
        m = build_silvertorch(
            data["embs"],
            k=K,
            n_lists=N_LISTS,
            n_probe=N_PROBE,
            filter_mode="bloom",
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

    def test_build_helper_torch_backend(self, data):
        m = build_silvertorch(
            data["embs"], k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3, backend="torch"
        )
        assert m.backend == "torch"
        ids, _ = m(data["query"])
        assert ids.shape == (B, K)

    def test_build_helper_with_exact(self, data):
        rev = torch.tensor([True, False], dtype=torch.bool, device="cuda")
        m = build_silvertorch(
            data["embs"],
            k=K,
            n_lists=N_LISTS,
            n_probe=N_PROBE,
            filter_mode="exact",
            n_iter=3,
            item_clause_attrs=data["attrs"],
            clause_is_reverse=rev,
        )
        assert isinstance(m, SilverTorch)
        assert m.filter_mode == "exact"
        ids, _ = m(data["query"], data["q_attrs"])
        assert ids.shape == (B, K)


@pytest.mark.parametrize("backend", BACKENDS)
class TestEdgeCases:
    def test_n_lists_equals_n(self, data, backend):
        """Degenerate clustering (one item per cluster); full probe → recall ≈ 1."""
        small_n = 256
        embs = data["embs"][:small_n]
        m = SilverTorch(k=K, n_lists=small_n, n_probe=small_n, n_iter=2, backend=backend)
        m.register_index(embs)
        ids, _ = m(data["query"])

        exact = FullScanKNN(k=K)
        exact.register_index(embs)
        ex_ids, _ = exact(data["query"])
        # int8 quant + kmeans degeneracy can shuffle near-ties; allow modest slack.
        assert recall_at_k(ids, ex_ids) >= 0.85
