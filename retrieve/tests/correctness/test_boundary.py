"""The library side of the library / harness contract (library-harness-boundary.md §4, gate
§5): for each algo-level module — ``SilverTorch``, ``LiNRV1``–``LiNRV4`` — on a tiny index,
the properties the harness measures through. ``official`` is covered where the clause names it
(``capturable``, ``DISPATCH``); its forward syncs by design (O D7)."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from retrieve import (
    ExactAttributeFilter,
    LiNRBuilder,
    LiNRV1,
    LiNRV2,
    LiNRV3,
    LiNRV4,
    SilverTorch,
    SilverTorchBuilder,
    modules as modules_pkg,
)
from retrieve.interfaces import DISPATCH
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs, require_official
from tests.parity.conftest import assert_ids_equal_up_to_ties

N, D, B, K = 512, 64, 8, 16
CLASSES = [SilverTorch, LiNRV1, LiNRV2, LiNRV3, LiNRV4]
BACKENDS = ["triton", "torch"]
ST = {"k": K, "n_lists": 8, "n_probe": 4, "n_iter": 2, "filter_mode": "exact"}
V3 = {"candidate_pool": 64, "seed": 1}


@pytest.fixture(scope="module")
def data():
    return {
        "embs": make_index(N, D),
        "query": make_query(B, D),
        "attrs": make_attrs(N, c=2, a_max=2, n_vocab=8),
        "q_attrs": make_query_attrs(B, c=2, n_vocab=8),
    }


def _builder(cls, backend):
    if cls is SilverTorch:
        return SilverTorchBuilder(**ST, backend=backend)
    kw = V3 if cls is LiNRV3 else {}
    b = LiNRBuilder(cls.__name__[-2:].lower(), k=K, backend=backend, **kw)
    return b.set_filter(ExactAttributeFilter(backend=backend))


def _build(cls, backend, data):
    b = _builder(cls, backend).set_item_embeddings(data["embs"])
    if cls is SilverTorch:
        b.set_item_attributes(data["attrs"])
    else:
        b.set_filter(ExactAttributeFilter(backend=backend), data["attrs"])
    return b.build()


class _Shapes(TorchDispatchMode):
    """Records the output shape of every op a forward dispatches. Custom ops are opaque here:
    it sees a ``retrieve::`` call and its outputs, not the kernel's own buffers."""

    def __init__(self):
        super().__init__()
        self.seen: list[tuple[str, tuple[int, ...]]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        outs = out if isinstance(out, (tuple, list)) else (out,)
        self.seen += [(str(func), tuple(t.shape)) for t in outs if isinstance(t, torch.Tensor)]
        return out


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cls", CLASSES)
class TestPerClass:
    def test_forward_materializes_no_wide_intermediate(self, data, cls, backend):
        """No >= 3-d intermediate above ``B·N`` elements (a ``[B, P, D]`` gather or a ``[B, N,
        C, A_max]`` broadcast) on Triton; the torch backend, which materializes them by design,
        is the control that proves the recorder sees them."""
        m = _build(cls, backend, data)
        m(data["query"], data["q_attrs"])
        with _Shapes() as rec:
            m(data["query"], data["q_attrs"])
        assert len(rec.seen) >= 3, rec.seen
        wide = [(op, s) for op, s in rec.seen if len(s) >= 3 and torch.Size(s).numel() > B * N]
        if backend == "triton":
            assert wide == [], wide
        else:
            assert wide, "the torch control materialized nothing wide: the recorder is blind"

    def test_k_mutation_changes_width_without_reregistration(self, data, cls, backend):
        m = _build(cls, backend, data)
        assert m(data["query"], data["q_attrs"])[0].shape == (B, K)
        before = {n: b.clone() for n, b in m.state_dict().items()}
        m.k = K // 2
        ids, scores = m(data["query"], data["q_attrs"])
        assert ids.shape == scores.shape == (B, K // 2)
        m.k = K
        assert m(data["query"], data["q_attrs"])[0].shape == (B, K)
        for name, b in m.state_dict().items():
            assert torch.equal(b, before[name]), name

    def test_buffers_cover_every_tensor_attribute(self, data, cls, backend):
        m = _build(cls, backend, data)
        for name, sub in m.named_modules():
            strays = [k for k, v in vars(sub).items() if isinstance(v, torch.Tensor)]
            assert not strays, (name, strays)
        assert set(m.state_dict()) == {n for n, _ in m.named_buffers()}
        assert sum(b.numel() * b.element_size() for b in m.buffers()) > 0

    def test_forward_is_sync_free(self, data, cls, backend):
        m = _build(cls, backend, data)
        m(data["query"], data["q_attrs"])  # first call: Triton JIT, cuBLAS handles
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            m(data["query"], data["q_attrs"])
            if cls is not LiNRV2:
                m(data["query"])
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_set_state_dict_round_trip_equals_a_fresh_build(self, data, cls, backend):
        src = _build(cls, backend, data)
        twin = _builder(cls, backend).set_state_dict(src.state_dict()).build()
        fresh = _build(cls, backend, data)
        assert list(twin.state_dict()) == list(fresh.state_dict())
        for (name, a), (_, b) in zip(fresh.named_buffers(), twin.named_buffers()):
            assert torch.equal(a, b), name
        ids_f, sc_f = fresh(data["query"], data["q_attrs"])
        ids_t, sc_t = twin(data["query"], data["q_attrs"])
        assert torch.equal(sc_f, sc_t)
        assert_ids_equal_up_to_ties(ids_t, ids_f, sc_f)

    def test_capturable_is_a_class_attribute(self, data, cls, backend):
        m = _build(cls, backend, data)
        assert isinstance(inspect.getattr_static(cls, "capturable"), (bool, property))
        assert "capturable" not in vars(m)
        assert m.capturable is True


def test_capturable_is_false_on_official():
    require_official()
    assert SilverTorch(k=1, n_lists=1, n_probe=1, backend="official").capturable is False


def test_dispatch_names_every_class_and_backend():
    assert {cls.__name__ for cls in CLASSES} <= set(DISPATCH)
    for name, row in DISPATCH.items():
        assert isinstance(getattr(modules_pkg, name), type), name
        assert list(row) == ["triton", "torch", "official"]
        for backend in BACKENDS:
            assert row[backend] in ("triton", "torch", "cublas")
    assert DISPATCH["SilverTorch"]["official"] == "official"
    for cls in CLASSES[1:]:
        assert DISPATCH[cls.__name__]["official"] is None
        with pytest.raises(ValueError, match="unknown backend"):
            cls(k=K, filter=ExactAttributeFilter(), backend="official")


class TestQueryParams:
    def test_silvertorch_n_probe_revalidates(self, data):
        m = _build(SilverTorch, "torch", data)
        with pytest.raises(ValueError, match="cannot exceed n_lists"):
            m.set_query_params(n_probe=9)
        m.k = int(m.cluster_sizes.topk(8).values.sum()) + 1  # one past the 8-probe width
        with pytest.raises(ValueError, match="probe pool"):
            m.set_query_params(n_probe=8)
        m.k = K
        m.set_query_params(n_probe=8)
        assert m.n_probe == 8
        ids, _ = m(data["query"], data["q_attrs"])
        assert ids.shape == (B, K)

    def test_linr_v3_candidate_pool(self, data):
        m = _build(LiNRV3, "torch", data)
        assert m.stage1.k == V3["candidate_pool"]
        m.set_query_params(candidate_pool=128)
        assert m.stage1.k == 128 and m(data["query"])[0].shape == (B, K)
        with pytest.raises(ValueError, match="exceeds N"):
            m.set_query_params(candidate_pool=N + 1)


def test_build_timings(data):
    m = _build(SilverTorch, "triton", data)
    assert list(m.build_timings) == ["kmeans_s", "assemble_s", "quantize_s", "filter_s"]
    assert all(isinstance(v, float) and v >= 0 for v in m.build_timings.values())
    assert m.build_timings["kmeans_s"] > 0
    assert SilverTorch(**ST).build_timings == {}
