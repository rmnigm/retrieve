"""``bench.algos.build``: every ``(algo, filter_kind)`` cell on the torch path returns a
module whose ``k`` setter slices the top-k without touching a buffer (H §7's "no layer bakes
``k``"), the filter is a submodule counted by ``index_bytes``, ``set_query_params``
re-validates, and ``build`` refuses ``None``-path cells and invalid combos. The three
SilverTorch modules are built once per session with ``n_iter=2`` (``silvertorch_modules``);
the tests that share them are read-only beyond ``.k``."""

from __future__ import annotations

import pytest
import torch

from bench import algos as A
from bench.measure import index_bytes
from retrieve import OfficialConfig, SilverTorch

N, D = 256, 64


def _prefix_equal(big: tuple, small: tuple) -> None:
    """``small`` (top-k') is the prefix of ``big`` (top-k): scores exactly, ids up to ties."""
    ids_b, sc_b = big
    ids_s, sc_s = small
    k = ids_s.shape[1]
    assert torch.equal(sc_s, sc_b[:, :k])
    for r in range(ids_b.shape[0]):
        for s in sc_s[r].unique():
            a = set(ids_s[r][sc_s[r] == s].tolist())
            g = set(ids_b[r][sc_b[r] == s].tolist())
            if s > sc_b[r, -1]:  # the tie group is complete in the big run
                assert a <= g
                if s > sc_s[r, -1]:  # and in the small run
                    assert a == g


def _data(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=g)
    q = torch.randn(6, D, generator=g)
    attrs = torch.randint(0, 3, (N, 2, 1), generator=g)
    attrs[:, 1] = torch.where(torch.rand(N, 1, generator=g) < 0.4, torch.tensor(-1), attrs[:, 1])
    qa = torch.tensor([[0, -1], [1, -1], [2, 0], [0, 1], [1, 2], [2, -1]])
    return x, q, attrs, qa


@pytest.mark.parametrize(
    ("params", "bloom_path"), [({}, "partial"), ({"bloom_path": "partial"}, "partial"),
                               ({"bloom_path": "full"}, "full")],
)  # fmt: skip
def test_official_config_from_bloom_path(params, bloom_path):
    cfg = A.official_config("silvertorch", "bloom", "official", params)
    assert (cfg or OfficialConfig()).bloom_path == bloom_path
    assert (cfg is None) == (params == {})


@pytest.mark.parametrize(
    ("algo", "filter_kind", "backend"),
    [("silvertorch", "bloom", "triton"), ("silvertorch", "clause", "official"),
     ("linr_v3", "bloom", "triton")],
)  # fmt: skip
def test_official_config_rejects_bloom_path_elsewhere(algo, filter_kind, backend):
    with pytest.raises(ValueError, match="bloom_path applies to"):
        A.official_config(algo, filter_kind, backend, {"bloom_path": "full"})
    assert A.official_config(algo, filter_kind, backend, {"n_probe": 4}) is None


ST_PARAMS = {"n_lists": 8, "n_probe": 4, "n_iter": 2}
BLOOM = {"m_bits": 64, "k_hash": 2}


@pytest.fixture(scope="session")
def silvertorch_modules() -> dict[str, SilverTorch]:
    x, _, attrs, _ = _data()
    out = {}
    for fk in A.FILTER_KINDS:
        f = A.build_filter(fk, attrs, backend="torch", **BLOOM)
        out[fk] = A.build(
            "silvertorch", x, k=8, backend="torch", filter_kind=fk, filter_mod=f,
            item_attrs=attrs, params={**ST_PARAMS, **(BLOOM if fk == "bloom" else {})},
        )  # fmt: skip
    return out


def _check_k_slice(module, call, k_big: int = 8, k_small: int = 3) -> None:
    module.k = k_big
    before = {n: b.clone() for n, b in module.state_dict().items()}
    big = call()
    module.k = k_small
    small = call()
    assert small[0].shape[1] == k_small
    _prefix_equal(big, small)
    after = module.state_dict()
    assert before.keys() == after.keys()
    for name, b in before.items():
        assert torch.equal(b, after[name]), f"buffer {name} changed with k"


def _cells():
    for algo in A.ALGOS:
        for fk in A.FILTER_KINDS:
            if A.PATHS[algo, fk, "torch"] is not None:
                yield algo, fk


@pytest.mark.parametrize("algo,fk", list(_cells()))
@torch.inference_mode()
def test_built_module_k_setter_slices(algo, fk, silvertorch_modules):
    x, q, attrs, qa = _data()
    if algo == "silvertorch":
        m = silvertorch_modules[fk]
        assert isinstance(m, SilverTorch) and m.filter_mode == A.FILTER_MODE[fk]
    else:
        f = A.build_filter(fk, attrs, backend="torch", m_bits=64, k_hash=2)
        params = {"linr_v3": {"candidate_pool": 64}}.get(algo, {})
        m = A.build(
            algo, x, k=8, backend="torch", filter_kind=fk, filter_mod=f, item_attrs=attrs,
            params=params,
        )  # fmt: skip
    assert m.backend == "torch" and m.capturable is True
    _check_k_slice(m, lambda: m(q, qa if fk != "none" else None))


def test_index_bytes_includes_filter_submodule(silvertorch_modules):
    x, _, attrs, _ = _data()
    f = A.build_filter("clause", attrs, backend="torch")
    m = A.build("linr_v1_filter_mask", x, k=4, backend="torch", filter_kind="clause", filter_mod=f)
    assert m.filter is f and index_bytes(m) == index_bytes(m.idx) + index_bytes(f) > 0
    st = silvertorch_modules["bloom"]
    assert getattr(st, "filter", None) is None and "bloom_transposed" in st.state_dict()
    assert A.build_filter("none", None) is None


def test_silvertorch_query_params_revalidate():
    x, q, _, _ = _data()
    st = A.build(
        "silvertorch", x, k=4, backend="torch", params={"n_lists": 8, "n_probe": 2, "n_iter": 2}
    )
    with pytest.raises(ValueError, match="cannot exceed n_lists"):
        st.set_query_params(n_probe=9)
    st.set_query_params(n_probe=8)
    assert st.n_probe == 8
    ids, _ = st(q)  # probing every list: every row is full
    assert ids.shape == (6, 4) and bool((ids >= 0).all())
    st.k = x.shape[0]  # the largest k the full probe pool admits ...
    with pytest.raises(ValueError, match="probe pool"):
        st.set_query_params(n_probe=1)  # ... which a single probed list cannot serve


def test_linr_v3_query_params():
    x, q, _, _ = _data()
    m = A.build("linr_v3", x, k=4, backend="torch", params={"candidate_pool": 16})
    assert m.stage1.k == 16
    m.set_query_params(candidate_pool=32)
    assert m.stage1.k == 32 and m(q)[0].shape == (6, 4)
    with pytest.raises(ValueError, match="exceeds N"):
        m.set_query_params(candidate_pool=N + 1)


def test_build_refusals():
    x, _, _, _ = _data()
    with pytest.raises(ValueError, match="no code path"):
        A.build("linr_v2", x, k=4, backend="torch", filter_kind="none")
    with pytest.raises(ValueError, match="no code path"):
        A.build("linr_v1_filter_mask", x, k=4, backend="official")
    with pytest.raises(ValueError, match="invalid params"):
        A.build("silvertorch", x, k=4, backend="torch", params={"n_lists": 4, "n_probe": 8})
    assert A.is_valid_combo("silvertorch", {"n_lists": 16, "n_probe": 16})
    assert not A.is_valid_combo("silvertorch", {"n_lists": 16})  # default n_probe=24 > 16
    assert A.is_valid_combo("linr_v3", {"n_probe": 100}) and A.is_valid_combo("silvertorch", {})
