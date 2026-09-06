"""CPU-only tests for ``retrieval.algos`` (harness v2 WP-1).

- ``PATHS`` covers ``ALGOS × FILTER_KINDS × BACKENDS`` and agrees with the dispatch table
  in ``docs/system/architecture.md`` (parsed from the markdown, not mirrored in a fixture,
  so the doc and the table cannot drift apart silently).
- ``k``-slice invariance: for every ``retrieve`` layer the harness uses, and for every
  wrapper, ``module.k = k'`` after ``register_index`` returns the top-``k'`` prefix of the
  top-``k`` result (scores ``torch.equal``, ids equal up to ties) and leaves every buffer
  untouched — the H §7 "no layer bakes ``k`` into a buffer" check, on ``backend="torch"``
  (the Triton kernels need CUDA; C4 covers them).
- ``build`` refuses cells ``PATHS`` marks ``None``, invalid combos and, until roadmap B1
  lands, ``backend="official"``.

The three SilverTorch wrappers (``none`` / ``clause`` / ``bloom``) are built once per session
with ``n_iter=2`` (``silvertorch_modules``): the k-means build is the cost, not the
assertions, and every test that shares them is read-only on the module beyond ``.k``, which
``_check_k_slice`` sets before it reads. ``test_silvertorch_query_params_revalidate`` mutates
``n_probe`` and builds its own.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest
import torch

from retrieval import algos as A
from retrieval.bench import index_bytes
from retrieve import OneBitKNN, PostfilterKNN, PostfilterKNNInt8, PrefilterKNN, SilverTorch
from retrieve.interfaces import SilverTorchBackend

ARCH_MD = Path(__file__).resolve().parents[3] / "docs" / "system" / "architecture.md"
N, D = 256, 64

# ----- PATHS ----------------------------------------------------------------------


def test_paths_cover_the_full_grid():
    grid = {(a, f, b) for a in A.ALGOS for f in A.FILTER_KINDS for b in A.BACKENDS}
    assert set(A.PATHS) == grid
    assert set(A.FILTER_BACKEND) == set(A.CAPTURABLE) == set(A.BACKENDS)
    # official: SilverTorch only, Triton filters, no graph mode (H amendment, O §6.2 / D7).
    for a in A.ALGOS:
        assert (A.PATHS[a, "clause", "official"] == "official") == (a == "silvertorch")
    assert A.FILTER_BACKEND["official"] == "triton" and A.CAPTURABLE["official"] is False
    assert A.PATHS["linr_v2", "none", "triton"] is None  # the candidate source is the filter
    # linr_v1 / linr_v4 on ``none`` collapse to one cuBLAS path.
    for b in ("triton", "torch"):
        assert A.PATHS["linr_v1_filter_mask", "none", b] == "cublas"
        assert A.PATHS["linr_v4", "none", b] == "cublas"


def _dispatch_table() -> tuple[list[str], dict[str, list[str]]]:
    """Parse the ``### Backend dispatch`` table: header cells and rows keyed by module cell."""
    text = ARCH_MD.read_text()
    body = text.split("### Backend dispatch", 1)[1]
    rows = [ln for ln in body.splitlines() if ln.startswith("|")]
    split = lambda ln: [c.strip() for c in ln.strip().strip("|").split("|")]  # noqa: E731
    header = [re.sub(r"[`\"]", "", c) for c in split(rows[0])]
    table = {split(ln)[0]: split(ln)[1:] for ln in rows[2:]}
    return header, table


def _classify(cell: str) -> str:
    c = cell.lower()
    if "valueerror" in c or "raises" in c:
        return "rejected"
    if "cublas" in c:
        return "cublas"
    if "eager" in c or "torch" in c:
        return "torch"
    return "triton"  # a fused Triton kernel name, or "fused Triton"


_MODULES = {  # library modules on each algo's path; the first is the final top-k owner
    "linr_v1_filter_mask": ["PostfilterKNN"],
    "linr_v2": ["PrefilterKNN"],
    "linr_v3": ["OneBitKNN", "PrefilterKNN"],
    "linr_v4": ["PostfilterKNNInt8"],
    "silvertorch": ["SilverTorch"],
}
_FILTER = {"clause": "ExactAttributeFilter", "bloom": "BloomFilter"}


def test_paths_agree_with_architecture_dispatch_table():
    header, table = _dispatch_table()
    cols = {name: i for i, name in enumerate(header[1:])}

    def row(module: str) -> list[str]:
        matches = [cells for key, cells in table.items() if module in key]
        assert len(matches) == 1, module
        cells = list(matches[0])
        for i, c in enumerate(cells):  # "same" = the flag is a no-op: inherit the triton cell
            cells[i] = cells[0] if c.lower() == "same" else c
        return cells

    for (algo, fk, backend), path in A.PATHS.items():
        if backend == "official":
            if algo != "silvertorch":
                # The table says these modules reject "official" at construction — no
                # official code — so PATHS refuses the cell rather than mislabelling a
                # torch/cuBLAS number.
                for m in _MODULES[algo]:
                    assert _classify(row(m)[cols["official"]]) == "rejected", (algo, m)
                assert path is None
            continue
        if algo == "linr_v2" and fk == "none":
            assert path is None
            continue
        parts = [_classify(row(m)[cols[backend]]) for m in _MODULES[algo]]
        assert len(set(parts)) == 1, (algo, backend, parts)
        if fk != "none" and algo != "silvertorch":  # SilverTorch fuses its predicate
            parts.append(_classify(row(_FILTER[fk])[cols[backend]]))
        expected = parts[0] if len(set(parts)) == 1 else "+".join(parts)
        assert path == expected, (algo, fk, backend, path, expected)


# ----- k-slice invariance ---------------------------------------------------------


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


ST_PARAMS = {"n_lists": 8, "n_probe": 4, "m_bits": 64, "k_hash": 2, "n_iter": 2}


@pytest.fixture(scope="session")
def silvertorch_modules() -> dict[str, A.Silvertorch]:
    """``filter_kind -> Silvertorch`` wrapper on the torch path over ``_data()``, built once."""
    x, _, attrs, _ = _data()
    out = {}
    for fk in A.FILTER_KINDS:
        f = A.build_filter(fk, attrs, backend="torch", m_bits=64, k_hash=2)
        out[fk] = A.build(
            "silvertorch", x, k=8, backend="torch", filter_kind=fk, filter_mod=f,
            item_attrs=attrs, params=ST_PARAMS,
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


@torch.inference_mode()
def test_layer_postfilter_knn_k_not_baked():
    x, q, attrs, _ = _data()
    m = PostfilterKNN(k=8, backend="torch")
    m.register_index(x)
    _check_k_slice(m, lambda: m(q))
    mask = torch.rand(6, N) < 0.5
    _check_k_slice(m, lambda: m(q, mask=mask))


@torch.inference_mode()
def test_layer_postfilter_knn_int8_k_not_baked():
    x, q, _, _ = _data()
    m = PostfilterKNNInt8(k=8, backend="torch")
    m.register_index(x)
    _check_k_slice(m, lambda: m(q))
    mask = torch.rand(6, N) < 0.5
    _check_k_slice(m, lambda: m(q, mask=mask))


@torch.inference_mode()
def test_layer_prefilter_knn_k_not_baked():
    x, q, _, _ = _data()
    m = PrefilterKNN(k=8, backend="torch")
    m.register_index(x)
    _check_k_slice(m, lambda: m(q))
    cand = torch.arange(40).repeat(6, 1)
    counts = torch.tensor([40, 30, 20, 10, 40, 5])
    _check_k_slice(m, lambda: m(q, candidate_ids=cand, counts=counts))


@torch.inference_mode()
def test_layer_one_bit_knn_k_not_baked():
    x, q, _, _ = _data()
    m = OneBitKNN(k=8, backend="torch")
    m.register_index(x)
    _check_k_slice(m, lambda: m(q))
    cand = torch.arange(40).repeat(6, 1)
    counts = torch.tensor([40, 30, 20, 10, 40, 5])
    _check_k_slice(m, lambda: m(q, candidate_ids=cand, counts=counts))


@pytest.mark.parametrize("fk", ["none", "clause", "bloom"])
@torch.inference_mode()
def test_layer_silvertorch_k_not_baked(fk, silvertorch_modules):
    """The ``SilverTorch`` layer itself (``filter_mode`` none / exact / bloom), reached through
    the session wrapper's ``.idx`` — the same registered layer, called directly."""
    _, q, _, qa = _data()
    m = silvertorch_modules[fk].idx
    assert isinstance(m, SilverTorch)
    assert m.filter_mode == {"none": "none", "clause": "exact", "bloom": "bloom"}[fk]
    if fk == "none":
        _check_k_slice(m, lambda: m(q))
    else:
        _check_k_slice(m, lambda: m(q, query_clause_attrs=qa))


def _cells():
    for algo in A.ALGOS:
        for fk in A.FILTER_KINDS:
            if A.PATHS[algo, fk, "torch"] is not None:
                yield algo, fk


@pytest.mark.parametrize("algo,fk", list(_cells()))
@torch.inference_mode()
def test_wrapper_k_setter_slices(algo, fk, silvertorch_modules):
    x, q, attrs, qa = _data()
    if algo == "silvertorch":
        m = silvertorch_modules[fk]
    else:
        f = A.build_filter(fk, attrs, backend="torch", m_bits=64, k_hash=2)
        params = {"linr_v3": {"candidate_pool": 64}}.get(algo, {})
        m = A.build(
            algo, x, k=8, backend="torch", filter_kind=fk, filter_mod=f, item_attrs=attrs,
            params=params,
        )  # fmt: skip
    assert m.backend == "torch" and m.capturable is True
    _check_k_slice(m, lambda: m(q, qa if fk != "none" else None))


# ----- memory, query params, build refusals -----------------------------------------


def test_index_bytes_includes_filter_submodule(silvertorch_modules):
    x, _, attrs, _ = _data()
    f = A.build_filter("clause", attrs, backend="torch")
    m = A.build("linr_v1_filter_mask", x, k=4, backend="torch", filter_kind="clause", filter_mod=f)
    assert m.filter is f and index_bytes(m) == index_bytes(m.idx) + index_bytes(f) > 0
    st = silvertorch_modules["bloom"]
    assert st.filter is None and "idx.bloom_sigs" in st.state_dict()
    assert A.build_filter("none", None) is None


def test_silvertorch_query_params_revalidate():
    x, q, _, _ = _data()
    st = A.build(
        "silvertorch", x, k=4, backend="torch", params={"n_lists": 8, "n_probe": 2, "n_iter": 2}
    )
    max_size = st.idx.padded_cluster_items.shape[1]
    with pytest.raises(ValueError, match="cannot exceed n_lists"):
        st.set_query_params(n_probe=9)
    with pytest.raises(ValueError, match="probe pool"):
        st.k = 8 * max_size + 1
    st.set_query_params(n_probe=8)
    assert st.idx.n_probe == 8
    ids, _ = st(q)  # probing every list: every row is full
    assert ids.shape == (6, 4) and bool((ids >= 0).all())
    st.k = 8 * max_size  # the largest k the full probe pool admits ...
    with pytest.raises(ValueError, match="probe pool"):
        st.set_query_params(n_probe=1)  # ... which a single probed list cannot serve


def test_silvertorch_plan_cache_is_official_only():
    """``set_plan_cache`` / ``cache_plans`` (the official timing rule of kernels.md) are a
    no-op / ``None`` on every other backend; on ``official`` — simulated here, the ops need
    CUDA — the flip replaces ``OfficialConfig.cache_plans`` in place and nothing else."""
    from retrieve.layers.silvertorch import OfficialConfig  # noqa: PLC0415

    x, _, attrs, _ = _data()
    m = A.build(
        "silvertorch", x, k=4, backend="torch", params={"n_lists": 8, "n_probe": 2, "n_iter": 2}
    )
    assert m.cache_plans is None
    m.set_plan_cache(False)
    assert m.cache_plans is None and m.idx.official.cache_plans is True  # untouched
    m.idx.backend = "official"
    m.idx.official = OfficialConfig(b_multiplier=4.0)
    assert m.cache_plans is True
    m.set_plan_cache(False)
    assert m.cache_plans is False and m.idx.official.cache_plans is False
    assert m.idx.official.b_multiplier == 4.0 and m.idx.official.score_path == "fp16"
    m.set_plan_cache(True)
    assert m.cache_plans is True


def test_linr_v3_query_params():
    x, q, _, _ = _data()
    m = A.build("linr_v3", x, k=4, backend="torch", params={"candidate_pool": 16})
    assert m.stage1.k == 16
    m.set_query_params(candidate_pool=32)
    assert m.stage1.k == 32 and m(q)[0].shape == (6, 4)
    with pytest.raises(ValueError, match="exceeds N"):
        m.set_query_params(candidate_pool=N + 1)


def test_build_refusals():
    x, _, attrs, _ = _data()
    with pytest.raises(ValueError, match="no code path"):
        A.build("linr_v2", x, k=4, backend="torch", filter_kind="none")
    with pytest.raises(ValueError, match="no code path"):
        A.build("linr_v1_filter_mask", x, k=4, backend="official")
    with pytest.raises(ValueError, match="invalid params"):
        A.build("silvertorch", x, k=4, backend="torch", params={"n_lists": 4, "n_probe": 8})
    assert A.is_valid_combo("silvertorch", {"n_lists": 16, "n_probe": 16})
    assert A.is_valid_combo("linr_v3", {"n_probe": 100}) and A.is_valid_combo("silvertorch", {})
    if "official" in get_args(SilverTorchBackend):
        pytest.skip("official backend integrated in retrieve — its cell is C4's gate")
    with pytest.raises(NotImplementedError, match="official"):
        A.build("silvertorch", x, k=4, backend="official", params={"n_lists": 8, "n_probe": 4})
