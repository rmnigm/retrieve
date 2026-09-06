"""CPU-only tests for ``retrieval.config``'s harness-v2 matrix (H §6 WP-2, §8.4).

Fixture side (``tests/data/``): job counts and keys per suite, the PATHS collapse and
``None``-path skips, ``disabled: true`` sweeps, the ``build:`` / ``query:`` split (a job is
one build carrying its query combos; ``n_probe > n_lists`` combos are dropped per build),
the seeds rule, ``{dim}`` templating (string and per-dim mapping), the CLI narrows, and the
``ConfigError`` messages. Real side: expanding ``config/goodreads.yaml`` / ``config/arxiv.yaml``
with ``config/suites.yaml`` reproduces the old ``config/<dataset>/d128-{filter,quality}.yaml``
cells (algos × backends × sweeps × ks × batch sizes), modulo the PATHS collapse on ``none``
and the ``official`` backend the old configs never had.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from retrieval.config import NONE_SWEEP, ConfigError, load_dataset, load_matrix

FIX = Path(__file__).parent / "data"
CFG = Path(__file__).resolve().parents[2] / "config"
MINI, TEXT, SUITES = FIX / "mini.yaml", FIX / "text.yaml", FIX / "suites.yaml"


# ----- dataset files ----------------------------------------------------------------


def test_load_dataset_templates_and_derived_paths():
    ds = load_dataset(MINI, 32)
    assert ds.name == "mini" and ds.dim == 32
    assert ds.checkpoint == Path("data/mini/checkpoints/d32/best_model.pt")
    assert ds.content_dir is None and ds.users_limit == 100
    assert ds.encode == {"batch_size": 8, "num_workers": 0, "max_seq_length": 20}
    assert ds.attrs == Path("data/mini/item_attrs_narrow.pt")
    assert ds.reverse == Path("data/mini/clause_is_reverse_narrow.pt")
    assert ds.gt_dir == Path("data/mini/gt_d32")
    # c9_off is disabled → gone here, once, not at every consumer.
    assert ds.clauses == {"clause": {"c0": (0,), "c0c1": (0, 1)}, "bloom": {"c0": (0,)}}


def test_load_dataset_per_dim_mapping():
    assert load_dataset(TEXT, 16).content_dir == Path("data/text/content_d16")
    ds = load_dataset(TEXT, 32)
    assert ds.content_dir == Path("data/text/content") and ds.checkpoint is None
    assert ds.attrs is None and ds.clauses == {}
    with pytest.raises(ConfigError, match="dim 64 not in dims"):
        load_dataset(TEXT, 64)


def test_load_dataset_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("data_dir: d\ndims: [16]\ncheckpoint: c\ncontent_dir: x\n")
    with pytest.raises(ConfigError, match="exactly one of checkpoint / content_dir"):
        load_dataset(bad, 16)
    bad.write_text("data_dir: d\ndims: [16]\ncheckpoint: c\nsplit: test\n")
    with pytest.raises(ConfigError, match=r"unknown keys \['split'\]"):
        load_dataset(bad, 16)
    bad.write_text(
        "data_dir: d\ndims: [16]\ncheckpoint: c\nfilters: {attrs: a, clause: {s: [0, x]}}\n"
    )
    with pytest.raises(ConfigError, match="clause/s: clauses must be ints"):
        load_dataset(bad, 16)
    bad.write_text("data_dir: d\ndims: [16]\ncheckpoint: c\nusers_limit: 0\n")
    with pytest.raises(ConfigError, match="users_limit"):
        load_dataset(bad, 16)


# ----- suite expansion on the fixture --------------------------------------------------


def _counts(jobs, *fields):
    return Counter(tuple(getattr(j, f) for f in fields) for j in jobs)


def test_quality_suite_collapses_same_path_backends():
    jobs = load_matrix(MINI, SUITES, "quality")
    # per dim: linr_v1 [triton, torch] → 1 (cublas), linr_v3 → 2, silvertorch → 3.
    assert len(jobs) == 2 * (1 + 2 + 3)
    assert _counts(jobs, "algo", "backend", "path") == Counter(
        {
            ("linr_v1_filter_mask", "triton", "cublas"): 2,
            ("linr_v3", "triton", "triton"): 2,
            ("linr_v3", "torch", "torch"): 2,
            ("silvertorch", "triton", "triton"): 2,
            ("silvertorch", "torch", "torch"): 2,
            ("silvertorch", "official", "official"): 2,
        }
    )
    j = jobs[0]
    assert (j.filter_kind, j.sweep, j.clauses, j.seed) == ("none", NONE_SWEEP, None, 0)
    assert j.ks == (10, 50) and j.batch_sizes == (1,) and j.build == {} and j.query == ({},)
    assert j.bloom == {"m_bits": 64, "k_hash": 3}  # top-level default, no suite override
    assert j.cells() == [{}]
    assert j.key() == {
        "dataset": "mini",
        "dim": 16,
        "suite": "quality",
        "filter_kind": "none",
        "sweep": NONE_SWEEP,
        "algo": "linr_v1_filter_mask",
        "backend": "triton",
        "params": {},
        "seed": 0,
    }
    assert j.group == ("mini", 16, "linr_v1_filter_mask", "triton")
    # Jobs come out grouped by the process boundary.
    groups = [j.group for j in jobs]
    assert groups == sorted(groups, key=groups.index)  # each group contiguous


def test_filter_suite_counts_keys_and_headline_seeds():
    jobs = load_matrix(MINI, SUITES, "filter")
    # (fk, sweep) pairs per dim: clause c0, c0c1 (c9_off disabled) + bloom c0 = 3.
    # algos per pair: linr_v2 2 paths, linr_v3 1, silvertorch 2 = 5.
    # seeds: c0 at dim 32 (clause and bloom) → 3 seeds; everything else 1.
    assert len(jobs) == 3 * 5 + (5 + 15 + 15)
    assert _counts(jobs, "dim", "filter_kind", "sweep") == Counter(
        {
            (16, "clause", "c0"): 5,
            (16, "clause", "c0c1"): 5,
            (16, "bloom", "c0"): 5,
            (32, "clause", "c0"): 15,
            (32, "clause", "c0c1"): 5,
            (32, "bloom", "c0"): 15,
        }
    )
    assert {j.seed for j in jobs if j.dim == 32 and j.sweep == "c0"} == {0, 1, 2}
    assert {j.seed for j in jobs if not (j.dim == 32 and j.sweep == "c0")} == {0}
    assert {(j.algo, j.backend, j.path) for j in jobs} == {
        ("linr_v2", "triton", "triton"),
        ("linr_v2", "torch", "torch"),
        ("linr_v3", "torch", "torch"),
        ("silvertorch", "triton", "triton"),
        ("silvertorch", "official", "official"),
    }
    assert all(j.ks == (10, 50) and j.batch_sizes == (1, 2) for j in jobs)
    assert {j.clauses for j in jobs if j.sweep == "c0c1"} == {(0, 1)}
    assert len(
        {(j.key()["sweep"], j.key()["seed"], j.algo, j.backend, j.dim, j.filter_kind) for j in jobs}
    ) == len(jobs)


def test_deep_suite_build_query_split():
    jobs = load_matrix(MINI, SUITES, "deep")
    st = [j for j in jobs if j.algo == "silvertorch"]
    v3 = [j for j in jobs if j.algo == "linr_v3"]
    # silvertorch: 2 builds × 2 seeds; n_probe ∈ {4, 16, 64} valid per build: {4} | {4, 16, 64}.
    assert len(st) == 4 and len(v3) == 2 and len(jobs) == 6
    by_build = {(j.build["n_lists"], j.seed): j.query for j in st}
    assert by_build[8, 0] == ({"n_probe": 4},)
    assert by_build[64, 1] == ({"n_probe": 4}, {"n_probe": 16}, {"n_probe": 64})
    assert st[-1].cells() == [
        {"n_lists": 64, "n_probe": 4},
        {"n_lists": 64, "n_probe": 16},
        {"n_lists": 64, "n_probe": 64},
    ]
    assert st[-1].key({"n_lists": 64, "n_probe": 16})["params"] == {"n_lists": 64, "n_probe": 16}
    assert v3[0].build == {} and v3[0].query == ({"candidate_pool": 20}, {"candidate_pool": 40})
    assert {j.dim for j in jobs} == {32} and {j.filter_kind for j in jobs} == {"bloom"}
    assert {j.seed for j in jobs} == {0, 1}
    assert jobs[0].bloom == {"m_bits": 128, "k_hash": 3}  # suite override on top of the default


def test_narrows_apply_before_collapse():
    jobs = load_matrix(
        MINI,
        SUITES,
        "quality",
        dims=[16],
        backends=["torch"],
        algos=["linr_v1_filter_mask", "silvertorch"],
    )
    assert [(j.algo, j.backend, j.path) for j in jobs] == [
        ("linr_v1_filter_mask", "torch", "cublas"),
        ("silvertorch", "torch", "torch"),
    ]
    jobs = load_matrix(MINI, SUITES, "filter", sweeps=["c0c1"], seeds=[1], ks=[10], batch_sizes=[2])
    assert jobs == [] or all(j.sweep == "c0c1" for j in jobs)
    assert jobs == []  # c0c1 only has seed 0
    jobs = load_matrix(
        MINI, SUITES, "filter", sweeps=["c0"], dims=[32], seeds=[1], ks=[10], batch_sizes=[2]
    )
    assert len(jobs) == 10 and all(
        j.ks == (10,) and j.batch_sizes == (2,) and j.seed == 1 for j in jobs
    )


def test_ks_and_batch_sizes_narrows_mark_jobs_narrowed():
    """``--k`` / ``--bs`` replace the suite's lists (ks [10, 50], batch_sizes [1, 2] in the
    filter suite): a different set is ``narrowed`` (→ ``partial`` records); the same set,
    in any order, is not. Cell-selecting narrows never are."""
    assert not any(j.narrowed for j in load_matrix(MINI, SUITES, "filter"))
    assert not any(j.narrowed for j in load_matrix(MINI, SUITES, "filter", dims=[16], seeds=[0]))
    assert not any(j.narrowed for j in load_matrix(MINI, SUITES, "filter", ks=[50, 10]))
    assert not any(j.narrowed for j in load_matrix(MINI, SUITES, "filter", batch_sizes=[2, 1]))
    assert all(j.narrowed for j in load_matrix(MINI, SUITES, "filter", ks=[10]))
    assert all(j.narrowed for j in load_matrix(MINI, SUITES, "filter", batch_sizes=[2]))
    assert all(j.narrowed for j in load_matrix(MINI, SUITES, "filter", ks=[10, 50, 99]))


def test_suite_errors_are_named():
    with pytest.raises(ConfigError, match="no suite 'nope'"):
        load_matrix(MINI, SUITES, "nope")
    with pytest.raises(ConfigError, match="dataset 'text' not in"):
        load_matrix(TEXT, SUITES, "filter")
    with pytest.raises(ConfigError, match="query params.*misplaced: build=\\['n_probe'\\]"):
        load_matrix(MINI, SUITES, "bad_params")
    with pytest.raises(ConfigError, match="--k"):
        load_matrix(MINI, SUITES, "quality", ks=[0])


# ----- the real configs against the old (deleted) d128 YAMLs ----------------------------

_OLD_FILTER = {  # config/<dataset>/d128-filter.yaml @ 06bc4c7: sweeps per filter kind
    "goodreads": {
        "clause": ["c0_genre", "c1_lang_reverse", "c2_format", "c3_year", "c0c1", "all4"],
        "bloom": ["c0_genre", "c2_format", "c3_year"],
        "data_dir": "data/goodreads-work-id",
        "checkpoint": "data/goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id/best_model.pt",
        "content_dir": None,
    },
    "arxiv": {
        "clause": ["c0_maincat", "c2_year", "c3_nversions", "c0c2", "all4"],
        "bloom": ["c0_maincat", "c2_year", "c3_nversions", "c0c2", "all4"],
        "data_dir": "data/arxiv-papers",
        "checkpoint": None,
        "content_dir": "data/arxiv-papers/content_d128",
    },
}
_OLD_ALGOS = ["linr_v1_filter_mask", "linr_v2", "linr_v3", "linr_v4", "silvertorch"]
_OLD_BACKENDS = ["triton", "torch"]


def _new_cells(jobs) -> set[tuple]:
    return {
        (j.algo, j.backend, j.filter_kind, j.sweep, k, bs)
        for j in jobs
        for k in j.ks
        for bs in j.batch_sizes
    }


@pytest.mark.parametrize("dataset", ["goodreads", "arxiv"])
def test_filter_suite_matches_old_d128_filter_config(dataset):
    old = _OLD_FILTER[dataset]
    jobs = [
        j
        for j in load_matrix(
            CFG / f"{dataset}.yaml", CFG / "suites.yaml", "filter", dims=[128], seeds=[0]
        )
        if j.backend != "official"
    ]
    assert _new_cells(jobs) == {
        (a, b, fk, sw, k, bs)
        for a in _OLD_ALGOS
        for b in _OLD_BACKENDS
        for fk in ("clause", "bloom")
        for sw in old[fk]
        for k in (100, 500, 1000)
        for bs in (1, 8, 16)
    }
    # Every filter cell has the same users_limit and paths the old config named.
    ds = jobs[0].data
    assert ds.users_limit == 10000
    assert str(ds.attrs) == f"{old['data_dir']}/item_attrs_narrow.pt"
    assert str(ds.reverse) == f"{old['data_dir']}/clause_is_reverse_narrow.pt"
    assert str(ds.data_dir) == old["data_dir"]
    assert ds.gt_dir == Path(old["data_dir"]) / "gt_d128"
    assert (str(ds.checkpoint) if ds.checkpoint else None) == old["checkpoint"]
    assert (str(ds.content_dir) if ds.content_dir else None) == old["content_dir"]
    # Headline sweeps carry seeds {0, 1, 2} at d128; the silvertorch n_probe grid is a query grid.
    all_jobs = load_matrix(CFG / f"{dataset}.yaml", CFG / "suites.yaml", "filter", dims=[128])
    headline = {"c0_genre", "c0_maincat", "all4"}
    assert {j.seed for j in all_jobs if j.sweep in headline} == {0, 1, 2}
    assert {j.seed for j in all_jobs if j.sweep not in headline} == {0}
    st = next(j for j in all_jobs if j.algo == "silvertorch")
    assert st.build == {} and st.query == ({"n_probe": 24}, {"n_probe": 32})
    assert st.bloom == {"m_bits": 1024, "k_hash": 5}


@pytest.mark.parametrize("dataset", ["goodreads", "arxiv"])
def test_quality_suite_matches_old_d128_quality_config_modulo_collapse(dataset):
    # Old: [linr_v1_filter_mask, linr_v4, silvertorch, linr_v3] × [triton, torch] × none;
    # linr_v1 / linr_v4 collapse to their first (cuBLAS) backend.
    jobs = [
        j
        for j in load_matrix(CFG / f"{dataset}.yaml", CFG / "suites.yaml", "quality", dims=[128])
        if j.backend != "official"
    ]
    kept = {("linr_v1_filter_mask", "triton"), ("linr_v4", "triton")} | {
        (a, b) for a in ("linr_v3", "silvertorch") for b in _OLD_BACKENDS
    }
    assert _new_cells(jobs) == {
        (a, b, "none", NONE_SWEEP, k, 1) for a, b in kept for k in (100, 200, 400)
    }
    assert len(jobs) == 6 and {j.path for j in jobs} == {"cublas", "triton", "torch"}


def test_deep_suite_builds_once_per_n_lists():
    jobs = load_matrix(
        CFG / "arxiv.yaml",
        CFG / "suites.yaml",
        "deep",
        backends=["triton"],
        seeds=[0],
        filter_kinds=["bloom"],
        sweeps=["c0_maincat"],
    )
    st = [j for j in jobs if j.algo == "silvertorch"]
    assert [j.build for j in st] == [{"n_lists": 1664}, {"n_lists": 8192}]  # 2 builds, not 12
    assert all(len(j.query) == 6 for j in st) and {q["n_probe"] for q in st[0].query} == {
        4,
        8,
        24,
        32,
        128,
        256,
    }
    assert [j.query for j in jobs if j.algo == "linr_v3"] == [
        tuple({"candidate_pool": c} for c in (2000, 4000, 8000, 16000, 32000))
    ]
    for name in ("yambda-500m", "yambda-5b"):
        q = load_matrix(CFG / f"{name}.yaml", CFG / "suites.yaml", "quality")
        assert q and all(j.filter_kind == "none" and j.data.attrs is None for j in q)
