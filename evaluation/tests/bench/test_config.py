"""CPU-only tests for ``bench.config``'s harness-v2 matrix (H §6 WP-2, §8.4).

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

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from bench import cli, run
from bench.algos import official_config
from bench.config import NONE_SWEEP, ConfigError, interleave_units, load_dataset, load_matrix
from bench.records import resume_key
from eval_datasets import synth_filter

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


@pytest.mark.parametrize("parent", ["goodreads", "arxiv", "yfcc10m"])
def test_synth_dataset_shares_the_parents_inputs(parent):
    """``<parent>-synth.yaml``: the parent's items, queries, encoder (so its encode cache) and
    users_limit; the synth attrs and query attrs; sweep names disjoint from the parent's in
    the shared ``gt_d{dim}``."""
    dim = load_dataset(CFG / f"{parent}.yaml", 128 if parent != "yfcc10m" else 192).dim
    p, s = (load_dataset(CFG / f"{n}.yaml", dim) for n in (parent, f"{parent}-synth"))
    for f in ("dim", "data_dir", "checkpoint", "content_dir", "users_limit", "encode", "inputs"):
        assert getattr(s, f) == getattr(p, f), f
    assert s.gt_dir == p.gt_dir and s.name == f"{parent}-synth"
    assert (s.attrs.name, s.query_attrs.name, s.reverse) == (
        "item_attrs_synth.pt",
        "query_attrs_synth.pt",
        None,
    )
    assert p.query_attrs == p.data_dir / "eval_split.parquet"
    names = ["p0001", "p0003", "p001", "p003", "p005", "p01", "p02", "p03", "p05", "p1"]
    assert len(names) == len(synth_filter.RATES)
    want = {r: (j,) for j, r in enumerate(names)}
    assert s.clauses == {"clause": want, "bloom": want}
    assert not {sw for fk in p.clauses.values() for sw in fk} & set(want)


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
        "inputs": "d16",
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


def test_filter_suite_arms_sweeps_and_seeds():
    jobs = load_matrix(MINI, SUITES, "filter")
    # (fk, sweep) pairs per dim: clause c0, c0c1 (c9_off disabled) + bloom c0 = 3. Per pair and
    # seed: linr_v2 2 paths on all 3; linr_v3 (sweeps [c0]) on 2; silvertorch triton on 3;
    # official (bloom only) on 1 → 6 + 2 + 3 + 1 = 12 per dim and seed.
    assert len(jobs) == 2 * 3 * 12
    assert _counts(jobs, "dim", "filter_kind", "sweep") == Counter(
        {
            (d, fk, sw): n
            for d in (16, 32)
            for (fk, sw), n in {
                ("clause", "c0"): 12,
                ("clause", "c0c1"): 9,
                ("bloom", "c0"): 15,
            }.items()
        }
    )
    assert {j.seed for j in jobs} == {0, 1, 2}
    assert not any(j.backend == "official" and j.filter_kind == "clause" for j in jobs)
    assert {j.sweep for j in jobs if j.algo == "linr_v3"} == {"c0"}
    assert all(j.ks == (10, 50) and j.batch_sizes == (1, 2) for j in jobs)
    assert {j.clauses for j in jobs if j.sweep == "c0c1"} == {(0, 1)}
    keys = [(j.dim, j.algo, j.backend, j.filter_kind, j.sweep, j.seed) for j in jobs]
    assert len(set(keys)) == len(jobs)


def test_widths_compile_frac_ks_by_sweep_and_empty_slots():
    jobs = load_matrix(MINI, SUITES, "widths")
    st_bloom = [
        j for j in jobs if (j.algo, j.backend, j.filter_kind) == ("silvertorch", "triton", "bloom")
    ]
    # m_bits gridded: the job's bloom carries it, the suite default fills k_hash.
    assert [(j.build, j.bloom) for j in st_bloom] == [
        ({"m_bits": 64}, {"m_bits": 64, "k_hash": 3}),
        ({"m_bits": 256}, {"m_bits": 256, "k_hash": 3}),
    ]
    # Not gridded: the key's params stay {} and the bloom is the default, as before.
    plain = [
        j for j in jobs if (j.algo, j.backend, j.filter_kind) == ("silvertorch", "triton", "clause")
    ]
    assert plain and all(j.build == {} for j in plain)
    assert all(j.bloom == {"m_bits": 64, "k_hash": 3} for j in plain)
    compiled = [j for j in jobs if j.backend == "torch" and j.algo == "silvertorch"]
    assert compiled and all(j.build == {"compile": "max-autotune"} for j in compiled)
    v3 = [j for j in jobs if j.algo == "linr_v3"]
    assert v3 and all(j.query == ({"candidate_pool_frac": 0.1},) for j in v3)
    assert not any(j.algo == "linr_v2" for j in jobs)  # `datasets: {}` runs nowhere
    assert {j.sweep: j.ks for j in jobs if j.filter_kind == "clause"} == {
        "c0": (10, 50),
        "c0c1": (10,),
    }
    assert {j.seed for j in jobs if j.sweep == "c0"} == {0}
    assert {j.seed for j in jobs if j.sweep == "c0c1"} == {0, 2}
    assert not any(j.narrowed for j in jobs)
    narrowed = load_matrix(MINI, SUITES, "widths", ks=[10])
    assert {j.sweep: j.narrowed for j in narrowed if j.filter_kind == "clause"} == {
        "c0": True,
        "c0c1": False,  # --k 10 is that sweep's own list
    }


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


def test_codesign_suite_carries_bloom_path_in_the_key():
    jobs = load_matrix(MINI, SUITES, "codesign")
    assert [(j.backend, j.filter_kind, j.build) for j in jobs] == [
        ("official", "bloom", {"n_lists": 8, "bloom_path": "partial"}),
        ("official", "bloom", {"n_lists": 8, "bloom_path": "full"}),
    ]
    assert jobs[1].key({"n_lists": 8, "bloom_path": "full", "n_probe": 4})["params"] == {
        "n_lists": 8,
        "bloom_path": "full",
        "n_probe": 4,
    }
    assert all("bloom_path" not in j.build for j in load_matrix(MINI, SUITES, "filter"))


def test_bloom_path_off_the_official_bloom_path_is_a_config_error():
    with pytest.raises(ConfigError, match="bloom_path applies to silvertorch/bloom on official"):
        load_matrix(MINI, SUITES, "bad_bloom_path")


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
    # per dim: linr_v2 on two paths + silvertorch triton (no bloom c0c1, linr_v3 is c0-only)
    assert len(jobs) == 6 and all(j.sweep == "c0c1" and j.seed == 1 for j in jobs)
    jobs = load_matrix(
        MINI, SUITES, "filter", sweeps=["c0"], dims=[32], seeds=[1], ks=[10], batch_sizes=[2]
    )
    assert len(jobs) == 9 and all(
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


@pytest.mark.parametrize(
    ("arms", "match"),
    [
        (
            "[{algo: silvertorch, backends: [triton], build: {m_bits: [64]}}]",
            "silvertorch/bloom only",
        ),
        (
            "[{algo: silvertorch, backends: [triton], build: {compile: [max-autotune]}}]",
            "torch arms only",
        ),
        (
            "[{algo: silvertorch, backends: [torch], build: {compile: [default]}}]",
            "torch arms only",
        ),
        (
            "[{algo: silvertorch, backends: [triton], query: {candidate_pool_frac: [0.1]}}]",
            "linr_v3",
        ),
        ("[{algo: linr_v2, backends: [triton], sweeps: [nope]}]", r"sweeps \['nope'\] not in mini"),
        (
            "[{algo: linr_v2, backends: [triton]}, {algo: linr_v2, backends: [triton]}]",
            "a cell twice",
        ),
        ("[{algo: linr_v2, backends: [triton], datasets: {other: {}}}]", "datasets \\['other'\\]"),
        (
            "[{algo: linr_v2, backends: [triton], filter_kinds: [exact]}]",
            "filter_kinds \\['exact'\\]",
        ),
        ("[{algo: linr_v9, backends: [triton]}]", "unknown algo 'linr_v9'"),
        (
            "[{algo: linr_v1_filter_mask, backends: [triton]},"
            " {algo: linr_v1_filter_mask, backends: [torch], filter_kinds: [none]}]",
            None,  # the cuBLAS path collapses across arms, logged, not an error
        ),
    ],
)
def test_arm_errors_are_named(tmp_path, arms, match):
    suites = tmp_path / "suites.yaml"
    suites.write_text(
        "s:\n  datasets: [mini]\n  filter_kinds: [none, clause, bloom]\n  ks: [10]\n"
        f"  batch_sizes: [1]\n  arms: {arms}\n"
    )
    if match is None:
        jobs = load_matrix(MINI, suites, "s", filter_kinds=["none"])
        assert [(j.algo, j.backend, j.path) for j in jobs] == [
            ("linr_v1_filter_mask", "triton", "cublas")
        ] * 2
        return
    with pytest.raises(ConfigError, match=match):
        load_matrix(MINI, suites, "s")


# ----- the real grid (campaign-v2, user decisions 2026-10-08) ---------------------------

GRID = {  # (suite, dataset): (jobs, cells), the planner's GPU-h input; change it deliberately
    ("h2h", "goodreads"): (30, 30),
    ("h2h", "arxiv"): (30, 30),
    ("filter", "goodreads"): (87, 114),
    ("filter", "arxiv"): (135, 180),
    ("filter", "yfcc10m"): (15, 21),
    ("filter", "pubmed"): (63, 90),
    ("deep", "goodreads"): (42, 210),
    ("deep", "arxiv"): (72, 360),
    ("deep", "yfcc10m"): (9, 45),
    ("synth", "goodreads-synth"): (312, 558),
    ("synth", "arxiv-synth"): (25, 46),
    ("synth", "arxiv-corr-synth"): (15, 42),
    ("synth", "yfcc10m-synth"): (15, 25),
    ("codesign", "arxiv"): (36, 108),  # official + triton (C5-OURS)
    ("codesign", "goodreads"): (36, 108),
    ("bloomwidth", "goodreads"): (42, 42),
    ("bloomwidth", "arxiv"): (126, 126),
    ("bloomwidth", "pubmed"): (42, 42),
    ("bloomwidth-timed", "goodreads"): (21, 21),
    ("bloomwidth-timed", "arxiv"): (63, 63),
    ("bloomwidth-timed", "pubmed"): (21, 21),
    ("v3bits", "goodreads-synth"): (84, 168),
    ("v3bits", "goodreads"): (24, 48),
    ("v3bits", "pubmed"): (18, 36),
    ("router", "goodreads"): (72, 72),
    ("router", "pubmed"): (36, 36),
    ("laion30m", "laion30m"): (6, 10),  # claims first: seed 0 (user 2026-10-10)
    ("laion30m-bs1", "laion30m"): (2, 6),
}
KEPT = {
    "goodreads": {"c0_genre", "c1_lang_reverse", "all4"},
    "arxiv": {"c3_nversions", "c0_maincat", "all4"},
    "pubmed": {"c0_mesh", "c3_journal_reverse", "all5"},
    "yfcc10m": {"tags_and"},
}
SYNTH_N = {"goodreads-synth": 797_084, "arxiv-synth": 2_988_996, "arxiv-corr-synth": 2_988_996,
           "yfcc10m-synth": 10_000_000}  # fmt: skip
RATE = {"p0001": 0.001, "p0003": 0.003, "p001": 0.01, "p003": 0.03, "p005": 0.05, "p01": 0.1,
        "p02": 0.2, "p03": 0.3, "p05": 0.5, "p1": 1.0,
        "c001": 0.01, "c003": 0.03, "c01": 0.1}  # fmt: skip


def _real(suite: str, dataset: str, **kw):
    return load_matrix(CFG / f"{dataset}.yaml", CFG / "suites.yaml", suite, **kw)


def test_grid_covers_every_suite_and_dataset():
    suites = yaml.safe_load((CFG / "suites.yaml").read_text())
    listed = {(s, d) for s, v in suites.items() if isinstance(v, dict) and "datasets" in v
              for d in v["datasets"]}  # fmt: skip
    assert listed == set(GRID)


@pytest.mark.parametrize(("suite", "dataset"), list(GRID))
def test_grid_counts_and_invariants(suite, dataset):
    """G-grid: the job / cell counts and the user's 2026-10-08 rules on every expanded cell."""
    jobs = _real(suite, dataset)
    assert (len(jobs), sum(len(j.query) for j in jobs)) == GRID[suite, dataset]
    cells = [(j, {**j.build, **q}) for j in jobs for q in j.query]
    by_seed: dict[tuple, set[int]] = {}
    for j, p in cells:
        cell = (j.dim, j.algo, j.backend, j.filter_kind, j.sweep, json.dumps(p, sort_keys=True))
        by_seed.setdefault(cell, set()).add(j.seed)
    spec = yaml.safe_load((CFG / "suites.yaml").read_text())[suite]
    by_sweep = (spec.get("seeds_by_sweep") or {}).get(dataset, {})
    for (
        cell,
        got,
    ) in by_seed.items():  # h2h: 5 repeats; synth: seed 0, 0-2 where variance is the question
        assert got == set(by_sweep.get(cell[4], spec["seeds"])), cell
    assert all(set(j.batch_sizes) <= {1, 16} and set(j.ks) <= {100, 1000} for j in jobs)
    if suite in ("filter", "deep", "synth", "codesign"):
        assert all(j.batch_sizes == (1, 16) for j in jobs)
    assert not any(j.backend == "official" and j.filter_kind == "clause" for j in jobs)
    assert {j.algo for j in jobs} <= {
        "linr_v1_filter_mask",
        "linr_v2",
        "linr_v3",
        "silvertorch",
        "postfilter",
        "router",
    }
    assert not any(p.get("n_probe") == 4 for _, p in cells)
    tuned = IVF.get(dataset, (None, None))[1]
    assert suite == "synth" or not any(p.get("n_probe") == 256 != tuned for _, p in cells)
    assert not any(j.narrowed for j in jobs)
    if suite in ("filter", "deep") and dataset in KEPT:
        assert {j.sweep for j in jobs} == KEPT[dataset] or (
            suite == "deep" and {j.sweep for j in jobs} <= KEPT[dataset]
        )
    c3_torch = {j.algo for j in jobs if j.backend == "torch" and j.algo != "postfilter"}
    if suite == "filter":  # addendum 2: C3's torch arms on goodreads + arxiv only
        assert bool(c3_torch) == (dataset in ("goodreads", "arxiv"))
    if suite == "synth":  # SYNTH-TRIM sized by claim (controller 2026-10-10)
        assert bool(c3_torch) == (dataset == "goodreads-synth")
        for j in jobs:
            big_enough = SYNTH_N[dataset] * RATE[j.sweep] >= 4 * 1000
            assert (1000 in j.ks) == (big_enough and dataset != "yfcc10m-synth"), j.sweep
            if j.backend == "torch" and j.algo != "postfilter":
                assert j.sweep in {"p001", "p01", "p1"}
        torch_arms = {(j.algo, json.dumps(j.build)) for j in jobs if j.backend == "torch"}
        assert torch_arms == {
            "goodreads-synth": {
                ("postfilter", "{}"),
                ("linr_v1_filter_mask", "{}"),
                ("linr_v2", "{}"),
                ("linr_v1_filter_mask", '{"compile": "max-autotune"}'),
                ("linr_v2", '{"compile": "max-autotune"}'),
            },
            "arxiv-corr-synth": {("postfilter", "{}")},
        }.get(dataset, set())
        assert (
            {j.algo for j in jobs if j.algo.startswith("linr_v")}
            & {"linr_v1_filter_mask", "linr_v2"}
        ) == (set() if dataset == "arxiv-corr-synth" else {"linr_v1_filter_mask", "linr_v2"})
        assert any(j.algo == "linr_v3" for j in jobs) == (
            dataset in ("goodreads-synth", "arxiv-corr-synth")
        )
        full = (24, 64, 128, 256, 512, 1024)
        st = {(j.backend, j.filter_kind): tuple(q["n_probe"] for q in j.query)
              for j in jobs if j.algo == "silvertorch"}  # fmt: skip
        assert st == {
            "goodreads-synth": {("triton", "clause"): full, ("triton", "bloom"): (24, 256),
                                ("official", "bloom"): (24, 256)},
            "arxiv-synth": {("triton", "clause"): (24, 64, 256, 1024), ("triton", "bloom"): (24,),
                            ("official", "bloom"): (24,)},
            "arxiv-corr-synth": {("triton", "clause"): full},
            "yfcc10m-synth": {("triton", "clause"): (24, 256, 1024)},
        }[dataset]  # fmt: skip
        bloom = {j.sweep for j in jobs if j.filter_kind == "bloom"}
        if dataset == "arxiv-synth":
            assert bloom == {"p001", "p1"}


IVF_ARXIV = ({"n_probe": 24}, {"n_probe": 256})


def test_filter_suite_arms():
    jobs = _real("filter", "arxiv", seeds=[0])
    arms = {
        (j.algo, j.backend, j.filter_kind, json.dumps(j.build), json.dumps(j.query)) for j in jobs
    }
    want = set()
    for fk in ("clause", "bloom"):
        for a in ("linr_v1_filter_mask", "linr_v2", "linr_v3"):
            want.add((a, "triton", fk, "{}", json.dumps(({},))))
        want.add(("silvertorch", "triton", fk, '{"n_lists": 2048}', json.dumps(IVF_ARXIV)))
        want.add(("silvertorch", "torch", fk, '{"n_lists": 2048}', json.dumps(({"n_probe": 24},))))
        want.add(
            (
                "silvertorch",
                "torch",
                fk,
                '{"compile": "max-autotune", "n_lists": 2048}',
                json.dumps(({"n_probe": 24},)),
            )
        )
        want.add(("postfilter", "torch", fk, "{}", json.dumps(({"alpha": 1}, {"alpha": 8}))))
    want.add(("silvertorch", "official", "bloom", '{"n_lists": 2048}', json.dumps(IVF_ARXIV)))
    assert arms == want
    ds = jobs[0].data
    assert ds.users_limit == 10000 and ds.gt_dir == Path("data/arxiv-papers/gt_d128")
    assert all(j.bloom == {"m_bits": 1024, "k_hash": 5} for j in jobs)


def test_deep_suite_per_dataset_n_lists_and_pool_fractions():
    n_lists = {"goodreads": [1024, 4096], "arxiv": [1664, 8192], "yfcc10m": [4096, 16384]}
    for dataset, want in n_lists.items():
        jobs = _real("deep", dataset, seeds=[0])
        st = [j for j in jobs if j.algo == "silvertorch"]
        assert sorted({j.build["n_lists"] for j in st}) == want
        assert all(j.query == tuple({"n_probe": n} for n in (8, 16, 32, 64, 128)) for j in st)
        v3 = [j for j in jobs if j.algo == "linr_v3"]
        assert v3 and all(
            j.query == tuple({"candidate_pool_frac": f} for f in (0.005, 0.01, 0.02, 0.05, 0.1))
            for j in v3
        )


def test_codesign_and_bloomwidth_suites():
    cd = _real("codesign", "goodreads", seeds=[0])
    assert {j.sweep for j in cd} == {"c0_genre", "c2_format", "c3_year"}
    assert {(j.backend, j.build["n_lists"], j.build["bloom_path"]) for j in cd} == {
        ("official", 1024, "partial"),
        ("official", 1024, "full"),
        ("triton", 1024, "partial"),
        ("triton", 1024, "full"),
    }
    assert {(j.ks, j.batch_sizes) for j in cd} == {((100,), (1, 16))}
    bw = _real("bloomwidth", "pubmed", seeds=[0])
    assert {j.sweep for j in bw} == {"c0_mesh"} and {j.backend for j in bw} == {
        "triton",
        "official",
    }
    tri = [j for j in bw if j.backend == "triton"]
    assert sorted({(j.bloom["m_bits"], j.bloom["k_hash"]) for j in tri}) == [
        (m, k) for m in (64, 128, 256, 512, 1024, 2048) for k in (3, 5)
    ]
    assert all(j.bloom == j.build and j.query == ({},) for j in tri)
    # official's width is OfficialConfig.b_multiplier, not m_bits: only k_hash is gridded
    assert sorted(j.build["k_hash"] for j in bw if j.backend == "official") == [3, 5]
    assert not any("m_bits" in j.build for j in bw if j.backend == "official")
    timed = _real("bloomwidth-timed", "pubmed", seeds=[0])
    assert {(j.ks, j.batch_sizes, j.bloom["k_hash"]) for j in timed} == {((100,), (16,), 5)}
    assert not any(j.timed for j in bw) and all(j.timed for j in timed + cd)


# G-key: resume keys of cells that exist before and after the redesign, as today's code
# (b96e1f2) computed them. Old records must keep matching (docs/system/evaluation.md § Resume).
OLD_KEYS = [
    ("filter", "arxiv", '{"algo":"linr_v3","backend":"triton","code_version":"CV","dataset":"arxiv","dim":128,"filter_kind":"bloom","inputs":"content_d128","params":{},"seed":1,"suite":"filter","sweep":"all4"}'),  # noqa: E501
    ("filter", "goodreads", '{"algo":"postfilter","backend":"torch","code_version":"CV","dataset":"goodreads","dim":128,"filter_kind":"clause","inputs":"sasrec-ssm-logq-d128","params":{"alpha":8},"seed":0,"suite":"filter","sweep":"c1_lang_reverse"}'),  # noqa: E501
    ("filter", "goodreads", '{"algo":"linr_v1_filter_mask","backend":"triton","code_version":"CV","dataset":"goodreads","dim":128,"filter_kind":"clause","inputs":"sasrec-ssm-logq-d128","params":{},"seed":2,"suite":"filter","sweep":"all4"}'),  # noqa: E501
    ("filter", "pubmed", '{"algo":"linr_v2","backend":"triton","code_version":"CV","dataset":"pubmed","dim":768,"filter_kind":"clause","inputs":"content_d768","params":{},"seed":0,"suite":"filter","sweep":"all5"}'),  # noqa: E501
    ("deep", "arxiv", '{"algo":"silvertorch","backend":"triton","code_version":"CV","dataset":"arxiv","dim":128,"filter_kind":"clause","inputs":"content_d128","params":{"n_lists":1664,"n_probe":8},"seed":2,"suite":"deep","sweep":"c0_maincat"}'),  # noqa: E501
    ("deep", "arxiv", '{"algo":"silvertorch","backend":"official","code_version":"CV","dataset":"arxiv","dim":128,"filter_kind":"bloom","inputs":"content_d128","params":{"n_lists":8192,"n_probe":128},"seed":1,"suite":"deep","sweep":"all4"}'),  # noqa: E501
    ("codesign", "arxiv", '{"algo":"silvertorch","backend":"official","code_version":"CV","dataset":"arxiv","dim":128,"filter_kind":"bloom","inputs":"content_d128","params":{"bloom_path":"full","n_lists":1664,"n_probe":32},"seed":0,"suite":"codesign","sweep":"c3_nversions"}'),  # noqa: E501
]  # fmt: skip


@pytest.mark.parametrize(("suite", "dataset", "key"), OLD_KEYS)
def test_cells_kept_by_the_redesign_keep_their_keys(suite, dataset, key):
    new = {resume_key(j.key(p), "CV") for j in _real(suite, dataset) for p in j.cells()}
    assert key in new


DATASET_YAMLS = sorted(p for p in CFG.glob("*.yaml") if p.name != "suites.yaml")
SUITE_NAMES = sorted(k for k, v in yaml.safe_load((CFG / "suites.yaml").read_text()).items()
                     if isinstance(v, dict) and "datasets" in v)  # fmt: skip


def test_the_config_globs_found_the_shipped_files():
    assert len(DATASET_YAMLS) >= 5 and len(SUITE_NAMES) >= 2, (DATASET_YAMLS, SUITE_NAMES)


@pytest.mark.parametrize("suite", SUITE_NAMES)
@pytest.mark.parametrize("dataset_yaml", DATASET_YAMLS, ids=lambda p: p.stem)
def test_every_shipped_config_parses(dataset_yaml, suite):
    """Every ``config/*.yaml`` against every suite through the real loader: a listed dataset
    expands to jobs at each of its dims the suite runs; an unlisted one is refused by name
    and still resolves at every dim it declares."""
    listed = yaml.safe_load((CFG / "suites.yaml").read_text())[suite]["datasets"]
    if dataset_yaml.stem in listed:
        jobs = load_matrix(dataset_yaml, CFG / "suites.yaml", suite)
        assert jobs and all(j.dataset == dataset_yaml.stem for j in jobs)
        return
    with pytest.raises(ConfigError, match=f"dataset {dataset_yaml.stem!r} not in"):
        load_matrix(dataset_yaml, CFG / "suites.yaml", suite)
    for dim in yaml.safe_load(dataset_yaml.read_text())["dims"]:
        assert load_dataset(dataset_yaml, dim).dim == dim


def test_interleave_groups_of_the_real_suites():
    """The comparison groups ``--interleave`` times round-robin: h2h's three arms per (filter
    kind, seed), codesign's partial / full, filter's V1 / V2 and triton / official bloom; a
    group is per seed, and every arm keeps the key it has without ``--interleave``."""
    units = interleave_units(_real("h2h", "goodreads"))
    groups = [m for by, m in units if by]
    assert len(groups) == len(units) == 2 * 5  # none + bloom, five seeds
    for m in groups:
        assert [(j.backend, j.build.get("score_path")) for j in m] == [
            ("triton", None), ("official", "fp16"), ("official", "int32")
        ]  # fmt: skip
        assert len({j.seed for j in m}) == 1 and all(j.query == ({"n_probe": 24},) for j in m)
    assert [run.group_label(j, ("backend", "score_path")) for j in groups[0]] == [
        "silvertorch/triton", "silvertorch/official/score_path=fp16",
        "silvertorch/official/score_path=int32",
    ]  # fmt: skip
    cd = [m for by, m in interleave_units(_real("codesign", "arxiv")) if by]
    assert len(cd) == 18 and all(
        [j.build["bloom_path"] for j in m] == ["partial", "full"]
        and len({j.backend for j in m}) == 1
        for m in cd
    )
    fil = [m for by, m in interleave_units(_real("filter", "arxiv")) if by]
    assert sorted({tuple(f"{j.algo}/{j.backend}" for j in m) for m in fil}) == [
        ("linr_v1_filter_mask/triton", "linr_v2/triton"),
        ("silvertorch/triton", "silvertorch/official"),
    ]
    assert not any(j.backend == "torch" for m in fil for j in m)  # values: [triton, official]
    assert cli._children(_real("filter", "arxiv"), True) == [
        ("arxiv", 128, ("linr_v1_filter_mask", "linr_v2"), ("triton",)),
        ("arxiv", 128, ("linr_v3",), ("triton",)),
        ("arxiv", 128, ("silvertorch",), ("triton", "official")),
        ("arxiv", 128, ("silvertorch",), ("torch",)),
        ("arxiv", 128, ("postfilter",), ("torch",)),
    ]
    assert len(cli._children(_real("filter", "arxiv"), False)) == 7  # one per Job.group


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("[{by: n_probe}]", "by is algo, backend or build params"),
        ("[{by: [algo, backend], values: [x]}]", "one by field only"),
        ("[{by: algo, extra: 1}]", "unknown keys"),
    ],
)
def test_interleave_errors_are_named(tmp_path, spec, match):
    suites = tmp_path / "suites.yaml"
    suites.write_text(
        "s:\n  datasets: [mini]\n  filter_kinds: [clause]\n  ks: [10]\n  batch_sizes: [1]\n"
        f"  interleave: {spec}\n  arms: [{{algo: linr_v2, backends: [triton]}}]\n"
    )
    with pytest.raises(ConfigError, match=match):
        load_matrix(MINI, suites, "s")


def test_a_job_in_two_groups_is_a_config_error(tmp_path):
    suites = tmp_path / "suites.yaml"
    suites.write_text(
        "s:\n  datasets: [mini]\n  filter_kinds: [clause]\n  ks: [10]\n  batch_sizes: [1]\n"
        "  interleave: [{by: algo}, {by: backend}]\n  arms:\n"
        "    - {algo: linr_v2, backends: [triton, torch]}\n"
        "    - {algo: linr_v1_filter_mask, backends: [triton]}\n"
    )
    with pytest.raises(ConfigError, match="in two interleave groups"):
        load_matrix(MINI, suites, "s")


def test_score_path_is_an_official_build_param():
    assert (
        official_config("silvertorch", "none", "official", {"score_path": "int32"}).score_path
        == "int32"
    )
    assert official_config("silvertorch", "bloom", "official", {}) is None
    with pytest.raises(ValueError, match="score_path applies to silvertorch/official only"):
        official_config("silvertorch", "bloom", "triton", {"score_path": "int32"})


# IVF-TUNE (docs/artifacts/campaign-v2/ivf-tune): SilverTorch n_lists / n95 per dataset; None = open
IVF = {
    "goodreads": (4096, 64),
    "arxiv": (2048, 256),
    "yfcc10m": (4096, 1024),
    "pubmed": (4096, 1024),
}


@pytest.mark.parametrize("dataset", list(IVF))
def test_filter_silvertorch_runs_at_the_tuned_ivf(dataset):
    n_lists, n95 = IVF[dataset]
    st = [j for j in _real("filter", dataset, seeds=[0]) if j.algo == "silvertorch"]
    assert st and {j.build.get("n_lists") for j in st} == {n_lists}
    for j in st:
        want = {24} if j.backend == "torch" or n95 is None else {24, n95}
        assert {q["n_probe"] for q in j.query} == want


@pytest.mark.parametrize("dataset", ["goodreads", "arxiv", "yfcc10m"])
def test_synth_silvertorch_runs_on_the_real_datasets_ivf(dataset):
    st = [j for j in _real("synth", f"{dataset}-synth", seeds=[0]) if j.algo == "silvertorch"]
    assert st and {j.build["n_lists"] for j in st} == {IVF[dataset][0]}


def test_v3bits_pubmed_is_clause_only_at_k_bits_256_and_768():
    """V3-BITS-PUBMED: PubMed d768 at k_bits 256 and 768, clause only; goodreads keeps 64 / 128."""
    pm = _real("v3bits", "pubmed")
    assert {(j.dim, j.algo, j.backend, j.filter_kind) for j in pm} == {
        (768, "linr_v3", "triton", "clause")
    }
    assert {j.build["k_bits"] for j in pm} == {256, 768}
    assert {j.sweep for j in pm} == KEPT["pubmed"]
    assert {q["candidate_pool_frac"] for j in pm for q in j.query} == {0.01, 0.05}
    gr = _real("v3bits", "goodreads")
    assert {j.build["k_bits"] for j in gr} == {64, 128}


def test_router_pubmed_runs_the_goodreads_threshold_beside_its_branches():
    """V-ROUTER PubMed: thresholds 0.05 / 0.2, n_lists 4096, both branches beside, clause."""
    pm = _real("router", "pubmed")
    assert {j.filter_kind for j in pm} == {"clause"} and {j.sweep for j in pm} == KEPT["pubmed"]
    assert {j.ks for j in pm} == {(100,)}
    assert {j.build["lq_threshold"] for j in pm if j.algo == "router"} == {0.05, 0.2}
    assert {j.build.get("n_lists") for j in pm if j.algo != "linr_v2"} == {4096}
    assert {j.algo for j in pm} == {"router", "linr_v2", "silvertorch"}
