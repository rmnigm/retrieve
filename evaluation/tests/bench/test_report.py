"""``bench.report``: the table-generating path over a synthetic results tree.

The gate this pins: the columns ``report.py`` reads out of ``records.aggregate`` still exist,
every emitted fragment is structurally balanced LaTeX with the thesis's labels, a
``failed`` record never reaches a number, and the citability marker is on unless a gate is
declared *and* the evidence allows it.
"""

from __future__ import annotations

import re

import pytest

from bench import records, report

# Everything a table reads out of results.parquet. A schema change that drops one breaks here.
REQUIRED_COLUMNS = (
    *records.KEY_FIELDS, "status", "path", "n_items", "pass_rate", "index_mib",
    "env_code_version", "env_commit", "env_dirty",
    "perf_k", "perf_bs", "perf_mode", "perf_median_ms", "perf_mean_ms", "perf_p99_ms",
    "perf_qps", "perf_spread", "perf_unstable", "perf_sm_mhz",
    "oracle_recall@100", "heldout_recall@100", "quality_jaccard_vs_first@100",
    "quality_score_max_abs_diff",
)  # fmt: skip

ENV = {
    "gpu": "NVIDIA A100-SXM4-80GB", "driver": "580.159.04", "cuda": "12.8",
    "torch": "2.10.0+cu128", "triton": "3.6.0", "commit": "abc1234", "dirty": False,
    "git_branch": "development", "code_version": "f" * 40, "host": "box", "python": "3.11.15",
    "started": "2026-09-15T08:00:00+00:00", "sm_mhz_idle": 1155.0, "sm_mhz_load": 1410.0,
    "clocks_drift": False,
}  # fmt: skip


def _perf(k, bs, mode, ms, unstable=False):
    return {
        "k": k, "bs": bs, "mode": mode, "n": 1000, "median_ms": ms, "mean_ms": ms,
        "p95_ms": ms * 1.1, "p99_ms": ms * 1.2, "min_ms": ms * 0.9, "iqr_ms": 0.01,
        "qps": 1000 * bs / (ms * 1e-3 * 1000), "host_gap_ms": 0.01, "spread":
        0.2 if unstable else 0.001, "unstable": unstable, "peak_fwd_mib": 4.0,
        "window_medians_ms": [ms, ms, ms], "sm_mhz": 1410.0, "load": "closed_loop",
    }  # fmt: skip


def _rec(dataset, algo, backend, *, filter_kind="clause", sweep="c0_genre", params=None,
         status="ok", seed=0, ms=1.0, unstable=False, recall=0.9, jaccard=None):
    perf = [_perf(k, bs, mode, ms * (1 + 0.1 * bs), unstable)
            for k in (100,) for bs in (1, 8, 16) for mode in ("eager", "graph")]
    return {
        "schema_version": records.SCHEMA_VERSION, "status": status, "partial_reasons":
        ["skip_perf"] if status == "partial" else None,
        "dataset": dataset, "dim": 128, "suite": "filter", "filter_kind": filter_kind,
        "sweep": sweep, "algo": algo, "backend": backend, "params": params or {},
        "seed": seed, "path": backend, "n_items": 797084, "n_queries": 10000, "n_kept": 9859,
        "n_queries_heldout": 9859, "n_queries_oracle": 9859, "pass_rate": 0.33,
        "bloom_fp_rate": None, "k_max": 100, "ks": [100], "batch_sizes": [1, 8, 16],
        "build_s": 1.0, "index_mib": 194.6, "filter_mib": 0.0,
        "quality": {
            "heldout": {"recall@100": 0.2, "n": 9859},
            "oracle": {"recall@100": recall, "ndcg@100": recall, "n": 9859},
            "jaccard_vs_first@100": jaccard,
            "score_max_abs_diff": None if jaccard is None else 9.77e-3,
            "parity": "reference" if jaccard is None else "vs_triton",
        },
        "perf": None if status == "failed" else perf,
        "unstable": unstable, "memory_reserved_mib": 1220.0, "elapsed_s": 10.0,
        "env": dict(ENV),
        **({"stage": "build", "error": "RuntimeError: boom"} if status == "failed" else {}),
    }  # fmt: skip


@pytest.fixture
def results(tmp_path):
    p = tmp_path / "results" / "filter" / "goodreads-d128.jsonl"
    for rec in (
        _rec("goodreads", "linr_v1_filter_mask", "triton", ms=0.5),
        _rec("goodreads", "linr_v1_filter_mask", "torch", ms=0.8, jaccard=1.0),
        _rec("goodreads", "silvertorch", "triton", params={"n_probe": 24}, ms=0.3,
             unstable=True, recall=0.91),
        _rec("goodreads", "silvertorch", "triton", params={"n_probe": 32}, ms=0.4,
             recall=0.94),
        _rec("goodreads", "linr_v2", "triton", status="failed"),
        _rec("goodreads", "linr_v3", "triton", status="partial", ms=1.2, recall=0.7),
    ):
        records.append_record(p, rec)
    q = tmp_path / "results" / "quality" / "goodreads-d128.jsonl"
    records.append_record(q, {**_rec("goodreads", "linr_v1_filter_mask", "triton",
                                     filter_kind="none", sweep="full_scan"), "suite": "quality"})
    samples = records.samples_path(p)
    records.append_record(samples, {
        "dataset": "goodreads", "dim": 128, "suite": "filter", "filter_kind": "clause",
        "sweep": "c0_genre", "algo": "linr_v1_filter_mask", "backend": "triton", "params": {},
        "seed": 0, "k": 100, "bs": 1, "mode": "eager", "ms": [0.5, 0.51, 0.49, 0.52],
    })
    return tmp_path / "results"


def _generate(results, out, **kw):
    return report.generate(results, out, dim=128, k=100, bs=1, compare_bs=16, mode="eager",
                           backend="triton", sweep=None, batch_dataset=None,
                           budgets=(1.0, 5.0), **kw)


def test_results_parquet_still_carries_every_column_the_tables_read(results, tmp_path):
    c = _generate(results, tmp_path / "out")
    missing = [col for col in REQUIRED_COLUMNS if col not in c.rows[0]]
    assert not missing, f"records.aggregate no longer emits: {missing}"


def test_every_artifact_is_emitted_and_the_latex_is_structurally_sound(results, tmp_path):
    c = _generate(results, tmp_path / "out")
    names = {p.name for p in c.written}
    assert {"results.parquet", "report.md", "methodology.tex"} <= names
    assert {f"tab-{n}.tex" for n in ("recall_nofilter", "pareto_goodreads", "batch_scaling",
                                     "memory", "backend_parity", "recall_at_budget",
                                     "paper_comparison")} <= names
    assert sum(1 for p in c.written if p.suffix == ".png") >= 5
    labels = set()
    for path in [p for p in c.written if p.suffix == ".tex"]:
        text = path.read_text()
        body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("%"))
        for env in ("table", "tabular", "itemize", "minipage"):
            assert body.count(f"\\begin{{{env}}}") == body.count(f"\\end{{{env}}}"), path
        assert body.count("{") == body.count("}"), f"unbalanced braces in {path}"
        assert body.count("$") % 2 == 0, f"unbalanced math in {path}"
        labels |= set(re.findall(r"\\label\{([^}]*)\}", body))
        stripped = re.sub(r"\$[^$]*\$|\\(?:label|texttt|ref)\{[^}]*\}", "", body)
        assert "_" not in stripped.replace("\\_", ""), f"unescaped underscore in {path}"
    assert {"tab:recall_nofilter", "tab:pareto_goodreads", "tab:batch_scaling",
            "tab:memory"} <= labels, "a thesis label went missing (docs/thesis/main.tex)"


def test_failed_is_excluded_partial_and_unstable_are_marked(results, tmp_path):
    c = _generate(results, tmp_path / "out")
    pareto = (tmp_path / "out" / "tables" / "tab-pareto_goodreads.tex").read_text()
    assert "LiNR V2" not in pareto  # a failed cell never reaches a number, backend or not
    assert "^{\\dagger}" in pareto  # the unstable silvertorch cell is marked
    assert "^{*}" in pareto  # the partial linr_v3 cell is marked
    md = (tmp_path / "out" / "report.md").read_text()
    assert "RuntimeError: boom" in md and "skip_perf" in md
    assert c.prov["status"] == {"failed": 1, "ok": 5, "partial": 1}


def test_citability_is_off_by_default_and_evidence_beats_the_gate(results, tmp_path):
    ungated = _generate(results, tmp_path / "a")
    assert not ungated.prov["citable"]
    gated = _generate(results, tmp_path / "b", gate="D1")
    assert not gated.prov["citable"], "a failed/partial record must veto --gate"
    assert any("failed" in b for b in gated.prov["blockers"])
    for out in (tmp_path / "a", tmp_path / "b"):
        assert "NOT CITABLE" in (out / "tables" / "tab-memory.tex").read_text()

    clean = tmp_path / "clean" / "filter" / "goodreads-d128.jsonl"
    records.append_record(clean, _rec("goodreads", "linr_v1_filter_mask", "triton"))
    c = _generate(tmp_path / "clean", tmp_path / "c", gate="D1")
    assert c.prov["citable"] and c.prov["blockers"] == []
    tex = (tmp_path / "c" / "tables" / "tab-memory.tex").read_text()
    assert "NOT CITABLE" not in tex and "gate D1" in tex

    dirty = tmp_path / "dirty" / "filter" / "goodreads-d128.jsonl"
    rec = _rec("goodreads", "linr_v1_filter_mask", "triton")
    rec["env"] = {**ENV, "dirty": True}
    records.append_record(dirty, rec)
    d = _generate(tmp_path / "dirty", tmp_path / "d", gate="D1")
    assert not d.prov["citable"] and any("dirty" in b for b in d.prov["blockers"])

    branch = tmp_path / "branch" / "filter" / "goodreads-d128.jsonl"
    rec = _rec("goodreads", "linr_v1_filter_mask", "triton")
    rec["env"] = {**ENV, "git_branch": "dev/c4-gate-rerun"}
    records.append_record(branch, rec)
    b = _generate(tmp_path / "branch", tmp_path / "e", gate="D1")
    assert not b.prov["citable"], "rule 2: a number from a dev/* branch is not citable"
    assert any("dev/c4-gate-rerun" in x for x in b.prov["blockers"])


def test_an_empty_results_tree_emits_placeholders_rather_than_failing(tmp_path):
    (tmp_path / "results" / "filter").mkdir(parents=True)
    c = _generate(tmp_path / "results", tmp_path / "out")
    assert c.prov["n_records"] == 0
    tex = (tmp_path / "out" / "tables" / "tab-memory.tex").read_text()
    assert "no matching cells" in tex and "NOT CITABLE" in tex


def test_it_reads_schema_1_records_without_the_c5_clock_fields(results, tmp_path):
    """C4's records are schema 1: no ``env.sm_mhz_load``, an ``env.sm_mhz`` that means a
    whole-run median. The tables must still build, and must not use that field."""
    p = tmp_path / "old" / "filter" / "goodreads-d128.jsonl"
    rec = _rec("goodreads", "silvertorch", "triton", params={"n_probe": 24})
    rec["schema_version"] = 1
    rec["env"] = {k: v for k, v in ENV.items() if k not in ("sm_mhz_load", "clocks_drift")}
    rec["env"]["sm_mhz"] = 1155.0
    records.append_record(p, rec)
    c = _generate(tmp_path / "old", tmp_path / "out")
    assert c.prov["schema_versions"] == [1]
    note = (tmp_path / "out" / "tables" / "tab-pareto_goodreads.tex").read_text()
    assert "1410--1410" in note  # the under-load per-variant sample, not the 1155 median
