"""``bench.report`` over a synthetic results tree: the citability verdict and its reasons, a
``failed`` record in no number, the cell lines and marks of ``report.md``, two encoders under one
dataset refused, an empty tree, a schema-1 record."""

from __future__ import annotations

import click
import pytest

from bench import records, report

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
        "qps": 1000 * bs / ms, "host_gap_ms": 0.01, "spread": 0.2 if unstable else 0.001,
        "unstable": unstable, "peak_fwd_mib": 4.0, "window_medians_ms": [ms, ms, ms],
        "sm_mhz": 1410.0, "load": "closed_loop",
    }  # fmt: skip


def _rec(dataset, algo, backend, *, params=None, status="ok", seed=0, ms=1.0, unstable=False,
         recall=0.9):  # fmt: skip
    perf = [_perf(100, bs, mode, ms * (1 + 0.1 * bs), unstable) for bs in (1, 16)
            for mode in ("eager", "graph")]  # fmt: skip
    return {
        "schema_version": records.SCHEMA_VERSION, "status": status,
        "partial_reasons": ["skip_perf"] if status == "partial" else None,
        "dataset": dataset, "dim": 128, "inputs": "sasrec-ssm-logq-d128", "suite": "filter",
        "filter_kind": "clause", "sweep": "c0_genre", "algo": algo, "backend": backend,
        "params": params or {}, "seed": seed, "path": backend, "n_items": 797084,
        "n_queries": 10000, "n_kept": 9859, "n_queries_heldout": 9859, "n_queries_oracle": 9859,
        "pass_rate": 0.33, "k_max": 100, "ks": [100], "batch_sizes": [1, 16], "build_s": 1.0,
        "index_mib": 194.6, "filter_mib": 0.0,
        "quality": {"heldout": {"recall@100": 0.2, "n": 9859},
                    "oracle": {"recall@100": recall, "ndcg@100": recall, "n": 9859}},
        "perf": None if status == "failed" else perf, "unstable": unstable,
        "memory_reserved_mib": 1220.0, "elapsed_s": 10.0, "env": dict(ENV),
        **({"stage": "build", "error": "RuntimeError: boom"} if status == "failed" else {}),
    }  # fmt: skip


@pytest.fixture
def results(tmp_path):
    p = tmp_path / "results" / "filter" / "goodreads-d128.jsonl"
    for rec in (
        _rec("goodreads", "linr_v1_filter_mask", "triton", ms=0.5),
        _rec(
            "goodreads",
            "silvertorch",
            "official-fork",
            params={"n_probe": 24},
            ms=0.3,
            unstable=True,
            recall=0.91,
        ),  # fmt: skip
        _rec("goodreads", "linr_v2", "triton", status="failed"),
        _rec("goodreads", "linr_v3", "triton", status="partial", ms=1.2, recall=0.7),
    ):
        records.append_record(p, rec)
    return tmp_path / "results"


def test_report_md_lists_every_cell_and_marks_failed_partial_unstable(results, tmp_path):
    prov = report.generate(results, tmp_path / "out")
    assert prov["status"] == {"failed": 1, "ok": 2, "partial": 1}
    md = (tmp_path / "out" / "report.md").read_text()
    cells = [ln for ln in md.split("## Cells")[1].split("## Failed")[0].splitlines()
             if ln.startswith("| goodreads")]  # fmt: skip
    assert len(cells) == 3  # the failed record is in no line
    assert "| 0.9100 | k100 bs1 eager 0.33†" in next(c for c in cells if "silvertorch" in c)
    assert "| partial* | 0.7000 |" in next(c for c in cells if "linr_v3" in c)
    assert "RuntimeError: boom" in md and "skip_perf" in md and "spread 20.0%" in md
    rows = records.read_table(tmp_path / "out" / "results.parquet")
    assert len(rows) == 3 * 4 + 1  # one row per perf entry, one for the failed record
    assert {r["perf_mode"] for r in rows} == {"eager", "graph", None}


def test_citability_is_off_by_default_and_evidence_beats_the_gate(results, tmp_path):
    assert not report.generate(results, tmp_path / "a")["citable"]
    gated = report.generate(results, tmp_path / "b", gate="D1")
    assert not gated["citable"] and any("failed" in b for b in gated["blockers"])
    for out in ("a", "b"):
        assert "NOT CITABLE" in (tmp_path / out / "report.md").read_text()

    clean = tmp_path / "clean" / "filter" / "goodreads-d128.jsonl"
    records.append_record(clean, _rec("goodreads", "linr_v1_filter_mask", "triton"))
    c = report.generate(tmp_path / "clean", tmp_path / "c", gate="D1")
    assert c["citable"] and c["blockers"] == []
    assert (
        "**CITABLE** — the caller declared gate `D1` green."
        in (tmp_path / "c" / "report.md").read_text()
    )

    for name, env, word in (
        ("dirty", {"dirty": True}, "dirty"),
        ("branch", {"git_branch": "dev/x"}, "dev/x"),
    ):
        rec = _rec("goodreads", "linr_v1_filter_mask", "triton")
        rec["env"] = {**ENV, **env}
        records.append_record(tmp_path / name / "filter" / "goodreads-d128.jsonl", rec)
        prov = report.generate(tmp_path / name, tmp_path / f"{name}-out", gate="D1")
        assert not prov["citable"] and any(word in b for b in prov["blockers"]), name


def test_each_reason_is_stated_in_the_verdict_and_the_report(results, tmp_path):
    rec = _rec("goodreads", "linr_v1_filter_mask", "triton")
    rec["env"] = {**ENV, "dirty": True, "git_branch": "dev/x"}
    records.append_record(results / "filter" / "goodreads-d128.jsonl", rec)
    prov = report.generate(results, tmp_path / "out")
    assert prov["blockers"] == [
        "no --gate given: no validation gate is declared passed for these records",
        "1 record(s) with status=failed",
        "1 record(s) with status=partial (partial_reasons: skip_perf 1)",
        "1 record(s) with env.dirty (library subtree was dirty)",
        "record(s) produced on dev/x — harness numbers from a branch are not paper material",
    ]
    assert prov["marks"] == [
        "gate not green", "1 failed", "1 partial: skip_perf 1", "1 dirty", "branch dev/x"
    ]  # fmt: skip
    md = (tmp_path / "out" / "report.md").read_text()
    assert all(f"- {b}" in md for b in prov["blockers"])


def test_two_encoders_under_one_dataset_and_dim_are_refused(results, tmp_path):
    legacy = _rec("goodreads", "linr_v2", "torch")
    del legacy["inputs"]  # a schema-2 goodreads record: gSASRec
    records.append_record(results / "filter" / "goodreads-d128.jsonl", legacy)
    with pytest.raises(click.ClickException, match="gsasrec-d128-drop0.5-id"):
        report.generate(results, tmp_path / "out")


def test_an_empty_results_tree_reports_no_records(tmp_path):
    (tmp_path / "results" / "filter").mkdir(parents=True)
    prov = report.generate(tmp_path / "results", tmp_path / "out")
    assert prov["n_records"] == 0 and "no records" in prov["blockers"]
    assert "NOT CITABLE" in (tmp_path / "out" / "report.md").read_text()


def test_it_reads_schema_1_records_without_the_under_load_clock_fields(tmp_path):
    p = tmp_path / "old" / "filter" / "goodreads-d128.jsonl"
    rec = _rec("goodreads", "silvertorch", "official", params={"n_probe": 24})
    rec["schema_version"] = 1
    rec["env"] = {k: v for k, v in ENV.items() if k not in ("sm_mhz_load", "clocks_drift")}
    rec["env"]["sm_mhz"] = 1155.0
    records.append_record(p, rec)
    prov = report.generate(tmp_path / "old", tmp_path / "out")
    assert prov["schema_versions"] == [1] and prov["n_records"] == 1
