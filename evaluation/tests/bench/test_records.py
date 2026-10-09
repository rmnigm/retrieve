"""``bench.records``: the resume key, the JSONL append / read round trip, one torn trailing
line, and ``aggregate`` → ``results.parquet`` (one row per perf entry, last record per key)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from bench import records, report
from bench.config import load_matrix

CONFIG = Path(__file__).resolve().parents[2] / "config"
FIX = Path(__file__).parent / "data"
# A real D1 arxiv record's key block and code_version (d1/arxiv, schema 2: no `inputs`).
D1_ARXIV = {
    "schema_version": 2, "dataset": "arxiv", "dim": 128, "suite": "filter",
    "filter_kind": "clause", "sweep": "c0_maincat", "algo": "linr_v1_filter_mask",
    "backend": "triton", "params": {}, "seed": 0,
    "env": {"code_version": "72e5a90c148435d070496ae59e8b4c26bb825d68"},
}  # fmt: skip

KEY = {
    "dataset": "goodreads", "dim": 128, "inputs": "sasrec-ssm-logq-d128", "suite": "filter",
    "filter_kind": "clause", "sweep": "c0_genre", "algo": "silvertorch", "backend": "triton",
    "params": {"n_probe": 24, "n_lists": 1024}, "seed": 0,
}  # fmt: skip


def test_resume_key_is_canonical_and_includes_code_version():
    a = records.resume_key(KEY, "abc")
    assert a == records.resume_key(dict(reversed(list(KEY.items()))), "abc")  # order-free
    assert a != records.resume_key(KEY, "abd")  # a kernel change invalidates the cell
    assert a != records.resume_key({**KEY, "params": {"n_probe": 32, "n_lists": 1024}}, "abc")
    assert set(json.loads(a)) == set(records.KEY_FIELDS) | {"code_version"}
    rec = {**KEY, "env": {"code_version": "abc"}}
    assert records.record_key(rec) == a


def test_append_read_and_a_torn_trailing_line(tmp_path):
    p = tmp_path / "x.jsonl"
    rec = {**KEY, "status": "ok", "env": {"code_version": "c"}, "nan": float("nan")}
    records.append_record(p, rec)
    records.append_record(p, {**rec, "seed": 1, "status": "failed", "t": torch.tensor([1, 2])})
    assert records.read_records(p)[0]["nan"] is None
    assert records.read_records(p)[1]["t"] == [1, 2]
    with open(p, "a") as f:
        f.write('{"dataset": "goodreads", "dim": 128, "sui')  # the crash
    keys = records.read_keys(p)
    assert len(keys) == 2 and keys[records.resume_key(KEY, "c")] == "ok"
    with open(p, "a") as f:
        f.write("\n" + json.dumps({**rec, "seed": 2, "nan": None}) + "\n")
    try:
        records.read_keys(p)
    except ValueError as exc:
        assert "malformed record on line 3" in str(exc)
    else:
        raise AssertionError("a torn line before the last one must raise")


def test_aggregate_one_row_per_perf_entry_last_record_per_key(tmp_path):
    p = tmp_path / "filter" / "goodreads-d128.jsonl"
    perf = [
        {"k": 100, "bs": 1, "mode": "eager", "median_ms": 1.5, "sm_mhz": 1410.0,
         "window_medians_ms": [1, 2, 3], "window_sm_mhz": [1410.0, 1410.0, 1395.0],
         "kernel_scopes": {"scorer": {"us": 30.0, "calls": 1}, "other": {"us": 5.0, "calls": 2}}},
        {"k": 100, "bs": 1, "mode": "graph", "median_ms": None, "reason": "cuda_unavailable"},
    ]  # fmt: skip
    base = {
        **KEY, "schema_version": records.SCHEMA_VERSION, "status": "ok", "path": "triton",
        "quality": {"heldout": {"recall@100": 0.5, "n": 3}, "oracle": {"recall@100": 0.9},
                    "jaccard_vs_first@100": None, "parity": "reference"},
        "perf": perf, "env": {"code_version": "c", "commit": "abc", "dirty": False,
                              "gpu": "cpu", "sm_mhz_load": 1410.0, "clocks_drift": False},
    }  # fmt: skip
    records.append_record(p, {**base, "status": "partial", "perf": None})
    records.append_record(p, base)  # the same key again: this one wins
    records.append_record(p, {**base, "seed": 1, "quality": None, "perf": None})
    records.append_record(records.samples_path(p), {**KEY, "ms": [1.0]})  # not a record
    out = records.aggregate(tmp_path)
    rows = records.read_table(out)
    assert out == tmp_path / "results.parquet" and len(rows) == 3
    seed0 = [r for r in rows if r["seed"] == 0]
    assert [r["perf_mode"] for r in seed0] == ["eager", "graph"]
    assert seed0[0]["perf_median_ms"] == 1.5 and seed0[1]["perf_median_ms"] is None
    assert seed0[0]["heldout_recall@100"] == 0.5 and seed0[0]["oracle_recall@100"] == 0.9
    assert seed0[0]["quality_parity"] == "reference" and seed0[0]["env_code_version"] == "c"
    assert json.loads(seed0[0]["params"]) == KEY["params"] and seed0[0]["status"] == "ok"
    assert seed0[0]["env_dirty"] is False and seed0[0]["perf_bs"] == 1  # typed, not strings
    assert seed0[0]["perf_window_medians_ms"] == [1, 2, 3] and "perf_window_sm_mhz" not in rows[0]
    # H-SCOPE: one us / calls column pair per kernel scope, no nested column
    assert seed0[0]["perf_kernels_scorer_us"] == 30.0 and seed0[0]["perf_kernels_other_calls"] == 2
    assert "perf_kernel_scopes" not in rows[0]
    (r1,) = [r for r in rows if r["seed"] == 1]
    assert r1["perf_mode"] is None and r1["heldout_recall@100"] is None


def test_a_pre_h2_text_record_keeps_the_key_a_run_computes_today():
    (job,) = load_matrix(
        CONFIG / "arxiv.yaml", CONFIG / "suites.yaml", "filter", dims=[128],
        algos=["linr_v1_filter_mask"], backends=["triton"], filter_kinds=["clause"],
        sweeps=["c0_maincat"], seeds=[0],
    )  # fmt: skip
    assert job.key()["inputs"] == "content_d128"
    assert records.record_key(D1_ARXIV) == records.resume_key(
        job.key(), D1_ARXIV["env"]["code_version"]
    )


def test_the_checkpoint_is_part_of_the_key():
    kw = {"dims": [32], "algos": ["linr_v2"], "backends": ["torch"], "sweeps": ["c0"], "seeds": [0]}
    (a,) = load_matrix(
        FIX / "mini.yaml", FIX / "suites.yaml", "filter", filter_kinds=["clause"], **kw
    )
    (b,) = load_matrix(
        FIX / "mini.yaml", FIX / "suites.yaml", "filter", filter_kinds=["clause"],
        checkpoint="data/mini/checkpoints/other-d{dim}/best_model.pt", **kw,
    )  # fmt: skip
    assert (a.key()["inputs"], b.key()["inputs"]) == ("d32", "other-d32")
    assert {**a.key(), "inputs": None} == {**b.key(), "inputs": None}
    assert records.resume_key(a.key(), "c") != records.resume_key(b.key(), "c")


def test_a_pre_h2_goodreads_record_is_gsasrec_not_today_s_encoder():
    old = {**KEY, "env": {"code_version": "c"}}
    del old["inputs"]
    assert records.inputs_of(old) == "gsasrec-d128-drop0.5-id"
    assert records.record_key(old) != records.resume_key(KEY, "c")


def test_a_pre_h7_empty_heldout_side_reads_as_null(tmp_path):
    """The d1/pubmed ``c3_journal_reverse`` shape: ``n == 0`` with 0.0 means."""
    held = {f"{m}@100": 0.0 for m in ("recall", "ndcg", "precision", "mrr")} | {"n": 0}
    rec = {**KEY, "status": "ok", "env": {"code_version": "c"}, "perf": None,
           "quality": {"heldout": held, "oracle": {"recall@100": 0.4, "n": 8600}}}  # fmt: skip
    records.append_record(tmp_path / "filter" / "goodreads-d128.jsonl", rec)
    (row,) = records.read_table(records.aggregate(tmp_path))
    assert row["heldout_recall@100"] is None and row["heldout_n"] == 0
    assert row["oracle_recall@100"] == 0.4


def test_schema_4_columns_and_the_windows_agree_with_the_nested_record(tmp_path):
    """The schema-4 body fields reach results.parquet, and the windows the report's bootstrap
    resamples are the same in the table as in the record ``report._attach`` joins (addendum 2).
    A schema-3 record reads every new column as null."""
    p = tmp_path / "filter" / "goodreads-d128.jsonl"
    entry = {"k": 100, "bs": 16, "mode": "graph", "median_ms": 1.0, "ids_sha256": "ab" * 32,
             "rounds": 3, "window_medians_ms": [0.9, 1.0, 1.1]}  # fmt: skip
    rec = {
        **KEY, "schema_version": 4, "status": "ok", "seed": 1, "seed_scope": "pool",
        "quality_source": {"seed": 0, "code_version": "c"}, "per_query": "filter/x.npz",
        "interleave": {"group": "g1", "arms": ["a", "b"], "position": 1}, "quality": None,
        "perf": [entry], "env": {"code_version": "c", "frac_windows_below_max": 0.25},
    }  # fmt: skip
    old = {**KEY, "schema_version": 3, "status": "ok", "quality": None, "perf": [entry],
           "env": {"code_version": "c"}}  # fmt: skip
    records.append_record(p, rec)
    records.append_record(p, old)
    rows = records.read_table(records.aggregate(tmp_path))
    new, three = sorted(rows, key=lambda r: -r["schema_version"])
    assert (new["seed_scope"], new["quality_source_seed"], new["per_query"]) == (
        "pool", 0, "filter/x.npz")  # fmt: skip
    assert (new["interleave_group"], new["interleave_position"]) == ("g1", 1)
    assert (new["perf_ids_sha256"], new["perf_rounds"]) == ("ab" * 32, 3)
    assert new["env_frac_windows_below_max"] == 0.25
    for c in ("seed_scope", "quality_source_seed", "per_query", "interleave_group",
              "interleave_position", "env_frac_windows_below_max"):  # fmt: skip
        assert three[c] is None
    report._attach(rows, records.latest(tmp_path))
    for r in rows:
        assert r["perf_window_medians_ms"] == r["_entry"]["window_medians_ms"]
