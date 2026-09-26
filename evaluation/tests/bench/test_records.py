"""``bench.records``: the resume key, the JSONL append / read round trip, one torn trailing
line, and ``flatten`` → ``flat.csv`` (one row per perf entry, last record per key)."""

from __future__ import annotations

import csv
import json

import torch

from bench import records

KEY = {
    "dataset": "goodreads", "dim": 128, "suite": "filter", "filter_kind": "clause",
    "sweep": "c0_genre", "algo": "silvertorch", "backend": "triton",
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


def test_flatten_one_row_per_perf_entry_last_record_per_key(tmp_path):
    p = tmp_path / "filter" / "goodreads-d128.jsonl"
    perf = [
        {"k": 100, "bs": 1, "mode": "eager", "median_ms": 1.5, "sm_mhz": 1410.0,
         "window_medians_ms": [1, 2, 3], "window_sm_mhz": [1410.0, 1410.0, 1395.0]},
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
    out = records.flatten(tmp_path)
    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert out == tmp_path / "flat.csv" and len(rows) == 3
    seed0 = [r for r in rows if r["seed"] == "0"]
    assert [r["perf_mode"] for r in seed0] == ["eager", "graph"]
    assert seed0[0]["perf_median_ms"] == "1.5" and seed0[1]["perf_median_ms"] == ""
    assert seed0[0]["heldout_recall@100"] == "0.5" and seed0[0]["oracle_recall@100"] == "0.9"
    assert seed0[0]["quality_parity"] == "reference" and seed0[0]["env_code_version"] == "c"
    assert json.loads(seed0[0]["params"]) == KEY["params"] and seed0[0]["status"] == "ok"
    assert "perf_window_medians_ms" not in rows[0] and "perf_window_sm_mhz" not in rows[0]
    (r1,) = [r for r in rows if r["seed"] == "1"]
    assert r1["perf_mode"] == "" and r1["heldout_recall@100"] == ""
