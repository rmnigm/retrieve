"""CPU end-to-end tests for ``retrieval.run`` (harness v2 WP-3) on the tiny fixture of
``conftest.py``: ``backend="torch"``, ``mode="eager"`` only (graph mode needs CUDA and is
C4's gate), shrunken latency windows.

Checked: the JSONL record schema (H §3.2 + §8.2 amendments: key block, ``schema_version``,
``status``, quality shapes per filter kind, one perf entry per ``(bs, k, mode)`` with the
§2.5 stats, ``env`` with ``code_version``), resume by key (a second run skips everything,
a changed ``code_version`` re-runs everything), a failing cell recorded as ``status: failed``
with its traceback while the loop continues, the exact-algo quality gate, the parity spill
file (second backend records ``jaccard_vs_first@k`` against the first), and the samples
sidecar. ``test_cli.py`` drives the same fixture through ``bench``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from retrieval import bench, oracle, run
from retrieval.config import NONE_SWEEP, load_matrix

LAT = {"warmup": 2, "n_min": 4, "n_max": 4, "windows": 3}
EAGER = ("eager",)


def _jobs(cfgs, **narrow):
    ds, suites = cfgs
    return load_matrix(ds, suites, "e2e", **narrow)


def _records(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_end_to_end_records(tiny_configs, tmp_path):
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"])
    assert [(j.filter_kind, j.sweep) for j in jobs] == [
        ("none", "full_scan"),
        ("clause", "c0"),
        ("clause", "c0c1"),
    ]
    out = tmp_path / "results"
    counts = run.run(jobs, out_dir=out, modes=EAGER, latency_kw=LAT, expected_sm_mhz=None)
    assert dict(counts) == {"ok": 3}
    path = out / "e2e" / "tiny-d8.jsonl"
    recs = _records(path)
    assert len(recs) == 3
    # Key block + status + provenance on every record.
    for rec, job in zip(recs, jobs):
        assert rec["schema_version"] == run.SCHEMA_VERSION and rec["status"] == "ok"
        assert {k: rec[k] for k in oracle.KEY_FIELDS} == job.key({})
        assert rec["path"] == "cublas" if job.filter_kind == "none" else "cublas+torch"
        assert rec["env"]["code_version"] == bench.code_version()
        assert {"gpu", "torch", "commit", "sm_mhz", "clocks_locked", "clocks_drift"} <= set(
            rec["env"]
        )
        assert rec["n_items"] == 24 and rec["n_queries"] == 8 and rec["k_max"] == 4
        assert rec["ks"] == [2, 4] and rec["batch_sizes"] == [1, 2]
        assert rec["build_s"] > 0 and rec["index_mib"] > 0 and rec["unstable"] in (True, False)
        assert rec["memory_reserved_mib"] is None  # no CUDA allocator on this box
        # perf: one entry per (bs, k, mode) with the §2.5 statistics.
        assert [(e["bs"], e["k"], e["mode"]) for e in rec["perf"]] == [
            (1, 2, "eager"), (1, 4, "eager"), (2, 2, "eager"), (2, 4, "eager")
        ]  # fmt: skip
        for e in rec["perf"]:
            assert set(run.PERF_STAT_KEYS) <= set(e) and e["n"] == 4 and e["load"] == "closed_loop"
            assert e["reason" if e["median_ms"] is None else "median_ms"] is not None
    none, c0, c0c1 = recs
    # Unfiltered: held-out only, everything passes, no filter memory.
    assert "oracle" not in none["quality"] and none["pass_rate"] == 1.0
    assert none["n_kept"] == 8 == none["n_queries_heldout"] and none["n_queries_oracle"] is None
    assert none["filter_mib"] == 0.0 and none["bloom"] is None
    metrics = {f"{m}@{k}" for m in ("recall", "ndcg", "precision", "mrr") for k in (2, 4)}
    assert set(none["quality"]["heldout"]) == metrics | {"n"}
    # Exact dense scan on 24 items at k=4: the oracle prefix is reproduced exactly.
    for rec in (c0, c0c1):
        assert rec["quality"]["oracle"]["recall@4"] == pytest.approx(1.0)
        assert rec["quality"]["oracle"]["recall@2"] == pytest.approx(1.0)
        assert rec["n_kept"] == 7 and rec["filter_mib"] > 0  # query 4 is skip-masked
        assert 0.0 < rec["pass_rate"] < 1.0 and rec["bloom_fp_rate"] is None
        assert rec["n_queries_heldout"] <= rec["n_kept"] and rec["n_queries_oracle"] == 7
        assert rec["quality"]["heldout"]["n"] == rec["n_queries_heldout"]
        assert rec["quality"]["parity"] == "reference"
        assert rec["quality"]["jaccard_vs_first@4"] is None
    assert c0["quality"]["heldout"]["recall@4"] >= 0.0
    # The samples sidecar: one line per perf entry carrying the key block.
    samples = _records(path.with_suffix(".samples.jsonl"))
    assert len(samples) == 3 * 4 and all(len(s["ms"]) == 4 for s in samples)
    assert {k: samples[0][k] for k in oracle.KEY_FIELDS} == jobs[0].key({})
    # The oracle blob was cached under the dataset's gt_dir.
    assert len(list(jobs[1].data.gt_dir.glob("oracle_v4_c0_*.pt"))) == 1


def test_resume_skips_ok_cells_and_code_version_change_reruns(tiny_configs, tmp_path, monkeypatch):
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP, "c0"])
    out = tmp_path / "results"
    kw = dict(out_dir=out, modes=EAGER, latency_kw=LAT, expected_sm_mhz=None)
    assert dict(run.run(jobs, **kw)) == {"ok": 2}
    assert dict(run.run(jobs, **kw)) == {"skipped": 2}
    assert dict(run.run(jobs, resume=False, **kw)) == {"ok": 2}
    path = out / "e2e" / "tiny-d8.jsonl"
    assert len(_records(path)) == 4
    monkeypatch.setattr(bench, "code_version", lambda: "0" * 40)
    assert dict(run.run(jobs, **kw)) == {"ok": 2}
    recs = _records(path)
    assert len(recs) == 6 and recs[-1]["env"]["code_version"] == "0" * 40
    assert len(run.read_keys(path)) == 4  # 2 keys × 2 code versions


def test_failed_cell_is_recorded_and_the_loop_continues(tiny_configs, tmp_path, monkeypatch):
    jobs = _jobs(tiny_configs, sweeps=[NONE_SWEEP, "c0"])  # v1 (none, c0), v4 (none, c0)
    assert [j.algo for j in jobs] == ["linr_v1_filter_mask"] * 2 + ["linr_v4"] * 2
    real = run.algos.build

    def flaky(algo, *a, **kw):
        if algo == "linr_v4":
            raise RuntimeError("boom: int8 path unavailable")
        return real(algo, *a, **kw)

    monkeypatch.setattr(run.algos, "build", flaky)
    out = tmp_path / "results"
    counts = run.run(jobs, out_dir=out, modes=EAGER, latency_kw=LAT, expected_sm_mhz=None)
    assert dict(counts) == {"ok": 2, "failed": 2}
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert [r["status"] for r in recs] == ["ok", "ok", "failed", "failed"]
    for r in recs[2:]:
        assert r["algo"] == "linr_v4" and r["stage"] == "build"
        assert "boom: int8 path unavailable" in r["error"] and "Traceback" in r["error"]
        assert r["env"]["code_version"] == bench.code_version()
    # A failed record does not count as done: resume re-runs it.
    monkeypatch.setattr(run.algos, "build", real)
    counts = run.run(jobs, out_dir=out, modes=EAGER, latency_kw=LAT, expected_sm_mhz=None)
    assert dict(counts) == {"skipped": 2, "ok": 2}


def test_quality_gate_kills_the_run_after_recording(tiny_configs, tmp_path, monkeypatch):
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP, "c0"])
    monkeypatch.setattr(run, "EXACT_MIN_RECALL", 1.5)  # unreachable → the gate must fire
    out = tmp_path / "results"
    with pytest.raises(run.QualityGateError, match="recall_oracle@4"):
        run.run(jobs, out_dir=out, modes=EAGER, latency_kw=LAT, expected_sm_mhz=None)
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert [r["status"] for r in recs] == ["ok", "failed"]  # none cell has no oracle gate
    assert recs[1]["stage"] == "quality" and "QualityGateError" in recs[1]["error"]


def test_parity_spill_compares_the_second_backend(tiny_configs, tmp_path, monkeypatch):
    ds, suites = tiny_configs
    jobs = load_matrix(ds, suites, "e2e", algos=["linr_v1_filter_mask"], sweeps=["c0"])
    clause = [j for j in jobs if j.filter_kind == "clause"]
    out = tmp_path / "results"
    kw = dict(out_dir=out, modes=EAGER, latency_kw=LAT, expected_sm_mhz=None, skip_perf=True)
    run.run(clause, **kw)
    ref = list((out / "_parity").glob("*.npz"))
    assert len(ref) == 1
    # A "second backend": the same cell relabelled, built on the torch path underneath.
    other = dataclasses.replace(clause[0], backend="torch2")
    real_build = run.build_module
    monkeypatch.setattr(
        run,
        "build_module",
        lambda job, *a, **kw: real_build(dataclasses.replace(job, backend="torch"), *a, **kw),
    )
    monkeypatch.setitem(run.FILTER_BACKEND, "torch2", "torch")
    run.run([other], **kw)
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert recs[0]["quality"]["parity"] == "reference"
    assert recs[1]["quality"]["parity"] == "vs_torch"
    assert recs[1]["quality"]["jaccard_vs_first@2"] == 1.0
    assert recs[1]["quality"]["jaccard_vs_first@4"] == 1.0
    assert recs[1]["quality"]["score_max_abs_diff"] == 0.0
    assert recs[1]["status"] == "partial" and recs[1]["perf"] is None  # skip_perf


def test_append_record_writes_valid_json_for_non_finite_and_tensors(tmp_path):
    p = tmp_path / "x.jsonl"
    run.append_record(p, {"a": float("nan"), "b": torch.tensor([1, 2]), "c": (1.0, float("inf"))})
    run.append_record(p, {"a": 1})
    assert _records(p) == [{"a": None, "b": [1, 2], "c": [1.0, None]}, {"a": 1}]
