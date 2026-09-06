"""CPU end-to-end tests for ``retrieval.run`` (harness v2 WP-3) on the tiny fixture of
``conftest.py``: ``backend="torch"``, both modes — ``graph`` needs CUDA, so on this box every
graph entry is the null entry with ``reason: cuda_unavailable`` (the exact path the
``official`` backend takes on the A100; capture itself is C4's gate) — shrunken latency
windows.

Checked: the JSONL record schema (H §3.2 + §8.2 amendments: key block, ``schema_version``,
``status``, quality shapes per filter kind, one perf entry per ``(bs, k, mode)`` with the
§2.5 stats, ``env`` with ``code_version``), resume by key (a second run skips everything,
a changed ``code_version`` re-runs everything), a failing cell recorded as ``status: failed``
with its traceback while the loop continues, the exact-algo quality gate, the parity spill
file (second backend records ``jaccard_vs_first@k`` against the first), the samples
sidecar, and the ``partial`` rule (a ``--mode`` subset or a ``--k`` / ``--bs`` narrow is
recorded as partial and re-run by resume). ``test_cli.py`` drives the same fixture through
``bench``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from retrieval import bench, oracle, run
from retrieval.config import NONE_SWEEP, load_matrix

LAT = {"warmup": 2, "n_min": 4, "n_max": 4, "windows": 3}
EAGER = ("eager",)
KW = dict(latency_kw=LAT, expected_sm_mhz=None)


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
    counts = run.run(jobs, out_dir=out, **KW)
    assert dict(counts) == {"ok": 3}
    path = out / "e2e" / "tiny-d8.jsonl"
    recs = _records(path)
    assert len(recs) == 3
    # Key block + status + provenance on every record.
    for rec, job in zip(recs, jobs):
        assert rec["schema_version"] == run.SCHEMA_VERSION and rec["status"] == "ok"
        assert rec["partial_reasons"] is None
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
        # perf: one entry per (bs, k, mode) — eager with the §2.5 statistics, graph as the
        # null entry (every stat key null + the reason) because there is no CUDA here.
        assert [(e["bs"], e["k"], e["mode"]) for e in rec["perf"]] == [
            (bs, k, mode) for bs in (1, 2) for k in (2, 4) for mode in ("eager", "graph")
        ]
        for e in rec["perf"]:
            assert set(run.PERF_STAT_KEYS) <= set(e)
            if e["mode"] == "eager":
                assert e["n"] == 4 and e["load"] == "closed_loop" and "reason" not in e
            else:
                assert e["reason"] == "cuda_unavailable"
                assert all(e[key] is None for key in run.PERF_STAT_KEYS)
    none, c0, c0c1 = recs
    # Unfiltered: held-out only, everything passes, no filter memory.
    assert "oracle" not in none["quality"] and none["pass_rate"] == 1.0
    assert none["n_kept"] == 8 == none["n_queries_heldout"] and none["n_queries_oracle"] is None
    assert none["n_targets_in_filter"] == 8  # one target per query, every one reachable
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
        assert rec["n_targets_in_filter"] == rec["n_queries_heldout"]  # 1 target, in filter
        assert rec["quality"]["parity"] == "reference"
        assert rec["quality"]["jaccard_vs_first@4"] is None
    assert c0["quality"]["heldout"]["recall@4"] >= 0.0
    # The samples sidecar: one line per *measured* perf entry carrying the key block.
    samples = _records(path.with_suffix(".samples.jsonl"))
    assert len(samples) == 3 * 4 and all(len(s["ms"]) == 4 for s in samples)
    assert {k: samples[0][k] for k in oracle.KEY_FIELDS} == jobs[0].key({})
    # The oracle blob was cached under the dataset's gt_dir.
    assert len(list(jobs[1].data.gt_dir.glob("oracle_v4_c0_*.pt"))) == 1


def test_resume_skips_ok_cells_and_code_version_change_reruns(tiny_configs, tmp_path, monkeypatch):
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP, "c0"])
    out = tmp_path / "results"
    kw = dict(out_dir=out, **KW)
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


def test_narrowed_runs_are_partial_and_resume_reruns_them(tiny_configs, tmp_path):
    """An iteration-day ``--mode eager`` / ``--k 2`` run must never make the next full run
    skip the cell: the record is ``partial`` (with the reasons), and only ``ok`` resumes."""
    out = tmp_path / "results"
    path = out / "e2e" / "tiny-d8.jsonl"
    full = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP])
    assert not full[0].narrowed
    # --mode eager: a subset of MODES.
    assert dict(run.run(full, out_dir=out, modes=EAGER, **KW)) == {"partial": 1}
    assert _records(path)[-1]["partial_reasons"] == ["modes"]
    assert [e["mode"] for e in _records(path)[-1]["perf"]] == ["eager"] * 4
    # --k 2 (the suite has [2, 4]) and --bs 1 (of [1, 2]): the suite's lists replaced.
    narrow = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP], ks=[2])
    assert narrow[0].narrowed and narrow[0].key({}) == full[0].key({})  # same key block
    assert dict(run.run(narrow, out_dir=out, skip_perf=True, **KW)) == {"partial": 1}
    rec = _records(path)[-1]
    assert rec["status"] == "partial" and rec["partial_reasons"] == ["skip_perf", "ks_bs"]
    assert rec["ks"] == [2] and rec["k_max"] == 2
    # The same narrow with the suite's own values is not narrowed at all.
    same = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP], ks=[4, 2])
    assert not same[0].narrowed
    # A full run afterwards does not resume over any of the partial records ...
    assert dict(run.run(full, out_dir=out, **KW)) == {"ok": 1}
    assert [r["status"] for r in _records(path)] == ["partial", "partial", "ok"]
    # ... and only the ok one is what resume sees.
    assert dict(run.run(full, out_dir=out, **KW)) == {"skipped": 1}
    assert dict(run.run(narrow, out_dir=out, **KW)) == {"skipped": 1}  # same key: ok wins


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
    counts = run.run(jobs, out_dir=out, **KW)
    assert dict(counts) == {"ok": 2, "failed": 2}
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert [r["status"] for r in recs] == ["ok", "ok", "failed", "failed"]
    for r in recs[2:]:
        assert r["algo"] == "linr_v4" and r["stage"] == "build"
        assert "boom: int8 path unavailable" in r["error"] and "Traceback" in r["error"]
        assert r["env"]["code_version"] == bench.code_version()
    # A failed record does not count as done: resume re-runs it.
    monkeypatch.setattr(run.algos, "build", real)
    counts = run.run(jobs, out_dir=out, **KW)
    assert dict(counts) == {"skipped": 2, "ok": 2}


def test_sticky_cuda_error_is_recorded_then_ends_the_process(tiny_configs, tmp_path, monkeypatch):
    """After an illegal memory access the context is dead: the cell is recorded as failed
    like any other, but the loop does not continue into cells that can only fail."""
    jobs = _jobs(tiny_configs, sweeps=[NONE_SWEEP])  # v1 none, v4 none
    real = run.algos.build

    def dead(algo, *a, **kw):
        if algo == "linr_v4":
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real(algo, *a, **kw)

    monkeypatch.setattr(run.algos, "build", dead)
    out = tmp_path / "results"
    with pytest.raises(RuntimeError, match="illegal memory access"):
        run.run(jobs, out_dir=out, **KW)
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert [(r["algo"], r["status"]) for r in recs] == [
        ("linr_v1_filter_mask", "ok"),
        ("linr_v4", "failed"),
    ]
    assert recs[1]["stage"] == "build" and "illegal memory access" in recs[1]["error"]
    assert run.is_sticky(RuntimeError("CUDA error: device-side assert triggered"))
    assert not run.is_sticky(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))


def test_read_keys_tolerates_one_torn_trailing_line(tmp_path):
    """The line in flight when the process died must not block ``--resume``; a malformed
    line anywhere else is corruption and raises."""
    p = tmp_path / "x.jsonl"
    key = {
        "dataset": "tiny", "dim": 8, "suite": "e2e", "filter_kind": "none", "sweep": "full_scan",
        "algo": "linr_v1_filter_mask", "backend": "torch", "params": {}, "seed": 0,
    }  # fmt: skip
    rec = {**key, "status": "ok", "env": {"code_version": "c"}}
    run.append_record(p, rec)
    run.append_record(p, {**rec, "seed": 1, "status": "failed"})
    with open(p, "a") as f:
        f.write('{"dataset": "tiny", "dim": 8, "sui')  # the crash
    keys = run.read_keys(p)
    assert len(keys) == 2 and set(keys.values()) == {"ok", "failed"}
    assert keys[oracle.resume_key(key, "c")] == "ok"
    with open(p, "a") as f:
        f.write("\n")
        f.write(json.dumps({**rec, "seed": 2}) + "\n")  # a record *after* the torn line
    with pytest.raises(ValueError, match="malformed record on line 3"):
        run.read_keys(p)


def test_quality_gate_kills_the_run_after_recording(tiny_configs, tmp_path, monkeypatch):
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP, "c0"])
    monkeypatch.setattr(run, "EXACT_MIN_RECALL", 1.5)  # unreachable → the gate must fire
    out = tmp_path / "results"
    with pytest.raises(run.QualityGateError, match="recall_oracle@4"):
        run.run(jobs, out_dir=out, **KW)
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert [r["status"] for r in recs] == ["ok", "failed"]  # none cell has no oracle gate
    assert recs[1]["stage"] == "quality" and "QualityGateError" in recs[1]["error"]


def test_parity_spill_compares_the_second_backend(tiny_configs, tmp_path, monkeypatch):
    ds, suites = tiny_configs
    jobs = load_matrix(ds, suites, "e2e", algos=["linr_v1_filter_mask"], sweeps=["c0"])
    clause = [j for j in jobs if j.filter_kind == "clause"]
    out = tmp_path / "results"
    kw = dict(out_dir=out, skip_perf=True, **KW)
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


class _Fixed(torch.nn.Module):
    """Returns items 0..3 for every query, whatever the attrs."""

    def forward(self, q, qa=None):
        ids = torch.tensor([[0, 1, 2, 3]]).repeat(q.shape[0], 1)
        return ids, torch.linspace(1.0, 0.7, 4).repeat(q.shape[0], 1)


def test_heldout_recall_counts_only_reachable_targets():
    """Review §2.3: on a filter cell a held-out target the exact mask excludes cannot be
    retrieved by any algo. Three queries with targets {0, 5}, {1, 5}, {5}; the mask admits
    items 0–3, so item 5 is unreachable everywhere and query 2 has no reachable target."""
    inputs = {
        "queries": torch.zeros(3, 2),
        "targets": torch.tensor([[0, 5], [1, 5], [5, -1]]),
        "n_targets": torch.tensor([2, 2, 1]),
    }
    keep = torch.ones(3, dtype=torch.bool)
    tif = torch.tensor([[True, False], [True, False], [False, False]])
    blob = {"topk": torch.tensor([[0, 1, 2, 3]] * 3), "targets_in_filter": tif}
    assets = {
        "qa_s": None,
        "keep": keep,
        "blob": blob,
        "oracle_rows": keep,
        "heldout_rows": keep & tif.any(dim=1),
    }
    out, ids, _ = run.quality(_Fixed(), inputs, assets, [2, 4], torch.device("cpu"))
    assert ids.shape == (3, 4)
    h = out["heldout"]
    assert h["n"] == 2  # query 2 has no reachable target: not scored
    # Each scored row has one reachable target, found: recall 1/1 (was 1/2 over both targets).
    assert h["recall@4"] == pytest.approx(1.0) and h["recall@2"] == pytest.approx(1.0)
    assert h["precision@4"] == pytest.approx(0.25) and h["mrr@4"] == pytest.approx(0.75)
    assert h["ndcg@2"] == pytest.approx((1.0 + 1.0 / math.log2(3)) / 2)  # ranks 1 and 2
    # The oracle side is untouched by the masking.
    assert out["oracle"]["recall@4"] == pytest.approx(1.0)
    # On a ``none`` cell every target is reachable and every one counts.
    assets_none = {**assets, "blob": None, "oracle_rows": None, "heldout_rows": keep}
    out, *_ = run.quality(_Fixed(), inputs, assets_none, [4], torch.device("cpu"))
    assert out["heldout"]["n"] == 3
    assert out["heldout"]["recall@4"] == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_append_record_writes_valid_json_for_non_finite_and_tensors(tmp_path):
    p = tmp_path / "x.jsonl"
    run.append_record(p, {"a": float("nan"), "b": torch.tensor([1, 2]), "c": (1.0, float("inf"))})
    run.append_record(p, {"a": 1})
    assert _records(p) == [{"a": None, "b": [1, 2], "c": [1.0, None]}, {"a": 1}]
