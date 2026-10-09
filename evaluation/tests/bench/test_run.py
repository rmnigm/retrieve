"""CPU end-to-end tests for ``bench.run`` on the tiny fixture of
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
import hashlib
import json
import math
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from bench import algos, records, run, upload
from bench import measure as bench
from bench.config import NONE_SWEEP, load_matrix

LAT = {"warmup": 2, "n_min": 4, "n_max": 4, "windows": 3}
EAGER = ("eager",)
KW = {"latency_kw": LAT}


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
    path = out / "e2e" / "tiny-d64.jsonl"
    recs = _records(path)
    assert len(recs) == 3
    # Key block + status + provenance on every record.
    for rec, job in zip(recs, jobs, strict=True):
        assert rec["schema_version"] == records.SCHEMA_VERSION and rec["status"] == "ok"
        assert rec["partial_reasons"] is None
        assert {k: rec[k] for k in records.KEY_FIELDS} == job.key({})
        assert rec["path"] == "cublas" if job.filter_kind == "none" else "cublas+torch"
        assert rec["env"]["code_version"] == bench.code_version()
        assert {"gpu", "torch", "commit", "sm_mhz_idle", "sm_mhz_load", "clocks_drift"} <= set(
            rec["env"]
        )
        assert rec["env"]["sm_mhz_load"] is None and rec["env"]["clocks_drift"] is False
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
            # The filter's query encoding is prepared outside the timed calls (none on "none").
            prep = e["query_prep_ms"]
            assert prep is None if job.filter_kind == "none" else prep >= 0
            if e["mode"] == "eager":
                assert e["n"] == 4 and e["load"] == "closed_loop" and "reason" not in e
                assert e["sm_mhz"] is None  # sampled under load on CUDA only
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
    assert {k: samples[0][k] for k in records.KEY_FIELDS} == jobs[0].key({})
    # The oracle blob was cached under the dataset's gt_dir.
    assert len(list(jobs[1].data.gt_dir.glob("oracle_v4_c0_*.pt"))) == 1


def test_resume_skips_ok_cells_and_code_version_change_reruns(tiny_configs, tmp_path, monkeypatch):
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP, "c0"])
    out = tmp_path / "results"
    kw = dict(out_dir=out, **KW)
    assert dict(run.run(jobs, **kw)) == {"ok": 2}
    assert dict(run.run(jobs, **kw)) == {"skipped": 2}
    assert dict(run.run(jobs, resume=False, **kw)) == {"ok": 2}
    path = out / "e2e" / "tiny-d64.jsonl"
    assert len(_records(path)) == 4
    monkeypatch.setattr(bench, "code_version", lambda: "0" * 40)
    assert dict(run.run(jobs, **kw)) == {"ok": 2}
    recs = _records(path)
    assert len(recs) == 6 and recs[-1]["env"]["code_version"] == "0" * 40
    assert len(records.read_keys(path)) == 4  # 2 keys × 2 code versions


def test_narrowed_runs_are_partial_and_resume_reruns_them(tiny_configs, tmp_path):
    """An iteration-day ``--mode eager`` / ``--k 2`` run must never make the next full run
    skip the cell: the record is ``partial`` (with the reasons), and only ``ok`` resumes."""
    out = tmp_path / "results"
    path = out / "e2e" / "tiny-d64.jsonl"
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
    jobs = _jobs(tiny_configs, sweeps=[NONE_SWEEP])  # v1 none, v3 none
    assert [j.algo for j in jobs] == ["linr_v1_filter_mask", "linr_v3"]
    real = run.algos.build

    def flaky(algo, *a, **kw):
        if algo == "linr_v3":
            raise RuntimeError("boom: OPORP build failed")
        return real(algo, *a, **kw)

    monkeypatch.setattr(run.algos, "build", flaky)
    out = tmp_path / "results"
    counts = run.run(jobs, out_dir=out, **KW)
    assert dict(counts) == {"ok": 1, "failed": 1}
    recs = _records(out / "e2e" / "tiny-d64.jsonl")
    assert [r["status"] for r in recs] == ["ok", "failed"]
    r = recs[1]
    assert r["algo"] == "linr_v3" and r["stage"] == "build"
    assert "boom: OPORP build failed" in r["error"] and "Traceback" in r["error"]
    assert r["env"]["code_version"] == bench.code_version()
    # A failed record does not count as done: resume re-runs it (v3 for real this time).
    monkeypatch.setattr(run.algos, "build", real)
    counts = run.run(jobs, out_dir=out, **KW)
    assert dict(counts) == {"skipped": 1, "ok": 1}


def test_sticky_cuda_error_is_recorded_then_ends_the_process(tiny_configs, tmp_path, monkeypatch):
    """After an illegal memory access the context is dead: the cell is recorded as failed
    like any other, but the loop does not continue into cells that can only fail."""
    jobs = _jobs(tiny_configs, sweeps=[NONE_SWEEP])  # v1 none, v3 none
    real = run.algos.build

    def dead(algo, *a, **kw):
        if algo == "linr_v3":
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real(algo, *a, **kw)

    monkeypatch.setattr(run.algos, "build", dead)
    out = tmp_path / "results"
    with pytest.raises(RuntimeError, match="illegal memory access"):
        run.run(jobs, out_dir=out, **KW)
    recs = _records(out / "e2e" / "tiny-d64.jsonl")
    assert [(r["algo"], r["status"]) for r in recs] == [
        ("linr_v1_filter_mask", "ok"),
        ("linr_v3", "failed"),
    ]
    assert recs[1]["stage"] == "build" and "illegal memory access" in recs[1]["error"]
    assert run.is_sticky(RuntimeError("CUDA error: device-side assert triggered"))
    assert not run.is_sticky(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))


def test_read_keys_tolerates_one_torn_trailing_line(tmp_path):
    """The line in flight when the process died must not block ``--resume``; a malformed
    line anywhere else is corruption and raises."""
    p = tmp_path / "x.jsonl"
    key = {
        "dataset": "tiny", "dim": 8, "inputs": "content", "suite": "e2e", "filter_kind": "none",
        "sweep": "full_scan", "algo": "linr_v1_filter_mask", "backend": "torch", "params": {},
        "seed": 0,
    }  # fmt: skip
    rec = {**key, "status": "ok", "env": {"code_version": "c"}}
    records.append_record(p, rec)
    records.append_record(p, {**rec, "seed": 1, "status": "failed"})
    with open(p, "a") as f:
        f.write('{"dataset": "tiny", "dim": 8, "sui')  # the crash
    keys = records.read_keys(p)
    assert len(keys) == 2 and set(keys.values()) == {"ok", "failed"}
    assert keys[records.resume_key(key, "c")] == "ok"
    with open(p, "a") as f:
        f.write("\n")
        f.write(json.dumps({**rec, "seed": 2}) + "\n")  # a record *after* the torn line
    with pytest.raises(ValueError, match="malformed record on line 3"):
        records.read_keys(p)


def test_quality_gate_kills_the_run_after_recording(tiny_configs, tmp_path, monkeypatch):
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP, "c0"])
    monkeypatch.setattr(run, "EXACT_MIN_RECALL", 1.5)  # unreachable → the gate must fire
    out = tmp_path / "results"
    with pytest.raises(run.QualityGateError, match="recall_oracle@4"):
        run.run(jobs, out_dir=out, **KW)
    recs = _records(out / "e2e" / "tiny-d64.jsonl")
    assert [r["status"] for r in recs] == ["ok", "failed"]  # none cell has no oracle gate
    assert recs[1]["stage"] == "quality" and "QualityGateError" in recs[1]["error"]


def test_parity_spill_compares_the_second_backend(tiny_configs, tmp_path, monkeypatch):
    ds, suites = tiny_configs
    jobs = load_matrix(ds, suites, "e2e", algos=["linr_v1_filter_mask"], sweeps=["c0"])
    clause = [j for j in jobs if j.filter_kind == "clause"]
    out = tmp_path / "results"
    kw = dict(out_dir=out, skip_perf=True, **KW)
    run.run(clause, **kw)
    ref = list((out / "_parity" / "tiny-d64_linr_v1_filter_mask").glob("*.npz"))
    assert len(ref) == 1
    # A "second backend": the same cell relabelled, built on the torch path underneath.
    other = dataclasses.replace(clause[0], backend="torch2")
    real_build = run.build_module
    monkeypatch.setattr(
        run,
        "build_module",
        lambda job, *a, **kw: real_build(dataclasses.replace(job, backend="torch"), *a, **kw),
    )
    monkeypatch.setattr(run.algos, "filter_backend", lambda b: "torch")
    run.run([other], **kw)
    recs = _records(out / "e2e" / "tiny-d64.jsonl")
    assert recs[0]["quality"]["parity"] == "reference"
    assert recs[1]["quality"]["parity"] == "vs_torch"
    assert recs[1]["quality"]["jaccard_vs_first@2"] == 1.0
    assert recs[1]["quality"]["jaccard_vs_first@4"] == 1.0
    assert recs[1]["quality"]["score_max_abs_diff"] == 0.0
    assert recs[1]["status"] == "partial" and recs[1]["perf"] is None  # skip_perf


def test_parity_spill_of_the_same_backend_is_rewritten(tiny_configs, tmp_path):
    """A kill after the spill but before the record: the re-run is not its own cross-check."""
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=["c0"])
    out = tmp_path / "results"
    run.run(jobs, out_dir=out, skip_perf=True, **KW)
    (out / "e2e" / "tiny-d64.jsonl").unlink()
    run.run(jobs, out_dir=out, skip_perf=True, **KW)
    recs = _records(out / "e2e" / "tiny-d64.jsonl")
    assert [r["quality"]["parity"] for r in recs] == ["reference"] * len(jobs)


class _Fixed(torch.nn.Module):
    """Returns items 0..3 for every query, whatever the attrs."""

    def forward(self, q, qa=None):
        ids = torch.tensor([[0, 1, 2, 3]]).repeat(q.shape[0], 1)
        return ids, torch.linspace(1.0, 0.7, 4).repeat(q.shape[0], 1)


def test_postfilter_records_oracle_recall_per_k(tiny_configs, tmp_path):
    """Roadmap D5-code: ``recall_oracle`` against the exact oracle, at each k from a run at
    that k (``PER_K_QUALITY``), never above the exact V1 of the same sweep."""
    ds, suites = tiny_configs
    jobs = load_matrix(ds, suites, "postfilter")
    out = tmp_path / "results"
    assert dict(run.run(jobs, out_dir=out, **KW)) == {"ok": 9}
    recs = _records(out / "postfilter" / "tiny-d64.jsonl")
    exact = {(r["filter_kind"], r["sweep"]): r for r in recs if r["algo"] != "postfilter"}
    post = [r for r in recs if r["algo"] == "postfilter"]
    assert sorted((r["filter_kind"], r["sweep"], r["params"]["alpha"]) for r in post) == [
        (fk, sw, a) for fk, sw in (("bloom", "c0"), ("clause", "c0"), ("clause", "c0c1"))
        for a in (1, 2)
    ]  # fmt: skip
    for r in post:
        assert r["backend"] == "torch" and r["path"] == "cublas+torch"
        for k in (2, 4):
            got = r["quality"]["oracle"][f"recall@{k}"]
            assert 0 <= got <= exact[r["filter_kind"], r["sweep"]]["quality"]["oracle"][
                f"recall@{k}"] + 1e-12  # fmt: skip
    job = next(j for j in jobs if j.algo == "postfilter" and j.sweep == "c0c1")
    rec = next(r for r in post if r["sweep"] == "c0c1" and r["params"]["alpha"] == 1)
    inp = run.inputs.load_inputs(job.data, torch.device("cpu"), with_filters=True)
    assets = run.sweep_assets(job, inp, 4, torch.device("cpu"))
    module = run.build_module(job, inp, assets, 4, {"alpha": 1})
    module.k = 2
    at_2, _, _, _ = run.quality(module, inp, assets, [2], torch.device("cpu"))
    assert rec["quality"]["oracle"]["recall@2"] == at_2["oracle"]["recall@2"]
    assert rec["quality"]["heldout"]["recall@2"] == at_2["heldout"]["recall@2"]


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
    out, _, ids, _ = run.quality(_Fixed(), inputs, assets, [2, 4], torch.device("cpu"))
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


def test_quality_holds_torch_cpu_threads_for_the_loop_and_restores_them():
    """H-QLOOP: the chunk loop runs at ``QUALITY_CPU_THREADS`` and gives the caller's count back."""
    seen = []

    class Probe(_Fixed):
        def forward(self, q, qa=None):
            seen.append(torch.get_num_threads())
            return super().forward(q, qa)

    rows = 40  # three chunks
    inputs = {"queries": torch.zeros(rows, 2), "targets": torch.zeros(rows, 1, dtype=torch.long)}
    keep = torch.ones(rows, dtype=torch.bool)
    assets = {"qa_s": None, "keep": keep, "blob": None, "oracle_rows": None, "heldout_rows": keep}
    before = torch.get_num_threads()
    torch.set_num_threads(max(2, before))
    try:
        run.quality(Probe(), inputs, assets, [4], torch.device("cpu"))
        assert seen == [run.QUALITY_CPU_THREADS] * 3
        assert torch.get_num_threads() == max(2, before)
    finally:
        torch.set_num_threads(before)


def test_quality_records_the_routers_local_pass_rate_and_route_per_query():
    """V-ROUTER: the sidecar arrays carry each kept query's l_q and route (1.0 = exact V2), and the
    record the share routed to the exact branch."""
    g = torch.Generator().manual_seed(0)
    n, d, rows = 256, 64, 20
    x, q = torch.randn(n, d, generator=g), torch.randn(rows, d, generator=g)
    attrs = torch.randint(0, 3, (n, 2, 1), generator=g)
    qa = torch.randint(0, 3, (rows, 2), generator=g)
    f = algos.build_filter("clause", attrs, backend="torch")
    params = {"n_lists": 8, "n_probe": 4, "n_iter": 2, "pre_n_probe": 8, "lq_threshold": 0.3}
    m = algos.build("router", x, k=4, backend="torch", filter_kind="clause", filter_mod=f,
                item_attrs=attrs, params=params)  # fmt: skip
    keep = torch.ones(rows, dtype=torch.bool)
    keep[3] = False  # a skipped row is not scored and not in the sidecar
    inputs = {"queries": q, "targets": torch.zeros(rows, 1, dtype=torch.long)}
    assets = {"qa_s": qa, "keep": keep, "blob": None, "oracle_rows": None, "heldout_rows": keep}
    out, per_q, _, _ = run.quality(m, inputs, assets, [4], torch.device("cpu"))
    lq = m.local_pass_rate(q[keep], m.filter.prepare_queries(qa[keep]))
    assert torch.equal(per_q["router_lq"], lq) and per_q["router_lq"].shape == (rows - 1,)
    assert torch.equal(per_q["router_exact"], (lq < 0.3).float())
    assert out["router_exact_share"] == pytest.approx(float((lq < 0.3).float().mean()))


class _Prepared(torch.nn.Module):
    """A module with a query-prep step, as ``SilverTorch``: counts its preps and records what each
    forward received."""

    backend = "official"

    def __init__(self) -> None:
        super().__init__()
        self.k = 4
        self.preps = 0
        self.seen: list = []

    def prepare_queries(self, qa):
        self.preps += 1
        return ("prepared", qa)

    def forward(self, q, prepared=None):
        self.seen.append(prepared[0] if prepared is not None else None)
        return torch.zeros(q.shape[0], self.k, dtype=torch.long), torch.zeros(q.shape[0], self.k)


def test_perf_prepares_the_pool_before_timing_and_records_it(tiny_configs, monkeypatch):
    """evaluation.md § Query preparation: every pool batch is prepared once before the timed calls
    (no forward prepares), each timed call gets the prepared batch, and every entry records
    ``query_prep_ms``; graph null entries carry it too."""
    job = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP])[0]
    n_pool = 6
    pool = torch.zeros(n_pool, 1, 8)
    qa_pool = torch.zeros(n_pool, 1, 2, dtype=torch.long)
    monkeypatch.setattr(run.inputs, "query_pool", lambda *a, **kw: (pool, qa_pool))
    m = _Prepared()
    ((entries, samples),) = run.perf(
        [m], {}, {"qa_s": None, "skip": None}, job, torch.device("cpu"), modes=EAGER,
        profile=False, latency_kw=LAT,
    )  # fmt: skip
    per_bs = len(job.batch_sizes)
    assert m.preps == n_pool * per_bs and set(m.seen) == {"prepared"}
    assert len(entries) == len(samples) == per_bs * len(job.ks)
    assert all(e["query_prep_ms"] is not None and e["query_prep_ms"] >= 0 for e in entries)
    ((entries, _),) = run.perf(
        [m], {}, {"qa_s": None, "skip": None}, job, torch.device("cpu"), modes=("graph",),
        profile=False, latency_kw=LAT,
    )  # fmt: skip
    assert all(e["reason"] == "cuda_unavailable" and e["query_prep_ms"] >= 0 for e in entries)


def test_eager_only_partial_is_per_job(tiny_configs, tmp_path, monkeypatch):
    """Roadmap H6: an eager-only run marks a capturable module's record ``partial`` for
    ``modes`` and not an ``official`` one, whose graph entry is ``not_capturable`` anyway.
    Both run the torch path underneath; the ``official`` build is flagged uncapturable."""
    job = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=[NONE_SWEEP])[0]
    jobs = [dataclasses.replace(job, backend=b) for b in ("triton", "official")]
    real_build = run.build_module

    def build(job, *a, **kw):
        m = real_build(dataclasses.replace(job, backend="torch"), *a, **kw)
        if job.backend == "official":
            m.capturable = False
        return m

    monkeypatch.setattr(run, "build_module", build)
    out = tmp_path / "results"
    assert dict(run.run(jobs, out_dir=out, modes=EAGER, **KW)) == {"partial": 1, "ok": 1}
    recs = {r["backend"]: r for r in _records(out / "e2e" / "tiny-d64.jsonl")}
    assert recs["triton"]["status"] == "partial"
    assert recs["triton"]["partial_reasons"] == ["modes"]
    assert recs["official"]["status"] == "ok" and recs["official"]["partial_reasons"] is None


def test_append_record_writes_valid_json_for_non_finite_and_tensors(tmp_path):
    p = tmp_path / "x.jsonl"
    records.append_record(
        p, {"a": float("nan"), "b": torch.tensor([1, 2]), "c": (1.0, float("inf"))}
    )
    records.append_record(p, {"a": 1})
    assert _records(p) == [{"a": None, "b": [1, 2], "c": [1.0, None]}, {"a": 1}]


def test_a_cell_with_no_in_filter_heldout_target_records_null(tiny_configs, tmp_path, monkeypatch):
    """Pubmed's reverse clause: no kept query has a held-out target the filter admits. The
    held-out side has no mean, so it is null with ``n == 0``, never 0.0 (roadmap H7)."""
    real = run.oracle.load_or_build

    def unreachable(*a, **kw):
        blob = real(*a, **kw)
        return {
            **blob,
            "targets_in_filter": torch.zeros_like(blob["targets_in_filter"]),
            "target_in_filter": torch.zeros_like(blob["target_in_filter"]),
        }

    monkeypatch.setattr(run.oracle, "load_or_build", unreachable)
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=["c0"])
    out = tmp_path / "results"
    assert dict(run.run(jobs, out_dir=out, skip_perf=True, **KW)) == {"partial": 1}
    (rec,) = _records(out / "e2e" / "tiny-d64.jsonl")
    held = rec["quality"]["heldout"]
    assert held["n"] == 0 and rec["n_queries_heldout"] == 0
    assert all(v is None for m, v in held.items() if m != "n")
    assert rec["quality"]["oracle"]["recall@4"] == pytest.approx(1.0)  # the oracle side scores


def _arms(cfgs, **narrow):
    ds, suites = cfgs
    return load_matrix(ds, suites, "arms", **narrow)


def test_gridded_bloom_width_is_the_cells_own(tiny_configs, tmp_path):
    """G-bloomwidth (CPU): ``m_bits`` in ``params`` sets that job's bloom — the module's index,
    the standalone filter's pass counts and ``bloom_fp_rate``, the record's ``bloom`` — while
    the ungridded cell keeps the suite default and a key without ``m_bits``."""
    jobs = _arms(tiny_configs, filter_kinds=["bloom"])
    out = tmp_path / "results"
    assert dict(run.run(jobs, out_dir=out, skip_perf=True, **KW)) == {"partial": 3}
    recs = _records(out / "arms" / "tiny-d64.jsonl")
    by = {r["params"].get("m_bits"): r for r in recs}
    assert by[64]["bloom"] == {"m_bits": 64, "k_hash": 2}
    assert by[256]["bloom"] == {"m_bits": 256, "k_hash": 2}
    assert by[None]["bloom"] == {"m_bits": 64, "k_hash": 2}  # the suite default
    assert by[None]["params"] == {"n_lists": 4, "n_probe": 2}
    assert by[256]["index_mib"] > by[64]["index_mib"]  # the bloom words live in the index
    assert all(r["bloom_fp_rate"] is not None for r in recs)
    assert by[256]["bloom_fp_rate"] <= by[64]["bloom_fp_rate"]


def test_resolve_pool_from_the_sweeps_pass_counts():
    """G-pool: ``max(POOL_MIN, round(frac × mean pass count over the oracle rows))``; on synth
    every query passes the same N·p items. Skipped rows (``-1``) do not count."""
    count = 79_708  # goodreads-synth p01: 797,084 items × 0.1
    blob = {"pass_counts": torch.tensor([count] * 5 + [-1])}
    assets = {"blob": blob, "oracle_rows": torch.tensor([True] * 5 + [False])}
    for frac in (0.005, 0.01, 0.05, 0.1):
        p = run.resolve_pool({"candidate_pool_frac": frac}, assets)
        assert p == {"candidate_pool": max(2000, round(frac * count))}
    assert run.resolve_pool({"candidate_pool_frac": 0.01}, assets)["candidate_pool"] == 2000
    assert run.resolve_pool({"n_probe": 8}, assets) == {"n_probe": 8}
    with pytest.raises(ValueError, match="oracle pass counts"):
        run.resolve_pool({"candidate_pool_frac": 0.1}, {"blob": None})


def test_pool_fraction_end_to_end(tiny_configs, tmp_path, monkeypatch):
    """The key keeps ``candidate_pool_frac``; the module and the record body get the int."""
    monkeypatch.setattr(run, "POOL_MIN", 4)
    jobs = _arms(tiny_configs, algos=["linr_v3"], sweeps=["c0"])
    out = tmp_path / "results"
    assert dict(run.run(jobs, out_dir=out, skip_perf=True, **KW)) == {"partial": 2}
    recs = _records(out / "arms" / "tiny-d64.jsonl")
    # clause c0: every live query passes 8 of the 24 items
    assert [(r["params"], r["candidate_pool"]) for r in recs] == [
        ({"candidate_pool_frac": 0.5}, 4),
        ({"candidate_pool_frac": 1.0}, 8),
    ]
    assert recs[1]["quality"]["oracle"]["recall@4"] == pytest.approx(1.0)  # the whole pass set


def test_compiled_arm_times_its_warmup_apart_and_skips_graph(tiny_configs, tmp_path, monkeypatch):
    """``compile`` is in the key, the module is compiled in place with its mode, the first
    forward is ``compile_s`` (not ``build_s``), and ``graph`` is a null ``not_capturable`` entry
    (max-autotune replays its own CUDA graphs). The compiler itself is faked on CPU."""
    calls = []
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, **kw: calls.append(kw))
    jobs = _arms(tiny_configs, filter_kinds=["clause"], algos=["silvertorch"])
    out = tmp_path / "results"
    assert dict(run.run(jobs, out_dir=out, **KW)) == {"ok": 2}
    assert calls == [{"mode": "max-autotune"}] * 2
    recs = _records(out / "arms" / "tiny-d64.jsonl")
    assert {r["sweep"] for r in recs} == {"c0", "c0c1"}
    for r in recs:
        assert r["params"] == {"n_lists": 4, "compile": "max-autotune", "n_probe": 2}
        assert r["compile_s"] > 0 and r["build_s"] > 0
        graph = [e for e in r["perf"] if e["mode"] == "graph"]
        assert graph and all(e["reason"] == "not_capturable" and e["median_ms"] is None
                             for e in graph)  # fmt: skip
    plain = _records(out / "arms" / "tiny-d64.jsonl")
    assert all(r["candidate_pool"] is None for r in plain)


def test_quality_cache_copies_seed_free_arms_only(tiny_configs, tmp_path, monkeypatch):
    """G-cache (CPU): V1 and the postfilter copy seed 0's quality to seeds 1-2 (perf still runs
    per seed); V3 and SilverTorch always recompute. A new code_version or other ks is a miss."""
    ds, suites = tiny_configs
    out = tmp_path / "results"
    path = out / "cache" / "tiny-d64.jsonl"
    assert dict(run.run(load_matrix(ds, suites, "cache"), out_dir=out, **KW)) == {"ok": 12}
    by = {(r["algo"], r["seed"]): r for r in _records(path)}
    for algo in ("linr_v1_filter_mask", "postfilter"):
        assert by[algo, 0]["quality_source"] is None and by[algo, 0]["seed_scope"] == "pool"
        for seed in (1, 2):
            r = by[algo, seed]
            assert r["quality_source"] == {"seed": 0, "code_version": bench.code_version()}
            assert r["quality"] == by[algo, 0]["quality"] and r["perf"]
    for algo in ("linr_v3", "silvertorch"):
        for seed in (0, 1, 2):
            r = by[algo, seed]
            assert r["quality_source"] is None and r["seed_scope"] == "pool+build"
    # Fresh quality at seed 1 equals what the cache copied.
    fresh = tmp_path / "fresh"
    jobs = load_matrix(ds, suites, "cache", algos=["linr_v1_filter_mask"], seeds=[1])
    run.run(jobs, out_dir=fresh, skip_perf=True, **KW)
    (r,) = _records(fresh / "cache" / "tiny-d64.jsonl")
    assert r["quality_source"] is None
    assert {k: v for k, v in r["quality"].items() if k != "parity"} == {
        k: v for k, v in by["linr_v1_filter_mask", 1]["quality"].items() if k != "parity"
    }
    # Other ks, or another code_version: no source, quality is computed.
    jobs = load_matrix(ds, suites, "cache", algos=["linr_v1_filter_mask"], seeds=[2], ks=[2])
    run.run(jobs, out_dir=out, skip_perf=True, **KW)
    assert _records(path)[-1]["quality_source"] is None
    monkeypatch.setattr(bench, "code_version", lambda: "0" * 40)
    jobs = load_matrix(ds, suites, "cache", algos=["linr_v1_filter_mask"], seeds=[1])
    run.run(jobs, out_dir=out, **KW)
    assert _records(path)[-1]["quality_source"] is None


def test_a_quality_only_suite_is_ok_without_perf(tiny_configs, tmp_path):
    """Addendum 1: ``perf: false`` (n95, bloomwidth) runs no perf and is ``ok``, not ``partial``
    for ``skip_perf``; ``--skip-perf`` or ``--mode eager`` on it drops nothing either."""
    ds, suites = tiny_configs
    jobs = load_matrix(ds, suites, "untimed")
    assert jobs and not any(j.timed for j in jobs)
    out = tmp_path / "results"
    assert dict(run.run(jobs, out_dir=out, **KW)) == {"ok": 2}
    recs = _records(out / "untimed" / "tiny-d64.jsonl")
    assert all(r["perf"] is None and r["quality"] and r["partial_reasons"] is None for r in recs)
    assert dict(run.run(jobs, out_dir=out, resume=False, skip_perf=True, modes=EAGER, **KW)) == {
        "ok": 2
    }
    assert not (out / "untimed" / "tiny-d64.samples.jsonl").exists()


def test_ids_sha256_is_stable_and_tracks_the_ids(tiny_configs, tmp_path):
    """G-ids (CPU): two runs of a deterministic cell record the same ``ids_sha256`` per
    ``(bs, k, mode)``; the hash is over the ids (int64, row-major) of the pool's first batches,
    so different ids hash differently and a null entry has none."""
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=["c0"])
    hashes = []
    for out in (tmp_path / "a", tmp_path / "b"):
        run.run(jobs, out_dir=out, **KW)
        (rec,) = _records(out / "e2e" / "tiny-d64.jsonl")
        hashes.append({(e["bs"], e["k"], e["mode"]): e["ids_sha256"] for e in rec["perf"]})
    assert hashes[0] == hashes[1]
    eager = {v for (_, _, m), v in hashes[0].items() if m == "eager"}
    assert len(eager) == 4 and all(len(h) == 64 for h in eager)  # every (bs, k) differs
    assert all(v is None for (_, _, m), v in hashes[0].items() if m == "graph")  # no CUDA
    pool = torch.zeros(10, 2, 8)
    fixed = run.ids_sha256(_Fixed(), pool, None)
    assert fixed == run.ids_sha256(_Fixed(), pool, torch.zeros(10, 2, 1))  # qa ignored here

    class _Shifted(_Fixed):
        def forward(self, q, qa=None):
            ids, sc = super().forward(q, qa)
            return ids.flip(1), sc

    assert all(a != b for a, b in zip(run.ids_sha256(_Shifted(), pool, None), fixed, strict=True))
    assert (
        fixed[0]
        == hashlib.sha256(
            torch.tensor([[0, 1, 2, 3]] * 2).numpy().tobytes() * run.IDS_PROBE_BATCHES
        ).hexdigest()
    )


class _Tied(torch.nn.Module):
    """Items 5 and 2 tie at 0.9: ``swap`` returns them in the other order, scores unchanged."""

    def __init__(self, swap: bool) -> None:
        super().__init__()
        self.ids = torch.tensor([7, 2, 5, 1] if swap else [7, 5, 2, 1])

    def forward(self, q, qa=None):
        sc = torch.tensor([1.0, 0.9, 0.9, 0.3])
        return self.ids.repeat(q.shape[0], 1), sc.repeat(q.shape[0], 1)


def test_canonical_ids_hash_ignores_the_order_of_tied_ids():
    """Follow-up 1: a tied pair in two orders hashes differently exactly (``ids_sha256``, the
    eager-vs-graph gate) and equal canonically (``ids_sha256_canon``, official vs Triton:
    rows re-ordered by score desc, then id asc); a real id difference changes both."""
    pool = torch.zeros(10, 2, 8)
    a, b = run.ids_sha256(_Tied(False), pool, None), run.ids_sha256(_Tied(True), pool, None)
    assert a[0] != b[0] and a[1] == b[1]
    assert (
        a[1]
        == hashlib.sha256(
            torch.tensor([[7, 2, 5, 1]] * 2).numpy().tobytes() * run.IDS_PROBE_BATCHES
        ).hexdigest()
    )  # ties to the lower id
    assert run.ids_sha256(_Fixed(), pool, None)[1] != a[1]


def test_per_query_sidecar_matches_the_record_and_ships(tiny_configs, tmp_path):
    """G-dump (CPU): one npz per record with quality, under ``<suite>/<ds>-d<dim>.perquery/``;
    its per-query recall averages to the record's (to float32 summation), ``pass_count`` is
    the blob's (``-1`` without a filter), a quality-cache copy points at its source's file,
    ``bench upload --dry-run`` lists it and ``records.aggregate`` ignores it."""
    ds, suites = tiny_configs
    out = tmp_path / "results"
    run.run(load_matrix(ds, suites, "e2e"), out_dir=out, skip_perf=True, **KW)
    run.run(load_matrix(ds, suites, "cache"), out_dir=out, skip_perf=True, **KW)
    recs = _records(out / "e2e" / "tiny-d64.jsonl") + _records(out / "cache" / "tiny-d64.jsonl")
    worst = 0.0
    for r in recs:
        z = np.load(out / r["per_query"])
        assert r["per_query"].startswith(f"{r['suite']}/tiny-d64.perquery/")
        assert z["rows"].dtype == np.int32 and len(z["rows"]) == r["n_kept"]
        assert z["pass_count"].dtype == np.int64
        assert (z["pass_count"] == -1).all() == (r["filter_kind"] == "none")
        for k in r["ks"]:
            for side, name in (("oracle", "recall_oracle"), ("heldout", "heldout_recall")):
                v = z[f"{name}@{k}"]
                assert v.dtype == np.float32
                if side not in r["quality"] or r["quality"][side]["n"] == 0:
                    assert np.isnan(v).all()
                    continue
                assert (~np.isnan(v)).sum() == r["quality"][side]["n"]
                diff = abs(
                    float(np.nanmean(v.astype(np.float64))) - r["quality"][side][f"recall@{k}"]
                )
                worst = max(worst, diff)
    assert worst < 1e-7
    by = {(r["algo"], r["seed"]): r for r in recs if r["suite"] == "cache"}
    assert by["postfilter", 2]["per_query"] == by["postfilter", 0]["per_query"]
    assert by["linr_v3", 2]["per_query"] != by["linr_v3", 0]["per_query"]
    cars = sorted(out.rglob("*.npz"))
    assert {p.relative_to(out).as_posix() for p in cars if "_parity" not in p.parts} == {
        r["per_query"] for r in recs
    }
    n_rows = len(records.read_table(records.aggregate(out)))
    assert n_rows == len({records.record_key(r) for r in recs})  # one row each: no perf
    res = CliRunner().invoke(
        upload.upload, ["--results", str(out), "--path-in-repo", "x", "--dry-run"]
    )
    assert res.exit_code == 0, res.output
    assert all(r["per_query"] in res.output for r in recs)


def test_interleaved_group_keys_rounds_and_resume(tiny_configs, tmp_path, monkeypatch):
    """G-interleave (CPU): with ``interleave`` each arm of a group writes the record (and key)
    it writes without it, plus the group block (one group per seed and sweep); both arms are
    timed in one ``latency_group`` call per variant with equal window counts (``rounds``); and
    resume re-runs a group whole when one arm's cell is missing."""
    ds, suites = tiny_configs
    jobs = load_matrix(ds, suites, "pair")
    plain, inter = tmp_path / "plain", tmp_path / "inter"
    assert dict(run.run(jobs, out_dir=plain, **KW)) == {"ok": 8}
    widths = []
    real = run.measure.latency_group
    monkeypatch.setattr(
        run.measure, "latency_group", lambda fns, **kw: widths.append(len(fns)) or real(fns, **kw)
    )
    assert dict(run.run(jobs, out_dir=inter, interleave=True, **KW)) == {"ok": 8}
    assert widths == [2] * (4 * 4)  # 4 groups x (2 bs x 2 k x eager); graph is null on CPU
    a = _records(plain / "pair" / "tiny-d64.jsonl")
    b = _records(inter / "pair" / "tiny-d64.jsonl")
    assert sorted(map(records.record_key, a)) == sorted(map(records.record_key, b))
    assert all(r["interleave"] is None for r in a)
    groups: dict[str, list] = {}
    for r in b:
        groups.setdefault(r["interleave"]["group"], []).append(r)
    assert len(groups) == 4  # 2 sweeps x 2 seeds
    for g in groups.values():
        assert [r["algo"] for r in g] == ["linr_v1_filter_mask", "linr_v2"]
        assert [r["interleave"]["position"] for r in g] == [0, 1]
        assert g[0]["interleave"]["arms"] == ["linr_v1_filter_mask/torch", "linr_v2/torch"]
        assert len({(r["seed"], r["sweep"]) for r in g}) == 1
        for e0, e1 in zip(g[0]["perf"], g[1]["perf"], strict=True):
            assert (e0["bs"], e0["k"], e0["mode"]) == (e1["bs"], e1["k"], e1["mode"])
            if e0["mode"] == "eager":
                assert e0["rounds"] == e1["rounds"] == 3 == len(e0["window_medians_ms"])
    assert {r["quality_source"] is None for r in b if r["seed"] == 0} == {True}
    assert all(r["quality_source"] == {"seed": 0, "code_version": bench.code_version()}
               for r in b if r["seed"] == 1)  # fmt: skip
    # Drop one arm's record: the next interleaved run re-runs that group whole, nothing else.
    path = inter / "pair" / "tiny-d64.jsonl"
    victim = next(r for r in b if r["algo"] == "linr_v2" and r["seed"] == 1)
    path.write_text("".join(json.dumps(r) + "\n" for r in b if r is not victim))
    assert dict(run.run(jobs, out_dir=inter, interleave=True, **KW)) == {"ok": 2, "skipped": 6}
    rerun = _records(path)[-2:]
    assert {(r["algo"], r["seed"], r["sweep"]) for r in rerun} == {
        (a, 1, victim["sweep"]) for a in ("linr_v1_filter_mask", "linr_v2")
    }
    assert rerun[0]["interleave"]["group"] == victim["interleave"]["group"]


def test_frac_windows_below_the_devices_max_clock(tiny_configs, tmp_path, monkeypatch):
    """Follow-up 2: ``env.frac_windows_below_max`` is the share of a record's window clock
    samples below the device's max SM clock (``env.sm_max_mhz``), whatever ran before: a cell
    whose windows all sample below it reads 1.0 first or second; no device max, no value."""
    real = bench.clocks
    monkeypatch.setattr(bench, "clocks", lambda: {**real(), "sm_max_mhz": 1410.0})
    low, mixed = [1140.0, 1140.0, 1140.0], [1410.0, 1395.0, 1305.0, 1410.0]
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=["c0", "c0c1"])
    for order, want in (((low, mixed), [1.0, 0.5]), ((mixed, low), [0.5, 1.0])):
        clocks = iter(order)

        def fake_perf(modules, *a, clocks=clocks, **k):
            win = next(clocks)
            e = {"k": 2, "bs": 1, "mode": "eager", "sm_mhz": win[-1], "window_sm_mhz": win}
            return [([e], []) for _ in modules]

        monkeypatch.setattr(run, "perf", fake_perf)
        out = tmp_path / str(want)
        run.run(jobs, out_dir=out, skip_quality=True, **KW)
        recs = _records(out / "e2e" / "tiny-d64.jsonl")
        assert [r["env"]["frac_windows_below_max"] for r in recs] == want
        assert all(r["env"]["sm_max_mhz"] == 1410.0 for r in recs)
    monkeypatch.setattr(bench, "clocks", lambda: {**real(), "sm_max_mhz": None})
    monkeypatch.setattr(run, "perf", lambda modules, *a, **k: [(
        [{"k": 2, "bs": 1, "mode": "eager", "sm_mhz": 1.0, "window_sm_mhz": [1.0]}], []
    ) for _ in modules])  # fmt: skip
    run.run(jobs[:1], out_dir=tmp_path / "none", skip_quality=True, **KW)
    (r,) = _records(tmp_path / "none" / "e2e" / "tiny-d64.jsonl")
    assert r["env"]["frac_windows_below_max"] is None
    assert bench.clock_histogram([1410.0, 1395.0, 1410.0]) == (
        "sm_mhz under load (n=3): 1410×2, 1395×1"
    )
    assert bench.clock_histogram([]) == "sm_mhz under load (n=0): no samples"


def test_score_path_arms_share_the_triton_parity_spill(tiny_configs, tmp_path):
    """h2h's official arms carry ``score_path`` in ``params``; the spill hash drops it with
    ``backend``, so they compare against the triton reference instead of each writing one."""
    job = _jobs(tiny_configs, algos=["linr_v1_filter_mask"], sweeps=["c0"])[0]
    ids, sc = torch.tensor([[0, 1], [2, 3]]), torch.tensor([[1.0, 0.5], [0.9, 0.1]])
    assert run.parity(tmp_path, job, {}, ids, sc, [2])["parity"] == "reference"
    off = dataclasses.replace(job, backend="official")
    for sp in ("fp16", "int32"):
        out = run.parity(tmp_path, off, {"score_path": sp}, ids, sc, [2])
        assert out["parity"] == "vs_torch" and out["jaccard_vs_first@2"] == 1.0


def test_each_finished_arm_is_freed_before_the_next_is_built(tiny_configs, tmp_path, monkeypatch):
    """Roadmap H-ARMFREE: no module of a finished unit is alive when the next unit builds, so
    one process can run several 10-30 M arms in a row."""
    jobs = _jobs(tiny_configs, algos=["linr_v1_filter_mask"])
    assert len(jobs) == 3
    real_build, built = run.build_module, []

    def build(*a, **kw):
        assert [r for r in built if r() is not None] == []
        m = real_build(*a, **kw)
        built.append(weakref.ref(m))
        return m

    monkeypatch.setattr(run, "build_module", build)
    assert dict(run.run(jobs, out_dir=tmp_path / "results", modes=EAGER, **KW)) == {"partial": 3}
    assert len(built) == 3
