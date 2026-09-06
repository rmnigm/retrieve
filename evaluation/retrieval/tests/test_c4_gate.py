"""CPU tests for the C4 golden-comparison script
(``docs/plans/evaluation-harness-v2-artifacts/c4_gate.py``, H §6 WP-4). Real golden files are
copied into ``tmp_path``; v2 records are synthesised to reproduce them exactly (quality to the
digit, graph medians equal, both clocks 1140 MHz, parity 1.0, an official cell at
``cache_plans: false``) so the gate passes, then each gate is broken one way at a time and
must be the one FAIL. Gate (6), kill-and-resume, is manual and not covered."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from retrieval import bench

SCRIPT = bench.ROOT / "docs" / "plans" / "evaluation-harness-v2-artifacts" / "c4_gate.py"
GOLDEN = bench.ROOT / "evaluation" / "golden"
FILES = [
    "goodreads-d128-c0_genre-silvertorch-triton.json",
    "goodreads-d128-c0_genre-silvertorch-torch.json",
    "goodreads-d128-c0_genre-linr_v2-triton.json",
    "goodreads-d128-c0_genre-linr_v2-torch.json",
]


@pytest.fixture(scope="module")
def c4():
    spec = importlib.util.spec_from_file_location("c4_gate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _golden_rows(name: str) -> list[dict]:
    return json.loads((GOLDEN / name).read_text())


def _synth(rows: list[dict], *, backend: str, parity: str, jaccard=1.0, sm_mhz=1140.0, **over):
    """A v2 record reproducing one golden file: quality to the digit, graph median == golden,
    eager median 3× (context only), stable, clocks at the golden's 1140 MHz."""
    r0 = rows[0]
    ks = sorted({r["k"] for r in rows})
    bss = sorted({r["batch_size"] for r in rows})
    by = {(r["k"], r["batch_size"]): r for r in rows}
    quality: dict = {"oracle": {}, "heldout": {}, "parity": parity, "score_max_abs_diff": 0.0}
    for k in ks:
        g = by[k, bss[0]]
        quality["oracle"][f"recall@{k}"] = g[f"recall@{k}"]
        quality["oracle"][f"ndcg@{k}"] = g[f"ndcg@{k}"]
        quality[f"jaccard_vs_first@{k}"] = None if parity == "reference" else jaccard
    perf = []
    for bs in bss:
        for k in ks:
            g = by[k, bs]["median_ms"]
            for mode, ms in (("eager", 3 * g), ("graph", g)):
                perf.append(
                    {"k": k, "bs": bs, "mode": mode, "median_ms": ms, "spread": 0.01,
                     "unstable": False, "sm_mhz": sm_mhz, "cache_plans": None}
                )  # fmt: skip
    rec = {
        "schema_version": 1, "status": "ok", "dataset": "goodreads", "dim": 128,
        "suite": "filter", "filter_kind": r0["filter_kind"], "sweep": r0["sweep"],
        "algo": r0["impl"], "backend": backend, "params": {}, "seed": 0, "quality": quality,
        "perf": perf, "unstable": False, "env": {"sm_mhz": 210.0, "clocks_drift": False},
    }  # fmt: skip
    rec.update(over)
    return rec


def _official(rows: list[dict]) -> dict:
    rec = _synth(rows, backend="official", parity="vs_triton", jaccard=0.995,
                 params={"n_probe": 24})  # fmt: skip
    for e in rec["perf"]:
        e["cache_plans"] = False
        if e["mode"] == "graph":
            e.update(median_ms=None, spread=None, unstable=None, sm_mhz=None,
                     reason="not_capturable")  # fmt: skip
    return rec


@pytest.fixture
def setup(tmp_path):
    golden = tmp_path / "golden"
    golden.mkdir()
    for f in FILES:
        shutil.copy(GOLDEN / f, golden / f)
    st_t, st_o, v2_t, v2_o = (_golden_rows(f) for f in FILES)
    recs = [
        _synth(st_t, backend="triton", parity="reference", params={"n_probe": 24}),
        _synth(st_o, backend="torch", parity="vs_triton", jaccard=0.98, params={"n_probe": 24}),
        _synth(st_t, backend="triton", parity="reference", params={"n_probe": 32}),  # no golden
        _official(st_t),
        _synth(v2_t, backend="triton", parity="reference"),
        _synth(v2_o, backend="torch", parity="vs_triton", jaccard=1.0),
        _synth(v2_t, backend="triton", parity="reference", seed=1),  # filtered out by --seed 0
    ]
    return golden, recs


def _write(tmp_path: Path, recs: list[dict]) -> Path:
    p = tmp_path / "goodreads-d128.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return p


def _fails(rows):
    return [(r["cell"].split("-s0-")[0].split("c0_genre-")[1], r["check"]) for r in rows
            if r["verdict"] == "FAIL"]  # fmt: skip


def test_reproducing_records_pass_every_gate(c4, setup, tmp_path, capsys):
    golden, recs = setup
    rows, counts = c4.gate([_write(tmp_path, recs)], golden)
    assert counts["FAIL"] == 0 and counts["PASS"] > 0, _fails(rows)
    checks = {(r["cell"].split("c0_genre-")[1], r["check"], r["verdict"]) for r in rows}
    # The four golden cells are covered — the silvertorch cell at the suite's n_probe=24 is
    # the one at the algo defaults; the n_probe=32 and official cells have no golden.
    st = {b: f'silvertorch-{b}-s0-{{"n_probe":24}}' for b in ("triton", "torch", "official")}
    assert (st["triton"], "quality recall@100", "PASS") in checks
    assert (st["torch"], "latency k=1000 bs=16", "PASS") in checks
    assert ('silvertorch-triton-s0-{"n_probe":32}', "quality", "INFO") in checks
    assert (st["official"], "quality", "INFO") in checks
    # Parity: exact algo gated, inexact reported, official ≥ 0.99.
    assert ("linr_v2-torch-s0-{}", "parity", "PASS") in checks
    assert (st["torch"], "parity", "INFO") in checks
    assert (st["official"], "parity", "PASS") in checks
    assert (st["official"], "graph", "PASS") in checks
    assert (st["official"], "official", "PASS") in checks
    assert not any("-s1-" in r["cell"] for r in rows)  # --seed 0 default
    # The CLI: exit 0, table on stdout.
    assert c4.main([str(_write(tmp_path, recs)), "--golden", str(golden)]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "FAIL 0" in out


def test_each_gate_fails_alone(c4, setup, tmp_path):
    golden, recs = setup

    def run(mutate):
        rs = json.loads(json.dumps(recs))
        mutate(rs)
        rows, _ = c4.gate([_write(tmp_path, rs)], golden)
        return _fails(rows)

    # (1) quality: 2e-6 off is a FAIL, 5e-7 is not.
    def off(rs, d):
        rs[0]["quality"]["oracle"]["recall@100"] += d

    assert run(lambda rs: off(rs, 2e-6)) == [("silvertorch-triton", "quality recall@100")]
    assert run(lambda rs: off(rs, 5e-7)) == []

    # (2) latency: +7 % on one graph entry fails that (k, bs) only ...
    def slow(rs, factor, sm=None):
        e = next(
            e for e in rs[0]["perf"] if e["mode"] == "graph" and e["k"] == 500 and e["bs"] == 8
        )
        e["median_ms"] *= factor
        if sm is not None:
            e["sm_mhz"] = sm

    assert run(lambda rs: slow(rs, 1.07)) == [("silvertorch-triton", "latency k=500 bs=8")]
    # ... and a faster clock is normalised: 1410 MHz with the median scaled by 1140/1410
    # is the same work, PASS; the same clock with that median would be a FAIL.
    assert run(lambda rs: slow(rs, 1140 / 1410, 1410.0)) == []
    assert run(lambda rs: slow(rs, 1140 / 1410)) == [("silvertorch-triton", "latency k=500 bs=8")]

    # (2b) a missing sm_mhz falls back to env.sm_mhz (idle 210 here → normalised → FAIL),
    # and no clock at all compares raw.
    def noclock(rs, env):
        for e in rs[4]["perf"]:
            e["sm_mhz"] = None
        rs[4]["env"]["sm_mhz"] = env

    assert ("linr_v2-triton", "latency k=100 bs=1") in run(lambda rs: noclock(rs, 210.0))
    assert run(lambda rs: noclock(rs, None)) == []

    # (3) graph: a cudagraph skip is a null entry with a reason → graph FAIL + that latency.
    def skip(rs):
        e = next(
            e for e in rs[5]["perf"] if e["mode"] == "graph" and e["k"] == 100 and e["bs"] == 1
        )
        e.update(median_ms=None, reason="cudagraph_skips=1")

    assert run(skip) == [("linr_v2-torch", "latency k=100 bs=1"), ("linr_v2-torch", "graph")]

    # (4) parity: an exact algo below 1.0; official below 0.99.
    def jac(rs, i, v):
        rs[i]["quality"]["jaccard_vs_first@100"] = v

    assert run(lambda rs: jac(rs, 5, 0.9999)) == [("linr_v2-torch", "parity")]
    assert run(lambda rs: jac(rs, 3, 0.98)) == [("silvertorch-official", "parity")]
    assert run(lambda rs: jac(rs, 1, 0.5)) == []  # inexact: reported only

    # (5) stable.
    def unstable(rs):
        rs[4]["unstable"] = True
        rs[4]["perf"][0]["unstable"] = True
        rs[4]["perf"][0]["spread"] = 0.2

    assert run(unstable) == [("linr_v2-triton", "stable")]

    # official: a timed entry with the plan cache on, or a failed status.
    def cached(rs):
        rs[3]["perf"][0]["cache_plans"] = True

    assert run(cached) == [("silvertorch-official", "official")]
    assert run(lambda rs: rs[3].update(status="failed", stage="build")) == [
        ("silvertorch-official", "status")
    ]
    # coverage: dropping the linr_v2/torch record leaves its golden cell unmatched.
    rows, _ = c4.gate([_write(tmp_path, recs[:5])], golden)
    assert [(r["cell"], r["check"]) for r in rows if r["verdict"] == "FAIL"] == [
        ("goodreads-d128-c0_genre-linr_v2-torch.json", "coverage")
    ]
    # --seed -1 keeps every seed (the seed-1 record then lacks a golden: INFO, not FAIL).
    rows, counts = c4.gate([_write(tmp_path, recs)], golden, seed=None)
    assert counts["FAIL"] == 0 and any("-s1-" in r["cell"] for r in rows)
