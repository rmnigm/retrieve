"""``bench merge`` on two tiny trees: the accept rule per cell (ok over failed, stable over
unstable, latest start), the samples pairing, a missing or orphaned sidecar, a sidecar line that
pairs wrongly, and the manifest's counts."""

from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

from bench import merge, records

ENV = {"code_version": "cv", "commit": "0000000", "git_branch": "HEAD", "gpu": "cpu",
       "host": "box-a", "started": "2026-10-14T10:00:00+00:00", "dirty": False,
       "official_build": {"so_sha256": "ab", "so_bytes": 1}}  # fmt: skip
KEY = {"dataset": "goodreads", "dim": 128, "inputs": "sasrec-ssm-logq-d128", "suite": "filter",
       "filter_kind": "clause", "sweep": "c0_genre", "algo": "linr_v2", "backend": "triton",
       "params": {}, "seed": 0}  # fmt: skip


def _entry(k, bs, ms, n=3):
    return {"k": k, "bs": bs, "mode": "eager", "n": n, "median_ms": ms, "unstable": False}


def _rec(status="ok", unstable=False, perf=True, **over):
    env = {**ENV, **over.pop("env", {})}
    rec = {
        "schema_version": records.SCHEMA_VERSION, "status": status, **KEY, "path": "triton",
        "unstable": unstable, "quality": {"oracle": {"recall@100": 0.9}}, "env": env,
        "bloom_fp_rate": 0.01,  # a retired field: dropped on read
        "perf": [_entry(100, 1, 1.0), _entry(100, 16, 2.0),
                 {"k": 100, "bs": 1, "mode": "graph", "reason": "cuda_unavailable"}]
        if perf else None,
    }  # fmt: skip
    if status == "failed":
        rec |= {"stage": "perf", "error": "OOM", "perf": None, "quality": None}
    return {**rec, **over}


def _sample(rec, e):
    return {**records.key_block(rec), "k": e["k"], "bs": e["bs"], "mode": e["mode"],
            "ms": [e["median_ms"]] * e["n"]}  # fmt: skip


def _write(tree, recs, sidecar=True):
    path = tree / "filter" / "goodreads-d128.jsonl"
    for rec in recs:
        if sidecar:
            for e in rec.get("perf") or []:
                if e.get("n") is not None:
                    records.append_record(records.samples_path(path), _sample(rec, e))
        records.append_record(path, rec)
    return tree


@pytest.fixture
def trees(tmp_path):
    later = {"started": "2026-10-14T12:00:00+00:00"}
    a = _write(
        tmp_path / "leg-s0",
        [
            _rec(unstable=True),  # X: unstable, replaced by the rerun's stable record
            _rec(seed=1),  # Y: stable; the rerun's unstable copy loses
            _rec(status="failed", seed=2),  # Z: an OOM line ...
            _rec(seed=2, env=later),  # ... superseded by its retry in the same file
            _rec(seed=3, perf=None),  # W: quality-only
            _rec(seed=4, env={"host": "box-b"}),  # V: run on another machine ...
            _rec(seed=5, filter_kind="bloom", algo="linr_v2"),  # a LiNR bloom cell: dropped
        ],
    )  # fmt: skip
    b = _write(
        tmp_path / "leg-s0-rerun",
        [
            _rec(env=later),  # X's stable rerun
            _rec(seed=1, unstable=True, env=later),  # Y's unstable rerun
            _rec(seed=4, env=later),  # ... and later on box-a: the later one wins
        ],
        sidecar=False,
    )
    return a, b


def test_accept_rule_tables_and_manifest(trees, tmp_path):
    a, b = trees
    out = tmp_path / "tables"
    r = CliRunner().invoke(merge.merge, [str(b), str(a), "--out", str(out)])
    assert r.exit_code == 0, r.output
    recs = records.read_table(out / "records.parquet")
    assert len(recs) == 9 and list(recs[0])[:3] == ["subtree", "cell", "accepted"]
    assert "bloom_fp_rate" not in recs[0] and not any(r["filter_kind"] == "bloom" for r in recs)
    by = {(r["subtree"], r["seed"], r["status"], r["line"]): r for r in recs}
    acc = {k: v["accepted"] for k, v in by.items()}
    assert acc == {
        ("leg-s0", 0, "ok", 1): False, ("leg-s0-rerun", 0, "ok", 1): True,
        ("leg-s0", 1, "ok", 2): True, ("leg-s0-rerun", 1, "ok", 2): False,
        ("leg-s0", 2, "failed", 3): False, ("leg-s0", 2, "ok", 4): True,
        ("leg-s0", 3, "ok", 5): True,
        ("leg-s0", 4, "ok", 6): False, ("leg-s0-rerun", 4, "ok", 3): True,
    }  # fmt: skip
    assert by["leg-s0", 0, "ok", 1]["superseded_by"] == "leg-s0-rerun"
    assert by["leg-s0-rerun", 1, "ok", 2]["superseded_by"] == "leg-s0"
    assert by["leg-s0", 2, "failed", 3]["superseded_by"] == "leg-s0"
    x = by["leg-s0-rerun", 0, "ok", 1]
    assert (x["env_host"], x["env_started"]) == ("box-a", "2026-10-14T12:00:00+00:00")
    assert x["tag"] is None  # commit 0000000 is at no tag of this checkout
    assert x["oracle_recall@100"] == 0.9 and x["env_official_build_so_sha256"] == "ab"
    assert x["env_code_version"] == "cv" and x["cell"] == by["leg-s0", 0, "ok", 1]["cell"]
    assert json.loads(x["params"]) == {} and x["file"] == "filter/goodreads-d128.jsonl"
    timings = records.read_table(out / "timings.parquet")
    assert len(timings) == 3 * 7  # every perf entry of every record with perf
    assert sum(t["accepted"] for t in timings) == 3 * 4
    assert {t["perf_mode"] for t in timings} == {"eager", "graph"}
    assert {t["perf_reason"] for t in timings if t["perf_mode"] == "graph"} == {"cuda_unavailable"}
    samples = records.read_table(out / "samples.parquet")
    assert len(samples) == 2 * 4  # leg-s0's four timed records; the rerun has no sidecar
    assert [s["accepted"] for s in sorted(samples, key=lambda s: (s["seed"], s["bs"]))] == [
        False, False, True, True, True, True, False, False
    ]  # fmt: skip
    assert samples[0]["ms"] == [1.0, 1.0, 1.0] and samples[0]["n"] == 3
    man = yaml.safe_load((out / "manifest.yaml").read_text())
    assert man["tables_version"] == merge.TABLES_VERSION
    assert man["rows"] == {"records": 9, "timings": 21, "samples": 8}
    s = man["subtrees"]
    assert list(s) == ["leg-s0", "leg-s0-rerun"]
    assert s["leg-s0"]["records"] == {"failed": 1, "ok": 5} and s["leg-s0"]["unstable"] == 1
    assert s["leg-s0"]["samples"] == 8  # the dropped LiNR bloom record's lines paired, then left
    assert (s["leg-s0"]["accepted"], s["leg-s0"]["superseded"]) == (3, 3)
    assert s["leg-s0"]["supersedes"] == ["leg-s0", "leg-s0-rerun"]
    assert s["leg-s0-rerun"]["supersedes"] == ["leg-s0"] and s["leg-s0-rerun"]["samples"] == 0
    assert s["leg-s0"]["hosts"] == ["box-a", "box-b"] and s["leg-s0"]["path_in_repo"] is None
    assert s["leg-s0"]["started"] == ["2026-10-14T10:00:00+00:00", "2026-10-14T12:00:00+00:00"]
    (cell,) = man["cells"]
    assert cell["subtrees"] == ["leg-s0", "leg-s0-rerun"] and cell["cells"] == 5
    assert (cell["accepted_ok"], cell["accepted_failed"], cell["seeds"]) == (5, 0, [0, 1, 2, 3, 4])
    assert "cross_host_duplicates" not in man


def test_a_sidecar_that_does_not_pair(tmp_path):
    """An orphan line (a cell killed between its samples and its record) costs that file its
    samples, not the merge; a line that pairs with the wrong record raises."""
    tree = _write(tmp_path / "leg", [_rec()])
    records.append_record(tree / "filter" / "goodreads-d128.samples.jsonl", {"ms": [1.0]})
    (rec,) = merge.load(tree)
    assert rec["_samples"] == [None, None, None]
    tree = _write(tmp_path / "leg2", [_rec(seed=5)], sidecar=False)
    sidecar = tree / "filter" / "goodreads-d128.samples.jsonl"
    records.append_record(sidecar, _sample(_rec(seed=6), _entry(100, 1, 1.0)))
    records.append_record(sidecar, _sample(_rec(seed=5), _entry(100, 16, 2.0)))
    with pytest.raises(ValueError, match="does not pair"):
        merge.load(tree)


def test_two_trees_with_one_name_are_refused(trees, tmp_path):
    a, _ = trees
    r = CliRunner().invoke(merge.merge, [str(a), str(a), "--out", str(tmp_path / "t")])
    assert r.exit_code != 0 and "share a name" in r.output
