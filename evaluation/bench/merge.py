"""``bench merge``: the records of several fetched results trees as three typed tables and a
manifest (docs/system/evaluation.md § Merged tables)."""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click
import yaml
from loguru import logger

from bench import records

EVAL_DIR = Path(__file__).resolve().parents[1]
TABLES_VERSION = 1
STATUS_RANK = {"ok": 2, "partial": 1, "failed": 0}
META_ENV = ("host", "started", "code_version", "commit")  # lifted into named columns by _meta


def load(tree: Path) -> list[dict[str, Any]]:
    """Every record of ``tree`` with ``_subtree``, ``_file``, ``_line``, ``_cell`` and the
    ``_samples`` list (one per-call vector per perf entry that ran, or ``None``). The retired
    ``bloom_fp_rate`` field is dropped on read, and so is every bloom record of an algo other than
    SilverTorch (LiNR's and the postfilter's)."""
    files = records.record_files(tree)
    if not files:
        logger.info("{}: no <suite>/*.jsonl record files (an untimed or config subtree)", tree)
    out = []
    for path in files:
        recs = records.read_records(path)
        for i, rec in enumerate(recs, 1):
            rec["_subtree"] = tree.name
            rec["_file"] = path.relative_to(tree).as_posix()
            rec["_line"] = i
            rec["_cell"] = hashlib.sha1(records.record_key(rec).encode()).hexdigest()[:16]
            rec["_samples"] = [None] * len(rec.get("perf") or [])
            rec.pop("bloom_fp_rate", None)
        samples = records.samples_path(path)
        if samples.exists():
            _pair_samples(samples, records.read_records(samples), recs)
        out += [r for r in recs if not (r["filter_kind"] == "bloom" and r["algo"] != "silvertorch")]
    return out


def _pair_samples(path: Path, lines: list[dict[str, Any]], recs: list[dict[str, Any]]) -> None:
    """The j-th sidecar line of a variant belongs to the j-th record of that key with that
    variant timed: both files are appended in cell order, samples first."""
    slots = [
        (rec, i)
        for rec in recs
        for i, e in enumerate(rec.get("perf") or [])
        if e.get("n") is not None
    ]
    if len(slots) != len(lines):
        logger.warning(
            "{}: {} sample lines for {} timed variants (a cell killed between its samples and its "
            "record): no samples from this file", path, len(lines), len(slots),
        )  # fmt: skip
        return
    for (rec, i), line in zip(slots, lines, strict=True):
        e = rec["perf"][i]
        want = {**records.key_block(rec), "k": e["k"], "bs": e["bs"], "mode": e["mode"]}
        got = {k: line[k] for k in want if k in line}
        if got != want:
            raise ValueError(f"{path}: a sample line does not pair with its record: {got}")
        rec["_samples"][i] = line["ms"]


def accept(recs: list[dict[str, Any]]) -> None:
    """Per cell, the accepted record: ``ok`` over ``partial`` over ``failed``, stable over unstable,
    the latest ``env.started``, the later line; the rest are ``_superseded_by`` it."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pos, rec in enumerate(recs):
        rec["_pos"] = pos
        by[rec["_cell"]].append(rec)
    for cands in by.values():
        best = max(
            cands,
            key=lambda r: (
                STATUS_RANK[r["status"]], not r.get("unstable"), r["env"]["started"], r["_pos"]
            ),
        )  # fmt: skip
        for r in cands:
            r["_accepted"] = r is best
            r["_superseded_by"] = None if r is best else best["_subtree"]


@functools.cache
def tags_at(commit: str | None) -> str | None:
    """The git tags at ``commit`` in the checkout the merge runs in, comma-joined; null without."""
    if commit is None:
        return None
    try:
        out = subprocess.run(
            ["git", "tag", "--points-at", commit], cwd=EVAL_DIR, capture_output=True, text=True,
            check=True,
        ).stdout.split()  # fmt: skip
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = []
    return ",".join(sorted(out)) or None


def _flat_env(env: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in env.items():
        if k in META_ENV or k in records.ENV_COLUMNS:
            continue
        if isinstance(v, dict):
            out.update({f"env_{k}_{kk}": vv for kk, vv in v.items()})
        else:
            out[f"env_{k}"] = v
    return out


def _meta(rec: dict[str, Any]) -> dict[str, Any]:
    """The columns every table row starts with: verdict, key block, provenance."""
    env = rec["env"]
    return {
        "subtree": rec["_subtree"],
        "cell": rec["_cell"],
        "accepted": rec["_accepted"],
        "status": rec["status"],
        "unstable": bool(rec.get("unstable")),
        **records.key_columns(rec),
        "env_host": env["host"],
        "env_started": env["started"],
        "env_code_version": env["code_version"],
        "env_commit": env.get("commit"),
        "tag": tags_at(env.get("commit")),
    }


def tables(recs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The ``records``, ``timings`` and ``samples`` rows of every record."""
    rows: dict[str, list[dict[str, Any]]] = {"records": [], "timings": [], "samples": []}
    for rec in recs:
        meta = _meta(rec)
        cols = records.record_columns(rec)
        rows["records"].append(
            {
                **meta,
                "file": rec["_file"],
                "line": rec["_line"],
                "superseded_by": rec["_superseded_by"],
                **{k: v for k, v in cols.items() if k not in meta},
                **_flat_env(rec["env"]),
            }
        )
        for e, ms in zip(rec.get("perf") or [], rec["_samples"], strict=True):
            rows["timings"].append({**meta, **records.perf_columns(e)})
            if ms is not None:
                rows["samples"].append(
                    {**meta, "k": e["k"], "bs": e["bs"], "mode": e["mode"], "n": len(ms), "ms": ms}
                )
    return rows


def _sorted_set(values) -> list:
    return sorted({v for v in values if v is not None})


def manifest(trees: list[Path], recs: list[dict[str, Any]], rows: dict) -> dict[str, Any]:
    """``subtrees`` (provenance and counts per input tree, with what it supersedes) and ``cells``
    (per dataset / suite / algo / backend, the subtrees the accepted records come from)."""
    by_tree: dict[str, list] = {t.name: [] for t in trees}
    supersedes: dict[str, set] = defaultdict(set)
    for r in recs:
        by_tree[r["_subtree"]].append(r)
        if r["_superseded_by"] is not None:
            supersedes[r["_superseded_by"]].add(r["_subtree"])
    n_samples = Counter(s["subtree"] for s in rows["samples"])
    tags: dict[str, set] = defaultdict(set)
    for t in rows["records"]:
        tags[t["subtree"]].add(t["tag"])
    subtrees = {}
    for tree in trees:
        rs = by_tree[tree.name]
        env = [r["env"] for r in rs]
        man = tree / "MANIFEST.json"
        m = json.loads(man.read_text()) if man.exists() else {}
        started = _sorted_set(e["started"] for e in env)
        subtrees[tree.name] = {
            "path_in_repo": m.get("path_in_repo"),
            "manifest_sha256": records.sha256(man) if man.exists() else None,
            "uploaded": m.get("uploaded"),
            "hosts": _sorted_set(e["host"] for e in env),
            "gpus": _sorted_set(e.get("gpu") for e in env),
            "commits": _sorted_set(e.get("commit") for e in env),
            "tags": _sorted_set(tags[tree.name]),
            "branches": _sorted_set(e.get("git_branch") for e in env),
            "code_versions": _sorted_set(e["code_version"] for e in env),
            "started": [started[0], started[-1]] if started else None,
            "records": dict(sorted(Counter(r["status"] for r in rs).items())),
            "unstable": sum(1 for r in rs if r.get("unstable")),
            "accepted": sum(1 for r in rs if r["_accepted"]),
            "superseded": sum(1 for r in rs if not r["_accepted"]),
            "supersedes": sorted(supersedes[tree.name]),
            "samples": n_samples[tree.name],
        }
    groups: dict[tuple, list] = defaultdict(list)
    for r in recs:
        if r["_accepted"]:
            groups[r["dataset"], r["suite"], r["algo"], r["backend"]].append(r)
    cells = []
    for (dataset, suite, algo, backend), rs in sorted(groups.items()):
        cells.append(
            {
                "dataset": dataset,
                "suite": suite,
                "algo": algo,
                "backend": backend,
                "subtrees": _sorted_set(r["_subtree"] for r in rs),
                "cells": len({r["_cell"] for r in rs}),
                **{f"accepted_{s}": sum(1 for r in rs if r["status"] == s) for s in STATUS_RANK},
                "seeds": _sorted_set(r["seed"] for r in rs),
                "code_versions": _sorted_set(r["env"]["code_version"] for r in rs),
                "filter_kinds": _sorted_set(r["filter_kind"] for r in rs),
                "sweeps": _sorted_set(r["sweep"] for r in rs),
            }
        )
    return {
        "tables_version": TABLES_VERSION,
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rows": {name: len(r) for name, r in rows.items()},
        "subtrees": subtrees,
        "cells": cells,
    }


def build(trees: list[Path], out: Path) -> dict[str, Any]:
    """The three tables and ``manifest.yaml`` under ``out``; returns the manifest."""
    recs = [r for t in trees for r in load(t)]
    accept(recs)
    rows = tables(recs)
    out.mkdir(parents=True, exist_ok=True)
    for name, r in rows.items():
        records.write_table(r, out / f"{name}.parquet")
    man = manifest(trees, recs, rows)
    (out / "manifest.yaml").write_text(yaml.safe_dump(man, sort_keys=False))
    return man


@click.command()
@click.argument(
    "trees", nargs=-1, required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)  # fmt: skip
@click.option("--out", required=True, type=click.Path(path_type=Path), help="the tables' directory")
def merge(trees: tuple[Path, ...], out: Path) -> None:
    """Merge fetched results trees into records / timings / samples tables and a manifest."""
    names = Counter(t.name for t in trees)
    if dup := [n for n, c in names.items() if c > 1]:
        raise click.ClickException(f"two trees share a name: {dup}")
    man = build(sorted(trees), out)
    for name, n in man["rows"].items():
        click.echo(f"{out / name}.parquet: {n} rows")
    click.echo(
        f"{out / 'manifest.yaml'}: {len(man['subtrees'])} subtrees, {len(man['cells'])} "
        "(dataset, suite, algo, backend) groups"
    )
