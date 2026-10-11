"""Gates 4-5 of the campaign-v2 freeze: every expected smoke cell has a record and none failed.

    uv run --directory evaluation python ../docs/artifacts/campaign-v2/freeze/check_records.py \
        RESULTS_DIR CONFIG_DIR [--timed]

Expected cells come from ``load_matrix`` over CONFIG_DIR's suites (written by suites_freeze.py)
for every (suite, dataset) that has a JSONL under RESULTS_DIR or is listed in the suite. Prints one
row per record: status, partial reasons, sm_mhz, unstable. A FAIL is a missing cell, a ``failed``
record, or a status other than ``ok`` / ``partial`` with only ``skip_perf`` (``ok`` is expected
where the suite declares no perf). With ``--timed`` every record must also be schema 4 and carry
``interleave`` (group, arms, position), per-entry ``ids_sha256`` and ``rounds`` with equal window
counts across the group, ``env.frac_windows_below_max`` and an existing per-query sidecar.
Exit 1 on any FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from bench.config import load_matrix


def records(root: Path) -> dict[tuple, dict]:
    out = {}
    for f in sorted(root.glob("*/*.jsonl")):
        if f.name.endswith(".samples.jsonl"):
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key = (r["suite"], r["dataset"], r["dim"], r["filter_kind"], r["sweep"], r["algo"],
                       r["backend"], json.dumps(r.get("params") or {}, sort_keys=True), r["seed"])  # fmt: skip
                out[key] = r
    return out


def expected(cfg: Path, suites: list[str] | None) -> set[tuple]:
    out = set()
    raw = yaml.safe_load((cfg / "suites.yaml").read_text())
    for sname, s in raw.items():
        if sname == "bloom" or (suites and sname not in suites):
            continue
        for ds in s["datasets"]:
            if not (cfg / f"{ds}.yaml").exists():
                continue
            for j in load_matrix(cfg / f"{ds}.yaml", cfg / "suites.yaml", sname):
                for q in j.query:
                    p = json.dumps({**j.build, **q}, sort_keys=True)
                    out.add((sname, ds, j.dim, j.filter_kind, j.sweep, j.algo, j.backend, p, j.seed))
    return out


def timed_problems(root: Path, r: dict, groups: dict) -> list[str]:
    bad = []
    if r.get("schema_version") != 4:
        bad.append(f"schema {r.get('schema_version')}")
    il = r.get("interleave")
    if not il or not {"group", "arms", "position"} <= set(il):
        bad.append(f"interleave {il}")
    else:
        groups[il["group"]].append(r)
    if (r.get("env") or {}).get("frac_windows_below_max") is None:
        bad.append("no env.frac_windows_below_max")
    pq = r.get("per_query")
    if not pq or not (root / pq).exists():
        bad.append(f"per_query sidecar missing ({pq})")
    for e in r.get("perf") or []:
        tag = f"k{e['k']} bs{e['bs']} {e['mode']}"
        if e.get("median_ms") is None:
            continue
        if not e.get("ids_sha256"):
            bad.append(f"{tag}: no ids_sha256")
        if not e.get("rounds"):
            bad.append(f"{tag}: no rounds")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("--timed", action="store_true")
    ap.add_argument("--suite", action="append")
    a = ap.parse_args()
    recs = records(a.results)
    exp = expected(a.config.resolve(), a.suite)
    fails = 0
    groups: dict[str, list] = defaultdict(list)
    for key in sorted(exp | set(recs), key=str):
        r = recs.get(key)
        cell = "/".join(str(x) for x in key)
        if r is None:
            print(f"FAIL missing  {cell}")
            fails += 1
            continue
        if key not in exp and (a.suite is None or key[0] in a.suite):
            print(f"INFO extra    {cell}")
        reasons = r.get("partial_reasons") or []
        ok = r["status"] == "ok" or (r["status"] == "partial" and set(reasons) <= {"skip_perf", "modes"})
        bad = [] if ok else [f"status {r['status']} {reasons} {r.get('stage', '')} {str(r.get('error', ''))[:200]}"]
        if a.timed:
            bad += timed_problems(a.results, r, groups)
        sm = sorted({e.get("sm_mhz") for e in r.get("perf") or [] if e.get("sm_mhz")})
        print(f"{'FAIL' if bad else 'ok  '} {r['status']:8s} {cell}  reasons={reasons} sm_mhz={sm} "
              f"unstable={r.get('unstable')} elapsed={r.get('elapsed_s', 0):.0f}s {'; '.join(bad)}")  # fmt: skip
        fails += bool(bad)
    for g, rs in groups.items():
        counts = {(e["k"], e["bs"], e["mode"], len(e.get("window_medians_ms") or []))
                  for r in rs for e in r.get("perf") or [] if e.get("median_ms") is not None}  # fmt: skip
        per_variant = defaultdict(set)
        for k, bs, mode, n in counts:
            per_variant[k, bs, mode].add(n)
        uneven = {v: n for v, n in per_variant.items() if len(n) > 1}
        arms = sorted(r["interleave"]["position"] for r in rs)
        print(f"group {g[:12]}: {len(rs)} arms positions {arms} windows {dict(per_variant)}"
              + (f" FAIL uneven {uneven}" if uneven else ""))  # fmt: skip
        fails += bool(uneven)
    print(f"\n{len(exp)} expected, {len(recs)} records, {fails} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
