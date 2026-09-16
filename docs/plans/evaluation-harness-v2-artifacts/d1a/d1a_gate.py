#!/usr/bin/env python3
"""d1a_gate.py — the WP-5 gate clauses that are decidable from the records alone, evaluated
over roadmap D1 stage **a** (goodreads + arxiv, `filter`, d128, seed 0, triton/torch/official).

    python3 d1a_gate.py evaluation/results/filter/goodreads-d128.jsonl \
                        evaluation/results/filter/arxiv-d128.jsonl [--expected-cells N]

Clauses, one section each. Exit 1 on any FAIL.

1. **status** — every cell of the slice `ok`; a `failed` cell prints its stage and traceback.
2. **completeness** — the recorded key set equals `load_matrix`'s for the slice (no missing cell).
3. **batch scaling** — `median_ms(bs=16) < 16 · median_ms(bs=1)` for every (cell, k, mode).
4. **capture** — every `graph` entry on a capturable backend was measured (a `cudagraph_skips
   > 0` capture is a null entry with `reason`, i.e. a FAIL); on `official` the only accepted
   reason is `not_capturable` (O D7).

The two clauses the records cannot decide — ids identical across `mode`, and a byte-identical
rerun — are `d1a_mode_ids.py` and the `--force` rerun; see the validation record.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

KEY = ("dataset", "dim", "suite", "filter_kind", "sweep", "algo", "backend", "params", "seed")


def key_of(r: dict) -> tuple:
    return tuple(json.dumps(r[f], sort_keys=True) if f == "params" else r[f] for f in KEY)


def load(paths: list[Path]) -> dict[tuple, dict]:
    last: dict[tuple, dict] = {}
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                last[key_of(r)] = r
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", type=Path)
    ap.add_argument("--expected-cells", type=int, default=266)
    a = ap.parse_args()
    recs = load(a.records)
    bad = 0

    print("=" * 78)
    print("1. status")
    st = Counter(r["status"] for r in recs.values())
    print(f"   {len(recs)} cells: {dict(st)}")
    for k, r in sorted(recs.items()):
        if r["status"] != "ok":
            bad += 1
            print(f"   FAIL {k} status={r['status']} stage={r.get('stage')} "
                  f"reasons={r.get('partial_reasons')}")
            if r.get("traceback"):
                print("        " + r["traceback"].replace("\n", "\n        ")[:1500])
    print(f"   -> {'PASS' if st.get('ok', 0) == len(recs) else 'FAIL'}")

    print("2. completeness")
    ok = st.get("ok", 0)
    print(f"   expected {a.expected_cells} cells for the slice, {ok} recorded ok")
    if ok != a.expected_cells:
        bad += 1
    print(f"   -> {'PASS' if ok == a.expected_cells else 'FAIL'}")

    print("3. batch scaling: median_ms(bs=16) < 16 x median_ms(bs=1)")
    worst = None
    n_cmp = 0
    fails = 0
    for k, r in sorted(recs.items()):
        by = {(p["k"], p["bs"], p["mode"]): p for p in r["perf"] or []}
        for kk, bs, mode in list(by):
            if bs != 1:
                continue
            hi = by.get((kk, 16, mode))
            lo = by[(kk, 1, mode)]
            if not hi or hi["median_ms"] is None or lo["median_ms"] is None:
                continue
            ratio = hi["median_ms"] / lo["median_ms"]
            n_cmp += 1
            if worst is None or ratio > worst[0]:
                worst = (ratio, k, kk, mode, lo["median_ms"], hi["median_ms"])
            if ratio >= 16.0:
                fails += 1
                print(f"   FAIL ratio={ratio:.2f} {k} k={kk} {mode} "
                      f"bs1={lo['median_ms']:.4f} bs16={hi['median_ms']:.4f}")
    print(f"   {n_cmp} comparisons, worst ratio {worst[0]:.3f} "
          f"({worst[1][0]} {worst[1][5]} {worst[1][6]} {worst[1][4]} k={worst[2]} {worst[3]}: "
          f"{worst[4]:.4f} -> {worst[5]:.4f} ms)" if worst else "   no comparisons")
    bad += fails
    print(f"   -> {'PASS' if not fails else 'FAIL'}")

    print("4. graph capture (cudagraph_skips == 0 on every capturable arm)")
    reasons: Counter = Counter()
    fails = 0
    for k, r in sorted(recs.items()):
        backend = r["backend"]
        for p in r["perf"] or []:
            if p["mode"] != "graph":
                continue
            reason = p.get("reason")
            if reason is None:
                reasons["measured"] += 1
                if backend == "official":
                    fails += 1
                    print(f"   FAIL official graph entry was measured: {k} k={p['k']} bs={p['bs']}")
                continue
            reasons[f"{backend}:{reason}"] += 1
            if not (backend == "official" and reason == "not_capturable"):
                fails += 1
                print(f"   FAIL {k} k={p['k']} bs={p['bs']} reason={reason}")
    print(f"   {dict(reasons)}")
    bad += fails
    print(f"   -> {'PASS' if not fails else 'FAIL'}")

    print("-" * 78)
    print("informational")
    loads = sorted({r["env"]["sm_mhz_load"] for r in recs.values() if r["env"].get("sm_mhz_load")})
    per_entry = [p["sm_mhz"] for r in recs.values() for p in r["perf"] or [] if p.get("sm_mhz")]
    print(f"   env.sm_mhz_load over cells: {loads}")
    print(f"   perf[].sm_mhz under load: min {min(per_entry)} median "
          f"{statistics.median(per_entry)} max {max(per_entry)} over {len(per_entry)} entries")
    print(f"   env.sm_mhz_idle: {sorted({r['env'].get('sm_mhz_idle') for r in recs.values()})}")
    drift = [k for k, r in recs.items() if r["env"].get("clocks_drift")]
    print(f"   clocks_drift cells: {len(drift)}")
    uns: dict = defaultdict(Counter)
    for k, r in recs.items():
        for p in r["perf"] or []:
            if p.get("unstable"):
                uns["bs"][p["bs"]] += 1
                uns["mode"][p["mode"]] += 1
                uns["algo"][k[5]] += 1
    tot = sum(uns["bs"].values())
    print(f"   unstable perf entries: {tot} by bs {dict(uns['bs'])} by mode {dict(uns['mode'])} "
          f"by algo {dict(uns['algo'])}")
    par = defaultdict(list)
    for k, r in recs.items():
        q = (r.get("quality") or {})
        if q.get("parity", "reference") != "reference" and "jaccard_vs_first@100" in q:
            par[(k[0], k[5], k[6], q["parity"])].append(
                (q["jaccard_vs_first@100"], q.get("score_max_abs_diff"))
            )
    print("   parity vs the group's reference backend (min jaccard@100, max |dscore|):")
    for kk, vals in sorted(par.items()):
        js = [v[0] for v in vals if v[0] is not None]
        ds = [v[1] for v in vals if v[1] is not None]
        print(f"     {kk[0]:10s} {kk[1]:20s} {kk[2]:9s} {kk[3]:12s} n={len(vals):3d} "
              f"jaccard_min={min(js):.6f} dscore_max={max(ds):.3e}")
    codes = {r["env"]["code_version"] for r in recs.values()}
    print(f"   code_version: {codes}")
    print(f"   dirty: {sorted({r['env'].get('dirty') for r in recs.values()})} "
          f"branch: {sorted({r['env'].get('git_branch') for r in recs.values()})}")
    print("=" * 78)
    print("VERDICT:", "PASS" if bad == 0 else f"FAIL ({bad} problems)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
