#!/usr/bin/env python3
"""c4_latency_evidence.py — the evidence WP-4 gate 2 asks for when a graph median misses 5 %.

``c4_gate.py`` prints the verdict; this prints what is needed to *explain* one. WP-4 (2) says
a larger delta "must be explained"; H §11 and the golden README both say the golden's latency
columns are context, not a target. The three views below are the ones that decide whether a
missed row is the harness or the measurement:

1. ``--golden-noise`` — the golden harness's own run-to-run spread, from the repeat cells of
   [a1-rederive/l1-gate](a1-rederive/l1-gate) (same code, same box, same data, 2-3 runs). A
   golden column whose own repeats span more than the 5 % gate cannot be a 5 % target.
2. ``--windows`` — the v2 run's own three window medians per perf entry (``window_medians_ms``,
   ``spread``), i.e. the within-run noise of the number being compared.
3. ``--by-bs`` — the miss rate and ratio range grouped by batch size, which is where the
   bs=1 launch-latency floor separates from the bs=8/16 kernel-bound rows.

    python3 c4_latency_evidence.py <run.jsonl>... --golden evaluation/golden [--golden-sm-mhz 1155]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def golden_cells(golden_dir: Path) -> dict[tuple, dict]:
    out = {}
    for path in sorted(golden_dir.glob("*.json")):
        dataset, d_dim = path.stem.split("-")[:2]
        rows = json.loads(path.read_text())
        r0 = rows[0]
        key = (dataset, int(d_dim.lstrip("d")), r0["sweep"], r0["impl"], r0["backend"])
        out[key] = {(int(r["k"]), int(r["batch_size"])): float(r["median_ms"]) for r in rows}
    return out


def run_cells(paths: list[Path]) -> dict[tuple, dict]:
    last: dict[tuple, dict] = {}
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok" or (rec.get("params") or {}):
                if rec.get("params"):
                    continue
            key = (rec["dataset"], rec["dim"], rec["sweep"], rec["algo"], rec["backend"])
            last[key] = rec
    return last


def golden_noise() -> None:
    """Run-to-run spread of the *golden* harness's own median_ms (a1-rederive/l1-gate)."""
    groups: dict[str, dict] = defaultdict(dict)
    d = HERE / "a1-rederive" / "l1-gate"
    for f in sorted(glob.glob(str(d / "*.json"))):
        base = os.path.basename(f)
        tag, cell = (base.split("-", 1) if base[0] in "r" else ("run1", base))
        cell = cell.replace(".json", "").replace("goodreads-d128-c0_genre-", "")
        for r in json.loads(Path(f).read_text()):
            groups[cell].setdefault((r["k"], r["batch_size"]), {})[tag] = r["median_ms"]
    print("== golden-harness run-to-run spread (same code, same box) ==")
    by_bs: dict[int, list[float]] = defaultdict(list)
    for cell, d2 in sorted(groups.items()):
        for (k, bs), v in sorted(d2.items()):
            vals = list(v.values())
            if len(vals) < 2:
                continue
            spread = (max(vals) - min(vals)) / min(vals)
            by_bs[bs].append(spread)
            print(f"  {cell:<22} k={k:<5} bs={bs:<3} n={len(vals)} spread {spread * 100:5.1f}%")
    print("  -- by batch size --")
    for bs, s in sorted(by_bs.items()):
        print(
            f"  bs={bs:<3} n={len(s):<3} max {max(s) * 100:5.1f}%  median {statistics.median(s) * 100:5.1f}%"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="*", type=Path)
    ap.add_argument("--golden", type=Path, default=Path("evaluation/golden"))
    ap.add_argument("--golden-sm-mhz", type=float, default=1155.0)
    ap.add_argument("--latency-tol", type=float, default=0.05)
    ap.add_argument("--golden-noise", action="store_true")
    ap.add_argument("--windows", action="store_true")
    ap.add_argument("--by-bs", action="store_true")
    a = ap.parse_args()
    if a.golden_noise:
        golden_noise()
    if not a.jsonl:
        return 0
    gold, runs = golden_cells(a.golden), run_cells(a.jsonl)
    rows = []
    for key, rec in sorted(runs.items()):
        g = gold.get(key)
        if g is None:
            continue
        perf = {(int(e["k"]), int(e["bs"]), e["mode"]): e for e in rec.get("perf") or []}
        for (k, bs), g_ms in sorted(g.items()):
            e = perf.get((k, bs, "graph"))
            if e is None or e.get("median_ms") is None:
                continue
            ms = float(e["median_ms"])
            sm = e.get("sm_mhz") or (rec.get("env") or {}).get("sm_mhz")
            raw = ms / g_ms
            ratio = raw if sm is None or abs(sm - a.golden_sm_mhz) <= 0.02 * a.golden_sm_mhz else (
                ms * float(sm) / a.golden_sm_mhz / g_ms
            )
            rows.append(
                {
                    "cell": f"{key[3]}-{key[4]}" + (f"@{key[0]}" if key[0] != "goodreads" else ""),
                    "k": k, "bs": bs, "v2": ms, "golden": g_ms, "raw": raw, "ratio": ratio,
                    "sm": sm, "spread": e.get("spread"), "wins": e.get("window_medians_ms"),
                    "eager": (perf.get((k, bs, "eager")) or {}).get("median_ms"),
                    "miss": abs(ratio - 1.0) > a.latency_tol,
                }
            )
    if a.windows:
        print("\n== v2 graph medians, their own three windows, vs golden ==")
        for r in rows:
            wins = ", ".join(f"{w:.4f}" for w in (r["wins"] or []))
            flag = "MISS" if r["miss"] else "ok  "
            print(
                f"  {flag} {r['cell']:<28} k={r['k']:<5} bs={r['bs']:<3} v2 {r['v2']:.4f} "
                f"golden {r['golden']:.4f} ratio {r['ratio']:.3f} spread {float(r['spread'] or 0) * 100:4.1f}% "
                f"windows [{wins}] sm {r['sm']}"
            )
    if a.by_bs:
        print("\n== gate-2 miss rate by batch size ==")
        for bs in sorted({r["bs"] for r in rows}):
            sub = [r for r in rows if r["bs"] == bs]
            miss = [r for r in sub if r["miss"]]
            rr = [r["ratio"] for r in sub]
            ws = [float(r["spread"] or 0) for r in sub]
            print(
                f"  bs={bs:<3} {len(miss):>2}/{len(sub):<2} miss   ratio {min(rr):.3f}-{max(rr):.3f}   "
                f"v2 window spread max {max(ws) * 100:4.1f}%"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
