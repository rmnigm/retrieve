"""C5's one-cell gate and L4-b, as numbers. ``gate``: every C5 record of goodreads-d128
c0_genre silvertorch/triton against the pre-rename C4 record with the same key block and
``code_version`` — the key block must match and ``quality`` must be equal to the digit.
``l4b``: the chunk-64 linr_v4/triton record against the re-derived golden (per k, the golden's
bs=1 row) and against C4's chunk-16 record. Prints a table, exits 1 on any FAIL."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C4 = ROOT / "docs/plans/evaluation-harness-v2-artifacts/c4/results/filter/goodreads-d128.jsonl"
GOLDEN = ROOT / "evaluation/golden/goodreads-d128-c0_genre-linr_v4-triton.json"
KEY = ("dataset", "dim", "suite", "filter_kind", "sweep", "algo", "backend", "params", "seed")


def records(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def key(r):
    return json.dumps({k: r[k] for k in KEY}, sort_keys=True)


def flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def gate(c5_path):
    c4 = {(key(r), r["env"]["code_version"]): r for r in records(C4)}
    fails = 0
    for r in records(c5_path):
        ref = c4.get((key(r), r["env"]["code_version"]))
        print(f"\n{r['algo']}/{r['backend']} seed={r['seed']} params={r['params']} code_version={r['env']['code_version']}")
        print(f"  schema {r['schema_version']} status {r['status']} unstable {r['unstable']} "
              f"clocks_drift {r['env']['clocks_drift']} sm_mhz_load {r['env']['sm_mhz_load']} "
              f"clocks_locked {'absent' if 'clocks_locked' not in r['env'] else r['env']['clocks_locked']}")
        if ref is None:
            print("  FAIL no C4 record with this key block and code_version"); fails += 1; continue
        print(f"  key block == C4 record: PASS (C4 status {ref['status']}, schema {ref['schema_version']})")
        q5, q4 = dict(flat(r["quality"])), dict(flat(ref["quality"]))
        diff = {k: (q5.get(k), q4.get(k)) for k in set(q5) | set(q4) if q5.get(k) != q4.get(k)}
        diff = {k: v for k, v in diff.items() if not k.startswith(("jaccard", "parity", "score_max"))}
        print(f"  quality: {len(q4)} fields, {'PASS equal to the digit' if not diff else 'FAIL ' + str(diff)}")
        fails += bool(diff)
        for e in r["perf"]:
            if e["mode"] == "graph" and e.get("reason"):
                print(f"  FAIL graph entry k={e['k']} bs={e['bs']}: {e['reason']}"); fails += 1
        loads = sorted({e["sm_mhz"] for e in r["perf"] if e.get("sm_mhz")})
        spreads = [(e["k"], e["bs"], e["mode"], round(e["spread"], 4)) for e in r["perf"] if e.get("unstable")]
        print(f"  under-load sm_mhz samples {loads}; window-spread flags {spreads or 'none'}")
    return fails


def l4b(c5_path):
    golden = json.loads(GOLDEN.read_text())
    gold = {row["k"]: row for row in golden if row["batch_size"] == 1}
    (r64,) = [r for r in records(c5_path) if r["algo"] == "linr_v4" and r["seed"] == 0]
    (r16,) = [r for r in records(C4) if r["algo"] == "linr_v4" and r["backend"] == "triton"]
    print(f"\nlinr_v4/triton c0_genre, quality chunk 64 (C5) vs 16 (C4) vs the golden (chunk 64):")
    print(f"  {'k':>5} {'metric':<7} {'chunk64':>16} {'golden':>16} {'|d| vs golden':>14} {'chunk16 (C4)':>16} {'|d| vs C4':>10}")
    worst = 0.0
    for k in sorted(gold):
        for m in ("recall", "ndcg"):
            a, g, b = r64["quality"]["oracle"][f"{m}@{k}"], gold[k][f"{m}@{k}"], r16["quality"]["oracle"][f"{m}@{k}"]
            worst = max(worst, abs(a - g))
            print(f"  {k:>5} {m:<7} {a:>16.12f} {g:>16.12f} {abs(a - g):>14.2e} {b:>16.12f} {abs(a - b):>10.2e}")
    print(f"  worst |chunk64 - golden| = {worst:.2e} -> {'exact reproduction' if worst == 0 else 'within 1e-6' if worst <= 1e-6 else 'NOT reproduced'}")
    return 0


if __name__ == "__main__":
    what, path = sys.argv[1], Path(sys.argv[2])
    sys.exit(1 if {"gate": gate, "l4b": l4b}[what](path) else 0)
