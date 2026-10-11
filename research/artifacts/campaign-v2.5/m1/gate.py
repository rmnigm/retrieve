"""M1 gate: per (algo, params, bs, mode), the median over the loaded runs of the per-run median_ms must lie
within the arm's own repeat noise, |med(L) - med(A)| <= max(A) - min(A) over the alone runs.
Writes gate.md and gate.json next to the runs."""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
runs = defaultdict(lambda: {"A": [], "L": []})
for f in sorted(root.glob("[AL]*/filter/arxiv-d128.jsonl")):
    cond = f.parts[-3][0]
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["status"] == "failed":
            sys.exit(f"failed record in {f}: {r.get('stage')}")
        for p in r["perf"]:
            if p.get("median_ms") is None:
                continue
            key = (r["algo"], json.dumps(r["params"], sort_keys=True), p["bs"], p["mode"])
            runs[key][cond].append((p["median_ms"], p["sm_mhz"], p["unstable"], r["env"]["code_version"]))

rows, ok = [], True
for key, d in sorted(runs.items()):
    a = [m for m, *_ in d["A"]]
    lo = [m for m, *_ in d["L"]]
    noise = max(a) - min(a)
    delta = statistics.median(lo) - statistics.median(a)
    passed = abs(delta) <= noise
    ok &= passed
    rows.append({
        "algo": key[0], "params": json.loads(key[1]), "bs": key[2], "mode": key[3],
        "alone_ms": a, "loaded_ms": lo, "noise_ms": noise, "delta_ms": delta,
        "delta_rel": delta / statistics.median(a), "noise_rel": noise / statistics.median(a), "pass": passed,
        "sm_mhz_alone": [s for _, s, *_ in d["A"]], "sm_mhz_loaded": [s for _, s, *_ in d["L"]],
        "unstable": sum(u for *_, u, _ in d["A"] + d["L"]),
        "code_versions": sorted({c for *_, c in d["A"] + d["L"]}),
    })

lines = ["| algo | params | bs | mode | med alone ms | med loaded ms | Δ % | noise % | pass | sm_mhz A / L | unstable |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(
        f"| {r['algo']} | {r['params'].get('n_probe', '')} | {r['bs']} | {r['mode']} | "
        f"{statistics.median(r['alone_ms']):.4f} | {statistics.median(r['loaded_ms']):.4f} | "
        f"{100 * r['delta_rel']:+.2f} | {100 * r['noise_rel']:.2f} | {'yes' if r['pass'] else 'NO'} | "
        f"{'/'.join(map(str, sorted(set(r['sm_mhz_alone']))))} / {'/'.join(map(str, sorted(set(r['sm_mhz_loaded']))))} | "
        f"{r['unstable']} |")
verdict = "PASS" if ok else "FAIL"
text = f"M1 {verdict}: {sum(r['pass'] for r in rows)}/{len(rows)} variants within noise\n\n" + "\n".join(lines) + "\n"
(root / "gate.md").write_text(text)
(root / "gate.json").write_text(json.dumps({"verdict": verdict, "rows": rows}, indent=1))
print(text)
