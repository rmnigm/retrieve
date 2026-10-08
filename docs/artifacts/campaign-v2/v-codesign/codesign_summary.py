"""V-CODESIGN summary from a codesign JSONL: per (sweep, n_probe, bs, mode) median over seeds of
partial / full median_ms and of their paired ratio (full / partial, same interleave group), recall,
clock windows. Usage: python codesign_summary.py <codesign/DATASET-d128.jsonl>"""
import json, statistics as st, sys
from collections import defaultdict

recs = [json.loads(l) for l in open(sys.argv[1])]
print("records:", len(recs), "ok:", sum(r["status"] == "ok" for r in recs),
      "unstable:", sum(bool(r.get("unstable")) for r in recs))
recs = [r for r in recs if r["status"] == "ok"]
t = defaultdict(dict)
rec = defaultdict(list)
for r in recs:
    bp, npb = r["params"]["bloom_path"], r["params"]["n_probe"]
    q = r["quality"]
    rec[(r["sweep"], npb, bp)].append((q.get("oracle") or {}).get("recall@100"))
    for e in r["perf"] or []:
        if e.get("median_ms") is not None:
            t[(r["sweep"], npb, e["bs"], e["mode"], r["seed"])][bp] = e["median_ms"]
c = defaultdict(lambda: defaultdict(list))
for (sw, npb, bs, mode, seed), a in t.items():
    for bp, v in a.items():
        c[(sw, npb, bs, mode)][bp].append(v)
    if len(a) == 2:
        c[(sw, npb, bs, mode)]["r"].append(a["full"] / a["partial"])
print(f"{'sweep':14} {'np':>4} {'bs':>2} {'mode':5} {'partial':>8} {'full':>8} {'full/partial':>18}")
for k in sorted(c):
    v = c[k]
    r = v["r"]
    print(f"{k[0]:14} {k[1]:>4} {k[2]:>2} {k[3]:5} {st.median(v['partial']):8.3f} {st.median(v['full']):8.3f} "
          f"{st.median(r):6.3f} [{min(r):.3f},{max(r):.3f}]" if r else f"{k} incomplete")
print("recall_oracle@100 (median over seeds):")
for k in sorted(rec):
    vals = [x for x in rec[k] if x is not None]
    print(" ", k, round(st.median(vals), 4) if vals else None)
win = [s for r in recs for e in r["perf"] or [] for s in (e.get("window_sm_mhz") or [])]
print(f"windows {len(win)}, below 1410 MHz {sum(s < 1410 for s in win)}, min {min(win) if win else None}")
