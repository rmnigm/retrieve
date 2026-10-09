"""C5-OURS cells summary from one codesign JSONL (our Triton SilverTorch, bloom_path partial vs full, interleaved).

Per (sweep, n_probe, bs, mode): median over seeds of partial / full median_ms, the paired full / partial ratio (same
seed = same interleave group; median [min, max] over seeds), partial / full peak_fwd_mib, and whether the two paths
returned the same ids (`ids_sha256_canon`) in every seed. Then recall@100 per path and the clock windows.

    python c5_summary.py <codesign/DATASET-dDIM.jsonl>
"""

import json
import statistics as st
import sys
from collections import defaultdict

recs = [json.loads(x) for x in open(sys.argv[1])]
print("records:", len(recs), "ok:", sum(r["status"] == "ok" for r in recs),
      "unstable:", sum(bool(r.get("unstable")) for r in recs),
      "code_version:", sorted({r["env"]["code_version"][:8] for r in recs}))  # fmt: skip
recs = [r for r in recs if r["status"] == "ok" and r["backend"] == "triton"]
cell = defaultdict(dict)
recall = defaultdict(list)
for r in recs:
    bp, npb = r["params"]["bloom_path"], r["params"]["n_probe"]
    recall[(r["sweep"], npb, bp)].append(
        (r["quality"].get("oracle") or {}).get("recall@100")
    )
    for e in r["perf"] or []:
        if e.get("median_ms") is not None:
            cell[(r["sweep"], npb, e["bs"], e["mode"], r["seed"])][bp] = e
rows = defaultdict(lambda: defaultdict(list))


def mib(xs):
    return f"{st.median(xs):.2f}" if xs else "-"


for (sw, npb, bs, mode, seed), a in cell.items():
    k = (sw, npb, bs, mode)
    for bp, e in a.items():
        rows[k][bp].append(e["median_ms"])
        if e.get("peak_fwd_mib") is not None:  # eager only
            rows[k][bp + "_mib"].append(e["peak_fwd_mib"])
    if len(a) == 2:
        rows[k]["r"].append(a["full"]["median_ms"] / a["partial"]["median_ms"])
        rows[k]["ids"].append(
            a["full"]["ids_sha256_canon"] == a["partial"]["ids_sha256_canon"]
        )
print(f"{'sweep':14} {'np':>5} {'bs':>2} {'mode':5} {'partial':>8} {'full':>8} {'full/partial':>22} "
      f"{'MiB p':>8} {'MiB f':>8} ids")  # fmt: skip
for k in sorted(rows):
    v = rows[k]
    if not v["r"]:
        print(k, "incomplete")
        continue
    r = v["r"]
    print(f"{k[0]:14} {k[1]:>5} {k[2]:>2} {k[3]:5} {st.median(v['partial']):8.3f} {st.median(v['full']):8.3f} "
          f"{st.median(r):8.3f} [{min(r):.3f},{max(r):.3f}] {mib(v['partial_mib']):>8} "
          f"{mib(v['full_mib']):>8} {'equal' if all(v['ids']) else 'DIFFER'}")  # fmt: skip
print("recall_oracle@100 (median over seeds):")
for k in sorted(recall):
    vals = [x for x in recall[k] if x is not None]
    print(" ", k, round(st.median(vals), 4) if vals else None)
win = [s for r in recs for e in r["perf"] or [] for s in (e.get("window_sm_mhz") or [])]
print(
    f"windows {len(win)}, below 1410 MHz {sum(s < 1410 for s in win)}, min {min(win) if win else None}"
)
