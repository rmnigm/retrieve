"""Tabulate h2h.py JSON: per (mode, bs, arm) the wall median, the device µs by class, and the
kernel-only gate ratio at bs=16, ours (scorer + mask) over official-fp16 (scorer + mask).

    python h2h_table.py a.json [b.json ...]
"""

import json
import sys

print("| dataset | tree | mode | bs | arm | width | wall ms | scorer µs | mask µs | topk µs | prep µs | device µs | sm_mhz | unstable |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
gates = []
for path in sys.argv[1:]:
    d = json.load(open(path))
    tree = "padded" if "/base/" in d["retrieve"] or "/p" in d["retrieve"].split("scratchpad")[-1][:4] else "compact"
    tree = d.get("tree", tree)
    ku = {}
    for r in d["rows"]:
        c = r["device_us_by_class"]
        ku[(r["mode"], r["bs"], r["arm"])] = c.get("scorer", 0) + c.get("mask", 0)
        print(f"| {d['dataset']} | {tree} | {r['mode']} | {r['bs']} | {r['arm']} | {r['width']} | "
              f"{r['median_ms']:.4f} | {c.get('scorer', 0):.1f} | {c.get('mask', 0):.1f} | "
              f"{c.get('topk', 0):.1f} | {c.get('prep', 0):.1f} | {r['device_us_total']:.1f} | "
              f"{r['sm_mhz']} | {r['unstable']} |")
    for mode in ("none", "bloom", "exact"):
        t, o = ku.get((mode, 16, "triton")), ku.get((mode, 16, "official-fp16"))
        if t and o:
            gates.append(f"| {d['dataset']} | {tree} | {mode} | {t:.1f} | {o:.1f} | {t / o:.2f} |")
    for p in d["parity"]:
        gates.append(f"| {d['dataset']} | {tree} | parity {p['mode']} | jaccard {p['jaccard_at_k']:.6f} | dmax {p['score_max_abs_diff']:.2e} | |")
print()
print("| dataset | tree | mode | ours scorer+mask µs (bs16) | official-fp16 scorer+mask µs | ours / official |")
print("|---|---|---|---|---|---|")
print("\n".join(gates))
