"""The follow-up plan: deeper SilverTorch points for every group where a setting the final run reached is
not reached on the same grid at the current code.

usage: python gen_followup.py PLAN_JSON RESELECT_CSV OUT_PLAN_JSON

Per such group and n_lists: n_probe doubling from the deepest timed point up to n_lists / 4, then
n_lists / 2 and 3 n_lists / 4 (the final run's follow-up depths), minus the points already timed. The
group's LiNR variants run again in the same process (a ratio is taken within one job). Feed the output to
`gen_suites.py ... strerun-fu-`.
"""

import csv
import json
import sys

plan = json.load(open(sys.argv[1]))
lost = {
    (r["dataset"], r["filter"], int(r["batch"]), int(r["k"]))
    for r in csv.DictReader(open(sys.argv[2]))
    if r["new_outcome"] == "not_reached" and r["old_outcome"] != "not_reached"
}
out = []
for e in plan:
    if (e["dataset"], e["sweep"], e["bs"], e["k"]) not in lost:
        continue
    have = {tuple(p) for p in e["points"]}
    new = []
    for n_lists in sorted({n for n, _ in have}):
        timed = {q for n, q in have if n == n_lists}
        if max(timed) > n_lists // 4:
            timed.add(n_lists // 4)  # gen_suites.py timed it
        ladder, q = set(), max(q for q in timed if q <= n_lists // 4)
        while q * 2 <= n_lists // 4:
            q *= 2
            ladder.add(q)
        ladder |= {n_lists // 2, 3 * n_lists // 4}
        new += [[n_lists, q] for q in sorted(ladder - timed)]
    e = {**e, "points": new}
    out.append(e)
json.dump(out, open(sys.argv[3], "w"), indent=1)
for e in out:
    print(e["dataset"], e["sweep"], e["bs"], e["k"], e["points"])
