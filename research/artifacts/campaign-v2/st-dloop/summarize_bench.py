"""ST-DLOOP: the interleaved PubMed before/after table from phase.sh's bench trees (rounds r1, r2 of
`before` = campaign-v2.1 and `after` = this tree, each a `bench run --interleave` of triton with official).

Per cell (filter kind, sweep, bs, mode) and backend: median of the 6 window medians (2 rounds × 3 windows)
per arm; after / before as the ratio of geometric means with a 95 % Welch t-interval over the log window
medians; official / Triton-after likewise; `ids` = whether every after record's `ids_sha256` equals the
before one (eager and graph). Prints a markdown table.

    python summarize_bench.py <phase out dir>
"""

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

T975 = {
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}

out = Path(sys.argv[1])
win = defaultdict(list)  # (arm, backend, fk, sweep, bs, mode) -> window medians
ids = defaultdict(set)
mhz = defaultdict(list)
for arm in ("before", "after"):
    for f in sorted(out.glob(f"bench-{arm}-r*/st-dloop/pubmed-d768.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["status"] != "ok":
                continue
            for p in r["perf"]:
                if not p.get(
                    "window_medians_ms"
                ):  # not run (official has no graph mode)
                    continue
                key = (r["backend"], r["filter_kind"], r["sweep"], p["bs"], p["mode"])
                win[(arm, *key)] += p["window_medians_ms"]
                mhz[(arm, *key)] += p["window_sm_mhz"]
                if r["backend"] == "triton":
                    ids[key].add((arm, p["ids_sha256"]))


def ratio_ci(num, den):
    a, b = [math.log(x) for x in num], [math.log(x) for x in den]
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (
        (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1)) if se > 0 else 10
    )
    t = T975[min(10, max(2, int(df)))]
    d = statistics.mean(a) - statistics.mean(b)
    return math.exp(d), math.exp(d - t * se), math.exp(d + t * se)


rows = []
print(
    "| kind | sweep | bs | mode | Triton before ms | Triton after ms | after / before [95 % CI] "
    "| official ms | official / after [95 % CI] | ids before = after | sm_mhz |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for key in sorted({k[1:] for k in win if k[0] == "after" and k[1] == "triton"},
                  key=lambda k: (k[1], k[2], k[3], k[4])):  # fmt: skip
    _, fk, sw, bs, mode = key
    b, a = win[("before", *key)], win[("after", *key)]
    r, lo, hi = ratio_ci(a, b)
    hashes = ids[key]
    same = len({h for _, h in hashes}) == 1 and {arm for arm, _ in hashes} == {
        "before",
        "after",
    }
    o = win.get(("after", "official", fk, sw, bs, mode))
    ocol = "–", "–"
    if o:
        ro, olo, ohi = ratio_ci(o, a)
        ocol = f"{statistics.median(o):.3f}", f"{ro:.2f} [{olo:.2f}, {ohi:.2f}]"
    clocks = mhz[("before", *key)] + mhz[("after", *key)]
    row = (fk, sw, bs, mode, f"{statistics.median(b):.3f}", f"{statistics.median(a):.3f}",
           f"{r:.3f} [{lo:.3f}, {hi:.3f}]", *ocol, "yes" if same else "NO",
           f"{min(clocks):.0f}-{max(clocks):.0f}")  # fmt: skip
    rows.append(row)
    print("| " + " | ".join(map(str, row)) + " |")
