"""V2-HIGHP: the PubMed d768 V2 before/after table from phase.sh's bench trees (`bench-before-r*`, `bench-after-r*`,
before = campaign-v2.2). Per (sweep, bs, mode): the median window time of each arm, after / before as the ratio of
geometric means over the window medians with a 95 % Welch t-interval on logs, and whether the after records'
`ids_sha256` equal the before ones. Prints a markdown table.

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
win, ids, mhz = defaultdict(list), defaultdict(set), defaultdict(list)
for arm in ("before", "after"):
    for f in sorted(
        Path(sys.argv[1]).glob(f"bench-{arm}-r*/v2-highp/pubmed-d768.jsonl")
    ):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r["status"] != "ok":
                continue
            for p in r["perf"]:
                if not p.get("window_medians_ms"):
                    continue
                key = (r["sweep"], r["pass_rate"], p["bs"], p["mode"])
                win[(arm, *key)] += p["window_medians_ms"]
                mhz[(arm, *key)] += p["window_sm_mhz"]
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


print(
    "| sweep | pass rate | bs | mode | V2 before ms | V2 after ms | after / before [95 % CI] | ids equal | sm_mhz |"
)
print("|---|---|---|---|---|---|---|---|---|")
for key in sorted(
    {k[1:] for k in win if k[0] == "after"}, key=lambda k: (-k[1], k[2], k[3])
):
    sw, pr, bs, mode = key
    b, a = win[("before", *key)], win[("after", *key)]
    r, lo, hi = ratio_ci(a, b)
    same = len({h for _, h in ids[key]}) == 1 and {x for x, _ in ids[key]} == {
        "before",
        "after",
    }
    clocks = mhz[("before", *key)] + mhz[("after", *key)]
    print(f"| {sw} | {pr:.4f} | {bs} | {mode} | {statistics.median(b):.3f} | {statistics.median(a):.3f} | "
          f"{r:.3f} [{lo:.3f}, {hi:.3f}] | {'yes' if same else 'NO'} | {min(clocks):.0f}-{max(clocks):.0f} |")  # fmt: skip
