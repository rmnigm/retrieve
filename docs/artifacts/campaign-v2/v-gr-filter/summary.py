"""V-GR-FILTER record summary: counts, clocks, wall time and per-arm quality / latency.

Usage: python summary.py /scratch/campaign-v2/results/filter/goodreads-d128.jsonl
"""

import json
import statistics as st
import sys
from collections import Counter, defaultdict

recs = [json.loads(line) for line in open(sys.argv[1])]


def arm(r):
    p = {k: v for k, v in r["params"].items() if k not in ("n_probe", "candidate_pool")}
    return f"{r['algo']}/{r['backend']}" + "".join(
        f" {k}={v}" for k, v in sorted(p.items())
    )


print("records", len(recs), dict(Counter(r["status"] for r in recs)))
print("code_version", dict(Counter(r["env"]["code_version"] for r in recs)))
print(
    "inputs",
    dict(Counter(r["inputs"] for r in recs)),
    "gpu",
    dict(Counter(r["env"]["gpu"] for r in recs)),
)
print("per arm x kind", dict(Counter((arm(r), r["filter_kind"]) for r in recs)))
print(
    "seeds",
    dict(Counter(r["seed"] for r in recs)),
    "sweeps",
    dict(Counter(r["sweep"] for r in recs)),
)
print(
    "unstable",
    sum(r["unstable"] for r in recs),
    "clocks_drift",
    sum(bool(r["env"]["clocks_drift"]) for r in recs),
)
win = [m for r in recs for p in r["perf"] for m in (p.get("window_sm_mhz") or [])]
print(
    f"timing windows {len(win)}, below 1410 MHz {sum(m < 1410 for m in win)}, sm_mhz {min(win)}-{max(win)}"
)
# an interleaved group's elapsed_s is the whole group cell's wall time: count it once
seen = set()
wall = 0.0
for r in recs:
    g = (r["interleave"] or {}).get("group") or id(r)
    if g not in seen:
        seen.add(g)
        wall += r["elapsed_s"]
print(f"sum of cell wall time {wall / 3600:.2f} h")

by = defaultdict(list)
for r in recs:
    by[(arm(r), r["filter_kind"], r["sweep"])].append(r)
print(
    "\narm | kind | sweep | n | recall_oracle@100 mean (min over seeds) | held-out recall@100 | ms bs1 eager / graph (median of seeds) | bs16 graph QPS"
)
for (a, kind, sweep), rs in sorted(by.items()):
    ro = [r["quality"]["oracle"]["recall@100"] for r in rs]
    ho = [r["quality"]["heldout"]["recall@100"] for r in rs]

    def ms(bs, mode):
        v = [
            p["median_ms"]
            for r in rs
            for p in r["perf"]
            if p["bs"] == bs
            and p["k"] == 100
            and p["mode"] == mode
            and p.get("median_ms")
        ]
        return st.median(v) if v else None

    def qps(bs, mode):
        v = [
            p["qps"]
            for r in rs
            for p in r["perf"]
            if p["bs"] == bs and p["k"] == 100 and p["mode"] == mode and p.get("qps")
        ]
        return st.median(v) if v else None

    fmt = lambda x, f: "—" if x is None else format(x, f)  # noqa: E731
    print(
        f"{a} | {kind} | {sweep} | {len(rs)} | {st.mean(ro):.4f} ({min(ro):.4f}) | {st.mean(ho):.4f} | "
        f"{fmt(ms(1, 'eager'), '.3f')} / {fmt(ms(1, 'graph'), '.3f')} | {fmt(qps(16, 'graph'), ',.0f')}"
    )
