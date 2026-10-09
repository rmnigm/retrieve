"""T3x: official vs Triton (`h2h`) with device time next to end to end (README.md § t3x).

usage: t3x.py OUT TREE [TREE ...]

Device time is the sum of the eager `--profile` call's top-8 kernels (`profile_once`), a lower bound,
so the host share 1 - device / p50 is an upper bound. Graph replays are not profiled.
"""

import collections
import csv
import statistics as st
import sys
from pathlib import Path

from load import cv, load

from bench import stats

ARMS = [
    ("triton", None, "eager"),
    ("triton", None, "graph"),
    ("official", "fp16", "eager"),
    ("official", "int32", "eager"),
]
HOST_BOUND = 0.5  # flag an arm whose host share exceeds half of its p50


def arm_of(r, mode):
    return (r["backend"], r["params"].get("score_path"), mode)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for r in load(sys.argv[2:]) if r["suite"] == "h2h" and r["status"] == "ok"]
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        for e in r["perf"] or []:
            if e.get("median_ms") is not None:
                key = (cv(r), r["dataset"], r["filter_kind"], e["k"], e["bs"])
                cells[key][arm_of(r, e["mode"])].append((r, e))
    rows = []
    for key, arms in sorted(cells.items()):
        base = {
            (r["seed"], (r.get("interleave") or {}).get("group")): e
            for r, e in arms.get(ARMS[0], [])
        }
        for a in ARMS:
            if a not in arms:
                continue
            es = arms[a]
            p50 = st.median(e["median_ms"] for _, e in es)
            ks = [e["kernels"] for _, e in es if e.get("kernels")]
            dev = st.median(sum(k["us"] for k in kk) for kk in ks) / 1000 if ks else None
            launches = st.median(sum(k["calls"] for k in kk) for kk in ks) if ks else None
            num, den = [], []
            for r, e in es:
                b = base.get((r["seed"], (r.get("interleave") or {}).get("group")))
                if b and len(b["window_medians_ms"]) == len(e["window_medians_ms"]):
                    num += e["window_medians_ms"]
                    den += b["window_medians_ms"]
            ci = stats.paired_ratio_ci(num, den) if num and a != ARMS[0] else None
            q = [r["quality"] for r, _ in es]
            ids = (
                [
                    e.get("ids_sha256_canon")
                    == base.get((r["seed"], (r.get("interleave") or {}).get("group")), {}).get(
                        "ids_sha256_canon"
                    )
                    for r, e in es
                ]
                if a[0] == "official"
                else []
            )
            rows.append(
                {
                    "code_version": key[0],
                    "dataset": key[1],
                    "filter": key[2],
                    "k": key[3],
                    "bs": key[4],
                    "arm": f"{a[0]}{'/' + a[1] if a[1] else ''} {a[2]}",
                    "seeds": len(es),
                    "p50_ms": round(p50, 4),
                    "over_triton_eager": ""
                    if not ci
                    else f"{ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}]",
                    "device_ms_top8": "" if dev is None else round(dev, 4),
                    "launches_top8": "" if launches is None else int(launches),
                    "host_share_max": "" if dev is None else round(1 - dev / p50, 3),
                    "host_bound": "" if dev is None else (1 - dev / p50) > HOST_BOUND,
                    "unstable": sum(bool(e.get("unstable")) for _, e in es),
                    "ids_canon_eq": "" if not ids else f"{sum(ids)}/{len(ids)}",
                    "jaccard_min": ""
                    if a[0] != "official"
                    else min(x.get(f"jaccard_vs_first@{key[3]}") or 1.0 for x in q),
                    "max_abs_ds": ""
                    if a[0] != "official"
                    else max(x.get("score_max_abs_diff") or 0.0 for x in q),
                }
            )
    dev_ratio = []
    by = collections.defaultdict(dict)
    for x in rows:
        by[(x["code_version"], x["dataset"], x["filter"], x["k"], x["bs"])][x["arm"]] = x
    for key, d in sorted(by.items()):
        t = d.get("triton eager", {}).get("device_ms_top8")
        for a in ("official/fp16 eager", "official/int32 eager"):
            o = d.get(a, {}).get("device_ms_top8")
            if t and o:
                dev_ratio.append((*key, a, round(o / t, 2)))
    with open(out / "t3x.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = [
        "# T3x: official vs Triton, end to end and device (NOT CITABLE)\n",
        "p50 = median over seeds (repeats) of the cell's p50; `/ triton eager` = paired per-round ratio over the "
        "interleaved group (seed x window), 95 % CI. Device = top-8 kernels of one profiled eager call (a lower "
        f"bound); host share = 1 - device / p50 (an upper bound); `host_bound` = host share > {HOST_BOUND}.\n",
        "| cv | dataset | filter | k | bs | arm | p50 ms | / triton eager | device ms | launches | host share | "
        "unstable | ids canon = | jaccard min | max abs ds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in rows:
        md.append(
            "| "
            + " | ".join(
                str(x[c])
                for c in (
                    "code_version",
                    "dataset",
                    "filter",
                    "k",
                    "bs",
                    "arm",
                    "p50_ms",
                    "over_triton_eager",
                    "device_ms_top8",
                    "launches_top8",
                    "host_share_max",
                    "unstable",
                    "ids_canon_eq",
                    "jaccard_min",
                    "max_abs_ds",
                )
            )
            + " |"
        )
    md += [
        "\n## Device time, official / Triton eager (top-8 sums)\n",
        "| cv | dataset | filter | k | bs | arm | ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in dev_ratio]
    (out / "t3x.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} rows, {len(dev_ratio)} device ratios")


if __name__ == "__main__":
    main()
