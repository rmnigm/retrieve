"""T3x: official vs Triton (`h2h`) with device time next to end to end (README.md § t3x).

usage: t3x.py OUT TREE [TREE ...]

Device time is `kernels_us` (every kernel of the profiled eager call, H-KSUM) wherever a record of the
same cell, arm and seed carries it, in any tree given (the profile-only pass); else the top-8 sum
(`kernels`), a lower bound, and the row says which (`device_src`). Graph replays are not profiled.
"""

import collections
import csv
import statistics as st
import sys
from pathlib import Path

from load import cv, load, official_build

from bench import stats

ARMS = [
    ("triton", None, "eager"),
    ("triton", None, "graph"),
    ("official", "fp16", "eager"),
    ("official", "int32", "eager"),
]
# C7's "scorer in isolation", like for like: ours fuses probe list, bloom test and scoring in one kernel;
# Meta runs them as separate kernels (pod b's split, V-PROF3 a's full table). A side counts only when
# every one of its kernels is in the entry's kernel list (top-8 lists usually miss Meta's small ones).
SCORER = {
    "triton": ("_codesigned_probe_score_kernel",),
    "official": (
        "process_cluster",
        "generate_warp_payload",
        "generate_remaining_payload",
        "generate_cluster_warp_size",
    ),
}
BLOOM_SEARCH = "bloom_search"
HOST_BOUND = 0.5  # flag an arm whose host share exceeds half of its p50


def scorer_us(kernels, backend, filter_kind):
    """The comparable scorer scope's µs, or None when a kernel of that scope is not in the list."""
    names = SCORER[backend] + (
        (BLOOM_SEARCH,) if backend == "official" and filter_kind == "bloom" else ()
    )
    hit = {n: sum(k["us"] for k in kernels if n in k["kernel"]) for n in names}
    return None if not all(hit.values()) else sum(hit.values())


def cell_key(r, e):
    return (
        cv(r),
        r["dataset"],
        r["filter_kind"],
        r["sweep"],
        r["backend"],
        r["params"].get("score_path"),
        r["seed"],
        e["k"],
        e["bs"],
    )


def arm_of(r, mode):
    return (r["backend"], r["params"].get("score_path"), mode)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    # the profile-only pass is `partial` (skip_quality) by design; it only supplies kernels_us
    allrecs = [
        r for r in load(sys.argv[2:]) if r["suite"] == "h2h" and r["status"] in ("ok", "partial")
    ]
    ksum = {}  # (cv, dataset, filter, sweep, backend, score_path, seed, k, bs) -> eager perf entry
    for r in allrecs:
        for e in r["perf"] or []:
            if e["mode"] == "eager" and e.get("kernels_us") is not None:
                ksum[cell_key(r, e)] = e
    # an `ok` record timed with --profile carries its own kernels_us; only `partial` ones are profile-only
    recs = [r for r in allrecs if r["status"] == "ok"]
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
            full = [
                ksum[cell_key(r, e)] for r, e in es if a[2] == "eager" and cell_key(r, e) in ksum
            ]
            ks = [e["kernels"] for _, e in es if e.get("kernels")]
            top8 = st.median(sum(k["us"] for k in kk) for kk in ks) / 1000 if ks else None
            sc = [
                scorer_us(f.get("kernels") or [], a[0], key[2])
                for f in (full or [e for _, e in es if e.get("kernels")])
            ]
            scorer = st.median(sc) / 1000 if sc and all(x is not None for x in sc) else None
            if full:
                src = f"all ({len(full)}/{len(es)})"
                dev = st.median(f["kernels_us"] for f in full) / 1000
                launches = st.median(f["kernels_calls"] for f in full)
            else:
                src = "top8" if ks else ""
                dev = top8
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
                    "official_build": "/".join(sorted({official_build(r) for r, _ in es})),
                    "p50_ms": round(p50, 4),
                    "over_triton_eager": ""
                    if not ci
                    else f"{ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}]",
                    "device_ms": "" if dev is None else round(dev, 4),
                    "device_src": src,
                    "top8_ms": "" if top8 is None else round(top8, 4),
                    "launches": "" if launches is None else int(launches),
                    "scorer_ms": "" if scorer is None else round(scorer, 4),
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
        t = d.get("triton eager", {}).get("device_ms")
        for a in ("official/fp16 eager", "official/int32 eager"):
            o = d.get(a, {}).get("device_ms")
            if t and o:
                src = (
                    d["triton eager"]["device_src"].split()[0]
                    + " / "
                    + d[a]["device_src"].split()[0]
                )
                ts, os_ = d["triton eager"].get("scorer_ms"), d[a].get("scorer_ms")
                sr = round(os_ / ts, 2) if ts and os_ else ""
                dev_ratio.append((*key, a, round(o / t, 2), sr, src))
    with open(out / "t3x.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = [
        "# T3x: official vs Triton, end to end and device (NOT CITABLE)\n",
        "p50 = median over seeds (repeats) of the cell's p50; `/ triton eager` = paired per-round ratio over the "
        "interleaved group (seed x window), 95 % CI. Device = `kernels_us` (every kernel, H-KSUM) where `src` "
        "says `all`, else the top-8 kernels of one profiled eager call (`top8`, a lower bound); host share = "
        f"1 - device / p50; `host_bound` = host share > {HOST_BOUND}.\n",
        "| cv | dataset | filter | k | bs | arm | p50 ms | / triton eager | device ms | src | top-8 ms | launches "
        "| scorer ms | host share | unstable | ids canon = | jaccard min | max abs ds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
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
                    "device_ms",
                    "device_src",
                    "top8_ms",
                    "launches",
                    "scorer_ms",
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
        "\n## Device time, official / Triton eager (source: triton / official)\n",
        "Scorer scope, like for like: ours = `_codesigned_probe_score_kernel` (probe list, bloom test and "
        "scoring fused); Meta = `process_cluster*` + `generate_*payload*` + `generate_cluster_warp_size` (+ "
        "`bloom_search` on bloom). Blank when a kernel of the scope is missing from the recorded list (the "
        "top-8 lists miss Meta's small payload and bloom kernels).\n",
        "| cv | dataset | filter | k | bs | arm | device ratio | scorer-scope ratio | src |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in dev_ratio]
    (out / "t3x.md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} rows, {len(dev_ratio)} device ratios")


if __name__ == "__main__":
    main()
