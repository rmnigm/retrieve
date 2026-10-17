"""Reselect SilverTorch's pick per at-risk setting on the rerun's records, with the final run's rule.

usage: python reselect.py H2H_CSV RESULTS_DIR OUT_CSV [--same-grid]   (--same-grid: leave the follow-up runs out)
H2H_CSV = head_to_head_probe_budgets.csv (all 312 settings; `at_risk` marks the selection's 112, the rest
are reported only where the rerun timed their old pick point and their LiNR variant).

Rule (results/build_tables.py on origin/export): graph-mode median latency at seed 0; the pick is the
fastest SilverTorch bloom point whose oracle recall@k reaches the target with n_probe <= n_lists / 4
(`within_quarter`), else the fastest reaching it at any depth (`deeper`), else `not_reached`. The LiNR side
is the final run's picked variant (V1 / V2, or V3 at its pool) timed in the same process as the pick (the
main run, or the follow-up run `strerun-fu-*` that added deeper points); the verdict
applies the dataset's tie band (half SilverTorch's seed spread in the final run). `best_linr_*` is the
fastest LiNR arm timed here reaching the target, for reference only.
"""

import csv
import json
import sys
from pathlib import Path

h2h, results, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
same_grid = "--same-grid" in sys.argv[4:]
ALGO = {"linr_v1_filter_mask": "V1", "linr_v2": "V2", "linr_v3": "V3"}

pts = []  # one per (record, graph perf entry)
for f in results.glob("*/*.jsonl"):
    if f.name.endswith(".samples.jsonl"):
        continue
    for line in open(f):
        r = json.loads(line)
        if r["status"] == "failed" or r["seed"] != 0 or not r.get("perf"):
            continue
        if same_grid and r["suite"].startswith("strerun-fu-"):
            continue
        for e in r["perf"]:
            if e["mode"] != "graph" or e.get("median_ms") is None:
                continue
            pts.append(
                {
                    "suite": r["suite"],
                    "dataset": r["dataset"],
                    "sweep": r["sweep"],
                    "algo": r["algo"],
                    "params": r["params"],
                    "bs": e["bs"],
                    "k": e["k"],
                    "ms": e["median_ms"],
                    "recall": r["quality"]["oracle"][f"recall@{e['k']}"],
                    "unstable": bool(e["unstable"] or r["env"].get("clocks_drift")),
                    "sm_mhz": e["sm_mhz"],
                }
            )


def verdict(ratio, band):
    if ratio is None:
        return "not_reached"
    return "silvertorch_faster" if ratio < 1 - band else "linr_faster" if ratio > 1 + band else "tie"


def fnum(s):
    return float(s) if s not in ("", None) else None


rows = []
for x in csv.DictReader(open(h2h)):
    if x["dataset"] == "yambda":
        continue
    risky = x["silvertorch_over_linr"] == "" or float(x["silvertorch_over_linr"]) > 0.7
    ds, sw, bs, k, tgt = x["dataset"], x["filter"], int(x["batch"]), int(x["k"]), float(x["recall_target"])
    mine = [p for p in pts if p["dataset"] == ds and p["sweep"] == sw and p["bs"] == bs and p["k"] == k]
    st = [p for p in mine if p["algo"] == "silvertorch"]
    reach = [p for p in st if p["recall"] >= tgt]
    quarter = [p for p in reach if p["params"]["n_probe"] <= p["params"]["n_lists"] / 4]
    pick = min(quarter or reach, key=lambda p: p["ms"], default=None)
    outcome = "within_quarter" if quarter else "deeper" if reach else "not_reached"
    want = x["linr_variant"]
    pool = fnum(x["linr_v3_pool"])
    linr = [
        p
        for p in mine
        if ALGO.get(p["algo"]) == want and (want != "V3" or p["params"].get("candidate_pool_frac") == pool)
    ]
    assert len({p["suite"] for p in linr}) == len(linr), (ds, sw, bs, k, want)
    # a follow-up point (strerun-fu-*) is set against the LiNR timed in its own process
    li = next((p for p in linr if pick and p["suite"] == pick["suite"]), linr[0] if linr else None)
    old_pt = (fnum(x["silvertorch_n_lists"]), fnum(x["silvertorch_n_probe"]))
    covered = bool(st) and li is not None and (
        risky or any((p["params"]["n_lists"], p["params"]["n_probe"]) == old_pt for p in st)
    )
    best = min(
        (p for p in mine if p["algo"] in ALGO and p["recall"] >= tgt), key=lambda p: p["ms"], default=None
    )
    band = float(x["tie_band"])
    ratio = pick["ms"] / li["ms"] if pick and li else None
    rows.append(
        {
            "dataset": ds,
            "at_risk": risky,
            "filter": sw,
            "pass_rate": x["pass_rate"],
            "batch": bs,
            "k": k,
            "recall_target": tgt,
            "old_outcome": x["outcome"],
            "new_outcome": outcome if covered else "not_run",
            "old_pick": f"{x['silvertorch_n_lists']}x{x['silvertorch_n_probe']}" if x["silvertorch_n_lists"] else "",
            "new_pick": f"{pick['params']['n_lists']}x{pick['params']['n_probe']}" if pick else "",
            "old_recall": x["silvertorch_recall"],
            "new_recall": round(pick["recall"], 4) if pick else "",
            "old_max_recall": x["silvertorch_max_recall"],
            "new_max_recall": round(max(p["recall"] for p in st), 4) if st else "",
            "old_silvertorch_ms": x["silvertorch_ms"],
            "new_silvertorch_ms": round(pick["ms"], 4) if pick else "",
            "linr_variant": want + (f"@{pool}" if pool else ""),
            "old_linr_ms": x["linr_ms"],
            "new_linr_ms": round(li["ms"], 4) if li else "",
            "new_linr_recall": round(li["recall"], 4) if li else "",
            "old_ratio": x["silvertorch_over_linr"],
            "new_ratio": round(ratio, 4) if ratio is not None else "",
            "tie_band": band,
            "old_verdict": x["verdict"],
            "new_verdict": verdict(ratio, band) if covered else "not_run",
            "old_unstable": x["timing_unstable"],
            "new_unstable": bool((pick and pick["unstable"]) or (li and li["unstable"])),
            "best_linr_here": (ALGO[best["algo"]] + (f"@{best['params']['candidate_pool_frac']}" if best["algo"] == "linr_v3" else ""))
            if best
            else "",
            "best_linr_ms": round(best["ms"], 4) if best else "",
            "sm_mhz": f"{pick['sm_mhz']}/{li['sm_mhz']}" if pick and li else "",
        }
    )
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
run = [r for r in rows if r["new_verdict"] != "not_run"]
flips = [r for r in run if r["new_verdict"] != r["old_verdict"]]
print(
    f"{sum(r['at_risk'] for r in run)} / {sum(r['at_risk'] for r in rows)} at-risk and "
    f"{sum(not r['at_risk'] for r in run)} safe settings rerun, {len(flips)} verdicts changed"
)
for r in flips:
    print(
        f"  {'' if r['at_risk'] else '[safe] '}{r['dataset']} {r['filter']} bs{r['batch']} k{r['k']} @{r['recall_target']}: "
        f"{r['old_verdict']} ({r['old_ratio']}) -> {r['new_verdict']} ({r['new_ratio']})"
    )
