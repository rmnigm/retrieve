"""plan.json -> the rerun's suites.yaml: one suite `strerun-<dataset>` per dataset.

usage: python gen_suites.py PLAN_JSON OUT_YAML [PREFIX]   (PREFIX strerun- by default; strerun-fu- for the
follow-up plan of gen_followup.py, which adds no n_lists / 4 point)

Per (dataset, sweep): SilverTorch on Meta's bloom (1,024 bits, 5 hashes, recheck off) at every
(n_lists, n_probe) the final run timed for the sweep's at-risk workloads, plus n_lists / 4 at each
n_lists with points deeper than that (so a within-quarter pick can appear); the LiNR variants the
final run picked (V3 at its pool), interleaved by [algo, n_lists] as the final run's suites were.
Seed 0 only: no table reads seeds 1-2.
"""

import json
import sys
from collections import defaultdict

import yaml

plan = json.load(open(sys.argv[1]))
prefix = sys.argv[3] if len(sys.argv) > 3 else "strerun-"
ALGO = {"V1": "linr_v1_filter_mask", "V2": "linr_v2", "V3": "linr_v3"}
by_ds = defaultdict(list)
for e in plan:
    by_ds[(e["dataset"], e["dim"])].append(e)

suites = {"bloom": {"m_bits": 1024, "k_hash": 5}}
for (ds, dim), es in by_ds.items():
    sweeps = sorted({e["sweep"] for e in es})
    ks_by_sweep = {s: sorted({e["k"] for e in es if e["sweep"] == s}) for s in sweeps}
    arms = []
    linr = defaultdict(set)  # algo -> sweeps
    v3 = defaultdict(set)  # sweep -> pools
    st = defaultdict(set)  # (sweep, n_lists) -> n_probe
    for e in es:
        for li in e["linr"]:
            if li["variant"] == "V3":
                v3[e["sweep"]].add(li["pool"])
            else:
                linr[ALGO[li["variant"]]].add(e["sweep"])
        for n_lists, n_probe in e["points"]:
            st[(e["sweep"], n_lists)].add(n_probe)
    for (s, n_lists), probes in st.items():
        if prefix == "strerun-" and max(probes) > n_lists // 4:
            probes.add(n_lists // 4)
    for algo in ("linr_v1_filter_mask", "linr_v2"):
        if linr[algo]:
            arms.append(
                {
                    "algo": algo,
                    "backends": ["triton"],
                    "filter_kinds": ["clause"],
                    "datasets": {ds: {"sweeps": sorted(linr[algo])}},
                }
            )
    for s, pools in sorted(v3.items()):
        arms.append(
            {
                "algo": "linr_v3",
                "backends": ["triton"],
                "filter_kinds": ["clause"],
                "datasets": {ds: {"sweeps": [s], "query": {"candidate_pool_frac": sorted(pools)}}},
            }
        )
    for (s, n_lists), probes in sorted(st.items()):
        arms.append(
            {
                "algo": "silvertorch",
                "backends": ["cuda"],
                "filter_kinds": ["bloom"],
                "build": {"n_lists": [n_lists], "recheck": [False]},
                "datasets": {ds: {"sweeps": [s], "query": {"n_probe": sorted(probes)}}},
            }
        )
    suites[f"{prefix}{ds}"] = {
        "datasets": [ds],
        "dims": [dim],
        "filter_kinds": ["bloom", "clause"],
        "ks": sorted({e["k"] for e in es}),
        "ks_by_sweep": {ds: ks_by_sweep},
        "batch_sizes": sorted({e["bs"] for e in es}),
        "seeds": [0],
        "sweeps": {ds: sweeps},
        "interleave": [{"by": ["algo", "n_lists"]}],
        "arms": arms,
    }
yaml.safe_dump(suites, open(sys.argv[2], "w"), sort_keys=False)
print(", ".join(k for k in suites if k != "bloom"))
