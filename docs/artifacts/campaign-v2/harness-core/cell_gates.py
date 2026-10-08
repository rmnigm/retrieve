"""G-cache, G-ids and G-dump read off real records (the GPU half of the CPU tests).

``CACHE`` holds V1 / V2 / postfilter at seeds 0-2 with perf (seeds 1-2 copy seed 0's quality),
``FRESH`` the same cells at seed 1 computed fresh (``--skip-perf``), any further trees more
records (silvertorch triton). Checks: the fresh seed-1 quality equals the copied one, metric by
metric (``==``); every copy names seed 0 and still has perf; seed-dependent arms never copy;
eager and graph ``ids_sha256`` per ``(bs, k)``; the largest gap between a sidecar's mean and
its record's ``recall@k``; the clock fields.

    python cell_gates.py CACHE FRESH [MORE ...] > out.json     (from evaluation/)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from bench import records, run


def main() -> None:
    cache, fresh, *more = map(Path, sys.argv[1:])
    recs = {
        p: [r for f in records.record_files(p) for r in records.read_records(f)]
        for p in (cache, fresh, *more)
    }
    out: dict = {"cache": [], "ids": [], "dump_max_abs_diff": 0.0, "clocks": []}
    by = {(r["algo"], json.dumps(r["params"]), r["seed"]): r for r in recs[cache]}
    for r in recs[fresh]:
        src = by[r["algo"], json.dumps(r["params"]), 0]
        copy = by[r["algo"], json.dumps(r["params"]), 1]
        sides = ("heldout", "oracle")
        out["cache"].append(
            {
                "algo": r["algo"],
                "params": r["params"],
                "fresh_seed1_equals_seed0": all(
                    r["quality"][s] == src["quality"][s] for s in sides
                ),
                "copies": [
                    by[r["algo"], json.dumps(r["params"]), s]["quality_source"] for s in (1, 2)
                ],
                "copies_have_perf": all(
                    by[r["algo"], json.dumps(r["params"]), s]["perf"] for s in (1, 2)
                ),
                "copy_equals_seed0": copy["quality"] == src["quality"],
            }
        )
    for root, rs in recs.items():
        for r in rs:
            if r["algo"] not in run.SEED_FREE_QUALITY:
                assert r["quality_source"] is None, r
            ent = {(e["bs"], e["k"], e["mode"]): e.get("ids_sha256") for e in r["perf"] or []}
            for bs, k, mode in ent:
                if mode == "eager" and (bs, k, "graph") in ent:
                    g = ent[bs, k, "graph"]
                    out["ids"].append(
                        {
                            "algo": r["algo"],
                            "backend": r["backend"],
                            "params": r["params"],
                            "seed": r["seed"],
                            "bs": bs,
                            "k": k,
                            "equal": None if g is None else g == ent[bs, k, mode],
                        }
                    )
            if r.get("per_query") and r["quality_source"] is None:
                z = np.load(root / r["per_query"])
                for side, name in (("oracle", "recall_oracle"), ("heldout", "heldout_recall")):
                    for k in r["ks"]:
                        v = r["quality"][side][f"recall@{k}"]
                        if v is not None:
                            m = float(np.nanmean(z[f"{name}@{k}"].astype(np.float64)))
                            out["dump_max_abs_diff"] = max(out["dump_max_abs_diff"], abs(m - v))
            if r["perf"]:
                out["clocks"].append(
                    {
                        "algo": r["algo"],
                        "seed": r["seed"],
                        "sm_mhz_load": r["env"]["sm_mhz_load"],
                        "frac_windows_below_max": r["env"]["frac_windows_below_max"],
                        "unstable": r["unstable"],
                    }
                )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
