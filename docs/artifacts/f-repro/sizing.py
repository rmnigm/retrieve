"""F-REPRO sizing (README.md): the cell inventory of the final pass and its GPU-h, from suites.yaml
expanded by the harness (`config.load_matrix`) plus the scratch suites the campaign ran (read from
their records), each cell matched to its newest record of the same key (`Job.key(params)` = the
record's key block) for its recorded `elapsed_s`.

Durations: the newest record at v2.9-v2.11 where one exists, else the newest older one (flagged);
timing windows of SilverTorch Triton cells on the redo ledger are rescaled by the ledger's measured
factor (`load.redo`: ST-WIDE-2 `wide` 0.97, `sparse?` 0.9 eager; ST-TOPK `topk` / `topk?` 0.8 at
bs >= 16). A cell without a record takes the median of its (suite, dataset, arm), then (suite, arm),
then arm. Process overhead (start, dataset load, oracle) is the per-leg wall / sum(elapsed) ratio
measured on the driver logs.

usage: sizing.py OUT LEGS_DIR
"""

import collections
import csv
import json
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "artifacts/campaign-v2/exhibits"))
sys.path.insert(0, str(HERE.parents[2] / "evaluation"))
import yaml  # noqa: E402
from load import CV_LABEL, redo  # noqa: E402

from bench import config, records  # noqa: E402

CFG = HERE.parents[2] / "evaluation" / "config"
ORDER = ["d1", "d1-c0e4", "v2", "v2.1", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v2.7", "v2.8",
         "v2.9", "v2.10", "v2.11"]  # fmt: skip
RECENT = {"v2.9", "v2.10", "v2.11"}
EXCLUDED_SCRATCH = {"stw2-laion30m", "stt-laion30m", "v1g"}  # library gates, not campaign cells
OBSOLETE_SCRATCH = {"ivf-tune": "superseded by `tune` / `tune-q`", "n95": "superseded by `tune`",
                    "router": "the router is dropped (decisions)", "c7-laion30m": "superseded by `c7-scorepath`"}
GROUP = collections.defaultdict(set)  # interleave group id -> its records' keys: they share its elapsed_s
LEDGER = {"wide": 0.97, "sparse?": 0.9, "topk": 0.8, "topk?": 0.8}


def family(suite, arm):
    """The comparison family a cell belongs to (one box each in the plan) and the T1 claims it feeds."""
    algo, backend = arm.split()[0].split("/")
    if suite in ("tune", "tune-q", "ivf-tune", "n95"):
        return "tune"
    if suite.startswith("bloomwidth"):
        return "C4 bloom width"
    if suite in ("h2h", "c7-scorepath", "c7-laion30m"):
        return "C7 ours/Meta"
    if suite.startswith("codesign") or "co-design" in arm:
        return "C5 partial/full"
    if backend == "official":
        return "C7 ours/Meta"
    if suite in ("c3", "c3-real") or backend == "torch":
        return "C3 torch/Triton"
    if algo == "linr_v3" or suite == "v3bits":
        return "C2 V3"
    if algo in ("linr_v1_filter_mask", "linr_v2"):
        return "C1 V1/V2"
    if algo == "silvertorch":
        return "C6 IVF"
    return "other"


def label(c):
    return CV_LABEL.get(c[:8], c[:8])


def canon(key):
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def arm_of(algo, backend, filter_kind, params):
    a = f"{algo}/{backend} {filter_kind}"
    if params.get("compile"):
        a += " compile"
    if "bloom_path" in params:
        a += " co-design"
    if "score_path" in params:
        a += f" {params['score_path']}"
    return a


def windows_ms(t):
    """measure.latency's cost of one timed variant at t ms (gpuh.py): 50 warm-up calls plus three
    windows of clamp(2 s / t, 1000, 5000) calls."""
    return 50 * t + 3 * min(5000, max(1000, 2000 / t)) * t


def own(r):
    """A record's share of its process time: an interleave group's records each carry the group's
    elapsed_s, so it is split evenly across them."""
    g = (r.get("interleave") or {}).get("group")
    return (r.get("elapsed_s") or 0.0) / (len(GROUP[g]) if g else 1)


def scaled(r):
    """own() with the ledger-affected timing windows rescaled to the newest code."""
    el = own(r)
    for e in r.get("perf") or []:
        t = e.get("median_ms")
        tag = redo(r, e["bs"], e["k"]) if t else ""
        f = 1.0
        for part in filter(None, tag.split("+")):
            if part == "sparse?" and e["mode"] != "eager":
                continue
            if part.startswith("topk") and e["bs"] < 16:
                continue
            f *= LEDGER[part]
        if f != 1.0:
            el -= (windows_ms(t) - windows_ms(t * f)) / 1000
    return el


def scan(legs):
    best, by_suite = {}, collections.defaultdict(dict)
    for f in Path(legs).rglob("*.jsonl"):
        if f.name.endswith(".samples.jsonl"):
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "suite" not in r or "env" not in r:
                continue
            r["_cv"] = label(r["env"]["code_version"])
            g = (r.get("interleave") or {}).get("group")
            if g:
                GROUP[g].add(canon(records.key_block(r)))
            r["_leg"] = f.relative_to(legs).parts[0]
            k = canon(records.key_block(r))
            rank = (r["status"] != "failed", ORDER.index(r["_cv"]) if r["_cv"] in ORDER else -1)
            if k not in best or rank > best[k][0]:
                best[k] = (rank, r)
    for k, (_, r) in best.items():
        by_suite[r["suite"]][k] = r
    return {k: r for k, (_, r) in best.items()}, by_suite


def overhead(legs):
    """Per-leg wall / sum(elapsed) where the driver log has a start and a `driver done` stamp and
    names one (dataset, suite)."""
    out = []
    for d in sorted(Path(legs).iterdir()):
        log = d / "logs" / "driver.log"
        if not log.exists():
            continue
        text = log.read_text(errors="replace").splitlines()
        stamps = [x.split()[0] for x in text if re.match(r"^\d{4}-\d\d-\d\dT", x)]
        done = [x for x in text if "driver done" in x]
        pairs = set(re.findall(r"load_matrix:\d+ - (\S+): \d+ jobs", "\n".join(text)))
        if len(stamps) < 2 or not done or len(pairs) != 1:
            continue
        wall = (datetime.fromisoformat(done[-1].split()[0]) - datetime.fromisoformat(stamps[0])).total_seconds()
        el = 0.0
        for f in d.rglob("*.jsonl"):
            if not f.name.endswith(".samples.jsonl"):
                el += sum(own(json.loads(x)) for x in open(f) if x.strip())
        if el > 0 and wall > 0:
            out.append((d.name, wall / el))
    return out


def main():
    out, legs = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    best, by_suite = scan(legs)
    suites = yaml.safe_load(open(CFG / "suites.yaml"))
    planned = []  # (suite, dataset, arm, key, seed)
    for s, spec in suites.items():
        if not isinstance(spec, dict) or "datasets" not in spec:
            continue
        for ds in spec["datasets"]:
            for job in config.load_matrix(CFG / f"{ds}.yaml", CFG / "suites.yaml", s):
                for p in job.cells():
                    planned.append(
                        (s, ds, arm_of(job.algo, job.backend, job.filter_kind, p), canon(job.key(p)), job.seed, job)
                    )
    rows = collections.defaultdict(lambda: {"cells": 0, "ran": 0, "recent": 0, "s": [], "est": 0, "cv": collections.Counter(), "s0": 0.0, "s0n": 0})
    durations = {}
    for s, ds, arm, k, seed, job in planned:
        g = rows[(s, ds, arm)]
        g["cells"] += 1
        r = best.get(k)
        if r:
            g["ran"] += 1
            g["recent"] += r["_cv"] in RECENT
            g["cv"][r["_cv"]] += 1
            durations[k] = scaled(r)
            g["s"].append(durations[k])
    med = {}
    for (s, ds, arm), g in rows.items():
        if g["s"]:
            med[(s, ds, arm)] = st.median(g["s"])
    med_sa = collections.defaultdict(list)
    med_a = collections.defaultdict(list)
    for (s, ds, arm), g in rows.items():
        med_sa[(s, arm)] += g["s"]
        med_a[arm] += g["s"]
    for s, ds, arm, k, seed, job in planned:
        if k in durations:
            if seed == 0:
                rows[(s, ds, arm)]["s0"] += durations[k]
                rows[(s, ds, arm)]["s0n"] += 1
            continue
        g = rows[(s, ds, arm)]
        v = med.get((s, ds, arm)) or (st.median(med_sa[(s, arm)]) if med_sa[(s, arm)] else None)
        v = v or (st.median(med_a[arm]) if med_a[arm] else None)
        g["est"] += 1
        g["s"].append(v or 0.0)
        if seed == 0:
            g["s0"] += v or 0.0
            g["s0n"] += 1
        g.setdefault("unknown", 0)
        g["unknown"] = g.get("unknown", 0) + (v is None)
    # records under suites.yaml suites that no planned cell matches (off the current grid)
    planned_keys = {k for _, _, _, k, _, _ in planned}
    off = collections.Counter()
    for s, recs in by_suite.items():
        if s in suites:
            for k, r in recs.items():
                if k not in planned_keys:
                    off[(s, r["dataset"], arm_of(r["algo"], r["backend"], r["filter_kind"], r["params"]))] += 1
    scratch = collections.defaultdict(lambda: {"cells": 0, "s": 0.0, "cv": collections.Counter()})
    for s, recs in by_suite.items():
        if s in suites:
            continue
        for k, r in recs.items():
            g = scratch[(s, r["dataset"], arm_of(r["algo"], r["backend"], r["filter_kind"], r["params"]))]
            g["cells"] += 1
            g["s"] += scaled(r)
            g["cv"][r["_cv"]] += 1
    ov = overhead(legs)
    with open(out / "inventory.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "family", "suite", "dataset", "arm", "cells", "with_record", "record_v2.9+",
                    "estimated", "gpu_h_cells", "seed0_cells", "seed0_gpu_h", "newest_cv"])  # fmt: skip
        for (s, ds, arm), g in sorted(rows.items()):
            w.writerow(["suites.yaml", family(s, arm), s, ds, arm, g["cells"], g["ran"], g["recent"], g["est"],
                        round(sum(g["s"]) / 3600, 3), g["s0n"], round(g["s0"] / 3600, 3),
                        " ".join(f"{c}:{n}" for c, n in sorted(g["cv"].items()))])  # fmt: skip
        for (s, ds, arm), g in sorted(scratch.items()):
            tag = " (excluded)" if s in EXCLUDED_SCRATCH else " (obsolete)" if s in OBSOLETE_SCRATCH else ""
            w.writerow(["scratch" + tag, family(s, arm), s, ds, arm, g["cells"], g["cells"],
                        sum(n for c, n in g["cv"].items() if c in RECENT), 0, round(g["s"] / 3600, 3),
                        g["cells"], round(g["s"] / 3600, 3), " ".join(f"{c}:{n}" for c, n in sorted(g["cv"].items()))])  # fmt: skip
    with open(out / "off-grid.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["suite", "dataset", "arm", "records_off_current_grid"])
        for (s, ds, arm), n in sorted(off.items()):
            w.writerow([s, ds, arm, n])
    with open(out / "overhead.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["leg", "wall_over_sum_elapsed"])
        w.writerows((a, round(b, 3)) for a, b in ov)
    tot = sum(sum(g["s"]) for g in rows.values()) / 3600
    print(f"{len(planned)} planned cells, {sum(g['ran'] for g in rows.values())} with a record, "
          f"{tot:.1f} cell-GPU-h; overhead median {st.median(b for _, b in ov):.2f} over {len(ov)} legs")  # fmt: skip


if __name__ == "__main__":
    main()
