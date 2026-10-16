"""FR-GATE step 4 (D1-G b): per FR leg, a few quality-only cells rerun at the tag, for compare_legs.py.

Runs in the tag tree's venv (`PYTHONPATH=<tree>/evaluation`, cwd `<tree>/evaluation`, GPU 0 under flock). Per leg and
dataset: one SilverTorch job (the smallest `n_lists`; on the >= 10 M sets only for the legs in ST_ON_BIG, a k-means build
there costs about two GPU-minutes) and the first job of every other (algo, backend); of each job only its first query
combo, so every rerun cell is one FR cell. `--skip-perf`, eager, `--resume` off (a fresh out dir), same seed. An oracle
this pod has not cached is never built: `bench.oracle.compute` is replaced by a guard, and a pre-pass over every chosen
job's sweep assets drops the cells whose oracle would have to be built (listed as `skipped` in `chosen-<tree>.json`;
compare_legs.py reports them per leg).

usage: quality_legs.py LEGS_YAML --tree fr|clause --out DIR [--dry] [--legs a,b,...]
"""

import dataclasses
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

from bench import inputs, oracle
from bench import run as run_mod
from bench.config import load_matrix

BIG = {
    "laion30m",
    "laion30m-synth",
    "laion30m-synth-s1",
    "laion30m-synth-s2",
    "laion10m",
    "laion10m-synth",
    "laion10m-synth-s1",
    "laion10m-synth-s2",
    "pubmed",
    "pubmed-synth",
    "pubmed-synth-s1",
    "pubmed-synth-s2",
}
ST_ON_BIG = {
    "curve-d256-pode-s0",
    "curve-synth-d256-pode-s0",
    "curve-d768-pod1-s0",
    "synth-mid-curve-pode-s0",
}
ST = {"silvertorch"}
MARK = "FR-GATE: oracle not cached on this pod; cell skipped"

args = sys.argv[1:]
legs = yaml.safe_load(Path(args[0]).read_text())
tree = args[args.index("--tree") + 1]
out = Path(args[args.index("--out") + 1])
dry = "--dry" in args
only = set(args[args.index("--legs") + 1].split(",")) if "--legs" in args else None
cfgs = {k: Path(v) for k, v in legs["config_dirs"].items()}
d = legs["defaults"]


def guard(*a, **k):
    raise RuntimeError(MARK)


oracle.compute = guard

chosen, per_leg = [], defaultdict(Counter)
for leg in legs["legs"]:
    if "files" in leg or leg.get("never_run") or leg.get("tree", d["tree"]) != tree:
        continue
    if only and leg["name"] not in only:
        continue
    cfg = cfgs[leg.get("config", d["config"])]
    seed = leg.get("seed", d["seed"])
    suites = yaml.safe_load((cfg / "suites.yaml").read_text())
    for ds in leg["datasets"]:
        jobs = []
        for suite in leg["suites"]:
            if ds in suites[suite]["datasets"]:
                jobs += load_matrix(
                    cfg / f"{ds}.yaml", cfg / "suites.yaml", suite, seeds=[seed]
                )
        by = defaultdict(list)
        for j in jobs:
            by[j.algo, j.backend].append(j)
        for (algo, backend), js in by.items():
            if algo in ST:
                if ds in BIG and leg["name"] not in ST_ON_BIG:
                    per_leg[leg["name"]]["st_not_rerun_big"] += 1
                    continue
                j = min(js, key=lambda j: (j.build.get("n_lists", 0), j.suite))
            else:
                j = js[0]
            chosen.append((leg["name"], dataclasses.replace(j, query=j.query[:1])))
            per_leg[leg["name"]][f"{algo}/{backend}"] += 1

chosen.sort(key=lambda t: (t[1].dataset, t[1].dim, t[0]))
for name, c in sorted(per_leg.items()):
    print(f"{name:48s} {dict(c)}")
print(
    f"{tree}: {len(per_leg)} legs, {len(chosen)} cells; datasets {sorted({j.dataset for _, j in chosen})}"
)
out.mkdir(parents=True, exist_ok=True)


def row(name, j, skipped):
    return {
        "leg": name,
        "suite": j.suite,
        "dataset": j.dataset,
        "dim": j.dim,
        "seed": j.seed,
        "skipped": skipped,
        "key": j.key(j.cells()[0]),
    }


if dry:
    (out / f"chosen-{tree}.json").write_text(
        json.dumps([row(n, j, False) for n, j in chosen], indent=1, default=str)
    )
    sys.exit(0)
t0 = time.time()
device = torch.device("cuda")
keep, rows = [], []
by_data = defaultdict(list)
for n, j in chosen:
    by_data[j.dataset, j.dim].append((n, j))
for (ds, dim), group in by_data.items():
    inp = inputs.load_inputs(group[0][1].data, device, with_filters=True)
    for n, j in group:
        try:
            run_mod.sweep_assets(j, inp, max(j.ks), device)
            keep.append((n, j))
            rows.append(row(n, j, False))
        except RuntimeError as exc:
            if MARK not in str(exc):
                raise
            rows.append(row(n, j, True))
            print(
                f"skip (oracle not cached here): {n} {j.suite}/{ds} {j.algo}/{j.backend} {j.sweep}"
            )
    del inp
    run_mod._release()
(out / f"chosen-{tree}.json").write_text(json.dumps(rows, indent=1, default=str))
print(
    f"pre-pass {(time.time() - t0) / 60:.1f} min: {len(keep)} cells to run, {len(chosen) - len(keep)} skipped"
)
counts = run_mod.run(
    [j for _, j in keep],
    out_dir=out / tree,
    resume=False,
    modes=["eager"],
    skip_perf=True,
    env_extra={"fr_gate": "quality subset"},
)
print(f"done in {(time.time() - t0) / 60:.1f} min: {dict(counts)}")
