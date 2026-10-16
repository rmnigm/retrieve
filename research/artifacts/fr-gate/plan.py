"""FR-GATE step 1: the planned cells of every record-bearing leg of `legs.yaml`, resolved with the tag's own
`bench.config.load_matrix` (run it with the tag tree's venv and `PYTHONPATH=<tree>/evaluation`, once per `tree`).
Writes one JSON line per planned cell: leg, the subtrees that may hold it, seed, suite, dataset, `timed`, the
expected tag, and `key` = the record's key block with `params` as canonical JSON (what `records.key_columns`
gives a record). `--tree fr|clause` selects the legs; `--check` also runs check_suites.py's dry resolve on every
planned suite of every config dir of that tree.

usage: plan.py LEGS_YAML --tree fr|clause --out PLAN_JSONL [--check]
"""

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from bench.config import load_matrix

args = sys.argv[1:]
legs = yaml.safe_load(Path(args[0]).read_text())
tree = args[args.index("--tree") + 1]
out = Path(args[args.index("--out") + 1])
check = "--check" in args
cfgs = {k: Path(v) for k, v in legs["config_dirs"].items()}
tag = legs["trees"][tree]["tag"]
d = legs["defaults"]

rows, per_leg, by_cfg = [], Counter(), defaultdict(set)
for leg in legs["legs"]:
    if "files" in leg or leg.get("tree", d["tree"]) != tree:
        continue
    cfg = cfgs[leg.get("config", d["config"])]
    seed = leg.get("seed", d["seed"])
    suites_yaml = cfg / "suites.yaml"
    suites = yaml.safe_load(suites_yaml.read_text())
    for suite in leg["suites"]:
        by_cfg[cfg].add(suite)
        for ds in leg["datasets"]:
            if ds not in suites[suite]["datasets"]:
                continue  # run_leg.sh's in_suite skip
            for job in load_matrix(
                cfg / f"{ds}.yaml", suites_yaml, suite, seeds=[seed]
            ):
                for params in job.cells():
                    kb = job.key(params)
                    key = {**kb, "params": json.dumps(kb["params"], sort_keys=True)}
                    rows.append(
                        {
                            "leg": leg["name"],
                            "subtrees": [leg["name"], *leg.get("supplements", [])],
                            "never_run": bool(leg.get("never_run")),
                            "seed": seed,
                            "suite": suite,
                            "dataset": ds,
                            "dim": job.dim,
                            "timed": job.timed,
                            "tag": tag,
                            "key": key,
                        }
                    )
                    per_leg[leg["name"]] += 1
with out.open("w") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")
for name, n in per_leg.items():
    print(f"{name:48s} {n:5d} cells")
print(f"{tree}: {len(per_leg)} legs, {len(rows)} planned cells -> {out}")

if check:
    here = Path(__file__).resolve().parent.parent / "full-rerun" / "check_suites.py"
    rc = 0
    for cfg, names in sorted(by_cfg.items()):
        r = subprocess.run(
            [
                sys.executable,
                str(here),
                str(cfg),
                str(cfg / "suites.yaml"),
                "--only",
                ",".join(sorted(names)),
            ],
            capture_output=True,
            text=True,
        )
        print(
            f"check_suites.py {cfg}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''} rc={r.returncode}"
        )
        for line in r.stdout.splitlines():
            if line.startswith("PROBLEM"):
                print("  ", line)
        if r.returncode:
            print(r.stderr[-2000:])
        rc |= r.returncode
    sys.exit(rc)
