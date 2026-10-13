"""Every FULL-RERUN job as resolved by this tree's `bench.config.load_matrix`, one sorted JSON line per job, for the release
cleanup's gate: the dump at `pre-release-cleanup` and at the cleaned tree must be identical.

usage: resolve_dump.py CONFIG_DIR SUITES_YAML > jobs.jsonl   (CONFIG_DIR, SUITES_YAML: prep_config.py's output)
"""

import json
import sys
from pathlib import Path

import yaml

from bench.config import load_matrix

cfg, suites_yaml = Path(sys.argv[1]), Path(sys.argv[2])
lines = []
for name, s in yaml.safe_load(suites_yaml.read_text()).items():
    if not isinstance(s, dict) or "datasets" not in s:
        continue
    for ds in s["datasets"]:
        for j in load_matrix(cfg / f"{ds}.yaml", suites_yaml, name):
            row = {
                f: getattr(j, f)
                for f in (
                    "dataset",
                    "dim",
                    "suite",
                    "filter_kind",
                    "sweep",
                    "clauses",
                    "algo",
                    "backend",
                    "path",
                    "build",
                    "query",
                    "ks",
                    "batch_sizes",
                    "seed",
                    "bloom",
                    "narrowed",
                    "timed",
                    "interleave",
                )
            }
            row["key"] = j.key()
            row["cells"] = j.cells()
            lines.append(json.dumps(row, sort_keys=True, default=str))
print("\n".join(sorted(lines)))
