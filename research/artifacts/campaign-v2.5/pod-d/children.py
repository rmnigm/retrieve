"""The campaign children of one (dataset, suite) under --interleave, one `ALGOS|BACKENDS` line each, in suite order;
exits non-zero unless they partition the suite's jobs exactly (every interleave unit inside one child).
Usage (from evaluation/): python children.py DATASET SUITE DIM"""

import sys
from pathlib import Path

from bench import cli
from bench.config import interleave_units, load_matrix

dataset, suite, dim = sys.argv[1], sys.argv[2], int(sys.argv[3])
jobs = load_matrix(Path(f"config/{dataset}.yaml"), Path("config/suites.yaml"), suite, dims=[dim])
children = cli._children(jobs, True)
owner = {}
for i, (_, _, algos, backends) in enumerate(children):
    for j in jobs:
        if j.algo in algos and j.backend in backends:
            assert id(j) not in owner, f"{j.key()} in two children"
            owner[id(j)] = i
assert len(owner) == len(jobs), f"{len(jobs) - len(owner)} jobs in no child"
for _, unit in interleave_units(jobs):
    assert len({owner[id(j)] for j in unit}) == 1, f"unit {unit[0].key()} split across children"
n_cells = sum(len(j.query) for j in jobs)
print(f"# {dataset}/{suite} d{dim}: {len(jobs)} jobs, {n_cells} cells, {len(children)} children", file=sys.stderr)
for _, _, algos, backends in children:
    print(f"{' '.join(algos)}|{' '.join(backends)}")
