"""Cells a `bench run --resume --interleave` would run: the run's own skip rule over the tree.
Usage (from evaluation/): python pending.py RESULTS CODE_VERSION DATASET SUITE [ALGO ...]"""

import json
import sys
from collections import Counter
from pathlib import Path

from bench import records
from bench.config import interleave_units, load_matrix

out, cv, dataset, suite, *algos = sys.argv[1:]
jobs = load_matrix(
    Path(f"config/{dataset}.yaml"), Path("config/suites.yaml"), suite, algos=algos or None
)
done: dict[Path, dict[str, str]] = {}
todo: Counter = Counter()
skipped = 0
for _, unit in interleave_units(jobs):
    path = records.record_path(Path(out), unit[0])
    if path not in done:
        done[path] = {records.record_key(r): r.get("status", "ok") for r in records.read_records(path)}
    for q in {json.dumps(q, sort_keys=True): q for j in unit for q in j.query}.values():
        cells = [(j, {**j.build, **q}) for j in unit if q in j.query]
        if all(done[path].get(records.resume_key(j.key(p), cv)) == "ok" for j, p in cells):
            skipped += len(cells)
            continue
        for j, p in cells:
            todo[j.algo, j.backend, j.filter_kind, j.sweep, json.dumps(p, sort_keys=True)] += 1
for k, n in sorted(todo.items()):
    print(n, *k)
print(f"{dataset}/{suite}: {sum(todo.values())} to run, {skipped} skipped (resume)")
