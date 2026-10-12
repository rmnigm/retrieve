"""GPU-hours of the final pass and the NUMBERS legs from their results.parquet (CPU only).

Cells: elapsed_s once per record, once per (interleave_group, seed) for interleaved records, whose
elapsed_s is the whole group's wall time. Builds: build_s once per (dataset, dim, algo, backend, seed,
build_s), since every cell of one build repeats its build_s. Records missing from a parquet (failed,
timed out) are not counted.

usage: python gpu_hours.py <leg>/results.parquet ...
"""

import sys

import pyarrow.parquet as pq

KEY = ["path", "dataset", "dim", "inputs", "suite", "filter_kind", "sweep", "algo", "backend", "params", "seed"]


def hours(path):
    names = pq.read_schema(path).names
    cols = [c for c in KEY + ["elapsed_s", "build_s", "interleave_group"] if c in names]
    records = {}
    for r in pq.read_table(path, columns=cols).to_pylist():
        records.setdefault(tuple(r.get(k) for k in KEY), r)
    solo, groups, builds = 0.0, {}, {}
    for r in records.values():
        group, elapsed = r.get("interleave_group"), r["elapsed_s"] or 0.0
        if group:
            key = (group, r.get("seed"))
            groups[key] = max(groups.get(key, 0.0), elapsed)
        else:
            solo += elapsed
        if r.get("build_s"):
            builds[(r["dataset"], r["dim"], r["algo"], r["backend"], r.get("seed"), round(r["build_s"], 6))] = r["build_s"]
    return len(records), (solo + sum(groups.values())) / 3600, sum(builds.values()) / 3600


if __name__ == "__main__":
    for p in sys.argv[1:]:
        n, cells, builds = hours(p)
        print(f"{p}\trecords {n}\tcells_h {cells:.2f}\tbuilds_h {builds:.2f}")
