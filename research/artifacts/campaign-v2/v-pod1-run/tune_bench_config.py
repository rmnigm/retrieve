"""Scratch config for night-queue item 8, quality phase on goodreads 0.8 M / arXiv 3 M d128 (controller 2026-10-12): suite `tune-q`
(perf: false), SilverTorch triton, n_lists {256 ... 16384} x n_probe {8 ... 4096} with n_probe <= n_lists / 4, k 100, seed 0; clause on the
real kept sweeps and on uniform synth p 0.01 / 0.1 / 1 (arXiv also its correlated synth c001 / c003 / c01), bloom wherever the dataset
config has a bloom block for that sweep (goodreads: c0_genre only, its config has no reverse clause). Usage (from evaluation/):
python tune_bench_config.py OUT {goodreads|arxiv}"""

import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

N_LISTS = [256, 512, 1024, 2048, 4096, 8192, 16384]
N_PROBE = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BENCH = {
    "goodreads": {
        "goodreads": (["c0_genre", "c1_lang_reverse", "all4"], ["c0_genre"]),
        "goodreads-synth": (["p001", "p01", "p1"], ["p001", "p01", "p1"]),
    },
    "arxiv": {
        "arxiv": (
            ["c3_nversions", "c0_maincat", "all4"],
            ["c3_nversions", "c0_maincat", "all4"],
        ),
        "arxiv-synth": (["p001", "p01", "p1"], ["p001", "p01", "p1"]),
        "arxiv-corr-synth": (["c001", "c003", "c01"], ["c001", "c003", "c01"]),
    },
}  # fmt: skip  (dataset: (clause sweeps, bloom sweeps))
out, bench = Path(sys.argv[1]), BENCH[sys.argv[2]]
real = yaml.safe_load(Path("config/suites.yaml").read_text())
arms = []
for nl in N_LISTS:
    probes = [p for p in N_PROBE if p <= nl // 4]
    for fk, idx in (("clause", 0), ("bloom", 1)):
        arms.append({"algo": "silvertorch", "backends": ["triton"], "filter_kinds": [fk], "build": {"n_lists": [nl]},
                     "query": {"n_probe": probes}, "datasets": {ds: {"sweeps": sw[idx]} for ds, sw in bench.items()}})  # fmt: skip
suite = {"datasets": list(bench), "dims": [128], "filter_kinds": ["clause", "bloom"], "ks": [100], "batch_sizes": [1, 16],
         "seeds": [0], "perf": False, "sweeps": {ds: sorted(set(c + b)) for ds, (c, b) in bench.items()}, "arms": arms}  # fmt: skip
out.mkdir(parents=True, exist_ok=True)
(out / "suites.yaml").write_text(
    yaml.safe_dump({"tune-q": suite, "bloom": real["bloom"]}, sort_keys=False)
)
for ds in Path("config").glob("*.yaml"):
    if ds.name != "suites.yaml" and not (out / ds.name).exists():
        (out / ds.name).symlink_to(ds.resolve())
for ds in bench:
    jobs = load_matrix(out / f"{ds}.yaml", out / "suites.yaml", "tune-q")
    print(ds, len(jobs), "jobs", sum(len(j.query) for j in jobs), "cells")
