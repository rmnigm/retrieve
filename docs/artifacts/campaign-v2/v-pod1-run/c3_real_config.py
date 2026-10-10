"""Scratch config for C3 coverage on real filters at campaign-v2.10 (controller 2026-10-12, as d-run's C3 leg on synth): suite `c3-real`,
LiNR V1 (linr_v1_filter_mask) Triton vs torch.compile max-autotune (no plain torch), clause, k 100, bs {1, 16}, seed 0, on the real kept
sweeps of goodreads / arXiv / YFCC and PubMed 10 M d768 (c0_mesh, all5, c3_journal_reverse). Usage (from evaluation/): python c3_real_config.py OUT"""

import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

SWEEPS = {"goodreads": ["c0_genre", "c1_lang_reverse", "all4"], "arxiv": ["c3_nversions", "c0_maincat", "all4"],
          "yfcc10m": ["tags_and"], "pubmed": ["c0_mesh", "all5", "c3_journal_reverse"]}  # fmt: skip
out = Path(sys.argv[1])
real = yaml.safe_load(Path("config/suites.yaml").read_text())
suite = {"datasets": list(SWEEPS), "dims": [128, 192, 768], "filter_kinds": ["clause"], "ks": [100], "batch_sizes": [1, 16], "seeds": [0],
         "sweeps": SWEEPS, "arms": [{"algo": "linr_v1_filter_mask", "backends": ["triton"]},
                                    {"algo": "linr_v1_filter_mask", "backends": ["torch"], "build": {"compile": ["max-autotune"]}}]}  # fmt: skip
out.mkdir(parents=True, exist_ok=True)
(out / "suites.yaml").write_text(
    yaml.safe_dump({"c3-real": suite, "bloom": real["bloom"]}, sort_keys=False)
)
for ds in Path("config").glob("*.yaml"):
    if ds.name != "suites.yaml" and not (out / ds.name).exists():
        (out / ds.name).symlink_to(ds.resolve())
for ds in SWEEPS:
    jobs = load_matrix(out / f"{ds}.yaml", out / "suites.yaml", "c3-real")
    print(
        ds,
        len(jobs),
        "jobs",
        sorted({(j.backend, str(j.build), j.sweep, j.dim) for j in jobs}),
    )
