"""CPU dry resolve of every FULL-RERUN suite (README.md): `bench.config.load_matrix` on each (suite, dataset), job and cell
counts per suite, and the study's arm rules on every resolved job. Exits 1 on a rule break or a suite that does not load.

Rules: SilverTorch bloom on the adapted backend runs with the recheck off (user, 2026-10-12); no LiNR arm filters with a
bloom (user, 2026-10-13); no YFCC dataset (user, 2026-10-13); no LiNR V2 at bs 256 on PubMed (controller C2, 2026-10-13).

usage: check_suites.py CONFIG_DIR SUITES_YAML [...]   (CONFIG_DIR: the dataset yamls; the suites files are merged)
"""

import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from bench.config import load_matrix

cfg, files = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
merged = {}
for f in files:
    merged.update(yaml.safe_load(f.read_text()))
problems, totals = [], Counter()
with tempfile.TemporaryDirectory() as tmp:
    suites_yaml = Path(tmp) / "suites.yaml"
    suites_yaml.write_text(yaml.safe_dump(merged))
    for name, s in merged.items():
        if not isinstance(s, dict) or "datasets" not in s:
            continue
        jobs = cells = 0
        for ds in s["datasets"]:
            if ds.startswith("yfcc"):
                problems.append(f"{name}: YFCC dataset {ds}")
            try:
                matrix = load_matrix(cfg / f"{ds}.yaml", suites_yaml, name)
            except Exception as exc:  # noqa: BLE001 — report every suite, then fail
                problems.append(f"{name}/{ds}: {type(exc).__name__}: {exc}")
                continue
            for j in matrix:
                jobs += 1
                cells += len(j.query) * (1 if j.timed is False else len(j.batch_sizes))
                st_bloom = (
                    j.algo == "silvertorch"
                    and j.filter_kind == "bloom"
                    and j.backend == "official-fork"
                )
                if st_bloom and j.build.get("recheck") is not False:
                    problems.append(
                        f"{name}/{ds}: SilverTorch bloom without recheck: false ({j.build})"
                    )
                if j.algo.startswith("linr") and j.filter_kind == "bloom":
                    problems.append(f"{name}/{ds}: LiNR bloom arm {j.algo}")
                if (
                    j.algo == "linr_v2"
                    and ds.startswith("pubmed")
                    and 256 in j.batch_sizes
                ):
                    problems.append(f"{name}/{ds}: LiNR V2 at bs 256 on PubMed")
        totals[name] = jobs
        print(f"{name:34s} jobs {jobs:5d}  cell x bs {cells:6d}")
print(f"{len(totals)} suites, {sum(totals.values())} jobs; {len(problems)} problem(s)")
for p in problems:
    print("PROBLEM", p)
sys.exit(1 if problems else 0)
