"""Scratch config for night-queue item 8 (controller 2026-10-10): ANN tuning on PubMed 10 M d768, quality phase. Suite `tune-q`
(`perf: false`): SilverTorch triton, n_probe {8 ... 4096} with n_probe <= n_lists / 4, k 100, seed 0; clause on the `filter` kept sweeps
(c0_mesh, c3_journal_reverse, all5), bloom on c0_mesh (pubmed.yaml's only kept sweep with a bloom block); no PubMed synth config.
One config dir per n_lists (OUT/nl<n>) so the driver runs n_lists-major and can stop between them; the suite and record keys are the
same in every dir. With DATA_DIR, pubmed.yaml is copied with `data_dir: DATA_DIR`: a mirror of the PubMed dir whose own `gt_d768`
keeps the v2.10 oracle blobs apart from the shared ones (record keys use only the `content_d768` name, unchanged).
Usage (from evaluation/): python tune_config.py OUT [DATA_DIR]"""

import copy
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

N_LISTS = [1024, 4096, 16384, 65536]
N_PROBE = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SWEEPS = {"pubmed": ["c0_mesh", "c3_journal_reverse", "all5"]}
out = Path(sys.argv[1])
real = yaml.safe_load(Path("config/suites.yaml").read_text())
for nl in N_LISTS:
    d = out / f"nl{nl}"
    probes = [p for p in N_PROBE if p <= nl // 4]
    arm = {"algo": "silvertorch", "backends": ["triton"], "build": {"n_lists": [nl]}, "query": {"n_probe": probes}}
    suites = {"bloom": real["bloom"], "tune-q": {
        "perf": False, "datasets": list(SWEEPS), "dims": [768], "filter_kinds": ["clause", "bloom"], "ks": [100],
        "batch_sizes": [16], "seeds": [0], "sweeps": copy.deepcopy(SWEEPS),
        "arms": [{**arm, "filter_kinds": ["clause"]},
                 {**copy.deepcopy(arm), "filter_kinds": ["bloom"], "datasets": {"pubmed": {"sweeps": ["c0_mesh"]}}}]}}  # fmt: skip
    d.mkdir(parents=True, exist_ok=True)
    (d / "suites.yaml").write_text(yaml.safe_dump(suites, sort_keys=False))
    for ds in Path("config").glob("*.yaml"):
        if ds.name != "suites.yaml" and not (d / ds.name).exists():
            (d / ds.name).symlink_to(ds.resolve())
    if len(sys.argv) > 2:
        raw = yaml.safe_load(Path("config/pubmed.yaml").read_text())
        (d / "pubmed.yaml").unlink()
        (d / "pubmed.yaml").write_text(yaml.safe_dump({**raw, "data_dir": sys.argv[2]}, sort_keys=False))
    jobs = load_matrix(d / "pubmed.yaml", d / "suites.yaml", "tune-q")
    print(f"nl{nl}", len(jobs), "jobs", sum(len(j.query) for j in jobs), "cells", sorted({(j.filter_kind, j.sweep) for j in jobs}))
