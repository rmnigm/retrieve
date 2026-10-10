"""Scratch config for the timed phase of night-queue item 8 on PubMed 10 M d768 (controller 2026-10-10): suite `tune-t`, the 35
candidates of `campaign-v2.10/pubmed-tune` (`candidates.json`: [[sweep, filter_kind, n_lists, n_probe], ...], the smallest n_probe per
recall target and the knee) timed: SilverTorch triton, k 100, seed 0, bs {1, 16, 64}, eager + graph (one pass: `--mode` narrows every
bs alike, and a second pass would rebuild every index). One config dir per n_lists (OUT/nl<n>) so the driver runs n_lists-major and can
stop between them; pubmed.yaml is copied with `data_dir: DATA_DIR`, a mirror whose own gt_d768 keeps the v2.11 oracle blobs apart
(record keys use only the `content_d768` name).
Usage (from evaluation/): python tune_timed_config.py CANDIDATES.json OUT DATA_DIR"""

import collections
import copy
import json
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

cands, out, data_dir = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), sys.argv[3]
real = yaml.safe_load(Path("config/suites.yaml").read_text())
by_nl = collections.defaultdict(lambda: collections.defaultdict(list))
for sweep, fk, nl, n_probe in cands:
    by_nl[nl][sweep, fk].append(n_probe)
for nl, groups in sorted(by_nl.items()):
    arms = [{"algo": "silvertorch", "backends": ["triton"], "filter_kinds": [fk], "build": {"n_lists": [nl]},
             "query": {"n_probe": sorted(probes)}, "datasets": {"pubmed": {"sweeps": [sweep]}}}
            for (sweep, fk), probes in sorted(groups.items())]  # fmt: skip
    suites = {"bloom": real["bloom"], "tune-t": {
        "datasets": ["pubmed"], "dims": [768], "filter_kinds": ["clause", "bloom"], "ks": [100], "batch_sizes": [1, 16, 64],
        "seeds": [0], "sweeps": {"pubmed": sorted({s for s, _ in groups})}, "arms": arms}}  # fmt: skip
    d = out / f"nl{nl}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "suites.yaml").write_text(yaml.safe_dump(copy.deepcopy(suites), sort_keys=False))
    for ds in Path("config").glob("*.yaml"):
        if ds.name not in ("suites.yaml", "pubmed.yaml") and not (d / ds.name).exists():
            (d / ds.name).symlink_to(ds.resolve())
    raw = yaml.safe_load(Path("config/pubmed.yaml").read_text())
    (d / "pubmed.yaml").write_text(yaml.safe_dump({**raw, "data_dir": data_dir}, sort_keys=False))
    jobs = load_matrix(d / "pubmed.yaml", d / "suites.yaml", "tune-t")
    print(f"nl{nl}", len(jobs), "jobs", sum(len(j.query) for j in jobs), "cells",
          [(j.filter_kind, j.sweep, [q["n_probe"] for q in j.query]) for j in jobs])  # fmt: skip
