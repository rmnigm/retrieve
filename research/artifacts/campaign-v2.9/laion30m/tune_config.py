"""Scratch config for night-queue item 8 on LAION 30 M d256 (controller 2026-10-12): suite `tune`, SilverTorch triton, n_lists
{8192, 16384, 32768, 65536} x n_probe {8 ... 4096} capped at n_lists / 4 (25 % scanned, user), k 100, seed 0; clause on laion30m
`tags4` / `c0_domain` and laion30m-synth p 0.01 / 0.1 / 1; bloom (m_bits 1024, k_hash 5) on the real sweeps only (the synth config has no
bloom block). Optional trailing args narrow N_LISTS (the driver writes one config dir per n_lists). Timed in the same pass (bs 1 / 16 / 64; run with --mode graph): every job rebuilds its index, so a quality-then-timed
split would build all 28 indexes twice. Usage (from evaluation/): python tune_config.py OUT [N_LISTS ...]"""

import copy
import sys
from pathlib import Path

import yaml
from bench.config import load_matrix

N_LISTS = [int(a) for a in sys.argv[2:]] or [8192, 16384, 32768, 65536]
N_PROBE = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SWEEPS = {"laion30m": ["tags4", "c0_domain"], "laion30m-synth": ["p001", "p01", "p1"]}
out = Path(sys.argv[1])
real = yaml.safe_load(Path("config/suites.yaml").read_text())
arms = []
for nl in N_LISTS:
    probes = [p for p in N_PROBE if p <= nl // 4]
    arms.append({"algo": "silvertorch", "backends": ["triton"], "filter_kinds": ["clause"], "build": {"n_lists": [nl]},
                 "query": {"n_probe": probes}})  # fmt: skip
    arms.append({"algo": "silvertorch", "backends": ["triton"], "filter_kinds": ["bloom"], "build": {"n_lists": [nl]},
                 "query": {"n_probe": probes}, "datasets": {"laion30m": {}}})  # fmt: skip
suites = {"bloom": real["bloom"], "tune": {"datasets": list(SWEEPS), "dims": [256], "filter_kinds": ["clause", "bloom"], "ks": [100],
          "batch_sizes": [1, 16, 64], "seeds": [0], "sweeps": copy.deepcopy(SWEEPS), "arms": arms}}  # fmt: skip
out.mkdir(parents=True, exist_ok=True)
(out / "suites.yaml").write_text(yaml.safe_dump(suites, sort_keys=False))
for ds in SWEEPS:
    if not (out / f"{ds}.yaml").exists():
        (out / f"{ds}.yaml").symlink_to(Path(f"config/{ds}.yaml").resolve())
for ds in SWEEPS:
    jobs = load_matrix(out / f"{ds}.yaml", out / "suites.yaml", "tune")
    print(
        "tune",
        ds,
        len(jobs),
        "jobs",
        sum(len(j.query) for j in jobs),
        "cells",
        file=sys.stderr,
    )
