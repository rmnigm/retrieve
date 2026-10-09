"""Move every oracle blob of the given datasets not built at CODE_VERSION into <gt_dir>/before-<CODE_VERSION[:8]>/
(controller: oracles are rebuilt at a leg's tag; old blobs are moved, never deleted).
Usage (from evaluation/): python move_old_oracles.py CODE_VERSION DATASET [DATASET ...]"""

import sys
from pathlib import Path

import torch
import yaml

cv, *datasets = sys.argv[1:]
dirs = {
    Path(yaml.safe_load(Path(f"config/{d}.yaml").read_text())["data_dir"])
    for d in datasets
}
for gt in sorted(g for d in dirs for g in d.glob("gt_d*")):
    for p in sorted(gt.glob("oracle_v4_*.pt")):
        built = torch.load(p, map_location="cpu", mmap=True, weights_only=True)[
            "code_version"
        ]
        if built != cv:
            (gt / f"before-{cv[:8]}").mkdir(exist_ok=True)
            p.rename(gt / f"before-{cv[:8]}" / p.name)
            print(f"moved {p} (built at {built[:8]})")
