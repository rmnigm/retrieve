"""A FULL-RERUN config dir (recipe.yaml, runbooks): the tree's dataset yamls, and one suites.yaml merged from the final pass's suites
(evaluation/config/suites.yaml), this plan's stage-1 suites and, once they exist, the picked timed suites. A suite name in two files
is refused; the shared top-level `bloom` width must agree. "Seed everything" (user, 2026-10-13): a final-pass suite with a random arm
(SilverTorch builds a k-means, LiNR V3 draws a projection) and fewer seeds runs at 0-2 here; its exact arms keep one seed's quality
(`SEED_FREE_QUALITY`) and are timed at every seed of their interleave group. Such a suite's synth datasets also gain their draws
`<ds>-s1`, `<ds>-s2` (same sweeps and arm overrides), which queue.sh runs at seeds 1 and 2.

usage: prep_config.py TREE CFG [PICKS_YAML]
"""

import shutil
import sys
from pathlib import Path

import yaml

tree, cfg = Path(sys.argv[1]), Path(sys.argv[2])
cfg.mkdir(parents=True, exist_ok=True)
for f in (tree / "evaluation" / "config").glob("*.yaml"):
    if f.name != "suites.yaml":
        shutil.copy(f, cfg / f.name)
SEEDS = [0, 1, 2]
RANDOM = {"silvertorch", "linr_v3"}


def add_draws(block):
    """Every `-synth` dataset of the suite also as its draws 1 and 2, with the same per-dataset entries."""
    for ds in [d for d in block["datasets"] if d.endswith("-synth")]:
        for d in SEEDS[1:]:
            name = f"{ds}-s{d}"
            block["datasets"].append(name)
            for key in ("sweeps", "ks_by_sweep", "seeds_by_sweep"):
                if ds in (block.get(key) or {}):
                    block[key][name] = block[key][ds]
            for arm in block.get("arms", []):
                if ds in (arm.get("datasets") or {}):
                    arm["datasets"][name] = arm["datasets"][ds]


merged = {}
final = tree / "evaluation/config/suites.yaml"
for f in [
    final,
    tree / "research/artifacts/full-rerun/suites.yaml",
    *map(Path, sys.argv[3:]),
]:
    for name, block in yaml.safe_load(f.read_text()).items():
        if name in merged and (name != "bloom" or merged[name] != block):
            sys.exit(f"{f}: {name} is already defined")
        if (
            f == final
            and isinstance(block, dict)
            and {a["algo"] for a in block.get("arms", [])} & RANDOM
        ):
            block["seeds"] = sorted(set(block.get("seeds", [])) | set(SEEDS))
            add_draws(block)
        merged[name] = block
(cfg / "suites.yaml").write_text(yaml.safe_dump(merged, sort_keys=False, width=140))
print(f"{cfg}/suites.yaml: {len(merged)} top-level blocks")
