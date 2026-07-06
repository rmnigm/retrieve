"""Restructure a single config's results into the campaign layout.

After ``run-evaluation`` (``retrieval.cli.run_evaluation``) writes one JSON
per algo to ``<output_dir>/<algo>.json``, this script:
  1. Concatenates the per-algo JSONs into <output_dir>.json (combined, by row).
  2. Renames the per-algo dir to <output_dir>.perkernel/.
  3. Copies the YAML to <output_dir>.yaml.

Idempotent: re-running on an already-staged config wipes the .perkernel dir
and rebuilds from whatever's in <output_dir>/. If <output_dir>/ is gone but
.perkernel exists, does nothing (already staged).

Usage: stage_results.py <config_path>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from retrieval.config import load_raw_config
from retrieval.results_io import load_rows


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: stage_results.py <config_path>", file=sys.stderr)
        return 2

    cfg_path = Path(sys.argv[1])
    cfg = load_raw_config(cfg_path)

    out_dir = Path(cfg["output"])
    combined_path = out_dir.parent / f"{out_dir.name}.json"
    perkernel_dir = out_dir.parent / f"{out_dir.name}.perkernel"
    yaml_dst = out_dir.parent / f"{out_dir.name}.yaml"

    if not out_dir.is_dir():
        if perkernel_dir.exists():
            print(f"stage_results: {out_dir} already staged (perkernel exists)")
            return 0
        print(f"stage_results: {out_dir} does not exist — nothing to stage", file=sys.stderr)
        return 1

    jsons = sorted(out_dir.glob("*.json"))
    if not jsons:
        print(f"stage_results: {out_dir} has no JSON files — nothing to stage", file=sys.stderr)
        return 1

    rows: list = []
    for p in jsons:
        data = load_rows(p)
        if data is None:
            print(f"stage_results: {p}: missing or not a list — skipping", file=sys.stderr)
            continue
        rows.extend(data)

    combined_path.write_text(json.dumps(rows))
    print(f"stage_results: wrote {combined_path} ({len(rows)} rows from {len(jsons)} files)")

    if perkernel_dir.exists():
        shutil.rmtree(perkernel_dir)
    out_dir.rename(perkernel_dir)
    print(f"stage_results: renamed {out_dir} → {perkernel_dir}")

    shutil.copy2(cfg_path, yaml_dst)
    print(f"stage_results: copied {cfg_path} → {yaml_dst}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
