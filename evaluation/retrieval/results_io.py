"""Concatenate per-algo JSONs from a run-directory back into a flat row list.

The retrieval CLI writes one ``<algo>.json`` per algo into the cfg.output
directory. Analysis code (notebooks, plotting) consumes the full row set
via ``load_results(dir)``.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_results(results_dir: str | Path) -> list[dict]:
    """Return the concatenation of every ``<results_dir>/*.json`` in name order."""
    rows: list[dict] = []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            rows.extend(json.load(f))
    return rows


__all__ = ["load_results"]
