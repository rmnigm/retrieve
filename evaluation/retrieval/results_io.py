"""Read per-algo result JSONs.

The retrieval CLI writes one ``<algo>.json`` per algo into the cfg.output
directory. Analysis code (notebooks, plotting) consumes the full row set
via :func:`load_results`. Tools that need to validate a single file
(orchestrator's resume check, staging, upload) use :func:`load_rows`.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_rows(path: str | Path) -> list[dict] | None:
    """Load one per-algo JSON. ``None`` if missing, unparseable, or not a list."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) else None


def load_results(results_dir: str | Path) -> list[dict]:
    """Return the concatenation of every ``<results_dir>/*.json`` in name order."""
    rows: list[dict] = []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            rows.extend(json.load(f))
    return rows


__all__ = ["load_rows", "load_results"]
