"""What a record *is*: the schema version, the key block, the resume key, the JSONL append
with its ``fsync``, the samples sidecar, the tolerant ``read_keys`` and ``flatten`` — one row
per ``(record, perf entry)`` as ``flat.csv`` for ``report.py`` (H §3.2, §8.2 B/C/G).

Schema 2 (C5): ``env.expected_sm_mhz`` and ``env.clocks_locked`` are gone (a box that cannot
lock clocks cannot record whether they are locked), ``env.sm_mhz`` is ``env.sm_mhz_idle`` (the
process-start sample) and ``env.sm_mhz_load`` is the median of the cell's under-load samples;
``clocks_drift`` compares under-load samples with under-load samples only.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

SCHEMA_VERSION = 2
KEY_FIELDS = (
    "dataset",
    "dim",
    "suite",
    "filter_kind",
    "sweep",
    "algo",
    "backend",
    "params",
    "seed",
)
# Record scalars and env fields that go into flat.csv next to the key block.
_RECORD_COLUMNS = (
    "status", "path", "n_items", "n_queries", "n_kept", "n_queries_heldout", "n_queries_oracle",
    "n_targets_in_filter", "pass_rate", "bloom_fp_rate", "k_max", "build_s", "index_mib",
    "filter_mib", "unstable", "memory_reserved_mib", "elapsed_s",
    "schema_version", "stage", "error", "partial_reasons",
)  # fmt: skip
# flat.csv is shipped as a paper artifact (P §B.7), so it has to carry why a row is
# incomplete, not only that it is: `stage`/`error` on a failure, `partial_reasons` on a
# narrowed cell, `schema_version` because v1 and v2 records coexist in one file.
_ENV_COLUMNS = (
    "code_version", "commit", "dirty", "gpu", "sm_mhz_load", "clocks_drift", "git_branch",
)  # fmt: skip
_PERF_SKIP = ("window_medians_ms", "kernels")


def resume_key(key: dict[str, Any], code_version: str) -> str:
    """Canonical JSON of ``Job.key(params)`` plus the library's ``code_version`` (H §8.2 B):
    a kernel change invalidates a cell instead of silently reusing it."""
    return json.dumps({**key, "code_version": code_version}, sort_keys=True, separators=(",", ":"))


def record_key(rec: dict[str, Any]) -> str:
    return resume_key({k: rec[k] for k in KEY_FIELDS}, rec["env"]["code_version"])


def record_path(out_dir: Path, job) -> Path:
    return Path(out_dir) / job.suite / f"{job.dataset}-d{job.dim}.jsonl"


def samples_path(path: Path) -> Path:
    return Path(path).with_suffix(".samples.jsonl")


def _clean(o: Any) -> Any:
    """JSON-safe: tensors → lists, paths → str, non-finite floats → null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, torch.Tensor):
        return _clean(o.tolist())
    if isinstance(o, np.floating | np.integer):
        return _clean(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def append_record(path: Path, rec: dict[str, Any]) -> None:
    """One ``write`` of one line, then ``fsync``: a crash loses at most the cell in flight."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_clean(rec), separators=(",", ":"), allow_nan=False) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every record in ``path``. A malformed *last* line — the cell in flight when the process
    died — is logged and dropped; a malformed line anywhere else is corruption and raises."""
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:
                logger.warning("{}: torn trailing line ignored ({})", path, exc)
                break
            raise ValueError(f"{path}: malformed record on line {i + 1}: {exc}") from exc
    return out


def read_keys(path: Path) -> dict[str, str]:
    """``{resume_key: status}``, last record per key wins."""
    return {record_key(rec): rec.get("status", "ok") for rec in read_records(path)}


def _row(rec: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {k: rec[k] for k in KEY_FIELDS}
    row["params"] = json.dumps(rec["params"], sort_keys=True)
    row.update({c: rec.get(c) for c in _RECORD_COLUMNS})
    row.update({f"env_{c}": rec["env"].get(c) for c in _ENV_COLUMNS})
    q = rec.get("quality") or {}
    for side in ("heldout", "oracle"):
        for m, v in (q.get(side) or {}).items():
            row[f"{side}_{m}"] = v
    for m, v in q.items():
        if not isinstance(v, dict):
            row[f"quality_{m}"] = v
    for m, v in (entry or {}).items():
        if m not in _PERF_SKIP:
            row[f"perf_{m}"] = v
    return row


def flatten(results_dir: Path, out: Path | None = None) -> Path:
    """``<results_dir>/flat.csv``: the last record per resume key of every
    ``<suite>/<dataset>-d<dim>.jsonl``, one row per perf entry (one row with empty perf
    columns when the record has none). Dicts are JSON strings; nulls are empty cells."""
    results_dir = Path(results_dir)
    out = out or results_dir / "flat.csv"
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*/*.jsonl")):
        if path.name.endswith(".samples.jsonl"):
            continue
        for rec in read_records(path):
            latest[record_key(rec)] = rec
    rows: list[dict[str, Any]] = []
    for rec in latest.values():
        rows.extend(_row(rec, e) for e in (rec.get("perf") or [None]))
    cols = list(dict.fromkeys(c for r in rows for c in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if v is None else v) for c, v in r.items()})
    return out


__all__ = [
    "KEY_FIELDS",
    "SCHEMA_VERSION",
    "append_record",
    "flatten",
    "read_keys",
    "read_records",
    "record_key",
    "record_path",
    "resume_key",
    "samples_path",
]
