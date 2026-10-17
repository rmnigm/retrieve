"""Shared helpers for the gpu-checks session scripts (research/artifacts/gpu-checks/README.md).

Run from evaluation/ with the workspace venv: the harness's own loaders, query pool, graph
capture and timing windows (docs/system/evaluation.md § Measurement protocol) do the work."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from bench import inputs as binputs
from bench import measure
from bench.config import load_dataset
from eval_datasets.synth_filter import SEED, synth_flags, synth_u

DATASETS = {  # name -> (config, dim): the paper's scale points that are on the Hub
    "goodreads": ("config/goodreads.yaml", 128),
    "arxiv": ("config/arxiv.yaml", 128),
    "pubmed": ("config/pubmed.yaml", 768),
}
OUT = Path(os.environ.get("GPU_CHECKS_OUT", "/scratch/gpu-checks/out"))


def load(name: str, device: torch.device) -> tuple[Any, dict[str, Any]]:
    path, dim = DATASETS[name]
    ds = load_dataset(Path(path), dim)
    return ds, binputs.load_inputs(ds, device)


def synth_attrs(n_items: int, rates: list[float], device: torch.device) -> torch.Tensor:
    """``[N, len(rates), 1]`` int64 pass flags of draw 0 (``u_i < p``, eval-data synth-filter's
    rule and seed), so every listed rate is the harness's synth sweep of that rate."""
    return synth_flags(synth_u(n_items, SEED), rates).to(device)


def env() -> dict[str, Any]:
    p = measure.provenance()
    p["clocks_start"] = measure.clocks()
    return p


class Records:
    """One JSONL file per check, appended and fsynced per record."""

    def __init__(self, check: str, tag: str) -> None:
        self.dir = OUT / check
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{tag}.jsonl"

    def done(self) -> set[str]:
        if not self.path.exists():
            return set()
        return {
            json.loads(line)["cell"]
            for line in self.path.read_text().splitlines()
            if line
        }

    def append(self, rec: dict[str, Any]) -> None:
        rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
            f.flush()
            os.fsync(f.fileno())


def log(*a: Any) -> None:
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)
