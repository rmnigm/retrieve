"""Context dataclasses threaded through the sweep driver.

SweepContext is built once per run (post users_limit). FilterAssets is built
once per filter_kind; its per-sweep fields are stamped by run_one_sweep via
dataclasses.replace. Both are frozen: a new input = a new field here, visible
to every loop level at once.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from retrieval.config import EvalConfig
from retrieve.interfaces import Backend, FilterModule


@dataclass(frozen=True)
class SweepContext:
    cfg: EvalConfig
    algorithms: tuple[str, ...]  # explicit — cli no longer mutates cfg
    item_embs: Tensor  # [N, D] on device
    queries: Tensor  # [U, D] cpu, post users_limit
    targets: Tensor  # [U, T] cpu
    n_targets: Tensor  # [U]    cpu
    qa_narrow_all: Tensor | None  # [U, C] cpu or None
    device: torch.device
    gt_dir: Path
    k_gt: int  # max(cfg.ks)
    suite: str  # "yambda" | "filter"
    backends: tuple[Backend, ...]
    skip_quality: bool
    sweep_filter: str | None  # --sweep CLI narrow
    filter_kinds: tuple[str, ...]  # --filter-kind CLI narrow; empty = all


@dataclass(frozen=True)
class FilterAssets:
    """Per-filter_kind modules and attrs; per-sweep fields default empty."""

    filter_mods: dict[Backend, FilterModule | None]
    oracle_filter: FilterModule | None
    item_attrs_narrow: Tensor | None
    clause_is_reverse: Tensor | None
    n_clauses: int
    # --- stamped per sweep by run_one_sweep ---
    qa_n_sweep: Tensor | None = None
    skip_mask: Tensor | None = None
    oracle_topk: Tensor | None = None
    n_kept: int = 0

    def for_sweep(self, *, qa_n_sweep, skip_mask, oracle_topk, n_kept) -> FilterAssets:
        return replace(
            self,
            qa_n_sweep=qa_n_sweep,
            skip_mask=skip_mask,
            oracle_topk=oracle_topk,
            n_kept=n_kept,
        )


EMPTY_ASSETS = FilterAssets(
    filter_mods={},
    oracle_filter=None,
    item_attrs_narrow=None,
    clause_is_reverse=None,
    n_clauses=0,
)


__all__ = ["EMPTY_ASSETS", "FilterAssets", "SweepContext"]
