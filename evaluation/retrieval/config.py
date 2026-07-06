"""YAML config for the yambda retrieval benchmark.

Boring dataclass + ``yaml.safe_load``. No env-var interpolation, no schema
versioning, no validation framework — bad keys raise ``TypeError`` from the
dataclass constructor with a useful message.

The optional ``filters:`` block (used only by `eval_goodreads_retrieval.py`)
mirrors the bench design in
[goodreads-filter-eval.md](../../docs/plans/goodreads-filter-eval.md):
each ``filter_kind ∈ {none, clause, bloom}`` lists one or more named
``sweeps``, plus the paths to the on-disk attribute tensors. The yambda
harness ignores the field entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from retrieve.interfaces import Backend

# The three filter kinds a sweep cell can have. YAML config dicts stay keyed
# by plain str; ``_select_filter_iter`` (sweep.py) narrows to this on load.
FilterKind = Literal["none", "clause", "bloom"]


@dataclass
class EncodeConfig:
    batch_size: int = 512
    num_workers: int = 8
    max_seq_length: int = 200


@dataclass
class FilterSweepCfg:
    """One filter sweep — a named subset of active narrow clauses."""

    name: str
    # Which clauses to activate (others passed through as -1, treated as
    # "always pass" by ExactAttributeFilter / "no bits queried" by BloomFilter).
    active_clauses: list[int] | None = None


@dataclass
class FilterCfg:
    """Configuration for one filter_kind."""

    sweeps: list[FilterSweepCfg] = field(default_factory=list)
    # narrow / clause: path to item_attrs_narrow.pt OR
    # bloom: path to item_attrs_wide.pt
    attrs_path: str | None = None
    # narrow only: path to clause_is_reverse_narrow.pt
    reverse_path: str | None = None
    # bloom params
    m_bits: int = 1024
    k_hash: int = 5


def filter_cfg_from_dict(d: dict[str, Any]) -> FilterCfg:
    """Build a FilterCfg from a YAML dict; sweeps are upgraded from bare
    dicts to FilterSweepCfg dataclasses so downstream consumers don't have
    to re-parse."""
    raw_sweeps = d.pop("sweeps", []) or []
    sweeps = [FilterSweepCfg(**s) for s in raw_sweeps]
    return FilterCfg(sweeps=sweeps, **d)


@dataclass
class EvalConfig:
    data_dir: str
    # Optional for harnesses that don't load a SASRec checkpoint
    # (e.g. eval_arxiv_retrieval.py uses pre-encoded text embeddings on disk).
    checkpoint: str | None = None
    # Optional path to a pre-encoded query embedding tensor; defaults to
    # ``<data_dir>/<content_subdir>/query_emb.pt`` when a harness needs one.
    query_emb_path: str | None = None
    # Subdir under data_dir that holds {text_emb,query_emb}.pt + meta sidecars.
    # Arxiv ships variants at content_d64 / content_d128 / content (= d=256);
    # other datasets keep the default "content".
    content_subdir: str = "content"
    # Subdir under data_dir that holds gt_topk_<sweep>.pt oracle caches. Must
    # vary with content_subdir (oracle scores depend on item_embs which depend
    # on dim), so set per-yaml when running multiple dims off the same data_dir.
    gt_subdir: str = "gt"
    output: str | None = None
    split: str = "test"
    device: str = "cuda"
    ks: list[int] = field(default_factory=lambda: [100, 500])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 16])
    seed: int = 0
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    algorithms: list[str] = field(default_factory=list)
    algo_params: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Backends to sweep over for each algo. Each backend adds a row per
    # cell (with the same `recall@k`/`ndcg@k` columns plus a `backend`
    # field).
    backends: list[Backend] = field(default_factory=lambda: ["triton"])
    # Optional filter-bench block; consumed by eval_goodreads_retrieval.py and
    # eval_arxiv_retrieval.py.
    filters: dict[str, FilterCfg] | None = None
    # Optional cap on the number of users for ALL cells (quality and
    # filter alike). Goodreads has 313k test users which makes the bs=1
    # quality stream the wall-clock bottleneck; cap to e.g. 50000 to
    # speed runs up. Leave null to use the full split.
    users_limit: int | None = None


def load_eval_config(path: Path) -> EvalConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    # YAML anchor scratch keys (e.g. `_defaults: &defaults …`) live at the
    # top level after parsing; drop them before constructing the dataclass.
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    encode = EncodeConfig(**(raw.pop("encode", None) or {}))
    raw_filters = raw.pop("filters", None)
    filters: dict[str, FilterCfg] | None = None
    if raw_filters is not None:
        filters = {kind: filter_cfg_from_dict(dict(d)) for kind, d in raw_filters.items()}
    return EvalConfig(encode=encode, filters=filters, **raw)


__all__ = [
    "EncodeConfig",
    "EvalConfig",
    "FilterCfg",
    "FilterKind",
    "FilterSweepCfg",
    "filter_cfg_from_dict",
    "load_eval_config",
]
