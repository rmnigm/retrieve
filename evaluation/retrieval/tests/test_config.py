"""CPU-only round-trip tests for ``load_eval_config``.

Loads ``tests/data/mini.yaml`` — a tiny self-contained config that mirrors
the shape of the shipped goodreads YAMLs (``_defaults`` anchor + ``<<:``
merge, clause/bloom ``filters:`` block) without referencing real data.
Locks the three loader behaviors the driver relies on: ``_``-prefixed
anchor scratch keys are stripped, the ``filters:`` block is upgraded to
dataclasses, and a missing ``encode:`` block yields ``EncodeConfig``
defaults.
"""

from __future__ import annotations

from pathlib import Path

from retrieval.config import (
    EncodeConfig,
    EvalConfig,
    FilterCfg,
    FilterSweepCfg,
    load_eval_config,
)

MINI_YAML = Path(__file__).parent / "data" / "mini.yaml"


def test_underscore_keys_stripped():
    # `_defaults` is a top-level anchor scratch key; if it survived the strip,
    # EvalConfig(**raw) would raise TypeError on the unexpected kwarg.
    cfg = load_eval_config(MINI_YAML)
    assert isinstance(cfg, EvalConfig)
    assert not hasattr(cfg, "_defaults")
    # ...while the values merged in via `<<: *defaults` are present.
    assert cfg.data_dir == "data/mini"
    assert cfg.checkpoint == "data/mini/checkpoints/best_model.pt"


def test_scalar_fields_round_trip():
    cfg = load_eval_config(MINI_YAML)
    assert cfg.output == "results/mini"
    assert cfg.gt_subdir == "gt_mini"
    assert cfg.split == "test"
    assert cfg.device == "cpu"
    assert cfg.ks == [10, 50]
    assert cfg.batch_sizes == [1, 2]
    assert cfg.seed == 7
    assert cfg.algorithms == ["linr_v3", "silvertorch"]
    assert cfg.backends == ["triton", "torch"]
    assert cfg.users_limit == 100


def test_filters_block_upgraded_to_dataclasses():
    cfg = load_eval_config(MINI_YAML)
    assert cfg.filters is not None
    assert set(cfg.filters) == {"clause", "bloom"}

    clause = cfg.filters["clause"]
    assert isinstance(clause, FilterCfg)
    assert clause.attrs_path == "data/mini/item_attrs_narrow.pt"
    assert clause.reverse_path == "data/mini/clause_is_reverse_narrow.pt"
    assert [type(s) for s in clause.sweeps] == [FilterSweepCfg, FilterSweepCfg]
    assert clause.sweeps[0] == FilterSweepCfg(name="c0", active_clauses=[0])
    assert clause.sweeps[1] == FilterSweepCfg(name="c0c1", active_clauses=[0, 1])

    bloom = cfg.filters["bloom"]
    assert isinstance(bloom, FilterCfg)
    assert bloom.reverse_path is None
    assert bloom.m_bits == 64
    assert bloom.k_hash == 3
    assert bloom.sweeps == [FilterSweepCfg(name="c0", active_clauses=[0])]


def test_encode_defaults():
    # mini.yaml has no `encode:` block → dataclass defaults apply.
    cfg = load_eval_config(MINI_YAML)
    assert cfg.encode == EncodeConfig()
    assert cfg.encode.batch_size == 512
    assert cfg.encode.num_workers == 8
    assert cfg.encode.max_seq_length == 200
