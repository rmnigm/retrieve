"""Shared CPU fixtures: a tiny pre-encoded (arxiv-shaped) dataset on disk (``write_tiny_dataset``,
parametrised so ``test_data.py`` uses the same writer) plus the matching harness-v2 dataset /
suites YAMLs, so ``run.run`` can be driven end to end through ``config.load_matrix`` without a
GPU or real data. The ``e2e`` suite has two algos (the in-process loop); ``e2e1`` has one, for
the campaign test's single child."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch

N, U, D, C = 24, 8, 8, 2


def write_tiny_dataset(
    data_dir: Path,
    *,
    n: int = N,
    u: int = U,
    doc_prefix: str = "search_document: ",
    n_split: int | None = None,
) -> Path:
    """``data_dir/`` with ``content/{text_emb,query_emb}.pt`` + meta sidecars,
    ``heldout.parquet``, narrow attrs (clause 0: three values; clause 1: two values,
    reverse) and ``eval_split.parquet`` (``n_split`` rows, default ``u``; query 4 has no
    live attrs). Returns ``data_dir``."""
    content = data_dir / "content"
    content.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    torch.save(torch.randn(n, D, generator=g).half(), content / "text_emb.pt")
    torch.save(torch.randn(u, D, generator=g).half(), content / "query_emb.pt")
    (content / "text_emb.meta.json").write_text(json.dumps({"prefix": doc_prefix}))
    (content / "query_emb.meta.json").write_text(json.dumps({"prefix": "search_query: "}))
    pl.DataFrame({"item_id": list(range(1, u + 1))}).write_parquet(data_dir / "heldout.parquet")
    attrs = torch.full((n, C, 1), -1, dtype=torch.long)
    attrs[:, 0, 0] = torch.arange(n) % 3
    attrs[:, 1, 0] = torch.arange(n) % 2
    torch.save(attrs, data_dir / "item_attrs_narrow.pt")
    torch.save(torch.tensor([False, True]), data_dir / "clause_is_reverse_narrow.pt")
    qa = [[i % 3, i % 2] if i != 4 else [-1, -1] for i in range(u if n_split is None else n_split)]
    pl.DataFrame({"query_attrs_narrow": qa}).write_parquet(data_dir / "eval_split.parquet")
    return data_dir


@pytest.fixture
def tiny_configs(tmp_path: Path) -> tuple[Path, Path]:
    """``(tiny.yaml, suites.yaml)`` over the on-disk fixture; the ``e2e`` suite runs
    ``none`` + ``clause`` cells on ``backend="torch"`` (the CPU-capable path)."""
    data_dir = write_tiny_dataset(tmp_path / "data" / "tiny")
    ds = tmp_path / "tiny.yaml"
    ds.write_text(
        f"data_dir: {data_dir}\ncontent_dir: content\ndims: [{D}]\n"
        "filters:\n  attrs: item_attrs_narrow.pt\n  reverse: clause_is_reverse_narrow.pt\n"
        "  clause: {c0: [0], c0c1: [0, 1]}\n  bloom: {c0: [0]}\n"
    )
    suites = tmp_path / "suites.yaml"
    suites.write_text(
        "e2e:\n  datasets: [tiny]\n  filter_kinds: [none, clause]\n  ks: [2, 4]\n"
        "  batch_sizes: [1, 2]\n"
        "  algos: {linr_v1_filter_mask: [torch], linr_v4: [torch]}\n"
        "e2e1:\n  datasets: [tiny]\n  filter_kinds: [none, clause]\n  ks: [2, 4]\n"
        "  batch_sizes: [1, 2]\n  algos: {linr_v1_filter_mask: [torch]}\n"
        "bloom: {m_bits: 64, k_hash: 2}\n"
    )
    return ds, suites
