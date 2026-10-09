"""Shared fixtures for the whole ``evaluation/tests`` tree: a tiny pre-encoded (arxiv-shaped)
dataset on disk (``write_tiny_dataset``, the one writer ``test_inputs.py`` and
``test_layout.py`` share) plus the matching dataset / suites YAMLs, so ``run.run`` can be
driven end to end through ``config.load_matrix`` without a GPU or real data. The ``e2e``
suite has two algos (the in-process loop); ``e2e1`` has one, for the campaign test's single
child; ``postfilter`` runs the baseline next to the exact V1 it is measured against; ``arms``
holds the campaign-v2 params (gridded bloom widths, ``compile``, ``candidate_pool_frac``);
``cache`` runs seeds 0-2 of seed-free and seed-dependent arms (the quality cache); ``untimed``
is a quality-only suite (``perf: false``); ``pair`` interleaves V1 and V2 (``--interleave``).
Tests marked ``gpu`` skip without CUDA."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest
import torch

N, U, D, C = 24, 8, 8, 2
D_E2E = 64  # LiNR V3's OPORP packs bits in words of 64


def pytest_configure(config):
    """H-INDCACHE: a fresh inductor cache per pytest run, so no compile gate replays a graph an
    earlier tree cached for the same custom op (testing.md § Running)."""
    config.inductor_dir = tempfile.mkdtemp(prefix="pytest-inductor-")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = config.inductor_dir


def pytest_unconfigure(config):
    shutil.rmtree(config.inductor_dir, ignore_errors=True)


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def write_tiny_dataset(
    data_dir: Path,
    *,
    n: int = N,
    u: int = U,
    d: int = D,
    doc_prefix: str = "search_document: ",
    n_split: int | None = None,
    legacy: bool = False,
) -> Path:
    """``data_dir/`` with ``content/{text_emb,query_emb}.pt`` + meta sidecars,
    ``heldout.parquet``, narrow attrs (clause 0: three values; clause 1: two values,
    reverse) and ``eval_split.parquet`` (``n_split`` rows, default ``u``; query 4 has no
    live attrs). ``legacy=True`` writes the pre-``3b1b5b3`` ``[N+1, …]`` layout the Hub
    copies still have — an all-zero row 0 in ``text_emb`` and an all-``-1`` row 0 in the
    attrs — over the *same* items (held-out ids are 1-indexed in both layouts), so a legacy
    load must equal a modern one. Returns ``data_dir``."""
    content = data_dir / "content"
    content.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    text_emb = torch.randn(n, d, generator=g).half()
    if legacy:
        text_emb = torch.cat([torch.zeros(1, d, dtype=text_emb.dtype), text_emb])
    torch.save(text_emb, content / "text_emb.pt")
    torch.save(torch.randn(u, d, generator=g).half(), content / "query_emb.pt")
    (content / "text_emb.meta.json").write_text(json.dumps({"prefix": doc_prefix}))
    (content / "query_emb.meta.json").write_text(json.dumps({"prefix": "search_query: "}))
    pl.DataFrame({"item_id": list(range(1, u + 1))}).write_parquet(data_dir / "heldout.parquet")
    attrs = torch.full((n, C, 1), -1, dtype=torch.long)
    attrs[:, 0, 0] = torch.arange(n) % 3
    attrs[:, 1, 0] = torch.arange(n) % 2
    if legacy:
        attrs = torch.cat([torch.full((1, C, 1), -1, dtype=torch.long), attrs])
    torch.save(attrs, data_dir / "item_attrs_narrow.pt")
    torch.save(torch.tensor([False, True]), data_dir / "clause_is_reverse_narrow.pt")
    qa = [[i % 3, i % 2] if i != 4 else [-1, -1] for i in range(u if n_split is None else n_split)]
    pl.DataFrame({"query_attrs_narrow": qa}).write_parquet(data_dir / "eval_split.parquet")
    return data_dir


@pytest.fixture
def tiny_configs(tmp_path: Path) -> tuple[Path, Path]:
    """``(tiny.yaml, suites.yaml)`` over the on-disk fixture; the ``e2e`` suite runs
    ``none`` + ``clause`` cells on ``backend="torch"`` (the CPU-capable path)."""
    data_dir = write_tiny_dataset(tmp_path / "data" / "tiny", d=D_E2E)
    ds = tmp_path / "tiny.yaml"
    ds.write_text(
        f"data_dir: {data_dir}\ncontent_dir: content\ndims: [{D_E2E}]\n"
        "filters:\n  attrs: item_attrs_narrow.pt\n  reverse: clause_is_reverse_narrow.pt\n"
        "  clause: {c0: [0], c0c1: [0, 1]}\n  bloom: {c0: [0]}\n"
    )
    suites = tmp_path / "suites.yaml"
    suites.write_text(
        "e2e:\n  datasets: [tiny]\n  filter_kinds: [none, clause]\n  ks: [2, 4]\n"
        "  batch_sizes: [1, 2]\n"
        "  arms:\n    - {algo: linr_v1_filter_mask, backends: [torch]}\n"
        "    - {algo: linr_v3, backends: [torch], query: {candidate_pool: [8]}}\n"
        "e2e1:\n  datasets: [tiny]\n  filter_kinds: [none, clause]\n  ks: [2, 4]\n"
        "  batch_sizes: [1, 2]\n  arms: [{algo: linr_v1_filter_mask, backends: [torch]}]\n"
        "postfilter:\n  datasets: [tiny]\n  filter_kinds: [clause, bloom]\n  ks: [2, 4]\n"
        "  batch_sizes: [1, 2]\n  arms:\n    - {algo: linr_v1_filter_mask, backends: [torch]}\n"
        "    - {algo: postfilter, backends: [torch], query: {alpha: [1, 2]}}\n"
        "arms:\n  datasets: [tiny]\n  filter_kinds: [clause, bloom]\n  ks: [2, 4]\n"
        "  batch_sizes: [1]\n  arms:\n"
        "    - {algo: silvertorch, backends: [torch], filter_kinds: [bloom],\n"
        "       build: {n_lists: [4], m_bits: [64, 256]}, query: {n_probe: [2]}}\n"
        "    - {algo: silvertorch, backends: [torch], filter_kinds: [bloom],\n"
        "       build: {n_lists: [4]}, query: {n_probe: [2]}}\n"
        "    - {algo: silvertorch, backends: [torch], filter_kinds: [clause],\n"
        "       build: {n_lists: [4], compile: [max-autotune]}, query: {n_probe: [2]}}\n"
        "    - {algo: linr_v3, backends: [torch], filter_kinds: [clause],\n"
        "       query: {candidate_pool_frac: [0.5, 1.0]}}\n"
        "untimed:\n  perf: false\n  datasets: [tiny]\n  filter_kinds: [clause]\n  ks: [2, 4]\n"
        "  batch_sizes: [1]\n  arms: [{algo: linr_v1_filter_mask, backends: [torch]}]\n"
        "pair:\n  datasets: [tiny]\n  filter_kinds: [clause]\n  ks: [2, 4]\n  batch_sizes: [1, 2]\n"
        "  seeds: [0, 1]\n  sweeps: {tiny: [c0, c0c1]}\n  interleave: [{by: algo}]\n  arms:\n"
        "    - {algo: linr_v1_filter_mask, backends: [torch]}\n"
        "    - {algo: linr_v2, backends: [torch]}\n"
        "cache:\n  datasets: [tiny]\n  filter_kinds: [clause]\n  ks: [2, 4]\n  batch_sizes: [1]\n"
        "  seeds: [0, 1, 2]\n  sweeps: {tiny: [c0]}\n  arms:\n"
        "    - {algo: linr_v1_filter_mask, backends: [torch]}\n"
        "    - {algo: postfilter, backends: [torch], query: {alpha: [1]}}\n"
        "    - {algo: linr_v3, backends: [torch], query: {candidate_pool: [8]}}\n"
        "    - {algo: silvertorch, backends: [torch], build: {n_lists: [4]},\n"
        "       query: {n_probe: [2]}}\n"
        "bloom: {m_bits: 64, k_hash: 2}\n"
    )
    return ds, suites
