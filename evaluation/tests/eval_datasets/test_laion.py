"""CPU-only tests for the Re-LAION loader (roadmap V-LAION30): registered domains, the tag
buckets, the exact top-1 target, and prep → (stand-in vectors) → targets → attrs →
``validate_layout`` on a tiny pair of parts in the upstream column schema."""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl
import pytest
import torch

from eval_datasets import layout
from eval_datasets.etl import laion


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("cdn.shopify.com", "shopify.com"),
        ("i.pinimg.com", "pinimg.com"),
        ("www.bbc.co.uk", "bbc.co.uk"),
        ("192.168.0.1", "192.168.0.1"),
    ],
)
def test_registered_domain(host, expected):
    assert laion.registered_domain(host) == expected


def test_bucket_edges_are_upper_exclusive_and_nan_is_missing():
    got = laion.bucket(pl.Series([0.0, 200.0, 299.0, 300.0, 5000.0, None]), laion.SIZE_EDGES)
    assert got.to_list() == [0, 1, 1, 2, 4, None]


def _part(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "url": pl.Utf8, "similarity": pl.Float64, "hash": pl.Int64, "pwatermark": pl.Float32,
            "punsafe": pl.Float32, "caption": pl.Utf8, "key": pl.Utf8, "status": pl.Utf8,
            "original_width": pl.Int32, "original_height": pl.Int32,
        },
    )  # fmt: skip


@pytest.fixture
def raw(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    hosts = ["cdn.shopify.com", "i.pinimg.com", "www.bbc.co.uk", "img.example.org"]
    for part in range(2):
        rows = []
        for r in range(40):
            i = part * 40 + r
            caption = f"caption {i % 70}" if i != 5 else "   "  # 10 duplicates, one blank
            rows.append({
                "url": f"https://{hosts[i % 4]}/p/{i}.jpg", "similarity": 0.28 + 0.1 * rng.random(),
                "hash": int(rng.integers(-(2**62), 2**62)), "pwatermark": float(rng.random()),
                "punsafe": float(rng.random() * 1e-2), "caption": caption, "key": f"{i:09d}",
                "status": "success",
                "original_width": None if i == 7 else int(rng.integers(50, 2000)),
                "original_height": int(rng.integers(50, 2000)),
            })  # fmt: skip
        _part(rows).write_parquet(tmp_path / f"part-{part:05d}-x-c000.snappy.parquet")
    monkeypatch.setattr(laion, "ROOT", tmp_path)
    monkeypatch.setattr(laion, "WORK", tmp_path / "work")
    return tmp_path


def test_prep_then_ingest_pass_validate_layout(raw, tmp_path):
    args = argparse.Namespace(parts="0-1", keep_items=50, n_heldout=8, seed=0)
    assert laion.cmd_prep(args) == 0
    items = pl.read_parquet(laion.WORK / "items.parquet")
    queries = pl.read_parquet(laion.WORK / "queries.parquet")
    assert items["item_id"].to_list() == list(range(1, 51))
    assert items["caption"].is_unique().all() and "" not in items["caption"].to_list()
    assert not set(queries["caption"]) & set(items["caption"])
    assert set(items["domain"]) <= {"shopify.com", "pinimg.com", "bbc.co.uk", "example.org"}
    assert items["size_bucket"].null_count() == int((items["original_width"].is_null()).sum())

    # Stand-in vectors in encode_text's output shape: the encoder is not run on CPU.
    g = np.random.default_rng(1)
    emb = g.standard_normal((50, laion.EMB_DIM)).astype(np.float16)
    np.save(laion.WORK / "items_000.npy", emb[:30])
    np.save(laion.WORK / "items_001.npy", emb[30:])
    np.save(laion.WORK / "queries.npy", emb[[4, 40, 0, 1, 2, 3, 5, 6]])
    out = tmp_path / "laion30m"
    args = argparse.Namespace(
        output_dir=str(out), config_dir=str(tmp_path / "config"), device="cpu"
    )
    assert laion.cmd_ingest(args) == 0
    content = out / f"content_d{laion.EMB_DIM}"
    assert layout.validate_layout(out, content) == []
    assert pl.read_parquet(out / "heldout.parquet")["item_id"].to_list()[:2] == [5, 41]
    attrs = torch.load(out / "item_attrs_narrow.pt")
    assert attrs.shape == (50, len(laion.CLAUSES), 1)
    assert torch.equal(attrs[:, 0], attrs[:, 3])  # domain and its reverse clause
