"""CPU-only tests for the Re-LAION loader (roadmap V-LAION30): registered domains, the tag
buckets, the exact top-1 target, and prep → (stand-in vectors) → targets → attrs →
``validate_layout`` on a tiny pair of parts in the upstream column schema."""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F

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
    got = laion.bucket(np.array([0.0, 200.0, 299.0, 300.0, 5000.0, np.nan]), laion.SIZE_EDGES)
    assert got.tolist() == [0, 1, 1, 2, 4, -1]


def test_nearest_items_is_exact_and_ties_go_to_the_lowest_id():
    g = torch.Generator().manual_seed(0)
    items = F.normalize(torch.randn(50, 8, generator=g), dim=-1)
    items[37] = items[3]  # a tie across two chunks
    items[12] = items[9]  # a tie inside one chunk
    queries = torch.cat([items[[3, 9]], F.normalize(torch.randn(6, 8, generator=g), dim=-1)])
    ids, scores = laion.nearest_items(items, queries, item_chunk=7, query_batch=3)
    # A chunked product rounds differently from one matmul: ids against fp64, scores to 1e-6.
    ref = queries.double() @ items.double().t()
    assert ids[:2].tolist() == [3, 9]
    assert torch.equal(ids[2:], ref[2:].argmax(dim=1))
    assert torch.allclose(scores.double(), ref.max(dim=1).values, atol=1e-6)


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
    return tmp_path


def test_prep_targets_attrs_pass_validate_layout(raw, tmp_path):
    out = tmp_path / "laion30m"
    args = argparse.Namespace(output_dir=str(out), parts="0-1", keep_items=50, n_heldout=8, seed=0)
    assert laion.cmd_prep(args) == 0
    items = pl.read_parquet(out / "items.parquet")
    queries = pl.read_parquet(out / "queries.parquet")
    assert items["item_id"].to_list() == list(range(1, 51))
    assert items["caption"].is_unique().all() and "" not in items["caption"].to_list()
    assert not set(queries["caption"]) & set(items["caption"])
    assert set(items["domain"]) <= {"shopify.com", "pinimg.com", "bbc.co.uk", "example.org"}

    # Stand-in vectors: the encoder is not run on CPU; targets reads whatever is staged.
    content = out / f"content_d{laion.EMB_DIM}"
    content.mkdir()
    g = torch.Generator().manual_seed(1)
    emb = F.normalize(torch.randn(50, laion.EMB_DIM, generator=g), dim=-1).half()
    torch.save(emb[:30], content / "text_emb_shard_000.pt")
    torch.save(emb[30:], content / "text_emb_shard_001.pt")
    shards = [{"filename": "text_emb_shard_000.pt", "start_id": 0, "n_rows": 30},
              {"filename": "text_emb_shard_001.pt", "start_id": 30, "n_rows": 20}]  # fmt: skip
    (content / "shard_index.json").write_text(
        json.dumps({"n_items": 50, "dim": laion.EMB_DIM, "dtype": "float16", "shards": shards})
    )
    torch.save(emb[[4, 40, 0, 1, 2, 3, 5, 6]], content / "query_emb.pt")
    for name, prefix in (("text_emb", laion.DOC_PREFIX), ("query_emb", laion.QUERY_PREFIX)):
        (content / f"{name}.meta.json").write_text(json.dumps({"prefix": prefix}))

    assert laion.cmd_targets(argparse.Namespace(output_dir=str(out), device="cpu")) == 0
    heldout = pl.read_parquet(out / "heldout.parquet")
    assert heldout["item_id"].to_list()[:2] == [5, 41]
    assert laion.cmd_attrs(argparse.Namespace(output_dir=str(out))) == 0
    assert layout.validate_layout(out, content) == []

    attrs = torch.load(out / "item_attrs_narrow.pt")
    assert attrs.shape == (50, len(laion.CLAUSE_NAMES), 1)
    assert torch.equal(attrs[:, 0], attrs[:, 3])  # domain and its reverse clause
    vocab = json.loads((out / "domain_vocab.json").read_text())
    assert vocab["counts"] == sorted(vocab["counts"], reverse=True)
    codes = attrs[:, 0, 0].numpy()
    assert [vocab["domains"][c] for c in codes] == items["domain"].to_list()
    qa = np.array(pl.read_parquet(out / "eval_split.parquet")["query_attrs_narrow"].to_list())
    assert [vocab["domains"][c] for c in qa[:, 0]] == queries["domain"].to_list()
