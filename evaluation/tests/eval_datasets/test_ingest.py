"""CPU-only tests for ``eval-data ingest`` (roadmap H-ADDDATA): the clause spec, the exact
top-1 target, and a tiny dataset ingested from npy + parquet that passes ``validate_layout``
and loads through the harness's readers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn.functional as F
import yaml

from eval_datasets import layout
from eval_datasets.ingest import ingest, nearest_items, parse_clause


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("site", ("site", "site", False)), ("other=site!", ("other", "site", True)),
     ("tag=tags", ("tag", "tags", False))],
)  # fmt: skip
def test_parse_clause(spec, expected):
    assert parse_clause(spec) == expected


def test_nearest_items_is_exact_and_ties_go_to_the_lowest_id():
    g = torch.Generator().manual_seed(0)
    items = F.normalize(torch.randn(50, 8, generator=g), dim=-1)
    items[37] = items[3]  # a tie across two chunks
    items[12] = items[9]  # a tie inside one chunk
    queries = torch.cat([items[[3, 9]], F.normalize(torch.randn(6, 8, generator=g), dim=-1)])
    ids, scores = nearest_items(items, queries, item_chunk=7, query_batch=3)
    # A chunked product rounds differently from one matmul: ids against fp64, scores to 1e-6.
    ref = queries.double() @ items.double().t()
    assert ids[:2].tolist() == [3, 9]
    assert torch.equal(ids[2:], ref[2:].argmax(dim=1))
    assert torch.allclose(scores.double(), ref.max(dim=1).values, atol=1e-6)


def test_ingest_stages_a_dataset_the_harness_reads(tmp_path):
    rng = np.random.default_rng(0)
    n, u, d = 40, 6, 16
    items = rng.standard_normal((n, d)).astype(np.float32)
    np.save(tmp_path / "a.npy", items[:25])
    np.save(tmp_path / "b.npy", items[25:].astype(np.float16))
    np.save(tmp_path / "q.npy", items[[30, 2, 7, 11, 19, 0]])
    sites = ["x.com", "y.org", "z.net"]
    pl.DataFrame({
        "site": [sites[i % 3] for i in range(n)],
        "tags": [[i % 4, (i + 1) % 4] if i % 5 else [i % 4] for i in range(n)],
        "year": [None if i == 3 else 2000 + i % 2 for i in range(n)],
    }).write_parquet(tmp_path / "items.parquet")  # fmt: skip
    pl.DataFrame({
        "site": ["x.com", "y.org", "nowhere", "z.net", "x.com", None],
        "tags": [1, 2, 3, 9, 0, 1], "year": [2000, 2001, 2000, 2001, 1999, 2000],
    }).write_parquet(tmp_path / "queries.parquet")  # fmt: skip
    out, cfg = tmp_path / "ds", tmp_path / "config"
    problems = ingest(
        "tiny", items=[tmp_path / "a.npy", tmp_path / "b.npy"],
        item_attrs=tmp_path / "items.parquet", queries=tmp_path / "q.npy",
        query_attrs=tmp_path / "queries.parquet",
        clauses=["site", "tag=tags", "other=site!", "year"],
        sweeps=["c0=site", "c2=other", "both=site,tag"], output_dir=out, config_dir=cfg,
        shard_rows=10,
    )  # fmt: skip
    assert problems == []
    content = out / "content_d16"
    emb = layout.load_text_items(content, torch.device("cpu"))
    assert torch.allclose(emb, F.normalize(torch.from_numpy(items), dim=-1), atol=1e-3)
    queries, targets, _ = layout.load_text_queries(out, content, d)
    assert targets[:, 0].tolist()[:3] == [30, 2, 7]  # each query is an item's own vector

    attrs = torch.load(out / "item_attrs_narrow.pt")
    assert attrs.shape == (n, 4, 2)  # the longest tags list is 2
    assert torch.equal(attrs[:, 0], attrs[:, 2])
    assert attrs[3, 3, 0] == -1 and attrs[0, 1, 1] == -1  # null year; one-tag row padded
    assert torch.load(out / "clause_is_reverse_narrow.pt").tolist() == [False, False, True, False]
    qa = layout.load_query_attrs(out / "eval_split.parquet", u)
    assert qa[2, 0] == -1 and qa[5, 0] == -1 and qa[3, 1] == -1 and qa[4, 3] == -1
    site_codes = {"x.com": 0, "y.org": 1, "z.net": 2}  # 14 / 13 / 13 items
    assert attrs[:, 0, 0].tolist() == [site_codes[sites[i % 3]] for i in range(n)]

    text = (cfg / "tiny.yaml").read_text()
    assert "&" not in text  # no yaml anchors between the clause and bloom blocks
    config = yaml.safe_load(text)
    assert config["data_dir"] == str(out) and config["dims"] == [16]
    assert config["filters"]["clause"] == {"c0": [0], "c2": [2], "both": [0, 1]}
    assert config["filters"]["bloom"] == {"c0": [0], "both": [0, 1]}
