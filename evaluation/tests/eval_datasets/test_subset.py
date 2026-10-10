"""CPU-only tests for ``eval-data subset`` (the fixed-d N-sweep): two builds of one seed are
byte-identical, the rows nest across n, every item-side file is the parent's rows, and the
targets are the exact top-1 over the subset."""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
import yaml

from eval_datasets import layout, synth_filter
from eval_datasets.ingest import ingest, nearest_items
from eval_datasets.subset import build, subset_rows, write_configs


def _parent(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 60, 16
    items = rng.standard_normal((n, d)).astype(np.float32)
    np.save(tmp_path / "items.npy", items)
    np.save(tmp_path / "q.npy", rng.standard_normal((8, d)).astype(np.float32))
    pl.DataFrame({"site": [f"s{i % 5}" for i in range(n)]}).write_parquet(tmp_path / "ia.parquet")
    pl.DataFrame({"site": [f"s{i % 5}" for i in range(8)]}).write_parquet(tmp_path / "qa.parquet")
    parent, cfg = tmp_path / "data" / "par", tmp_path / "config"
    problems = ingest(
        "par", items=[tmp_path / "items.npy"], item_attrs=tmp_path / "ia.parquet",
        queries=tmp_path / "q.npy", query_attrs=tmp_path / "qa.parquet", clauses=["site"],
        sweeps=["c0=site"], output_dir=parent, config_dir=cfg, shard_rows=13,
    )  # fmt: skip
    assert problems == []
    synth_filter.build(parent)
    return parent, cfg


def test_two_builds_of_one_seed_are_byte_identical_and_seeds_differ(tmp_path):
    parent, _ = _parent(tmp_path)
    a = build(parent, tmp_path / "data" / "a", 25, seed=7, shard_rows=10)
    b = build(parent, tmp_path / "data" / "b", 25, seed=7, shard_rows=10)
    c = build(parent, tmp_path / "data" / "c", 25, seed=8, shard_rows=10)
    assert a["problems"] == [] and a["rows_sha256"] == b["rows_sha256"] != c["rows_sha256"]
    man = (tmp_path / "data" / "a" / "MANIFEST.sha256").read_text()
    assert man == (tmp_path / "data" / "b" / "MANIFEST.sha256").read_text()
    assert man != (tmp_path / "data" / "c" / "MANIFEST.sha256").read_text()


def test_rows_nest_and_item_files_are_the_parents_rows(tmp_path):
    parent, cfg = _parent(tmp_path)
    small, big = subset_rows(60, 20, seed=3), subset_rows(60, 40, seed=3)
    assert set(small.tolist()) <= set(big.tolist()) and torch.equal(small, small.sort().values)
    out = tmp_path / "data" / "sub"
    build(parent, out, 40, seed=3, shard_rows=10)
    rows = big
    pe = layout.load_text_items(parent / "content_d16", torch.device("cpu"))
    se = layout.load_text_items(out / "content_d16", torch.device("cpu"))
    assert torch.equal(se, pe[rows])
    for f in ("item_attrs_narrow.pt", synth_filter.ITEM_ATTRS, synth_filter.U_FILE):
        assert torch.equal(torch.load(out / f), torch.load(parent / f)[rows]), f
    for f in ("clause_is_reverse_narrow.pt", synth_filter.QUERY_ATTRS, "content_d16/query_emb.pt"):
        assert (out / f).read_bytes() == (parent / f).read_bytes(), f
    want, _ = nearest_items(se.float(), torch.load(out / "content_d16/query_emb.pt").float())
    assert pl.read_parquet(out / "heldout.parquet")["item_id"].to_list() == (want + 1).tolist()
    side = yaml.safe_load((out / synth_filter.SIDECAR).read_text())
    assert side["n_items"] == 40 and side["pass_counts"][-1] == 40
    write_configs("par", "sub", cfg)
    assert yaml.safe_load((cfg / "sub.yaml").read_text())["data_dir"] == "data/sub"
