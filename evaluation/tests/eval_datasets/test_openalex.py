"""CPU-only tests for the OpenAlex loader (roadmap E3): the inverted-index abstract, the C1
"strictly earlier era" encoding, the stream filter over the snapshot's projected parquet
schema, and prep → attrs → ``validate_layout`` on a tiny citation-connected staging dir.
The parser was checked against live snapshot rows and the API (docs/system/datasets.md
§ openalex); these pin the behaviour offline."""

from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch

from eval_datasets import layout
from eval_datasets.etl import openalex


@pytest.mark.parametrize(
    ("inverted", "expected"),
    [
        (None, ""),
        ("", ""),
        ("{}", ""),
        ('{"world": [1], "hello": [0]}', "hello world"),
        ('{"a": [0, 2], "b": [1], "c": [3]}', "a b a c"),
        ('{"a": [0], "b', None),
    ],
)
def test_abstract_text(inverted, expected):
    assert openalex.abstract_text(inverted) == expected


def test_earlier_era_slots_pass_exactly_the_strictly_earlier_eras():
    n = len(openalex.ERA_NAMES)
    slots = openalex.earlier_era_slots(np.arange(n))
    assert slots.shape == (n, openalex.A_MAX_NARROW)
    for item_era in range(n):
        for query_era in range(n):
            assert (query_era in slots[item_era]) == (item_era < query_era)


def _work(wid, year=2020, lang="en", kind="article", xpac=False, abstract=True, refs=(), field=17):
    return {
        "id": f"https://openalex.org/W{wid}",
        "publication_year": year,
        "language": lang,
        "type": kind,
        "is_paratext": False,
        "is_retracted": False,
        "is_xpac": xpac,
        "primary_topic": {
            "field": {"id": f"https://openalex.org/fields/{field}"},
            "subfield": {"id": f"https://openalex.org/subfields/{field}02"},
        },
        "primary_location": {"source": {"id": f"https://openalex.org/S{wid % 3 + 1}"}},
        "open_access": {"is_oa": wid % 2 == 0},
        "title": f"Paper {wid}",
        "abstract_inverted_index": json.dumps({f"w{wid}": [0], "text": [1]}) if abstract else "",
        "referenced_works": [f"https://openalex.org/W{r}" for r in refs],
    }


def test_stage_table_applies_every_filter_and_parses_ids():
    rows = [
        _work(1, refs=[7, 8]),
        _work(2, year=1999),
        _work(3, lang="de"),
        _work(4, kind="dataset"),
        _work(5, xpac=True),
        _work(6, abstract=False),
    ]
    staged, stats = openalex.stage_table(
        pa.Table.from_pylist(rows), max_year=2026, sample_rate=1.0, seed=0
    )
    assert stats == {
        "rows": 6, "after_year": 5, "after_language": 4, "after_type": 3, "after_flags": 2,
        "after_sample": 2, "abstract_truncated": 0, "after_abstract": 1,
    }  # fmt: skip
    row = staged.row(0, named=True)
    assert (row["work_id"], row["field"], row["subfield"], row["source"]) == (1, 17, 1702, 2)
    assert row["refs"] == [7, 8] and row["abstract"] == "w1 text"


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """40 cited papers (2005) and 20 citing ones (2023), all in field 17, one staging file."""
    monkeypatch.setattr(openalex, "STAGING", tmp_path / "staging")
    monkeypatch.setattr(openalex, "MANIFEST_PATH", tmp_path / "manifest.json")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"date": "2026-09-23", "entities": [{"entity": "works", "files": []}]})
    )
    rows = [_work(w, year=2005) for w in range(1, 41)]
    rows += [_work(w, year=2023, refs=range(1, 41)) for w in range(101, 121)]
    df, _ = openalex.stage_table(pa.Table.from_pylist(rows), max_year=2026, sample_rate=1.0, seed=0)
    (tmp_path / "staging").mkdir()
    df.write_parquet(tmp_path / "staging" / "2026-01-01_part_0000.parquet")
    (tmp_path / "staging" / "params.json").write_text(json.dumps({"seed": 0}))
    return tmp_path


def test_prep_attrs_hold_out_citing_papers_and_pass_validate_layout(staged):
    out = staged / "openalex"
    args = argparse.Namespace(
        output_dir=str(out), keep_items=45, n_heldout=100, seed=0, row_group_rows=10,
        source_vocab=2,
    )  # fmt: skip
    assert openalex.cmd_prep(args) == 0
    assert openalex.cmd_attrs(args) == 0

    id_map = json.loads((out / "item_id_map.json").read_text())
    heldout = pl.read_parquet(out / "heldout.parquet")
    assert len(id_map) == 45 and heldout.height >= 1
    for row in heldout.iter_rows(named=True):
        assert row["query_work_id"] not in id_map
        assert 101 <= int(row["query_work_id"][1:]) <= 120
        assert row["query_era"] == 4 and 1 <= row["item_id"] <= 45

    attrs = torch.load(out / "item_attrs_narrow.pt")
    split = pl.read_parquet(out / "eval_split.parquet")
    qa = torch.tensor(split["query_attrs_narrow"].to_list())
    target = torch.tensor(heldout["item_id"].to_list()) - 1
    assert attrs.shape == (45, openalex.C_NARROW, openalex.A_MAX_NARROW)
    assert torch.load(out / "clause_is_reverse_narrow.pt").tolist() == openalex.CLAUSE_IS_REVERSE
    for c in (0, 1):  # the target passes the query's own field and era
        assert (attrs[target, c, :] == qa[:, c : c + 1]).any(dim=1).all()

    content = out / "content_d768"
    content.mkdir()
    emb = torch.nn.functional.normalize(torch.randn(45, 768), dim=-1).half()
    torch.save(emb, content / "text_emb_shard_000.pt")
    torch.save(emb[: heldout.height], content / "query_emb.pt")
    (content / "shard_index.json").write_text(
        json.dumps({"n_items": 45, "dim": 768, "dtype": "float16",
                    "shards": [{"filename": "text_emb_shard_000.pt", "start_id": 0, "n_rows": 45}]})
    )  # fmt: skip
    for name, prefix in (("text_emb", openalex.DOC_PREFIX), ("query_emb", openalex.QUERY_PREFIX)):
        (content / f"{name}.meta.json").write_text(json.dumps({"prefix": prefix}))
    assert layout.validate_layout(out, content) == []


def test_reshard_gathers_a_smaller_catalogs_vectors_by_work_id(staged):
    big, small = staged / "big", staged / "small"
    prep = {"n_heldout": 100, "seed": 0, "row_group_rows": 10}
    assert openalex.cmd_prep(argparse.Namespace(output_dir=str(big), keep_items=45, **prep)) == 0
    big_ids = pl.read_parquet(big / "papers.parquet")["work_id"].to_list()
    content = big / "content_d768"
    content.mkdir()
    emb = torch.tensor(big_ids, dtype=torch.float16)[:, None].expand(45, 768).contiguous()
    torch.save(emb[:20], content / "text_emb_shard_000.pt")
    torch.save(emb[20:], content / "text_emb_shard_001.pt")
    shards = [{"filename": f"text_emb_shard_00{i}.pt", "start_id": s, "n_rows": n}
              for i, (s, n) in enumerate(((0, 20), (20, 25)))]  # fmt: skip
    (content / "shard_index.json").write_text(
        json.dumps({"n_items": 45, "dim": 768, "dtype": "float16", "shards": shards})
    )
    (content / "encode_params.json").write_text(json.dumps({"prefix": openalex.DOC_PREFIX}))

    assert openalex.cmd_prep(argparse.Namespace(output_dir=str(small), keep_items=30, **prep)) == 0
    args = argparse.Namespace(output_dir=str(small), from_dir=str(big), shard_rows=7)
    assert openalex.cmd_reshard(args) == 0
    got = layout.load_sharded(small / "content_d768" / "shard_index.json", torch.device("cpu"))
    small_ids = pl.read_parquet(small / "papers.parquet")["work_id"].to_list()
    assert set(small_ids) < set(big_ids)
    assert torch.equal(got, torch.tensor(small_ids, dtype=torch.float16)[:, None].expand(30, 768))
