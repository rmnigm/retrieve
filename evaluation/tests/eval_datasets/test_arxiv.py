"""arXiv `attrs` determinism: two builds of one synthetic paper table are byte-identical.

Sized so polars' parallel ``group_by`` returns its groups in a different order on each run
(thousands of tied authors and leaf categories): with a count-only sort the vocab ids then
differ between builds. CPU only.
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from eval_datasets.etl import arxiv

N_PAPERS = 3_000
MAINS = ["cs", "math", "physics", "stat"]


def _write_inputs(out):
    out.mkdir(parents=True)
    ids = range(N_PAPERS)
    # main categories and leaves equally frequent: all tied
    cats = [f"{MAINS[i % 4]}.L{i % 750} {MAINS[(i + 1) % 4]}.L{(i + 1) % 750}" for i in ids]
    authors = [json.dumps([[f"Last{i}", "A", ""], [f"Last{i + 1}", "B", ""]]) for i in ids]
    cc_by = "http://creativecommons.org/licenses/by/4.0/"
    pl.DataFrame(
        {
            "item_id": [i + 1 for i in ids],
            "categories": cats,
            "authors_parsed": authors,
            "license": [None if i % 3 else cc_by for i in ids],
            "update_date": [f"{2000 + i % 24}-01-01" for i in ids],
            "versions": [json.dumps([{"version": "v1"}] * (1 + i % 3)) for i in ids],
        },
        schema={
            "item_id": pl.Int64,
            "categories": pl.Utf8,
            "authors_parsed": pl.Utf8,
            "license": pl.Utf8,
            "update_date": pl.Utf8,
            "versions": pl.Utf8,
        },
    ).write_parquet(out / "papers.parquet")
    pl.DataFrame({"item_id": list(range(1, 51))}, schema={"item_id": pl.Int64}).write_parquet(
        out / "heldout.parquet"
    )
    (out / "item_id_map.json").write_text(json.dumps({f"p{i}": i for i in range(1, N_PAPERS + 1)}))


def test_attrs_two_builds_are_byte_identical(tmp_path):
    builds = []
    for name in ("a", "b"):
        out = tmp_path / name
        _write_inputs(out)
        assert (
            arxiv.cmd_attrs(argparse.Namespace(output_dir=str(out), seed=0, wide_min_count=1)) == 0
        )
        builds.append(out)
    inputs = {"papers.parquet", "heldout.parquet", "item_id_map.json", "prep_log.json"}
    files = sorted(p.name for p in builds[0].iterdir() if p.name not in inputs)
    assert {"cat_main_vocab.json", "item_attrs_narrow.pt", "eval_split.parquet"} <= set(files)
    for f in files:
        assert (builds[0] / f).read_bytes() == (builds[1] / f).read_bytes(), f
    assert json.loads((builds[0] / "cat_main_vocab.json").read_text())["names"] == sorted(MAINS)
