"""goodreads `attrs` determinism: two builds of one synthetic catalog are byte-identical.

The fixture is sized so polars' parallel ``group_by`` hands back its groups in a different order
on each run (a few thousand tied authors / shelves): with a count-only sort the vocab ids then
differ between builds. CPU only.
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from eval_datasets.etl import goodreads

N_BOOKS = 3_000
LANGS = ["eng", "fre", "ger", "spa", "ita", "por"]


def _write_inputs(processed, out):
    processed.mkdir(parents=True)
    out.mkdir(parents=True)
    books = pl.DataFrame(
        {
            "book_id": [str(b) for b in range(1, N_BOOKS + 1)],
            # every language and author occurs equally often: all tied
            "language_code": [LANGS[b % len(LANGS)] for b in range(N_BOOKS)],
            "format": ["Paperback" if b % 2 else "Hardcover" for b in range(N_BOOKS)],
            "publication_year": [str(1980 + b % 40) for b in range(N_BOOKS)],
            "authors": [[{"author_id": f"a{b}", "role": ""}] for b in range(N_BOOKS)],
            "popular_shelves": [
                [{"count": "3", "name": f"shelf-{b}"}, {"count": "3", "name": f"shelf-{b + 1}"}]
                for b in range(N_BOOKS)
            ],
        }
    )
    books.write_parquet(processed / "goodreads_books.parquet")
    # five genres tied at the same count per book: the top-4 cut has to pick by id
    genres = pl.DataFrame(
        {
            "book_id": [str(b) for b in range(1, N_BOOKS + 1)],
            "genres": [{k: (7 if i < 5 else None) for i, k in enumerate(goodreads.GENRE_KEYS)}]
            * N_BOOKS,
        }
    )
    genres.write_parquet(processed / "goodreads_book_genres_initial.parquet")
    # two editions per work
    pl.DataFrame(
        {
            "book_id": list(range(1, N_BOOKS + 1)),
            "work_id": [(b + 1) // 2 for b in range(1, N_BOOKS + 1)],
        },
        schema={"book_id": pl.Int64, "work_id": pl.Int64},
    ).write_parquet(out / "book_to_work.parquet")
    n_works = N_BOOKS // 2
    (out / "item_id_map.json").write_text(json.dumps({str(w): w for w in range(1, n_works + 1)}))
    pl.DataFrame(
        {"targets": [[w] for w in range(1, 51)]}, schema={"targets": pl.List(pl.Int64)}
    ).write_parquet(out / "test.parquet")


def test_attrs_two_builds_are_byte_identical(tmp_path):
    builds = []
    for name in ("a", "b"):
        processed, out = tmp_path / name / "processed", tmp_path / name / "out"
        _write_inputs(processed, out)
        args = argparse.Namespace(processed_dir=str(processed), output_dir=str(out), seed=0)
        assert goodreads.cmd_attrs(args) == 0
        builds.append(out)
    inputs = {"book_to_work.parquet", "item_id_map.json", "test.parquet", "prep_log.json"}
    files = sorted(p.name for p in builds[0].iterdir() if p.name not in inputs)
    assert {
        "author_vocab.json",
        "lang_vocab.json",
        "item_attrs_narrow.pt",
        "eval_split.parquet",
    } <= set(files)
    for f in files:
        assert (builds[0] / f).read_bytes() == (builds[1] / f).read_bytes(), f
    assert json.loads((builds[0] / "lang_vocab.json").read_text()) == {
        c: i for i, c in enumerate(sorted(LANGS))
    }
