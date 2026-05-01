#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyarrow>=15",
#   "polars>=1.0",
#   "duckdb>=0.10",
# ]
# ///
"""End-to-end mirror of the UCSD Book Graph (a.k.a. UCSD Goodreads) dataset.

Subcommands:
  download   Pull 13 top-level files via HTTPS into raw/. Resumable via
             Range header. Writes raw/MANIFEST.txt with sha256s.
  convert    Stream raw → ZSTD parquet under processed/. Idempotent.
  sort       External-sort configured parquets via duckdb (spill to disk).
             Idempotent via .sorted markers.
  all        download → convert → sort.

Examples::

    uv run goodreads.py all
    uv run goodreads.py download --force goodreads_books.json.gz
    uv run goodreads.py sort goodreads_interactions_dedup.parquet

Layout::

    ~/datasets/goodreads-ucsd/
    ├── raw/         <- mirror destination for source files + MANIFEST.txt
    └── processed/   <- parquet outputs + .sorted markers + logs
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path.home() / "datasets" / "goodreads-ucsd"
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"

DOWNLOAD_FILES = [
    "goodreads_books.json.gz",
    "goodreads_book_authors.json.gz",
    "goodreads_book_works.json.gz",
    "goodreads_book_series.json.gz",
    "goodreads_book_genres_initial.json.gz",
    "goodreads_interactions.csv",
    "goodreads_interactions_dedup.json.gz",
    "book_id_map.csv",
    "user_id_map.csv",
    "book_clubs.json",
    "goodreads_reviews_dedup.json.gz",
    "goodreads_reviews_spoiler.json.gz",
    "goodreads_reviews_spoiler_raw.json.gz",
]

# Plain NDJSON.gz (straight pass-through). Spoiler is handled separately.
NDJSON_GZ = [
    "goodreads_books.json.gz",
    "goodreads_book_authors.json.gz",
    "goodreads_book_works.json.gz",
    "goodreads_book_series.json.gz",
    "goodreads_book_genres_initial.json.gz",
    "goodreads_interactions_dedup.json.gz",
    "goodreads_reviews_dedup.json.gz",
    "goodreads_reviews_spoiler_raw.json.gz",
]
SPOILER = "goodreads_reviews_spoiler.json.gz"
CSVS = {
    "goodreads_interactions.csv": {
        "user_id": pl.UInt32,
        "book_id": pl.UInt32,
        "is_read": pl.UInt8,
        "rating": pl.UInt8,
        "is_reviewed": pl.UInt8,
    },
    "book_id_map.csv": None,
    "user_id_map.csv": None,
}
BOOK_CLUBS = "book_clubs.json"

SORT_KEYS: dict[str, list[str]] = {
    "goodreads_interactions_dedup.parquet": ["user_id", "book_id"],
    "goodreads_interactions.parquet": ["user_id", "book_id"],
}

FIRST_BATCH = 200_000
BATCH = 50_000


def fmt(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i, f = 0, float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f} {units[i]}"


# ----- download --------------------------------------------------------------


def http_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except (urllib.error.URLError, TimeoutError):
        return None


def download_one(url: str, dest: Path, *, retries: int = 5) -> int:
    """Resumable HTTPS pull. Honors HTTP 206 Partial Content; falls back to
    full re-download if the server returns 200 to a Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = http_size(url)
    last_err: Exception | None = None
    for attempt in range(retries):
        existing = dest.stat().st_size if dest.exists() else 0
        if expected is not None and existing == expected:
            return existing
        if expected is not None and existing > expected:
            dest.unlink()
            existing = 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                # Detect server ignoring Range → restart from scratch.
                if existing > 0 and r.status != 206:
                    mode = "wb"
                    existing = 0
                else:
                    mode = "ab"
                with open(dest, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            existing = dest.stat().st_size
            if expected is None or existing == expected:
                return existing
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt == retries - 1:
                raise
            time.sleep(5)
    if last_err is not None:
        raise last_err
    return dest.stat().st_size


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cmd_download(args) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = args.files or DOWNLOAD_FILES
    overall_t0 = time.monotonic()

    manifest_lines: list[str] = []
    for i, name in enumerate(files, 1):
        url = f"{BASE_URL}/{name}"
        dest = RAW_DIR / name
        size = dest.stat().st_size if dest.exists() else 0
        if dest.exists() and not args.force:
            expected = http_size(url)
            if expected is None or size == expected:
                sha = sha256_of(dest)
                manifest_lines.append(f"{name}  {size}  {sha}")
                print(f"SKIP [{i}/{len(files)}] {name} ({fmt(size)})", flush=True)
                continue
        print(f"START [{i}/{len(files)}] {name}", flush=True)
        t0 = time.monotonic()
        try:
            size = download_one(url, dest)
        except Exception as e:
            print(f"ERROR [{i}/{len(files)}] {name}: {e}", flush=True)
            continue
        sha = sha256_of(dest)
        manifest_lines.append(f"{name}  {size}  {sha}")
        print(
            f"DONE [{i}/{len(files)}] {name} {fmt(size)} in " f"{time.monotonic() - t0:.0f}s",
            flush=True,
        )

    (RAW_DIR / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n")

    fail = 0
    for f in sorted(RAW_DIR.glob("*.json.gz")):
        try:
            with gzip.open(f, "rb") as g:
                while g.read(1 << 24):
                    pass
        except Exception as e:
            print(f"ERROR gzip {f.name}: {e}", flush=True)
            fail += 1
    print(
        f"ALL DONE download in {time.monotonic() - overall_t0:.0f}s, " f"gzip failures={fail}",
        flush=True,
    )
    return 0 if fail == 0 else 1


# ----- convert ---------------------------------------------------------------


def stream_ndjson_gz(src: Path, dst: Path, *, transform=None) -> int:
    rows: list[dict] = []
    total = 0
    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None

    def flush() -> None:
        nonlocal writer, schema
        if not rows:
            return
        if schema is None:
            table = pa.Table.from_pylist(rows)
            schema = table.schema
            writer = pq.ParquetWriter(dst, schema, compression="zstd")
        else:
            table = pa.Table.from_pylist(rows, schema=schema)
        assert writer is not None
        writer.write_table(table)
        rows.clear()

    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if transform is not None:
                obj = transform(obj)
            rows.append(obj)
            total += 1
            cap = FIRST_BATCH if schema is None else BATCH
            if len(rows) >= cap:
                flush()

    flush()
    if writer is not None:
        writer.close()
    return total


def spoiler_transform(obj: dict) -> dict:
    sents = obj.get("review_sentences") or []
    obj["review_sentences"] = [
        {"is_spoiler": bool(s[0]), "text": s[1]}
        for s in sents
        if isinstance(s, list) and len(s) == 2
    ]
    return obj


def convert_csv(src: Path, dst: Path, schema) -> int:
    lf = pl.scan_csv(src, schema=schema) if schema else pl.scan_csv(src)
    lf.sink_parquet(dst, compression="zstd")
    return pl.scan_parquet(dst).select(pl.len()).collect().item()


def convert_book_clubs(src: Path, dst: Path) -> int:
    text = src.read_text(encoding="utf-8")
    first = text.split("\n", 1)[0].strip()
    try:
        json.loads(first)
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    except json.JSONDecodeError:
        obj = json.loads(text)
        rows = obj if isinstance(obj, list) else [obj]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, dst, compression="zstd")
    return len(rows)


def convert_one(label: str, src: Path, dst: Path, fn) -> None:
    if not src.exists():
        print(f"SKIP {label} (missing)", flush=True)
        return
    if dst.exists():
        print(
            f"SKIP {label} (already converted, {fmt(dst.stat().st_size)})",
            flush=True,
        )
        return
    print(f"START {label}", flush=True)
    t0 = time.monotonic()
    try:
        rows = fn(src, dst)
        in_b, out_b = src.stat().st_size, dst.stat().st_size
        ratio = 100 * out_b / in_b if in_b else 0
        print(
            f"DONE {label} {rows:,} rows {fmt(in_b)}→{fmt(out_b)} "
            f"({ratio:.0f}%) in {time.monotonic() - t0:.0f}s",
            flush=True,
        )
    except Exception as e:
        if dst.exists():
            dst.unlink()
        print(f"ERROR {label}: {type(e).__name__}: {e}", flush=True)


def cmd_convert(args) -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    overall_t0 = time.monotonic()

    for name in NDJSON_GZ:
        convert_one(
            name,
            RAW_DIR / name,
            PROCESSED_DIR / name.replace(".json.gz", ".parquet"),
            stream_ndjson_gz,
        )

    convert_one(
        SPOILER,
        RAW_DIR / SPOILER,
        PROCESSED_DIR / SPOILER.replace(".json.gz", ".parquet"),
        lambda s, d: stream_ndjson_gz(s, d, transform=spoiler_transform),
    )

    for name, schema in CSVS.items():
        convert_one(
            name,
            RAW_DIR / name,
            PROCESSED_DIR / name.replace(".csv", ".parquet"),
            lambda s, d, sch=schema: convert_csv(s, d, sch),
        )

    convert_one(
        BOOK_CLUBS,
        RAW_DIR / BOOK_CLUBS,
        PROCESSED_DIR / BOOK_CLUBS.replace(".json", ".parquet"),
        convert_book_clubs,
    )

    print(f"ALL DONE convert in {time.monotonic() - overall_t0:.0f}s", flush=True)
    return 0


# ----- sort ------------------------------------------------------------------


def sort_one_duckdb(path: Path, keys: list[str]) -> None:
    marker = path.with_suffix(path.suffix + ".sorted")
    if marker.exists():
        print(f"SKIP-SORT {path.name} (already sorted)", flush=True)
        return
    if not path.exists():
        print(f"ERROR {path.name} (missing)", flush=True)
        return

    tmp_dir = path.parent / "_duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_out = path.with_suffix(".sorted.parquet.tmp")
    if tmp_out.exists():
        tmp_out.unlink()

    print(f"SORT {path.name} by {keys} (duckdb external)", flush=True)
    t0 = time.monotonic()
    before = path.stat().st_size

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}'")
    con.execute("PRAGMA threads=4")

    src_sql = str(path).replace("'", "''")
    dst_sql = str(tmp_out).replace("'", "''")
    order = ", ".join(keys)
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{src_sql}') ORDER BY {order}) "
        f"TO '{dst_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()

    tmp_out.replace(path)
    marker.write_text("sorted by " + ",".join(keys) + " (duckdb)\n")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    after = path.stat().st_size
    pct = 100 * after / before if before else 0
    print(
        f"SORTED {path.name} {fmt(before)}→{fmt(after)} ({pct:.0f}%) "
        f"in {time.monotonic() - t0:.0f}s",
        flush=True,
    )


def cmd_sort(args) -> int:
    overall_t0 = time.monotonic()
    targets = args.files or list(SORT_KEYS.keys())
    for name in targets:
        keys = SORT_KEYS.get(name)
        if keys is None:
            print(f"ERROR no SORT_KEYS entry for {name}", flush=True)
            continue
        sort_one_duckdb(PROCESSED_DIR / name, keys)
    print(f"ALL DONE sort in {time.monotonic() - overall_t0:.0f}s", flush=True)
    return 0


# ----- all -------------------------------------------------------------------


def cmd_all(args) -> int:
    rc = cmd_download(argparse.Namespace(files=None, force=False))
    if rc != 0:
        return rc
    rc = cmd_convert(args)
    if rc != 0:
        return rc
    return cmd_sort(argparse.Namespace(files=None))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror the UCSD Goodreads dataset (download → parquet → sort).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_dl = sub.add_parser("download", help="fetch raw files into raw/")
    sp_dl.add_argument("files", nargs="*", help="optional subset of filenames")
    sp_dl.add_argument("--force", action="store_true", help="re-download even if present")
    sp_dl.set_defaults(func=cmd_download)

    sp_cv = sub.add_parser("convert", help="raw/ → processed/ parquet")
    sp_cv.set_defaults(func=cmd_convert)

    sp_sr = sub.add_parser("sort", help="external-sort configured parquets via duckdb")
    sp_sr.add_argument("files", nargs="*", help="parquet basenames; default = SORT_KEYS")
    sp_sr.set_defaults(func=cmd_sort)

    sp_al = sub.add_parser("all", help="download → convert → sort")
    sp_al.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
