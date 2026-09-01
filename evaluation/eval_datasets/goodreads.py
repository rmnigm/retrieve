#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyarrow>=15",
#   "polars>=1.0",
# ]
# ///
"""End-to-end mirror of the UCSD Book Graph (a.k.a. UCSD Goodreads) dataset.

Subcommands:
  download   Pull top-level files via HTTPS into raw/. Resumable via Range
             header. Writes raw/MANIFEST.txt with sha256s.
  convert    Stream raw → ZSTD parquet under processed/. Idempotent.
  prep       processed/ → yambda-format trainer inputs (train/val/test/item_id_map).
  attrs      Build narrow + wide attribute tensors + eval_split for the filter bench.
  all        download → convert.

Examples::

    uv run goodreads all
    uv run goodreads download --force goodreads_books.json.gz
    uv run goodreads prep \\
      --processed-dir ~/datasets/goodreads-ucsd/processed \\
      --output-dir data/goodreads/work-id

Layout::

    ~/datasets/goodreads-ucsd/
    ├── raw/         <- mirror destination for source files + MANIFEST.txt
    └── processed/   <- parquet outputs + logs
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from .common import sample_rare_biased_wide, synthesize_qa_narrow

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


# ----- all -------------------------------------------------------------------


def cmd_all(args) -> int:
    rc = cmd_download(argparse.Namespace(files=None, force=False))
    if rc != 0:
        return rc
    return cmd_convert(args)


# ----- prep ------------------------------------------------------------------
#
# processed/ -> yambda-shaped trainer inputs. Output file list and semantics:
# docs/system/datasets.md § goodreads.


def _import_timesplit():
    """Resolve the package-relative timesplit import in both module and script modes."""
    try:
        from .timesplit import sequential_split_train_val_test as f
    except ImportError:
        # uv-run-script mode: `eval_datasets` isn't on sys.path as a package.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from eval_datasets.timesplit import sequential_split_train_val_test as f
    return f


def cmd_prep(args) -> int:
    sequential_split_train_val_test = _import_timesplit()

    DAY = 24 * 60 * 60
    processed = Path(args.processed_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    interactions_path = processed / "goodreads_interactions_dedup.parquet"
    books_path = processed / "goodreads_books.parquet"
    if not interactions_path.exists():
        print(f"ERROR missing {interactions_path}", flush=True)
        return 1
    if not books_path.exists():
        print(f"ERROR missing {books_path}", flush=True)
        return 1

    log: dict = {"output_dir": str(output)}
    overall_t0 = time.monotonic()

    # ---- step 1-3: scan, filter is_read, parse date_added ----
    print("STEP scan + filter is_read + parse date_added", flush=True)
    # user_id in the JSON-dedup variant is a 32-char hex string (the CSV variant
    # has an int map but we don't use it). Keep it as-is — downstream code only
    # uses it for grouping/joining, not arithmetic.
    inter = pl.scan_parquet(str(interactions_path)).select(
        pl.col("user_id"),
        pl.col("book_id").cast(pl.Int64),
        pl.col("is_read"),
        pl.col("date_added"),
    )
    inter = inter.filter(pl.col("is_read").cast(pl.Boolean))
    inter = inter.filter(
        pl.col("date_added").is_not_null() & (pl.col("date_added").cast(pl.Utf8).str.len_chars() > 0)
    )

    n_pre_parse = inter.select(pl.len()).collect(engine="streaming").item()
    log["n_after_is_read"] = int(n_pre_parse)

    inter = inter.with_columns(
        pl.col("date_added")
        .str.strptime(pl.Datetime, "%a %b %d %H:%M:%S %z %Y", strict=False)
        .alias("dt")
    )
    inter = inter.with_columns(
        pl.col("dt").dt.epoch(time_unit="s").cast(pl.Int64).alias("ts"),
    )

    n_after_parse = inter.filter(pl.col("dt").is_not_null()).select(pl.len()).collect(
        engine="streaming"
    ).item()
    parse_fail = n_pre_parse - n_after_parse
    parse_fail_pct = (100.0 * parse_fail / n_pre_parse) if n_pre_parse else 0.0
    log["parse_fail_pct"] = round(parse_fail_pct, 4)
    print(f"  parse_fail_pct = {parse_fail_pct:.3f}%", flush=True)
    if parse_fail_pct > 5.0:
        print(
            f"ERROR parse_fail_pct {parse_fail_pct:.3f}% exceeds 5% floor — "
            "format drift; aborting.",
            flush=True,
        )
        return 1

    inter = inter.filter(pl.col("dt").is_not_null() & (pl.col("dt").dt.year() >= 2007))
    inter = inter.select("user_id", "book_id", "ts")

    # ---- step 4: book_id → work_id ----
    # In `goodreads_books.parquet`, both book_id and work_id are strings (some
    # empty). The interactions side has book_id as int. Filter empties, cast
    # both to Int64 (strict=False makes any leftover non-numeric → null), then
    # drop nulls so the join key types match.
    print("STEP join book_id → work_id", flush=True)
    books = (
        pl.scan_parquet(str(books_path))
        .select(pl.col("book_id"), pl.col("work_id"))
        .filter(
            pl.col("book_id").is_not_null()
            & pl.col("work_id").is_not_null()
            & (pl.col("book_id").cast(pl.Utf8) != "")
            & (pl.col("work_id").cast(pl.Utf8) != "")
        )
        .with_columns(
            pl.col("book_id").cast(pl.Int64, strict=False),
            pl.col("work_id").cast(pl.Int64, strict=False),
        )
        .filter(pl.col("book_id").is_not_null() & pl.col("work_id").is_not_null())
    )
    books_collected = books.unique(subset=["book_id"]).collect(engine="streaming")
    books_collected.write_parquet(output / "book_to_work.parquet", compression="zstd")
    print(f"  wrote book_to_work.parquet ({books_collected.height} rows)", flush=True)

    inter = inter.join(books_collected.lazy(), on="book_id", how="inner").select(
        "user_id", "work_id", "ts"
    )

    # ---- step 5: iterative n-core filter ----
    print("STEP iterative n-core filter", flush=True)
    df = inter.collect(engine="streaming")
    log["n_before_core"] = df.height

    five_core_trace: list[dict] = []
    n_core = args.n_core
    eps = 1e-3  # stop if a round drops <0.1 % of rows
    last_drop = float("inf")
    n_users = n_works = 0
    for it in range(10):
        n_before = df.height
        u_keep = df.group_by("user_id").len().filter(pl.col("len") >= n_core).select("user_id")
        df = df.join(u_keep, on="user_id", how="inner")
        i_keep = df.group_by("work_id").len().filter(pl.col("len") >= n_core).select("work_id")
        df = df.join(i_keep, on="work_id", how="inner")
        n_after = df.height
        n_users = int(df["user_id"].n_unique())
        n_works = int(df["work_id"].n_unique())
        five_core_trace.append(
            {
                "iter": it,
                "n_users": n_users,
                "n_works": n_works,
                "n_rows": n_after,
                "dropped": n_before - n_after,
            }
        )
        last_drop = (n_before - n_after) / max(n_before, 1)
        print(
            f"  iter {it}: users={n_users:,} works={n_works:,} rows={n_after:,} "
            f"(dropped {n_before - n_after:,})",
            flush=True,
        )
        if last_drop < eps:
            break

    log["five_core_trace"] = five_core_trace
    log["n_users"] = n_users
    log["n_items"] = n_works
    log["n_interactions"] = int(df.height)

    # ---- step 6: dense work_id → idx, 1-indexed ----
    print("STEP dense item_id map", flush=True)
    work_ids = df["work_id"].unique().sort().to_list()
    item_id_to_idx = {int(wid): i + 1 for i, wid in enumerate(work_ids)}
    with open(output / "item_id_map.json", "w") as f:
        json.dump({str(k): v for k, v in item_id_to_idx.items()}, f)
    print(f"  wrote item_id_map.json ({len(item_id_to_idx)} entries)", flush=True)

    # ---- step 7: group + sequence per user ----
    # Compute test_timestamp on the still-flat df (cheap — one quantile pass)
    # before we throw it away. Stream the groupby so we don't hold both the
    # flat df and the grouped seq in memory at once.
    test_timestamp = int(df["ts"].quantile(0.96))
    log["test_timestamp_unix"] = test_timestamp
    print(f"STEP group + sequence per user (test_timestamp={test_timestamp})", flush=True)
    max_seq = args.max_seq_len

    df = df.with_columns(pl.col("work_id").replace_strict(item_id_to_idx).alias("item_id"))
    seq = (
        df.lazy()
        .sort(["user_id", "ts"])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id"), pl.col("ts").alias("timestamp"))
        .with_columns(
            pl.col("item_id").list.tail(max_seq + 1),
            pl.col("timestamp").list.tail(max_seq + 1),
        )
        .collect(engine="streaming")
    )
    # df is no longer needed — free its 12M-row footprint before time-split runs.
    del df

    seq_lens = seq["item_id"].list.len()
    log["median_seq_len"] = int(seq_lens.median())
    log["p99_seq_len"] = int(seq_lens.quantile(0.99))

    # ---- step 8: time-split via shared util ----
    print("STEP time-split (sequential_split_train_val_test)", flush=True)

    # drop_non_train_items=False mirrors yambda.preprocess. Setting it True
    # triggers a polars `is_in(<imploded set>)` *inside* a `list.eval(arg_where)`
    # per user — polars re-evaluates the imploded set per row and the prep
    # process hits the 129 GB cgroup limit. Items only seen in val/test are
    # still present in `item_id_map.json` (built from the full post-5-core
    # frame), so the trainer can score them; they just have unsupervised input
    # embeddings, which is what yambda also accepts.
    seq_lf = seq.rename({"user_id": "uid"}).lazy()
    train_lf, val_lf, test_lf = sequential_split_train_val_test(
        seq_lf,
        test_timestamp=test_timestamp,
        val_size=30 * DAY,
        gap_size=7 * DAY,
        drop_non_train_items=False,
    )

    # ---- step 9: compose train/val/test parquets (yambda layout) ----
    # Collect train/val/test eagerly first — once the time-split chain has run,
    # operating on small DataFrames (one row per user) is cheap. Doing the
    # joins lazily through the streaming engine triggered list-op explosions
    # past the 129 GB cgroup limit.
    print("STEP compose train/val/test parquets", flush=True)
    train_df = (
        train_lf.select("uid", pl.col("item_id").list.tail(max_seq).alias("item_ids"))
        .collect(engine="streaming")
    )
    print(f"  collected train_df (n_users={train_df.height})", flush=True)
    val_df = val_lf.collect(engine="streaming")
    print(f"  collected val_df (n_users={val_df.height})", flush=True)
    test_df = test_lf.collect(engine="streaming")
    print(f"  collected test_df (n_users={test_df.height})", flush=True)

    train_df.select("item_ids").write_parquet(output / "train.parquet", compression="zstd")
    print(f"  wrote train.parquet (n_users={train_df.height})", flush=True)

    val_join = (
        train_df.join(
            val_df.select("uid", pl.col("item_id").alias("targets")),
            on="uid",
            how="inner",
        )
        .select("item_ids", "targets")
    )
    val_join.write_parquet(output / "val.parquet", compression="zstd")
    print(f"  wrote val.parquet (n_users={val_join.height})", flush=True)

    test_join = (
        train_df.join(
            val_df.select("uid", pl.col("item_id").alias("val_items")),
            on="uid",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("val_items").is_null())
            .then(pl.col("item_ids"))
            .otherwise(pl.col("item_ids").list.concat(pl.col("val_items")).list.tail(max_seq))
            .alias("item_ids")
        )
        .join(
            test_df.select("uid", pl.col("item_id").alias("targets")),
            on="uid",
            how="inner",
        )
        .select("item_ids", "targets")
    )
    test_join.write_parquet(output / "test.parquet", compression="zstd")
    print(f"  wrote test.parquet (n_users={test_join.height})", flush=True)

    log["train_rows"] = int(train_df.height)
    log["val_rows"] = int(val_join.height)
    log["test_rows"] = int(test_join.height)
    log["wall_clock_sec"] = round(time.monotonic() - overall_t0, 1)

    with open(output / "prep_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(
        f"ALL DONE prep in {log['wall_clock_sec']:.0f}s — "
        f"users={log['n_users']:,} items={log['n_items']:,} "
        f"interactions={log['n_interactions']:,} "
        f"train/val/test={log['train_rows']:,}/{log['val_rows']:,}/{log['test_rows']:,}",
        flush=True,
    )
    return 0


# ----- attrs -----------------------------------------------------------------
#
# Per-work narrow + wide attribute tensors for the filter bench, from `prep`
# outputs + the catalog parquets. Input/output file list, tensor shapes, and the
# reverse-clause convention: docs/system/datasets.md § goodreads.

# 10 fixed buckets in goodreads_book_genres_initial. Order is the dense id.
GENRE_KEYS = [
    "children",
    "comics, graphic",
    "fantasy, paranormal",
    "fiction",
    "history, historical fiction, biography",
    "mystery, thriller, crime",
    "non-fiction",
    "poetry",
    "romance",
    "young-adult",
]

# 5 buckets for `format`. Index = dense id.
FORMAT_BUCKETS = ["paperback", "hardcover", "ebook", "audio", "other"]

# Wide-shelf blocklist: shelves that say "I want to read this", "I own this",
# or "this is a book", carrying no topical signal.
SHELF_BLOCKLIST = frozenset({
    "to-read", "currently-reading", "owned", "owned-books", "books-i-own",
    "default", "favorites", "favourites", "kindle", "ebook", "audiobook",
    "library", "library-book", "wishlist", "want-to-read", "dnf",
    "did-not-finish", "unread", "read", "my-books", "my-library",
    "all-books", "books", "fiction", "non-fiction",
})

# Regex drops applied AFTER the lowercase-name lookup against SHELF_BLOCKLIST.
SHELF_RE_DROPS = (
    re.compile(r"^read-(in-)?\d{4}$"),
    re.compile(r"^\d-?stars?$"),
    re.compile(r"^[a-z]{2}-\d{4}$"),
    re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$"),
)

C_NARROW = 5  # genre, lang, format, year, author
A_MAX_NARROW = 4  # widest = genre top-4
WIDE_BAG_SIZE = 32


def _format_to_bucket(s: str | None) -> int:
    """Map a raw `format` string to a bucket id, or -1 if unknown/empty."""
    if not s:
        return -1
    n = s.lower().strip()
    if not n:
        return -1
    if "audio" in n or "cassette" in n or "cd-audio" in n or n in {"cd"}:
        return FORMAT_BUCKETS.index("audio")
    if (
        "ebook" in n
        or "e-book" in n
        or "kindle" in n
        or "epub" in n
        or "digital" in n
        or "nook" in n
    ):
        return FORMAT_BUCKETS.index("ebook")
    if "hardcover" in n or "hardback" in n or "library binding" in n:
        return FORMAT_BUCKETS.index("hardcover")
    if (
        "paperback" in n
        or "softcover" in n
        or "mass market" in n
        or n in {"paper", "trade pb"}
    ):
        return FORMAT_BUCKETS.index("paperback")
    return FORMAT_BUCKETS.index("other")


def _year_to_bucket(y: int | None) -> int:
    """Map a publication_year int to one of {0,1,2,3} or -1 if missing/oor.

    Buckets: <1990, 1990-2000, 2001-2010, 2011-2017+. Years before 1500 or
    after 2025 are treated as parse-junk → -1 (Goodreads has a long tail
    of obviously-wrong entries — placeholder dates, future-dated reissues).
    """
    if y is None:
        return -1
    if y < 1500 or y > 2025:
        return -1
    if y < 1990:
        return 0
    if y <= 2000:
        return 1
    if y <= 2010:
        return 2
    return 3  # 2011+ folds 2018-2025 into the 2011-2017 bucket


def _shelf_keep(name: str | None) -> bool:
    """Apply blocklist + regex drops; return True iff this shelf survives."""
    if not name:
        return False
    n = name.lower().strip()
    if not n or n in SHELF_BLOCKLIST:
        return False
    for r in SHELF_RE_DROPS:
        if r.match(n):
            return False
    return True


def cmd_attrs(args) -> int:
    """Build per-work narrow + wide attribute tensors + eval_split."""
    import torch

    processed = Path(args.processed_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser()
    if not output.exists():
        print(f"ERROR output dir {output} does not exist (run `prep` first)", flush=True)
        return 1

    books_path = processed / "goodreads_books.parquet"
    genres_path = processed / "goodreads_book_genres_initial.parquet"
    book_to_work_path = output / "book_to_work.parquet"
    item_id_map_path = output / "item_id_map.json"
    test_path = output / "test.parquet"
    for p in (books_path, genres_path, book_to_work_path, item_id_map_path, test_path):
        if not p.exists():
            print(f"ERROR missing {p}", flush=True)
            return 1

    overall_t0 = time.monotonic()
    log: dict = {}

    # ---- catalog map ------------------------------------------------------
    print("STEP load item_id_map + book_to_work", flush=True)
    with open(item_id_map_path) as f:
        item_id_map = {int(k): int(v) for k, v in json.load(f).items()}
    n_items = len(item_id_map)
    log["n_items"] = n_items
    print(f"  catalog has {n_items:,} works", flush=True)

    iim_df = pl.DataFrame(
        {
            "work_id": list(item_id_map.keys()),
            "item_id": list(item_id_map.values()),
        },
        schema={"work_id": pl.Int64, "item_id": pl.Int64},
    )
    book_to_work = (
        pl.read_parquet(book_to_work_path)
        .join(iim_df, on="work_id", how="inner")
        .select("book_id", "item_id")
    )
    print(f"  in-catalog book→work pairs: {book_to_work.height:,}", flush=True)

    # ---- per-book frame: project narrow+wide source columns ----------------
    print("STEP load books and join to catalog", flush=True)
    books = (
        pl.read_parquet(
            books_path,
            columns=[
                "book_id",
                "language_code",
                "format",
                "publication_year",
                "authors",
                "popular_shelves",
            ],
        )
        .with_columns(pl.col("book_id").cast(pl.Int64, strict=False))
        .filter(pl.col("book_id").is_not_null())
        .join(book_to_work, on="book_id", how="inner")
    )
    print(f"  books in-catalog: {books.height:,}", flush=True)

    # ---- C1 lang vocab + per-book c1 ---------------------------------------
    print("STEP C1 lang vocab", flush=True)
    lang_freq = (
        books.filter(pl.col("language_code") != "")
        .group_by("language_code")
        .len()
        .sort("len", descending=True)
        .head(30)
    )
    lang_codes_top30 = lang_freq["language_code"].to_list()
    lang_vocab = {c: i for i, c in enumerate(lang_codes_top30)}
    with open(output / "lang_vocab.json", "w") as f:
        json.dump(lang_vocab, f, indent=2)
    log["lang_vocab_size"] = len(lang_vocab)
    print(f"  top-30 lang vocab[:5] = {lang_codes_top30[:5]}", flush=True)

    # ---- C2 format vocab + per-book c2 -------------------------------------
    print("STEP C2 format buckets", flush=True)
    fmt_unique = books.select(pl.col("format")).unique()["format"].to_list()
    fmt_lookup = {f: _format_to_bucket(f) for f in fmt_unique}
    with open(output / "format_vocab.json", "w") as f:
        json.dump(
            {
                "buckets": FORMAT_BUCKETS,
                "string_to_id": {(k or ""): v for k, v in fmt_lookup.items()},
            },
            f,
            indent=2,
        )
    log["format_vocab_size"] = len(FORMAT_BUCKETS)

    # ---- C4 author dense remap (over in-catalog editions only) -------------
    print("STEP C4 author vocab (top-2 per book)", flush=True)
    # explode author lists, dense-remap by global frequency
    ab_pairs = (
        books.select(
            pl.col("book_id"),
            pl.col("item_id"),
            pl.col("authors").list.head(2).alias("authors_top2"),
        )
        .explode("authors_top2")
        .filter(pl.col("authors_top2").is_not_null())
        .with_columns(
            pl.col("authors_top2").struct.field("author_id").alias("author_id_str")
        )
        .filter(pl.col("author_id_str").is_not_null() & (pl.col("author_id_str") != ""))
    )
    # global author frequency (only counting in-catalog editions)
    auth_freq = (
        ab_pairs.group_by("author_id_str")
        .agg(pl.len().alias("freq"))
        .sort("freq", descending=True)
    )
    author_ids_str = auth_freq["author_id_str"].to_list()
    # 0-indexed dense ids; -1 reserved as padding sentinel.
    author_vocab = {a: i for i, a in enumerate(author_ids_str)}
    with open(output / "author_vocab.json", "w") as f:
        json.dump({"size": len(author_vocab), "ids": author_ids_str}, f)
    log["author_vocab_size"] = len(author_vocab)
    print(f"  author vocab size = {len(author_vocab):,}", flush=True)

    # ---- per-book attribute frame ------------------------------------------
    print("STEP per-book narrow attrs", flush=True)
    # Pre-build replace-tables as small DataFrames for join (faster than
    # replace_strict on 830k+ keys).
    auth_remap = pl.DataFrame(
        {"author_id_str": author_ids_str, "author_id": list(range(len(author_ids_str)))},
        schema={"author_id_str": pl.Utf8, "author_id": pl.Int64},
    )
    fmt_remap = pl.DataFrame(
        {
            "format": list(fmt_lookup.keys()),
            "format_id": [fmt_lookup[k] for k in fmt_lookup.keys()],
        },
        schema={"format": pl.Utf8, "format_id": pl.Int64},
    )
    lang_remap = pl.DataFrame(
        {
            "language_code": lang_codes_top30,
            "lang_id": list(range(len(lang_codes_top30))),
        },
        schema={"language_code": pl.Utf8, "lang_id": pl.Int64},
    )

    # Build c1, c2, c3 per book (single value each).
    pb = (
        books.select(
            "book_id",
            "item_id",
            "language_code",
            "format",
            pl.col("publication_year").cast(pl.Int64, strict=False).alias("year_int"),
        )
        .join(lang_remap, on="language_code", how="left")
        .join(fmt_remap, on="format", how="left")
        .with_columns(
            pl.col("lang_id").fill_null(-1).cast(pl.Int64),
            pl.col("format_id").fill_null(-1).cast(pl.Int64),
        )
    )
    pb = pb.with_columns(
        pl.col("year_int")
        .map_elements(_year_to_bucket, return_dtype=pl.Int64)
        .alias("year_id")
    ).select("book_id", "item_id", "lang_id", "format_id", "year_id")

    # Top-2 author ids per book.
    pb_authors = (
        ab_pairs.join(auth_remap, on="author_id_str", how="inner")
        .group_by("book_id", maintain_order=True)
        .agg(pl.col("author_id").head(2).alias("author_ids"))
    )
    pb = pb.join(pb_authors, on="book_id", how="left")
    pb = pb.with_columns(
        pl.col("author_ids").fill_null([]).alias("author_ids"),
    )

    # ---- per-book genres top-4 (separate parquet) --------------------------
    print("STEP per-book genres top-4", flush=True)
    genres = (
        pl.read_parquet(genres_path)
        .with_columns(pl.col("book_id").cast(pl.Int64, strict=False))
        .filter(pl.col("book_id").is_not_null())
        .join(book_to_work, on="book_id", how="inner")
    )
    # `genres` column is a struct with the 10 keys above (counts may be null).
    # Unnest → long-form (book_id, item_id, genre_name, count) → top-4.
    genres = genres.unnest("genres")
    genre_value_cols = [c for c in genres.columns if c not in ("book_id", "item_id")]
    glong = genres.unpivot(
        index=["book_id", "item_id"],
        on=genre_value_cols,
        variable_name="genre_name",
        value_name="count",
    ).filter(pl.col("count").is_not_null() & (pl.col("count") > 0))
    genre_remap = pl.DataFrame(
        {"genre_name": GENRE_KEYS, "genre_id": list(range(len(GENRE_KEYS)))},
        schema={"genre_name": pl.Utf8, "genre_id": pl.Int64},
    )
    glong = glong.join(genre_remap, on="genre_name", how="inner")
    # per-book top-4 by count
    glong = glong.sort(["book_id", "count"], descending=[False, True])
    pb_genres = glong.group_by("book_id", maintain_order=True).agg(
        pl.col("genre_id").head(4).alias("genre_ids"),
        pl.col("count").head(4).alias("genre_counts"),
    )
    pb = pb.join(pb_genres, on="book_id", how="left").with_columns(
        pl.col("genre_ids").fill_null([]).alias("genre_ids"),
        pl.col("genre_counts").fill_null([]).alias("genre_counts"),
    )

    # ---- per-work narrow aggregation --------------------------------------
    print("STEP per-work narrow aggregation", flush=True)
    # Group editions by item_id; aggregate per-clause.
    # C0 genre: union top-4 across editions, sum counts per genre, take top-4 by total.
    # C1 lang / C2 format / C3 year: mode (most-common non-`-1` value), break ties by first.
    # C4 author: union top-2 across editions, take 2 by in-catalog frequency rank
    #   (already encoded — lower id = more frequent).

    # genre roll-up
    g_explode = pb.select("item_id", "genre_ids", "genre_counts").explode(
        ["genre_ids", "genre_counts"]
    ).filter(pl.col("genre_ids").is_not_null())
    g_per_work = (
        g_explode.group_by(["item_id", "genre_ids"])
        .agg(pl.col("genre_counts").sum().alias("total"))
        .sort(["item_id", "total"], descending=[False, True])
        .group_by("item_id", maintain_order=True)
        .agg(pl.col("genre_ids").head(4).alias("c0_genre"))
    )

    # author roll-up: union, dedup, sort by author_id asc (smaller id = more-frequent),
    # take 2.
    a_explode = pb.select("item_id", "author_ids").explode("author_ids").filter(
        pl.col("author_ids").is_not_null()
    )
    a_per_work = (
        a_explode.unique(subset=["item_id", "author_ids"])
        .sort(["item_id", "author_ids"])
        .group_by("item_id", maintain_order=True)
        .agg(pl.col("author_ids").head(2).alias("c4_author"))
    )

    # lang/format/year mode: count occurrences of each non-`-1` value per work,
    # take the value with the highest count (ties: smallest id).
    def _mode_perwork(df: pl.DataFrame, value_col: str, out_name: str) -> pl.DataFrame:
        nz = df.filter(pl.col(value_col) != -1)
        if nz.height == 0:
            return pl.DataFrame(
                {"item_id": [], out_name: []},
                schema={"item_id": pl.Int64, out_name: pl.Int64},
            )
        return (
            nz.group_by(["item_id", value_col])
            .agg(pl.len().alias("ct"))
            .sort(["item_id", "ct", value_col], descending=[False, True, False])
            .group_by("item_id", maintain_order=True)
            .agg(pl.col(value_col).first().alias(out_name))
        )

    c1 = _mode_perwork(pb.select("item_id", "lang_id"), "lang_id", "c1_lang")
    c2 = _mode_perwork(pb.select("item_id", "format_id"), "format_id", "c2_format")
    c3 = _mode_perwork(pb.select("item_id", "year_id"), "year_id", "c3_year")

    # ---- assemble per-work narrow tensor ----------------------------------
    print("STEP assemble item_attrs_narrow", flush=True)
    # Start from a frame of every catalog item_id (so works with no editions
    # in-catalog still get a row of -1s — should be 0 in practice).
    all_items = pl.DataFrame(
        {"item_id": list(range(1, n_items + 1))}, schema={"item_id": pl.Int64}
    )
    narrow = (
        all_items.join(g_per_work, on="item_id", how="left")
        .join(c1, on="item_id", how="left")
        .join(c2, on="item_id", how="left")
        .join(c3, on="item_id", how="left")
        .join(a_per_work, on="item_id", how="left")
    )

    # [N, 5, 4] 0-indexed dense (row i = item_id i+1); no padding row.
    narrow_t = torch.full((n_items, C_NARROW, A_MAX_NARROW), -1, dtype=torch.long)
    item_id_arr = narrow["item_id"].to_numpy()
    g0 = narrow["c0_genre"].to_list()
    c1l = narrow["c1_lang"].to_list()
    c2f = narrow["c2_format"].to_list()
    c3y = narrow["c3_year"].to_list()
    c4a = narrow["c4_author"].to_list()
    cov = [0, 0, 0, 0, 0]
    for row_i, item_id in enumerate(item_id_arr.tolist()):
        pos = item_id - 1
        # C0
        gv = g0[row_i] or []
        if gv:
            cov[0] += 1
            for j, v in enumerate(gv[:A_MAX_NARROW]):
                narrow_t[pos, 0, j] = int(v)
        # C1
        v = c1l[row_i]
        if v is not None:
            cov[1] += 1
            narrow_t[pos, 1, 0] = int(v)
        # C2
        v = c2f[row_i]
        if v is not None:
            cov[2] += 1
            narrow_t[pos, 2, 0] = int(v)
        # C3
        v = c3y[row_i]
        if v is not None:
            cov[3] += 1
            narrow_t[pos, 3, 0] = int(v)
        # C4
        av = c4a[row_i] or []
        if av:
            cov[4] += 1
            for j, v in enumerate(av[:A_MAX_NARROW]):
                narrow_t[pos, 4, j] = int(v)

    coverage = {f"c{i}": round(cov[i] / max(n_items, 1), 4) for i in range(5)}
    log["narrow_coverage"] = coverage
    print(f"  per-clause coverage = {coverage}", flush=True)

    torch.save(narrow_t, output / "item_attrs_narrow.pt")
    clause_is_reverse_narrow = torch.tensor(
        [False, True, False, False, False], dtype=torch.bool
    )
    torch.save(clause_is_reverse_narrow, output / "clause_is_reverse_narrow.pt")
    print(
        f"  wrote item_attrs_narrow.pt {tuple(narrow_t.shape)} + clause_is_reverse_narrow.pt",
        flush=True,
    )

    # ---- wide shelves ------------------------------------------------------
    print("STEP wide shelves (per-book filter + per-work top-32)", flush=True)
    # Explode popular_shelves struct per book; apply blocklist + regex drops;
    # parse count → int. Then aggregate to per-work sum, take top-32 by sum.
    shelves_long = (
        books.select("item_id", "popular_shelves")
        .explode("popular_shelves")
        .filter(pl.col("popular_shelves").is_not_null())
        .with_columns(
            pl.col("popular_shelves").struct.field("name").alias("name"),
            pl.col("popular_shelves").struct.field("count").alias("count_str"),
        )
        .with_columns(
            pl.col("count_str").cast(pl.Int64, strict=False).fill_null(0).alias("count"),
        )
        .select("item_id", "name", "count")
    )
    # Apply Python predicate via map_elements — the regex+blocklist is awkward
    # to express purely in polars.  ~5M rows after explode; ~30 s on the user's
    # box, fine for one-shot.
    shelves_long = shelves_long.with_columns(
        pl.col("name")
        .map_elements(_shelf_keep, return_dtype=pl.Boolean)
        .alias("keep")
    ).filter(pl.col("keep")).drop("keep")
    # lower-case the surviving names (blocklist matched lowercase but the raw
    # name might mix case; canonicalize for the per-work group-by).
    shelves_long = shelves_long.with_columns(pl.col("name").str.to_lowercase().alias("name"))

    # per-work total per shelf-name → top-32 by count
    work_shelf = (
        shelves_long.group_by(["item_id", "name"])
        .agg(pl.col("count").sum().alias("count"))
        .sort(["item_id", "count"], descending=[False, True])
    )
    # Build wide-shelf vocab from the union of all per-work top-32 names
    # (sorted by global aggregated count desc; 0-indexed dense ids).
    work_shelf_top = (
        work_shelf.group_by("item_id", maintain_order=True)
        .agg(
            pl.col("name").head(WIDE_BAG_SIZE).alias("names"),
            pl.col("count").head(WIDE_BAG_SIZE).alias("counts"),
        )
    )
    used_names_freq = (
        work_shelf_top.select(pl.col("names").alias("name"), pl.col("counts").alias("count"))
        .explode(["name", "count"])
        .group_by("name")
        .agg(pl.col("count").sum().alias("global_count"))
        .sort("global_count", descending=True)
    )
    wide_names = used_names_freq["name"].to_list()
    wide_counts = used_names_freq["global_count"].to_list()
    wide_vocab = {n: i for i, n in enumerate(wide_names)}
    with open(output / "wide_shelf_vocab.json", "w") as f:
        json.dump({"size": len(wide_vocab), "names": wide_names}, f)
    torch.save(
        torch.tensor(wide_counts, dtype=torch.long),
        output / "wide_shelf_global_freq.pt",
    )
    log["wide_vocab_size"] = len(wide_vocab)
    print(f"  wide vocab size = {len(wide_vocab):,}", flush=True)

    # Materialize per-work [item_id, names_top32 (ids), counts_top32].
    print("STEP assemble item_attrs_wide", flush=True)
    wide_remap = pl.DataFrame(
        {"name": wide_names, "shelf_id": list(range(len(wide_names)))},
        schema={"name": pl.Utf8, "shelf_id": pl.Int64},
    )
    work_shelf_ids = (
        work_shelf.join(wide_remap, on="name", how="inner")
        .group_by("item_id", maintain_order=True)
        .agg(pl.col("shelf_id").head(WIDE_BAG_SIZE).alias("shelf_ids"))
    )
    # [N, 1, BAG] 0-indexed dense (row i = item_id i+1); no padding row.
    wide_t = torch.full((n_items, 1, WIDE_BAG_SIZE), -1, dtype=torch.long)
    wide_cov = 0
    wide_bag_size_hist = [0] * (WIDE_BAG_SIZE + 1)
    for row in work_shelf_ids.iter_rows(named=True):
        item_id = int(row["item_id"])
        ids = row["shelf_ids"] or []
        bag = ids[:WIDE_BAG_SIZE]
        if bag:
            wide_cov += 1
        wide_bag_size_hist[min(len(bag), WIDE_BAG_SIZE)] += 1
        for j, v in enumerate(bag):
            wide_t[item_id - 1, 0, j] = int(v)
    log["wide_coverage"] = round(wide_cov / max(n_items, 1), 4)
    log["wide_bag_size_hist"] = wide_bag_size_hist
    print(
        f"  wide coverage = {log['wide_coverage']}; "
        f"avg bag size = "
        f"{sum(i * c for i, c in enumerate(wide_bag_size_hist)) / max(n_items, 1):.1f}",
        flush=True,
    )

    torch.save(wide_t, output / "item_attrs_wide.pt")
    print(f"  wrote item_attrs_wide.pt {tuple(wide_t.shape)}", flush=True)

    # ---- eval_split.parquet -----------------------------------------------
    print("STEP build eval_split.parquet (rare-biased wide sampling)", flush=True)
    test_df = pl.read_parquet(test_path)
    n_users = test_df.height
    print(f"  test users: {n_users:,}", flush=True)

    targets_lists = test_df["targets"].to_list()
    target_first = [t[0] if t else 0 for t in targets_lists]

    qa_narrow = synthesize_qa_narrow(target_first, narrow_t, C_NARROW)
    qa_wide_1, qa_wide_2, n_drop_1, n_drop_2 = sample_rare_biased_wide(
        target_first, wide_t, wide_counts, seed=args.seed
    )

    log["eval_n_users"] = n_users
    log["eval_drop_1shelf"] = int(n_drop_1)
    log["eval_drop_2shelf"] = int(n_drop_2)
    print(
        f"  1-shelf drops = {n_drop_1:,} ({100 * n_drop_1 / max(n_users, 1):.1f}%); "
        f"2-shelf drops = {n_drop_2:,} ({100 * n_drop_2 / max(n_users, 1):.1f}%)",
        flush=True,
    )

    eval_split = pl.DataFrame(
        {
            "target_id": target_first,
            "query_attrs_narrow": qa_narrow.tolist(),
            "query_attrs_wide_1shelf": qa_wide_1.tolist(),
            "query_attrs_wide_2shelf": qa_wide_2.tolist(),
        },
        schema={
            "target_id": pl.Int64,
            "query_attrs_narrow": pl.List(pl.Int64),
            "query_attrs_wide_1shelf": pl.Int64,
            "query_attrs_wide_2shelf": pl.List(pl.Int64),
        },
    )
    eval_split.write_parquet(output / "eval_split.parquet", compression="zstd")
    print(f"  wrote eval_split.parquet ({eval_split.height} rows)", flush=True)

    # ---- log file ---------------------------------------------------------
    log["wall_clock_sec"] = round(time.monotonic() - overall_t0, 1)
    prep_log_path = output / "prep_log.json"
    if prep_log_path.exists():
        with open(prep_log_path) as f:
            existing = json.load(f)
        existing["attrs"] = log
        with open(prep_log_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(prep_log_path, "w") as f:
            json.dump({"attrs": log}, f, indent=2)

    print(
        f"ALL DONE attrs in {log['wall_clock_sec']:.0f}s — narrow_cov={coverage} "
        f"wide_cov={log['wide_coverage']} "
        f"vocab(lang/fmt/auth/wide)="
        f"{log['lang_vocab_size']}/{log['format_vocab_size']}/"
        f"{log['author_vocab_size']}/{log['wide_vocab_size']}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror the UCSD Goodreads dataset (download → parquet → trainer prep).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_dl = sub.add_parser("download", help="fetch raw files into raw/")
    sp_dl.add_argument("files", nargs="*", help="optional subset of filenames")
    sp_dl.add_argument("--force", action="store_true", help="re-download even if present")
    sp_dl.set_defaults(func=cmd_download)

    sp_cv = sub.add_parser("convert", help="raw/ → processed/ parquet")
    sp_cv.set_defaults(func=cmd_convert)

    sp_al = sub.add_parser("all", help="download → convert")
    sp_al.set_defaults(func=cmd_all)

    sp_pp = sub.add_parser(
        "prep",
        help="processed parquets → yambda-format trainer inputs (train/val/test/item_id_map)",
    )
    sp_pp.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="path to the directory `convert` wrote into (must contain "
        "goodreads_interactions_dedup.parquet and goodreads_books.parquet)",
    )
    sp_pp.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="where to write train.parquet/val.parquet/test.parquet/item_id_map.json",
    )
    sp_pp.add_argument("--n-core", type=int, default=5)
    sp_pp.add_argument("--max-seq-len", type=int, default=200)
    sp_pp.set_defaults(func=cmd_prep)

    sp_at = sub.add_parser(
        "attrs",
        help="build narrow + wide attribute tensors + eval_split for the filter bench",
    )
    sp_at.add_argument(
        "--processed-dir",
        type=str,
        required=True,
        help="path to the directory `convert` wrote into (must contain "
        "goodreads_books.parquet and goodreads_book_genres_initial.parquet)",
    )
    sp_at.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="same dir prep wrote into; reuses book_to_work.parquet, "
        "item_id_map.json, test.parquet",
    )
    sp_at.add_argument(
        "--seed",
        type=int,
        default=0,
        help="rng seed for wide-eval shelf sampling",
    )
    sp_at.set_defaults(func=cmd_attrs)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
