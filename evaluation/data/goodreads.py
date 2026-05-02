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
  all        download → convert.

Examples::

    uv run goodreads.py all
    uv run goodreads.py download --force goodreads_books.json.gz
    python -m data.goodreads prep \\
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
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

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
# Turns the staged parquets (interactions_dedup + books) into the yambda-shaped
# trainer inputs:
#
#     <output-dir>/train.parquet         (item_ids: list[int64])
#     <output-dir>/val.parquet           (item_ids, targets)
#     <output-dir>/test.parquet          (item_ids, targets)
#     <output-dir>/item_id_map.json      {work_id: 1-indexed dense int}
#     <output-dir>/book_to_work.parquet  (book_id, work_id) — reused by filter eval
#     <output-dir>/prep_log.json         stats
#
# Filters Listen+-equivalent (`is_read=true`), parses `date_added`, collapses to
# work_id catalog, runs an iterative n-core, then time-splits via
# `data.timesplit.sequential_split_train_val_test`.


def _import_timesplit():
    """Resolve the package-relative timesplit import in both module and script modes."""
    try:
        from .timesplit import sequential_split_train_val_test as f
    except ImportError:
        # uv-run-script mode: `data` isn't on sys.path as a package.
        # Add the parent of `data/` so `from data.timesplit import …` works.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from data.timesplit import sequential_split_train_val_test as f
    return f


def cmd_prep(args) -> int:
    sequential_split_train_val_test = _import_timesplit()

    DAY = 24 * 60 * 60
    processed = Path(args.processed_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser()
    if args.smoke:
        output = output.parent / (output.name + "-smoke")
    output.mkdir(parents=True, exist_ok=True)

    interactions_path = processed / "goodreads_interactions_dedup.parquet"
    books_path = processed / "goodreads_books.parquet"
    if not interactions_path.exists():
        print(f"ERROR missing {interactions_path}", flush=True)
        return 1
    if not books_path.exists():
        print(f"ERROR missing {books_path}", flush=True)
        return 1

    log: dict = {"output_dir": str(output), "smoke": bool(args.smoke)}
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

    # ---- smoke subsample BEFORE the heavy join + n-core ----
    if args.smoke:
        print("STEP smoke subsample (100k random users)", flush=True)
        u_unique = inter.select("user_id").unique().collect(engine="streaming")
        n_take = min(100_000, u_unique.height)
        u_sample = u_unique.sample(n=n_take, seed=42)
        inter = inter.join(u_sample.lazy(), on="user_id", how="inner")

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
    # past the 129 GB cgroup limit on the smoke dataset.
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
    sp_pp.add_argument(
        "--smoke",
        action="store_true",
        help="subsample to 100k random users; output goes to <output-dir>-smoke/",
    )
    sp_pp.set_defaults(func=cmd_prep)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
