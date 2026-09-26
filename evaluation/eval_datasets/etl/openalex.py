#!/usr/bin/env -S uv run --script
"""End-to-end ETL for the OpenAlex filter-bench dataset (roadmap E3, the OpenAlex fallback of
the Semantic Scholar SPECTER2 slice).

Source: the OpenAlex quarterly snapshot on public S3, anonymous, CC0 —
``s3://openalex/data/parquet/works/updated_date=*/part_*.parquet``, listed by
``data/parquet/manifest.json`` (written last; present = release complete). The parquet copy is
read rather than the ``data/jsonl/`` one because it allows column projection: the columns below
are a fraction of the bytes, and no record is JSON-parsed except the abstracts that are kept.
Measured sizes and rates are in docs/system/datasets.md § openalex.

Subcommands::

    plan            Dry run: manifest totals, projected bytes from every file footer, filter
                    rates on a seeded sample of row groups → --sample-rate, disk and wall time.
    download        Pin the release: fetch manifest.json into _raw/openalex/.
    convert         **Streaming**: every works file of the pinned manifest, projected columns,
                    row group by row group → filter (year, English, type, flags, hash sample,
                    abstract present) → _raw/openalex/staging/<date>_<part>.parquet.
                    Resumable per file; the snapshot is never landed.
    prep            staging/ → item_id_map.json, papers.parquet (the N smallest work-id hashes),
                    heldout.parquet + queries.parquet + qrels.parquet (held-out papers, their
                    in-catalog references as relevance).
    encode_text     nomic-embed-text-v1.5 at its native 768 dims, "search_document: " prefix →
                    content_d768/text_emb_shard_NNN.pt + shard_index.json. Resumable per shard.
    encode_queries  Same encoder, "search_query: " prefix → content_d768/query_emb.pt.
    attrs           item_attrs_narrow.pt [N, 5, 4], clause_is_reverse_narrow.pt, vocabs,
                    eval_split.parquet.
    all             download → convert → prep → encode_text → encode_queries → attrs.

Layout (under hub.data_root(): $RETRIEVE_DATA_ROOT, default evaluation/data)::

    data/_raw/openalex/manifest.json        the pinned release
    data/_raw/openalex/staging/             filtered, hash-sampled rows per source file
    data/openalex/                          the bench-side output dir
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from eval_datasets.etl.pubmed import pmid_hash, select_pmids
from eval_datasets.hub import data_root, raw_dir
from eval_datasets.layout import atomic_write

MANIFEST_URL = "https://openalex.s3.amazonaws.com/data/parquet/manifest.json"
S3_REGION = "us-east-1"

ROOT = raw_dir("openalex")
MANIFEST_PATH = ROOT / "manifest.json"
STAGING = ROOT / "staging"

# The only columns read from the snapshot; nested leaves are projected, not their structs.
COLUMNS = [
    "id",
    "publication_year",
    "language",
    "type",
    "is_paratext",
    "is_retracted",
    "is_xpac",
    "primary_topic.field.id",
    "primary_topic.subfield.id",
    "primary_location.source.id",
    "open_access.is_oa",
    "title",
    "abstract_inverted_index",
    "referenced_works",
]

# Scholarly text types. `dataset`, `other`, `paratext`, ... carry boilerplate abstracts (e.g.
# every CCDC crystal-structure entry shares one), i.e. exact-duplicate vectors.
WORK_TYPES = ["article", "preprint", "review", "conference-paper", "book-chapter", "dissertation"]
MIN_YEAR = 2000

# Narrow attribute layout — same [N, 5, 4] shape as goodreads / arxiv / pubmed.
#
#   C0 field        primary_topic.field (26)                      "same field"
#   C1 earlier_era  the eras strictly after the item's own         "published before the query"
#   C2 subfield     primary_topic.subfield (~250)
#   C3 source       top-`--source-vocab` primary_location.source, REVERSE
#   C4 is_oa        open_access.is_oa (null → 0)
C_NARROW = 5
A_MAX_NARROW = 4
CLAUSE_NAMES = ["field", "earlier_era", "subfield", "source", "is_oa"]
CLAUSE_IS_REVERSE = [False, False, False, True, False]

# Clauses are equality-only, so "item year < query year" is encoded over eras: an item of era b
# lists b+1 .. last in its C1 slots and a query sends its own era, which passes exactly the
# items of strictly earlier eras. That needs len(ERA_NAMES) - 1 == A_MAX_NARROW slots.
ERA_EDGES = [2010, 2015, 2019, 2022]
ERA_NAMES = ["2000-2009", "2010-2014", "2015-2018", "2019-2021", ">=2022"]

EMB_DIM_NATIVE = 768
ENCODER = "nomic-ai/nomic-embed-text-v1.5"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
TEXT_TEMPLATE = "{prefix}{title}. {abstract}"

_STAGED_SCHEMA = {
    "work_id": pl.Int64,
    "year": pl.Int16,
    "field": pl.Int16,
    "subfield": pl.Int16,
    "source": pl.Int64,
    "is_oa": pl.Boolean,
    "type": pl.Utf8,
    "title": pl.Utf8,
    "abstract": pl.Utf8,
    "refs": pl.List(pl.Int64),
}


# ----- record parsing ------------------------------------------------------------------------


def abstract_text(inverted: str | None) -> str | None:
    """OpenAlex ships abstracts as ``{word: [positions]}`` (a JSON string in the parquet copy);
    the text is the words in position order. Missing (``None``, ``""``, ``{}``) is ``""``;
    ``None`` means the string is cut off — the parquet copy caps strings just under 32,767
    chars, which truncates about one index in 7 M mid-JSON."""
    if not inverted:
        return ""
    try:
        index = json.loads(inverted)
    except json.JSONDecodeError:
        return None
    if not index:
        return ""
    words = sorted((p, w) for w, positions in index.items() for p in positions)
    return " ".join(w for _, w in words)


def _trailing_int(col: pl.Expr) -> pl.Expr:
    """``https://openalex.org/W123`` / ``…/fields/17`` / ``…/S42`` → the trailing integer."""
    return col.str.extract(r"(\d+)$").cast(pl.Int64)


def era_of(years: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(ERA_EDGES), years, side="right").astype(np.int64)


def stage_table(
    tbl: pa.Table, *, max_year: int, sample_rate: float, seed: int
) -> tuple[pl.DataFrame, dict]:
    """One row group of projected ``COLUMNS`` → the staged rows (``_STAGED_SCHEMA``) and the row
    count after each filter, in order: year in [MIN_YEAR, max_year], ``language == "en"``,
    ``type`` in WORK_TYPES, not paratext / retracted / xpac, the work-id hash sample, abstract
    present. The hash sample runs before the abstract step so only kept rows are JSON-parsed."""
    pt, loc = pl.col("primary_topic"), pl.col("primary_location")
    df = pl.from_arrow(tbl).select(
        _trailing_int(pl.col("id")).alias("work_id"),
        pl.col("publication_year").alias("year"),
        "language",
        "type",
        "is_paratext",
        "is_retracted",
        "is_xpac",
        _trailing_int(pt.struct.field("field").struct.field("id")).alias("field"),
        _trailing_int(pt.struct.field("subfield").struct.field("id")).alias("subfield"),
        _trailing_int(loc.struct.field("source").struct.field("id")).alias("source"),
        pl.col("open_access").struct.field("is_oa").fill_null(False).alias("is_oa"),
        pl.col("title").fill_null("").str.strip_chars(),
        "abstract_inverted_index",
        pl.col("referenced_works")
        .list.eval(_trailing_int(pl.element()))
        .fill_null(pl.lit([], dtype=pl.List(pl.Int64)))
        .alias("refs"),
    )
    stats = {"rows": df.height}
    flags = pl.col("is_paratext") | pl.col("is_retracted") | pl.col("is_xpac")
    for name, keep in (
        ("year", pl.col("year").is_between(MIN_YEAR, max_year)),
        ("language", pl.col("language") == "en"),
        ("type", pl.col("type").is_in(WORK_TYPES)),
        ("flags", ~flags.fill_null(False)),
    ):
        df = df.filter(keep.fill_null(False))
        stats[f"after_{name}"] = df.height
    if sample_rate < 1.0:
        threshold = np.uint64(int(sample_rate * 2.0**64))
        df = df.filter(pl.Series(pmid_hash(df["work_id"].to_numpy(), seed) < threshold))
    stats["after_sample"] = df.height
    abstracts = [abstract_text(s) for s in df["abstract_inverted_index"].to_list()]
    stats["abstract_truncated"] = abstracts.count(None)
    df = df.with_columns(pl.Series("abstract", abstracts, dtype=pl.Utf8)).filter(
        pl.col("abstract").fill_null("") != ""
    )
    stats["after_abstract"] = df.height
    return df.select(list(_STAGED_SCHEMA)).cast(_STAGED_SCHEMA), stats


# ----- manifest ------------------------------------------------------------------------------


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"ERROR missing {MANIFEST_PATH} (run `download` first)")
    return json.loads(MANIFEST_PATH.read_text())


def works_files(manifest: dict) -> list[dict]:
    """The ``works`` entries of a manifest as ``{key, name, records, bytes}``, in (date, part)
    order — the order every later step walks, and so the item-id order."""
    works = next(e for e in manifest["entities"] if e["entity"] == "works")
    out = []
    for f in works["files"]:
        key = f["url"].removeprefix("s3://")
        date, part = key.split("/")[-2:]
        out.append(
            {
                "key": key,
                "name": f"{date.removeprefix('updated_date=')}_{part}",
                "records": int(f["meta"]["record_count"]),
                "bytes": int(f["meta"]["content_length"]),
            }
        )
    return sorted(out, key=lambda f: f["name"])


def _parse_files(spec: str | None, n: int) -> list[int]:
    """``"0-3,7"`` → ``[0, 1, 2, 3, 7]`` (indices into ``works_files``); ``None`` → all."""
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def _s3() -> pafs.S3FileSystem:
    return pafs.S3FileSystem(
        anonymous=True,
        region=S3_REGION,
        retry_strategy=pafs.AwsStandardS3RetryStrategy(max_attempts=10),
    )


# Leaf paths in the parquet schema: list elements sit under `referenced_works.list.element`.
_LEAF_PREFIXES = [c for c in COLUMNS if c != "referenced_works"] + ["referenced_works.list"]


def _projected_bytes(md: pq.FileMetaData, row_groups: range | None = None) -> int:
    """Compressed bytes of the ``COLUMNS`` leaves of ``row_groups`` (default: all)."""
    total = 0
    for rg in row_groups if row_groups is not None else range(md.num_row_groups):
        r = md.row_group(rg)
        for c in range(r.num_columns):
            path = r.column(c).path_in_schema
            if any(path == p or path.startswith(p + ".") for p in _LEAF_PREFIXES):
                total += r.column(c).total_compressed_size
    return total


def _check_params(path: Path, params: dict) -> None:
    """A resumable step records its parameters on first run and refuses to resume under
    different ones: the finished pieces on disk would silently mix two configurations."""
    if path.exists():
        prev = json.loads(path.read_text())
        if prev != params:
            raise SystemExit(f"ERROR {path} was written with {prev}, now {params}: clear it first")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2))


def _merge_log(output: Path, key: str, payload: dict) -> None:
    path = output / "prep_log.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[key] = payload
    path.write_text(json.dumps(existing, indent=2))


# ----- plan ----------------------------------------------------------------------------------


def _footer(key: str) -> tuple[int, int]:
    md = pq.ParquetFile(key, filesystem=_s3()).metadata
    return md.num_row_groups, _projected_bytes(md)


def _sample_row_group(key: str, rg: int, max_year: int) -> dict:
    pf = pq.ParquetFile(key, filesystem=_s3())
    t0 = time.monotonic()
    tbl = pf.read_row_group(rg, columns=COLUMNS)
    t_read = time.monotonic() - t0
    nbytes = _projected_bytes(pf.metadata, range(rg, rg + 1))
    t0 = time.monotonic()
    staged, stats = stage_table(tbl, max_year=max_year, sample_rate=1.0, seed=0)
    t_stage = time.monotonic() - t0
    buf = io.BytesIO()
    staged.write_parquet(buf, compression="zstd")
    english = tbl.filter(pc.equal(tbl["language"], "en"))
    return {
        "stats": stats,
        "bytes": nbytes,
        "t_read": t_read,
        "t_stage": t_stage,
        "staged_bytes": buf.tell(),
        "types_en": Counter(english["type"].to_pylist()),
        "languages": Counter(tbl["language"].to_pylist()),
        "years": Counter(staged["year"].to_list()),
        "abstract_chars": int(staged["abstract"].str.len_chars().sum() or 0),
        "refs": int(staged["refs"].list.len().sum() or 0),
    }


def cmd_plan(args) -> int:
    manifest = _load_manifest() if MANIFEST_PATH.exists() else _fetch_json(MANIFEST_URL)
    files = works_files(manifest)
    max_year = int(manifest["date"][:4])
    n_records = sum(f["records"] for f in files)
    n_bytes = sum(f["bytes"] for f in files)
    print(
        f"release {manifest['date']}: {len(files):,} works files, {n_records:,} records, "
        f"{n_bytes / 1e9:.1f} GB",
        flush=True,
    )
    print(f"STEP footers of {len(files):,} files ({args.threads} threads)", flush=True)
    t0 = time.monotonic()
    with ThreadPoolExecutor(args.threads) as ex:
        footers = list(ex.map(_footer, [f["key"] for f in files]))
    projected = sum(b for _, b in footers)
    print(
        f"  projected columns: {projected / 1e9:.1f} GB of {n_bytes / 1e9:.1f} GB "
        f"({100 * projected / n_bytes:.1f} %) in {time.monotonic() - t0:.0f}s",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    weights = np.array([f["records"] for f in files], dtype=np.float64)
    picks = rng.choice(
        len(files), size=args.sample_row_groups, replace=True, p=weights / weights.sum()
    )
    jobs = [(files[i]["key"], int(rng.integers(footers[i][0]))) for i in picks]
    print(f"STEP filter rates on {len(jobs)} sampled row groups", flush=True)
    t0 = time.monotonic()
    with ThreadPoolExecutor(args.threads) as ex:
        samples = list(ex.map(lambda j: _sample_row_group(*j, max_year), jobs))
    t_wall = time.monotonic() - t0

    stats: Counter = Counter()
    types: Counter = Counter()
    langs: Counter = Counter()
    years: Counter = Counter()
    for s in samples:
        stats.update(s["stats"])
        types.update(s["types_en"])
        langs.update(s["languages"])
        years.update(s["years"])
    rows, kept = stats["rows"], stats["after_abstract"]
    frac = kept / rows
    sample_bytes = sum(s["bytes"] for s in samples)
    stage_cpu = sum(s["t_stage"] for s in samples)
    est_eligible = frac * n_records
    sample_rate = min(1.0, args.keep_items * (1 + args.margin) / est_eligible)
    staged_rows = est_eligible * sample_rate
    staged_bpr = sum(s["staged_bytes"] for s in samples) / kept
    # Threads share the GIL with the abstract parsing, so this understates `convert`'s spawned
    # processes several-fold (docs/artifacts/e3-openalex/stream_probe.py measures those).
    mbps = sample_bytes / t_wall / 1e6
    cpu_per_row = stage_cpu / rows  # at sample rate 1; the real run parses fewer abstracts
    report = {
        "release": manifest["date"],
        "works_files": len(files),
        "records": n_records,
        "bytes": n_bytes,
        "projected_bytes": projected,
        "sample": {
            "row_groups": len(jobs),
            "rows": rows,
            "filter_steps": {k: stats[k] for k in stats if k.startswith("after_")},
            "eligible_fraction": round(frac, 4),
            "languages_top10": dict(langs.most_common(10)),
            "types_among_english": dict(types.most_common(15)),
            "eligible_year_histogram": dict(sorted(years.items())),
            "eligible_by_era": {
                name: int(sum(c for y, c in years.items() if era_of(np.array([y]))[0] == e))
                for e, name in enumerate(ERA_NAMES)
            },
            "mean_abstract_chars": round(sum(s["abstract_chars"] for s in samples) / kept, 1),
            "mean_refs": round(sum(s["refs"] for s in samples) / kept, 2),
            "staged_bytes_per_row": round(staged_bpr, 1),
            "threaded_read_mb_per_s_lower_bound": round(mbps, 1),
            "threads": args.threads,
            "stage_cpu_s_per_row": cpu_per_row,
        },
        "budget": {
            "keep_items": args.keep_items,
            "margin": args.margin,
            "est_eligible": int(est_eligible),
            "sample_rate": round(sample_rate, 4),
            "staged_rows": int(staged_rows),
            "staging_gb": round(staged_rows * staged_bpr / 1e9, 1),
            "papers_parquet_gb": round(args.keep_items * staged_bpr / 1e9, 1),
            "fp16_items_gb": round(args.keep_items * EMB_DIM_NATIVE * 2 / 1e9, 1),
            "fp32_items_on_device_gb": round(args.keep_items * EMB_DIM_NATIVE * 4 / 1e9, 1),
            "download_gb": round(projected / 1e9, 1),
            "stream_hours_upper_bound": round(projected / 1e6 / mbps / 3600, 2),
        },
    }
    report["budget"]["peak_disk_gb"] = round(
        report["budget"]["staging_gb"]
        + report["budget"]["papers_parquet_gb"]
        + report["budget"]["fp16_items_gb"],
        1,
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    return 0


# ----- download ------------------------------------------------------------------------------


def cmd_download(args) -> int:
    if MANIFEST_PATH.exists() and not args.force:
        m = _load_manifest()
        print(f"SKIP download: {MANIFEST_PATH} pins release {m['date']} (--force to re-pin)")
        return 0
    m = _fetch_json(MANIFEST_URL)
    files = works_files(m)
    atomic_write(MANIFEST_PATH, lambda fh: fh.write(json.dumps(m).encode()))
    print(
        f"DONE pinned release {m['date']}: {len(files):,} works files, "
        f"{sum(f['records'] for f in files):,} records → {MANIFEST_PATH}",
        flush=True,
    )
    return 0


# ----- convert -------------------------------------------------------------------------------


def stage_file(key: str, out: Path, max_year: int, sample_rate: float, seed: int) -> dict:
    """One snapshot file, row group by row group, → ``out`` (written atomically). Returns the
    summed filter counts plus the projected bytes read."""
    pf = pq.ParquetFile(key, filesystem=_s3())
    parts = []
    stats: Counter = Counter()
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=COLUMNS)
        df, s = stage_table(tbl, max_year=max_year, sample_rate=sample_rate, seed=seed)
        parts.append(df)
        stats.update(s)
        del tbl
    staged = pl.concat(parts) if parts else pl.DataFrame(schema=_STAGED_SCHEMA)
    atomic_write(out, lambda fh: staged.write_parquet(fh, compression="zstd"))
    return dict(stats) | {"bytes_read": _projected_bytes(pf.metadata)}


def cmd_convert(args) -> int:
    manifest = _load_manifest()
    files = works_files(manifest)
    params = {
        "release": manifest["date"],
        "min_year": MIN_YEAR,
        "max_year": int(manifest["date"][:4]),
        "work_types": WORK_TYPES,
        "sample_rate": args.sample_rate,
        "seed": args.seed,
    }
    _check_params(STAGING / "params.json", params)
    selected = [files[i] for i in _parse_files(args.files, len(files))]
    todo = [f for f in selected if not (STAGING / f"{f['name']}.parquet").exists()]
    print(
        f"START convert release {params['release']}: {len(selected):,} files selected, "
        f"{len(selected) - len(todo):,} already staged, {len(todo):,} to stream "
        f"({sum(f['bytes'] for f in todo) / 1e9:.1f} GB full-width) with {args.workers} workers",
        flush=True,
    )
    t0 = time.monotonic()
    done_bytes = 0
    failed: list[str] = []
    log_path = STAGING / "convert_log.jsonl"
    ctx = multiprocessing.get_context("spawn")  # a forked pyarrow S3 client can deadlock
    with ProcessPoolExecutor(args.workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(
                stage_file,
                f["key"],
                STAGING / f"{f['name']}.parquet",
                params["max_year"],
                args.sample_rate,
                args.seed,
            ): f
            for f in todo
        }
        for n, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            try:
                stats = fut.result()
            except (OSError, pa.ArrowException) as e:
                failed.append(f["name"])
                print(f"  FAIL {f['name']}: {type(e).__name__}: {e}", flush=True)
                continue
            done_bytes += stats["bytes_read"]
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"name": f["name"]} | stats) + "\n")
            dt = time.monotonic() - t0
            print(
                f"  [{n}/{len(todo)}] {f['name']}: {stats['rows']:,} → {stats['after_abstract']:,}"
                f"  ({done_bytes / 1e9:.1f} GB read, {done_bytes / 1e6 / dt:.0f} MB/s)",
                flush=True,
            )
    if failed:
        print(f"ERROR {len(failed)} files failed (rerun resumes them): {failed}", flush=True)
        return 1
    print(f"ALL DONE convert in {time.monotonic() - t0:.0f}s", flush=True)
    return 0


# ----- prep ----------------------------------------------------------------------------------


def _staged_paths() -> list[Path]:
    """The staged files in manifest order (their names sort as (date, part))."""
    return sorted(STAGING.glob("*.parquet"))


def cmd_prep(args) -> int:
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    params = json.loads((STAGING / "params.json").read_text())
    paths = _staged_paths()
    n_manifest = len(works_files(_load_manifest()))
    if len(paths) < n_manifest:
        print(f"WARN only {len(paths):,} of {n_manifest:,} manifest files are staged", flush=True)
    t0 = time.monotonic()
    log: dict = {"staging": params, "n_files_staged": len(paths), "n_files_manifest": n_manifest}

    print("STEP staged ids → the catalog (N smallest work-id hashes) and the held-out pool")
    meta = pl.scan_parquet(paths).select("work_id", "year", "field").collect()
    ids = meta["work_id"].to_numpy()
    # A work id seen twice keeps its newest copy: the files are in updated_date order.
    _, last_rev = np.unique(ids[::-1], return_index=True)
    newest = np.zeros(ids.shape[0], dtype=bool)
    newest[ids.shape[0] - 1 - last_rev] = True
    keep = np.zeros_like(newest)
    keep[newest] = select_pmids(ids[newest], args.keep_items, args.seed)
    n_items = int(keep.sum())
    if args.keep_items is not None and n_items < args.keep_items:
        print(
            f"ERROR only {n_items:,} eligible works staged, --keep-items {args.keep_items:,}: "
            "re-run convert with a higher --sample-rate",
            flush=True,
        )
        return 1
    pool = newest & ~keep
    log.update(
        n_staged=int(ids.shape[0]), n_duplicates=int((~newest).sum()), n_items=n_items,
        n_pool=int(pool.sum()),
    )  # fmt: skip
    print(f"  {ids.shape[0]:,} staged → {n_items:,} items, {int(pool.sum()):,} in the pool")

    print("STEP write item_id_map.json + papers.parquet (item-id order)")
    writer = None
    offset = 0
    next_id = 1
    with open(output / "item_id_map.json", "w") as fmap:
        fmap.write("{")
        for p in paths:
            df = pl.read_parquet(p)
            k = keep[offset : offset + df.height]
            offset += df.height
            df = df.filter(pl.Series(k)).with_columns(
                pl.int_range(next_id, next_id + int(k.sum()), dtype=pl.Int64).alias("item_id")
            )
            if df.height == 0:
                continue
            fmap.write(("," if next_id > 1 else "") + ",".join(
                f'"W{w}":{i}' for w, i in zip(df["work_id"].to_list(), df["item_id"].to_list(),
                                              strict=True)
            ))  # fmt: skip
            next_id += df.height
            table = df.select("item_id", *_STAGED_SCHEMA).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(
                    output / "papers.parquet", table.schema, compression="zstd"
                )
            writer.write_table(table, row_group_size=args.row_group_rows)
        fmap.write("}")
    if writer is not None:
        writer.close()

    print("STEP held-out queries: pool papers whose references reach the catalog")
    kept_ids = ids[keep]
    order = np.argsort(kept_ids, kind="stable")
    sorted_ids = kept_ids[order]
    item_era = era_of(meta["year"].to_numpy()[keep])
    item_field = meta["field"].fill_null(-1).to_numpy()[keep]
    pool_rows = np.nonzero(pool)[0]
    edges = (
        pl.scan_parquet(paths)
        .with_row_index("row")
        .join(pl.LazyFrame({"row": pool_rows.astype(np.uint32)}), on="row", how="semi")
        .select("row", "year", pl.col("field").fill_null(-1), "refs")
        .explode("refs")
        .drop_nulls("refs")
        .collect()
    )
    ref_ids = edges["refs"].to_numpy()
    pos = np.clip(np.searchsorted(sorted_ids, ref_ids), 0, max(len(sorted_ids) - 1, 0))
    hit = (sorted_ids[pos] == ref_ids) if len(sorted_ids) else np.zeros(len(ref_ids), bool)
    edges = edges.filter(pl.Series(hit)).with_columns(
        pl.Series("item_id", order[pos[hit]] + 1, dtype=pl.Int64)
    ).unique(["row", "item_id"]).sort("row", "item_id")  # fmt: skip
    iid = edges["item_id"].to_numpy() - 1
    q_era = era_of(edges["year"].to_numpy())
    passes = (item_field[iid] == edges["field"].to_numpy()) & (edges["field"].to_numpy() >= 0)
    passes &= item_era[iid] < q_era
    edges = edges.with_columns(pl.Series("passes_field_era", passes))
    per_q = edges.group_by("row").agg(
        pl.len().alias("n_relevant"), pl.col("passes_field_era").sum().alias("n_relevant_field_era")
    )
    eligible = per_q.filter(pl.col("n_relevant_field_era") > 0)["row"].sort().to_numpy()
    rng = np.random.default_rng(args.seed)
    n_q = min(args.n_heldout, eligible.shape[0])
    chosen = np.sort(rng.choice(eligible, size=n_q, replace=False)) if n_q else eligible[:0]
    log.update(
        n_pool_with_catalog_refs=per_q.height, n_pool_eligible=int(eligible.shape[0]),
        n_heldout=int(n_q), n_citation_edges_in_catalog=edges.height,
    )  # fmt: skip
    print(
        f"  {per_q.height:,} pool papers cite the catalog, {eligible.shape[0]:,} with a "
        f"same-field earlier-era reference; {n_q:,} held out",
        flush=True,
    )
    if n_q == 0:
        print("ERROR no eligible held-out paper (catalog too small or too sparse)", flush=True)
        return 1

    query_row = pl.DataFrame({"row": chosen.astype(np.uint32), "query_row": np.arange(n_q)})
    qrels = (
        edges.join(query_row, on="row")
        .select("query_row", "item_id", "passes_field_era")
        .sort("query_row", "item_id")
    )
    passing = (
        qrels.filter("passes_field_era").group_by("query_row", maintain_order=True).agg("item_id")
    )
    targets = np.array(
        [int(rng.choice(np.asarray(v))) for v in passing.sort("query_row")["item_id"].to_list()]
    )
    queries = (
        pl.scan_parquet(paths)
        .with_row_index("row")
        .join(query_row.lazy(), on="row")
        .join(per_q.lazy(), on="row")
        .sort("query_row")
        .collect()
    )
    heldout = queries.select(
        pl.Series("item_id", targets, dtype=pl.Int64),
        pl.format("W{}", "work_id").alias("query_work_id"),
        pl.col("year").alias("query_year"),
        pl.Series("query_era", era_of(queries["year"].to_numpy())),
        pl.col("field").alias("query_field"),
        pl.col("subfield").alias("query_subfield"),
        pl.col("source").alias("query_source"),
        pl.col("is_oa").alias("query_is_oa"),
        "n_relevant",
        "n_relevant_field_era",
    )
    heldout.write_parquet(output / "heldout.parquet", compression="zstd")
    queries.select(
        pl.format("W{}", "work_id").alias("query_work_id"), "title", "abstract"
    ).write_parquet(output / "queries.parquet", compression="zstd")
    qrels.write_parquet(output / "qrels.parquet", compression="zstd")
    log["mean_relevant_per_query"] = round(float(heldout["n_relevant"].mean()), 2)
    log["mean_relevant_field_era_per_query"] = round(
        float(heldout["n_relevant_field_era"].mean()), 2
    )
    log["seed"] = args.seed
    log["wall_clock_sec"] = round(time.monotonic() - t0, 1)
    _merge_log(output, "prep", log)
    print(f"ALL DONE prep in {log['wall_clock_sec']:.0f}s — n_items={n_items:,} n_heldout={n_q:,}")
    return 0


# ----- encode --------------------------------------------------------------------------------


def _texts(df: pl.DataFrame, prefix: str, description_chars: int) -> list[str]:
    return [
        TEXT_TEMPLATE.format(prefix=prefix, title=t, abstract=a[:description_chars])
        for t, a in zip(df["title"].to_list(), df["abstract"].to_list(), strict=True)
    ]


def _load_encoder(args):
    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR --device cuda but no CUDA device visible")
    model = SentenceTransformer(args.encoder, device=args.device, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length
    if model.get_embedding_dimension() != EMB_DIM_NATIVE:
        raise SystemExit(f"ERROR {args.encoder} is not {EMB_DIM_NATIVE}-d")
    # bf16 weights, not autocast: 898 vs 755 docs/s on the A100, min cosine 0.99989 between
    # the two (docs/system/datasets.md § openalex).
    return (model.to(torch.bfloat16) if args.device == "cuda" else model).eval()


def _encode(model, texts: list[str], args):
    """``[len(texts), 768]`` fp16, L2-normalised."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        emb = model.encode(
            texts, batch_size=args.batch_size, convert_to_tensor=True, show_progress_bar=False
        )
    return F.normalize(emb.float(), dim=-1).half().cpu()


def _encode_meta(args, prefix: str, n_rows: int) -> dict:
    return {
        "prefix": prefix,
        "encoder": args.encoder,
        "dim": EMB_DIM_NATIVE,
        "reduction": "none",
        "normalization": "l2",
        "text_template": TEXT_TEMPLATE,
        "description_chars": args.description_chars,
        "max_seq_length": args.max_seq_length,
        "compute_dtype": "bfloat16 weights" if args.device == "cuda" else "float32",
        "n_rows": n_rows,
        "shape": [n_rows, EMB_DIM_NATIVE],
        "dtype": "float16",
    }


def cmd_encode_text(args) -> int:
    import torch

    output = Path(args.output_dir).expanduser()
    papers = output / "papers.parquet"
    content = output / f"content_d{EMB_DIM_NATIVE}"
    n_items = pl.scan_parquet(papers).select(pl.len()).collect().item()
    meta = _encode_meta(args, DOC_PREFIX, n_items) | {"shard_rows": args.shard_rows}
    _check_params(content / "encode_params.json", meta)
    model = _load_encoder(args)
    entries = []
    t0 = time.monotonic()
    n_done = 0
    for i, start in enumerate(range(0, n_items, args.shard_rows)):
        n = min(args.shard_rows, n_items - start)
        name = f"text_emb_shard_{i:03d}.pt"
        entries.append({"filename": name, "start_id": start, "n_rows": n})
        if (content / name).exists():
            print(f"  shard {i}: already encoded", flush=True)
            continue
        rows = pl.scan_parquet(papers).slice(start, n).select("title", "abstract").collect()
        emb = _encode(model, _texts(rows, DOC_PREFIX, args.description_chars), args)
        atomic_write(content / name, lambda fh, emb=emb: torch.save(emb, fh))
        n_done += n
        rate = n_done / (time.monotonic() - t0)
        print(
            f"  shard {i}: {n:,} rows → {name}  ({rate:,.0f} docs/s, "
            f"{(n_items - start - n) / rate / 3600:.2f} h left)",
            flush=True,
        )
    # Written last: the index is what the loader reads, so it only exists once every shard does.
    (content / "shard_index.json").write_text(
        json.dumps(
            {
                "n_items": n_items,
                "dim": EMB_DIM_NATIVE,
                "dtype": "float16",
                "n_shards": len(entries),
                "shards": entries,
                "order": "item_id - 1 (papers.parquet row order)",
            },
            indent=2,
        )
    )
    (content / "text_emb.meta.json").write_text(
        json.dumps(meta | {"layout": "sharded (shard_index.json)"}, indent=2)
    )
    print(f"ALL DONE encode_text in {time.monotonic() - t0:.0f}s — {n_items:,} items", flush=True)
    return 0


def cmd_encode_queries(args) -> int:
    import torch

    output = Path(args.output_dir).expanduser()
    content = output / f"content_d{EMB_DIM_NATIVE}"
    content.mkdir(parents=True, exist_ok=True)
    queries = pl.read_parquet(output / "queries.parquet")
    model = _load_encoder(args)
    emb = _encode(model, _texts(queries, QUERY_PREFIX, args.description_chars), args)
    atomic_write(content / "query_emb.pt", lambda fh: torch.save(emb, fh))
    (content / "query_emb.meta.json").write_text(
        json.dumps(_encode_meta(args, QUERY_PREFIX, emb.shape[0]), indent=2)
    )
    print(f"ALL DONE encode_queries — {tuple(emb.shape)} → {content / 'query_emb.pt'}", flush=True)
    return 0


# ----- attrs ---------------------------------------------------------------------------------


def _lookup(values: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """Dense code of each value in ``vocab`` (its position), ``-1`` where absent."""
    if vocab.size == 0:
        return np.full(values.shape, -1, dtype=np.int64)
    order = np.argsort(vocab, kind="stable")
    pos = np.clip(np.searchsorted(vocab[order], values), 0, vocab.size - 1)
    found = vocab[order][pos] == values
    return np.where(found, order[pos], -1).astype(np.int64)


def earlier_era_slots(era: np.ndarray) -> np.ndarray:
    """``[N, A_MAX_NARROW]``: the eras after each item's own, ``-1`` padded (C1's encoding)."""
    slots = era[:, None] + 1 + np.arange(A_MAX_NARROW)[None, :]
    return np.where(slots < len(ERA_NAMES), slots, -1)


def cmd_attrs(args) -> int:
    import torch

    output = Path(args.output_dir).expanduser()
    t0 = time.monotonic()
    papers = pl.read_parquet(
        output / "papers.parquet",
        columns=["item_id", "year", "field", "subfield", "source", "is_oa"],
    )
    n = papers.height
    if not (papers["item_id"].to_numpy() == np.arange(1, n + 1)).all():
        raise SystemExit("ERROR papers.parquet is not in item-id order (re-run prep)")

    field = papers["field"].fill_null(-1).to_numpy().astype(np.int64)
    subfield = papers["subfield"].fill_null(-1).to_numpy().astype(np.int64)
    source = papers["source"].fill_null(-1).to_numpy()
    field_vocab = np.unique(field[field >= 0])
    subfield_vocab = np.unique(subfield[subfield >= 0])
    src = (
        papers.filter(pl.col("source").is_not_null())
        .group_by("source")
        .len()
        .sort(["len", "source"], descending=[True, False])
        .head(args.source_vocab)
    )
    source_vocab = src["source"].to_numpy()

    narrow = np.full((n, C_NARROW, A_MAX_NARROW), -1, dtype=np.int64)
    narrow[:, 0, 0] = _lookup(field, field_vocab)
    narrow[:, 1, :] = earlier_era_slots(era_of(papers["year"].to_numpy()))
    narrow[:, 2, 0] = _lookup(subfield, subfield_vocab)
    narrow[:, 3, 0] = _lookup(source, source_vocab)
    narrow[:, 4, 0] = papers["is_oa"].to_numpy().astype(np.int64)
    torch.save(torch.from_numpy(narrow), output / "item_attrs_narrow.pt")
    torch.save(torch.tensor(CLAUSE_IS_REVERSE), output / "clause_is_reverse_narrow.pt")

    for name, vocab in (("field", field_vocab), ("subfield", subfield_vocab)):
        (output / f"{name}_vocab.json").write_text(
            json.dumps({"size": int(vocab.size), "openalex_ids": vocab.tolist()})
        )
    (output / "source_vocab.json").write_text(
        json.dumps(
            {
                "size": int(source_vocab.size),
                "openalex_ids": source_vocab.tolist(),
                "counts": src["len"].to_list(),
            }
        )  # fmt: skip
    )
    (output / "era_vocab.json").write_text(json.dumps({"edges": ERA_EDGES, "names": ERA_NAMES}))

    # Query side: every value is the held-out paper's own, so "same field, earlier era, same
    # subfield, not the query's venue"; C4 asks for open-access items.
    heldout = pl.read_parquet(output / "heldout.parquet")
    qa = np.stack(
        [
            _lookup(heldout["query_field"].fill_null(-1).to_numpy(), field_vocab),
            heldout["query_era"].to_numpy().astype(np.int64),
            _lookup(heldout["query_subfield"].fill_null(-1).to_numpy(), subfield_vocab),
            _lookup(heldout["query_source"].fill_null(-1).to_numpy(), source_vocab),
            np.ones(heldout.height, dtype=np.int64),
        ],
        axis=1,
    )
    pl.DataFrame(
        {"target_id": heldout["item_id"], "query_attrs_narrow": qa.tolist()},
        schema={"target_id": pl.Int64, "query_attrs_narrow": pl.List(pl.Int64)},
    ).write_parquet(output / "eval_split.parquet", compression="zstd")

    target = heldout["item_id"].to_numpy() - 1
    active = qa >= 0
    target_passes = [
        float(((narrow[target, c, :] == qa[:, c : c + 1]).any(axis=1) != CLAUSE_IS_REVERSE[c])[
            active[:, c]
        ].mean())
        if active[:, c].any()
        else None
        for c in range(C_NARROW)
    ]  # fmt: skip
    log = {
        "n_items": n,
        "clauses": CLAUSE_NAMES,
        "clause_is_reverse": CLAUSE_IS_REVERSE,
        "vocab_sizes": {
            "field": int(field_vocab.size),
            "subfield": int(subfield_vocab.size),
            "source": int(source_vocab.size),
            "era": len(ERA_NAMES),
        },
        "coverage": {
            name: round(float((narrow[:, c, 0] >= 0).mean()), 4) if n else 0.0
            for c, name in enumerate(CLAUSE_NAMES)
        },
        "query_clause_active": {
            name: round(float(active[:, c].mean()), 4) for c, name in enumerate(CLAUSE_NAMES)
        },
        "target_passes_clause": dict(zip(CLAUSE_NAMES, target_passes, strict=True)),
        "wall_clock_sec": round(time.monotonic() - t0, 1),
    }
    _merge_log(output, "attrs", log)
    print(json.dumps(log, indent=2), flush=True)
    return 0


# ----- all -----------------------------------------------------------------------------------


def cmd_all(args) -> int:
    for step in (
        cmd_download,
        cmd_convert,
        cmd_prep,
        cmd_encode_text,
        cmd_encode_queries,
        cmd_attrs,
    ):
        rc = step(args)
        if rc != 0:
            return rc
    return 0


# ----- main ----------------------------------------------------------------------------------


def _add_convert_args(p) -> None:
    p.add_argument(
        "--files",
        type=str,
        default=None,
        help='indices into the manifest\'s works files, e.g. "0-3,7" (default: all)',
    )
    p.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="keep works whose seeded id hash is below this fraction (from `plan`)",
    )
    p.add_argument("--workers", type=int, default=32)  # fmt: skip


def _add_prep_args(p) -> None:
    p.add_argument(
        "--keep-items",
        type=int,
        default=None,
        help="catalog size: the N smallest work-id hashes (default: all staged)",
    )
    p.add_argument("--n-heldout", type=int, default=10_000)
    p.add_argument("--row-group-rows", type=int, default=100_000)  # fmt: skip


def _add_encode_args(p) -> None:
    p.add_argument("--encoder", type=str, default=ENCODER)
    p.add_argument("--description-chars", type=int, default=1500)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--shard-rows", type=int, default=1_000_000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETL for the OpenAlex filter-bench dataset (E3).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_out = str(data_root() / "openalex")

    sp = sub.add_parser("plan", help="dry run: footers + sampled filter rates → budget")
    sp.add_argument("--keep-items", type=int, default=50_000_000)
    sp.add_argument("--margin", type=float, default=0.05, help="over-sample for the held-out pool")
    sp.add_argument("--sample-row-groups", type=int, default=32)
    sp.add_argument("--threads", type=int, default=32)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--report", type=str, default="", help="write the JSON budget here")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("download", help="pin the release: fetch manifest.json")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("convert", help="stream the works files → filtered staging parquet")
    _add_convert_args(sp)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_convert)

    sp = sub.add_parser("prep", help="staging → item_id_map, papers, heldout, queries, qrels")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--seed", type=int, default=0)
    _add_prep_args(sp)
    sp.set_defaults(func=cmd_prep)

    sp = sub.add_parser("encode_text", help="items → content_d768/text_emb_shard_*.pt")
    sp.add_argument("--output-dir", type=str, default=default_out)
    _add_encode_args(sp)
    sp.set_defaults(func=cmd_encode_text)

    sp = sub.add_parser("encode_queries", help="held-out papers → content_d768/query_emb.pt")
    sp.add_argument("--output-dir", type=str, default=default_out)
    _add_encode_args(sp)
    sp.set_defaults(func=cmd_encode_queries)

    sp = sub.add_parser("attrs", help="narrow clause tensor + vocabs + eval_split")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--source-vocab", type=int, default=5_000)
    sp.set_defaults(func=cmd_attrs)

    sp = sub.add_parser(
        "all", help="download → convert → prep → encode_text → encode_queries → attrs"
    )
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--source-vocab", type=int, default=5_000)
    _add_convert_args(sp)
    _add_prep_args(sp)
    _add_encode_args(sp)
    sp.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["abstract_text", "earlier_era_slots", "era_of", "main", "stage_table", "works_files"]
