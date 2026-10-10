#!/usr/bin/env -S uv run --script
"""ETL for the Re-LAION 30 M filter-bench dataset (roadmap V-LAION30): captions as text
items, their metadata as clause tags.

Source: ``laion/relaion2B-en-research-safe`` on the Hub (gated, auto-approved; the token in
the environment reads it), the first two parquet parts (2 × ~16.4 M rows: url, caption,
similarity, pwatermark, punsafe, original_width / original_height, hash, key, ...). No
embeddings ship with it; the captions are encoded here with nomic-embed-text-v1.5 at its
Matryoshka d256. Measured counts and rates are in docs/system/datasets.md § laion30m.

Subcommands::

    download        The parts (``--parts``) into _raw/relaion/.
    prep            parts → items.parquet (the catalog: one row per distinct caption, the N
                    smallest seeded row hashes, source order) + queries.parquet (held-out
                    captions drawn from the rest), each with its url domain and tag buckets.
    encode_text     nomic-embed-text-v1.5, "search_document: ", Matryoshka d256 →
                    items_NNN.npy (fp16, 1 M rows each). Resumable per shard.
    encode_queries  Same encoder, "search_query: " → queries.npy.
    ingest          ``eval-data ingest laion30m`` over the above: the staged dataset, its
                    yaml and the layout check (targets: each query's unfiltered top-1).
    all             download → prep → encode_text → encode_queries → ingest.

Layout (under hub.data_root(): $RETRIEVE_DATA_ROOT, default evaluation/data)::

    data/_raw/relaion/part-0000{0,1}-*.parquet
    data/_raw/relaion/laion30m/             prep + encode outputs (the ingest's inputs)
    data/laion30m/                          the staged dataset (written by the ingest)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import tldextract

from eval_datasets.common import merge_prep_log, parse_ranges, pmid_hash
from eval_datasets.hub import data_root, raw_dir
from eval_datasets.ingest import CONFIG_DIR, ingest
from eval_datasets.layout import atomic_write

REPO = "laion/relaion2B-en-research-safe"
ROOT = raw_dir("relaion")
PART_GLOB = "part-{:05d}-*.parquet"
COLUMNS = [
    "key", "hash", "url", "caption", "similarity", "pwatermark", "punsafe",
    "original_width", "original_height",
]  # fmt: skip

WORK = ROOT / "laion30m"
NAME = "laion30m"
# The ingest's clauses (NAME[=COLUMN][!], ! = reverse) and sweeps; C0 and C3 share a column.
CLAUSES = [
    "domain", "size=size_bucket", "similarity=similarity_bucket", "domain_reverse=domain!",
    "watermark=watermark_bucket", "unsafe=unsafe_bucket",
]  # fmt: skip
SWEEPS = [
    "c0_domain=domain", "c3_domain_reverse=domain_reverse",
    "tags4=size,similarity,watermark,unsafe", "all_fwd=domain,size,similarity,watermark,unsafe",
]  # fmt: skip
# Upper bucket edges (np.searchsorted side="right"), picked near the quintiles of the two
# parts so that each bucket clause passes ~20 %; the measured shares are in prep_log.json.
SIZE_EDGES = [200, 300, 500, 800]  # max(original_width, original_height), px
SIMILARITY_EDGES = [0.30, 0.315, 0.33, 0.35]  # the CLIP image-caption cosine
WATERMARK_EDGES = [0.1, 0.2, 0.35, 0.6]  # pwatermark
UNSAFE_EDGES = [1e-5, 1e-4, 1e-3, 1e-2]  # punsafe (research-safe caps it below 0.45)

EMB_DIM_NATIVE = 768
EMB_DIM = 256
ENCODER = "nomic-ai/nomic-embed-text-v1.5"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
TEXT_TEMPLATE = "{prefix}{caption}"

_PSL = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)  # the bundled snapshot


def registered_domain(host: str) -> str:
    """The registrable domain of ``host`` (``cdn.shopify.com`` → ``shopify.com``, by the
    public suffix list tldextract bundles); the host itself when it has none (an IP)."""
    ext = _PSL(host)
    return ext.top_domain_under_public_suffix or host


def bucket(values: pl.Series, edges: list[float]) -> pl.Series:
    """``len(edges) + 1`` buckets of ``values``, null where the value is missing."""
    v = values.cast(pl.Float64).to_numpy()
    out = np.searchsorted(np.asarray(edges), v, side="right").astype(np.int64)
    return pl.Series(out).set(pl.Series(np.isnan(v)), None)


def _with_tags(df: pl.DataFrame) -> pl.DataFrame:
    side = df.select(pl.max_horizontal("original_width", "original_height")).to_series()
    return df.with_columns(
        size_bucket=bucket(side, SIZE_EDGES),
        similarity_bucket=bucket(df["similarity"], SIMILARITY_EDGES),
        watermark_bucket=bucket(df["pwatermark"], WATERMARK_EDGES),
        unsafe_bucket=bucket(df["punsafe"], UNSAFE_EDGES),
    )


def _parts(parts: str) -> list[Path]:
    paths = []
    for i in parse_ranges(parts, 0):
        hits = sorted(ROOT.glob(PART_GLOB.format(i)))
        if len(hits) != 1:
            raise SystemExit(f"ERROR part {i}: {len(hits)} files under {ROOT} (run download)")
        paths.append(hits[0])
    return paths


def cmd_download(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    patterns = [PART_GLOB.format(i) for i in parse_ranges(args.parts, 0)]
    snapshot_download(REPO, repo_type="dataset", local_dir=ROOT, allow_patterns=patterns)
    print(f"ALL DONE download — {[p.name for p in _parts(args.parts)]}", flush=True)
    return 0


def _with_domain(df: pl.DataFrame) -> pl.DataFrame:
    host = df["url"].str.extract(r"^[A-Za-z][A-Za-z0-9+.-]*://([^/:?#@]+)", 1).str.to_lowercase()
    hosts = host.unique().drop_nulls()
    domains = pl.DataFrame(
        {"host": hosts, "domain": [registered_domain(h) for h in hosts.to_list()]}
    )
    # prep then picks items and queries by position: the join must keep the rank order
    joined = df.with_columns(host=host).join(domains, on="host", how="left", maintain_order="left")
    return joined.drop("host")


def cmd_prep(args: argparse.Namespace) -> int:
    output = WORK
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    paths = _parts(args.parts)
    print(f"STEP read {len(paths)} parts")
    df = (
        pl.scan_parquet(paths)
        .select(COLUMNS)
        .with_row_index("src_row")
        .with_columns(pl.col("caption").str.strip_chars())
        .collect()
    )
    n_rows = df.height
    df = df.filter(pl.col("caption").is_not_null() & (pl.col("caption").str.len_chars() > 0))
    n_captioned = df.height
    rank = pmid_hash(df["hash"].to_numpy().view(np.uint64), args.seed)
    # One row per distinct caption (identical text is an identical vector): the smallest rank.
    df = (
        df.with_columns(pl.Series("rank", rank))
        .sort("rank", "src_row")
        .unique("caption", keep="first", maintain_order=True)
    )
    n_distinct = df.height
    if n_distinct < args.keep_items + args.n_heldout:
        print(f"ERROR {n_distinct:,} distinct captions < keep + held-out", flush=True)
        return 1
    print(f"  {n_rows:,} rows → {n_captioned:,} captioned → {n_distinct:,} distinct captions")

    print("STEP url domains")
    df = _with_tags(_with_domain(df.drop("rank")))
    rng = np.random.default_rng(args.seed)
    pool = np.arange(args.keep_items, n_distinct)
    chosen = np.sort(rng.choice(pool, size=args.n_heldout, replace=False))
    tags = [
        "key", "url", "domain", "caption", "similarity", "pwatermark", "punsafe",
        "original_width", "original_height", "size_bucket", "similarity_bucket",
        "watermark_bucket", "unsafe_bucket",
    ]  # fmt: skip
    items = (
        df.head(args.keep_items)
        .sort("src_row")
        .with_row_index("item_id", offset=1)
        .with_columns(pl.col("item_id").cast(pl.Int64))
        .select("item_id", *tags)
    )
    items.write_parquet(output / "items.parquet", compression="zstd", row_group_size=1_000_000)
    queries = df[chosen].with_row_index("query_row").select("query_row", *tags)
    queries.write_parquet(output / "queries.parquet", compression="zstd")
    log = {
        "parts": [p.name for p in paths],
        "n_rows": n_rows,
        "n_captioned": n_captioned,
        "n_distinct_captions": n_distinct,
        "n_items": items.height,
        "n_pool": int(pool.size),
        "n_heldout": int(chosen.size),
        "n_domains_items": items["domain"].n_unique(),
        "seed": args.seed,
        "wall_clock_sec": round(time.monotonic() - t0, 1),
    }
    merge_prep_log(output, "prep", log)
    print(f"ALL DONE prep — {json.dumps(log)}", flush=True)
    return 0


def _texts(df: pl.DataFrame, prefix: str) -> list[str]:
    return [TEXT_TEMPLATE.format(prefix=prefix, caption=c) for c in df["caption"].to_list()]


def _load_encoder(args: argparse.Namespace):
    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ERROR --device cuda but no CUDA device visible")
    model = SentenceTransformer(args.encoder, device=args.device, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length
    if model.get_embedding_dimension() != EMB_DIM_NATIVE:
        raise SystemExit(f"ERROR {args.encoder} is not {EMB_DIM_NATIVE}-d")
    return (model.to(torch.bfloat16) if args.device == "cuda" else model).eval()


def _encode(model, texts: list[str], args: argparse.Namespace) -> np.ndarray:
    """``[len(texts), 256]`` fp16: the model card's Matryoshka recipe — layer norm over the
    768 dims, the first 256 kept, L2-normalised."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        emb = model.encode(
            texts, batch_size=args.batch_size, convert_to_tensor=True, show_progress_bar=False
        ).float()
    emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))[:, :EMB_DIM]
    return F.normalize(emb, dim=-1).half().cpu().numpy()


def _encode_params(args: argparse.Namespace) -> dict:
    return {
        "encoder": args.encoder,
        "dim": EMB_DIM,
        "reduction": "matryoshka: layer_norm(768) → [:256] (model card)",
        "text_template": TEXT_TEMPLATE,
        "doc_prefix": DOC_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "max_seq_length": args.max_seq_length,
        "compute_dtype": "bfloat16 weights" if args.device == "cuda" else "float32",
        "shard_rows": args.shard_rows,
    }


def _check_params(path: Path, params: dict) -> None:
    if path.exists():
        old = json.loads(path.read_text())
        if old != params:
            raise SystemExit(f"ERROR {path} pins other parameters: {old} != {params}")
    else:
        path.write_text(json.dumps(params, indent=2))


def _item_shards() -> list[Path]:
    return sorted(WORK.glob("items_*.npy"))


def cmd_encode_text(args: argparse.Namespace) -> int:
    items = WORK / "items.parquet"
    n_items = pl.scan_parquet(items).select(pl.len()).collect().item()
    _check_params(WORK / "encode_params.json", _encode_params(args))
    model = _load_encoder(args)
    t0 = time.monotonic()
    n_done = 0
    for i, start in enumerate(range(0, n_items, args.shard_rows)):
        n = min(args.shard_rows, n_items - start)
        path = WORK / f"items_{i:03d}.npy"
        if path.exists():
            print(f"  shard {i}: already encoded", flush=True)
            continue
        rows = pl.scan_parquet(items).slice(start, n).select("caption").collect()
        emb = _encode(model, _texts(rows, DOC_PREFIX), args)
        atomic_write(path, partial(np.save, arr=emb))
        n_done += n
        rate = n_done / (time.monotonic() - t0)
        print(
            f"  shard {i}: {n:,} rows → {path.name}  ({rate:,.0f} docs/s, "
            f"{(n_items - start - n) / rate / 3600:.2f} h left)",
            flush=True,
        )
    print(f"ALL DONE encode_text in {time.monotonic() - t0:.0f}s — {n_items:,} items", flush=True)
    return 0


def cmd_encode_queries(args: argparse.Namespace) -> int:
    _check_params(WORK / "encode_params.json", _encode_params(args))
    queries = pl.read_parquet(WORK / "queries.parquet", columns=["caption"])
    emb = _encode(_load_encoder(args), _texts(queries, QUERY_PREFIX), args)
    atomic_write(WORK / "queries.npy", partial(np.save, arr=emb))
    print(f"ALL DONE encode_queries — {emb.shape} → {WORK / 'queries.npy'}", flush=True)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    problems = ingest(
        NAME,
        items=_item_shards(),
        item_attrs=WORK / "items.parquet",
        queries=WORK / "queries.npy",
        query_attrs=WORK / "queries.parquet",
        clauses=CLAUSES,
        sweeps=SWEEPS,
        doc_prefix=DOC_PREFIX,
        query_prefix=QUERY_PREFIX,
        output_dir=Path(args.output_dir).expanduser(),
        config_dir=Path(args.config_dir),
        device=args.device,
    )
    print("\n".join(f"PROBLEM {p}" for p in problems) or f"ALL DONE ingest — {NAME}: ok")
    return 1 if problems else 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_download, cmd_prep, cmd_encode_text, cmd_encode_queries, cmd_ingest):
        rc = step(args)
        if rc != 0:
            return rc
    return 0


def _add_encode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--encoder", type=str, default=ENCODER)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--shard-rows", type=int, default=1_000_000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETL for the Re-LAION 30 M dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add(name, func, help, prep=False, encode=False, device=False, output=False):
        sp = sub.add_parser(name, help=help)
        if prep:
            sp.add_argument("--parts", type=str, default="0-1", help='part indices, e.g. "0-1"')
            sp.add_argument("--keep-items", type=int, default=30_000_000)
            sp.add_argument("--n-heldout", type=int, default=10_000)
            sp.add_argument("--seed", type=int, default=0)
        if encode:
            _add_encode_args(sp)
        if device:
            sp.add_argument("--device", type=str, default="cuda")
        if output:
            sp.add_argument("--output-dir", type=str, default=str(data_root() / NAME))
            sp.add_argument("--config-dir", type=str, default=str(CONFIG_DIR))
        sp.set_defaults(func=func)

    _add("download", cmd_download, "the parquet parts → _raw/relaion/", prep=True)
    _add("prep", cmd_prep, "parts → items.parquet + queries.parquet (with tags)", prep=True)
    _add("encode_text", cmd_encode_text, "items → items_NNN.npy", encode=True, device=True)
    _add("encode_queries", cmd_encode_queries, "queries → queries.npy", encode=True, device=True)
    _add("ingest", cmd_ingest, "eval-data ingest laion30m", device=True, output=True)
    _add("all", cmd_all, "download → prep → encode → ingest", prep=True, encode=True,
         device=True, output=True)  # fmt: skip
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["bucket", "main", "registered_domain"]
