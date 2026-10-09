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
                    captions drawn from the rest), each with its url domain and tag values.
    encode_text     nomic-embed-text-v1.5, "search_document: ", Matryoshka d256 →
                    content_d256/text_emb_shard_NNN.pt + shard_index.json. Resumable per shard.
    encode_queries  Same encoder, "search_query: " → content_d256/query_emb.pt.
    targets         heldout.parquet: each query's target is its exact unfiltered nearest item
                    (cosine, ties to the lowest id) — the yfcc10m convention.
    attrs           item_attrs_narrow.pt [N, 6, 1], clause_is_reverse_narrow.pt, vocabs,
                    eval_split.parquet.
    all             download → prep → encode_text → encode_queries → targets → attrs.

Layout (under hub.data_root(): $RETRIEVE_DATA_ROOT, default evaluation/data)::

    data/_raw/relaion/part-0000{0,1}-*.parquet
    data/laion30m/                          the bench-side output dir
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
from eval_datasets.layout import atomic_write, load_sharded

REPO = "laion/relaion2B-en-research-safe"
ROOT = raw_dir("relaion")
PART_GLOB = "part-{:05d}-*.parquet"
COLUMNS = [
    "key", "hash", "url", "caption", "similarity", "pwatermark", "punsafe",
    "original_width", "original_height",
]  # fmt: skip

CLAUSE_NAMES = ["domain", "size", "similarity", "domain_reverse", "watermark", "unsafe"]
CLAUSE_IS_REVERSE = [False, False, False, True, False, False]
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


def bucket(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """``len(edges) + 1`` buckets of ``values``; ``nan`` (a missing value) → ``-1``."""
    out = np.searchsorted(np.asarray(edges), values, side="right").astype(np.int64)
    return np.where(np.isnan(values), -1, out)


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
    return df.with_columns(host=host).join(domains, on="host", how="left").drop("host")


def cmd_prep(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).expanduser()
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
    df = _with_domain(df.drop("rank"))
    rng = np.random.default_rng(args.seed)
    pool = np.arange(args.keep_items, n_distinct)
    chosen = np.sort(rng.choice(pool, size=args.n_heldout, replace=False))
    tags = [
        "key", "url", "domain", "caption", "similarity", "pwatermark", "punsafe",
        "original_width", "original_height",
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


def _encode(model, texts: list[str], args: argparse.Namespace):
    """``[len(texts), 256]`` fp16: the model card's Matryoshka recipe — layer norm over the
    768 dims, the first 256 kept, L2-normalised."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        emb = model.encode(
            texts, batch_size=args.batch_size, convert_to_tensor=True, show_progress_bar=False
        ).float()
    emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))[:, :EMB_DIM]
    return F.normalize(emb, dim=-1).half().cpu()


def _encode_meta(args: argparse.Namespace, prefix: str, n_rows: int) -> dict:
    return {
        "prefix": prefix,
        "encoder": args.encoder,
        "dim": EMB_DIM,
        "reduction": "matryoshka: layer_norm(768) → [:256] (model card)",
        "normalization": "l2",
        "text_template": TEXT_TEMPLATE,
        "max_seq_length": args.max_seq_length,
        "compute_dtype": "bfloat16 weights" if args.device == "cuda" else "float32",
        "n_rows": n_rows,
        "shape": [n_rows, EMB_DIM],
        "dtype": "float16",
    }


def _check_params(path: Path, params: dict) -> None:
    if path.exists():
        old = json.loads(path.read_text())
        if old != params:
            raise SystemExit(f"ERROR {path} pins other parameters: {old} != {params}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(params, indent=2))


def cmd_encode_text(args: argparse.Namespace) -> int:
    import torch

    output = Path(args.output_dir).expanduser()
    items = output / "items.parquet"
    content = output / f"content_d{EMB_DIM}"
    n_items = pl.scan_parquet(items).select(pl.len()).collect().item()
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
        rows = pl.scan_parquet(items).slice(start, n).select("caption").collect()
        emb = _encode(model, _texts(rows, DOC_PREFIX), args)
        atomic_write(content / name, partial(torch.save, emb))
        n_done += n
        rate = n_done / (time.monotonic() - t0)
        print(
            f"  shard {i}: {n:,} rows → {name}  ({rate:,.0f} docs/s, "
            f"{(n_items - start - n) / rate / 3600:.2f} h left)",
            flush=True,
        )
    # Written last: the index is what the loader reads, so it exists only once every shard does.
    (content / "shard_index.json").write_text(
        json.dumps(
            {
                "n_items": n_items,
                "dim": EMB_DIM,
                "dtype": "float16",
                "n_shards": len(entries),
                "shards": entries,
                "order": "item_id - 1 (items.parquet row order)",
            },
            indent=2,
        )
    )
    (content / "text_emb.meta.json").write_text(
        json.dumps(meta | {"layout": "sharded (shard_index.json)"}, indent=2)
    )
    print(f"ALL DONE encode_text in {time.monotonic() - t0:.0f}s — {n_items:,} items", flush=True)
    return 0


def cmd_encode_queries(args: argparse.Namespace) -> int:
    import torch

    output = Path(args.output_dir).expanduser()
    content = output / f"content_d{EMB_DIM}"
    content.mkdir(parents=True, exist_ok=True)
    queries = pl.read_parquet(output / "queries.parquet", columns=["caption"])
    model = _load_encoder(args)
    emb = _encode(model, _texts(queries, QUERY_PREFIX), args)
    atomic_write(content / "query_emb.pt", lambda fh: torch.save(emb, fh))
    (content / "query_emb.meta.json").write_text(
        json.dumps(_encode_meta(args, QUERY_PREFIX, emb.shape[0]), indent=2)
    )
    print(f"ALL DONE encode_queries — {tuple(emb.shape)} → {content / 'query_emb.pt'}", flush=True)
    return 0


def nearest_items(items, queries, *, item_chunk: int = 2_000_000, query_batch: int = 256):
    """``(ids [U] 0-indexed, scores [U])``: each query's exact top-1 item by inner product in
    fp32, equal scores to the lowest id."""
    import torch

    ids = torch.empty(queries.shape[0], dtype=torch.long)
    best = torch.empty(queries.shape[0], dtype=torch.float32)
    for qs in range(0, queries.shape[0], query_batch):
        q = queries[qs : qs + query_batch].to(items.device)
        b_s = torch.full((q.shape[0],), float("-inf"), device=items.device)
        b_i = torch.zeros(q.shape[0], dtype=torch.long, device=items.device)
        for start in range(0, items.shape[0], item_chunk):
            s = q @ items[start : start + item_chunk].t()
            m = s.max(dim=1).values
            pos = torch.arange(s.shape[1], device=items.device).expand_as(s)
            first = torch.where(s == m[:, None], pos, s.shape[1]).min(dim=1).values + start
            better = m > b_s  # strictly: an equal score in a later chunk has a higher id
            b_s, b_i = torch.where(better, m, b_s), torch.where(better, first, b_i)
        ids[qs : qs + q.shape[0]], best[qs : qs + q.shape[0]] = b_i.cpu(), b_s.cpu()
    return ids, best


def cmd_targets(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F

    output = Path(args.output_dir).expanduser()
    content = output / f"content_d{EMB_DIM}"
    t0 = time.monotonic()
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    items = load_sharded(content / "shard_index.json", device, normalize=True)
    queries = F.normalize(torch.load(str(content / "query_emb.pt")).float(), dim=-1)
    ids, scores = nearest_items(items, queries)
    keys = pl.read_parquet(output / "queries.parquet", columns=["key"])["key"]
    pl.DataFrame(
        {
            "item_id": (ids + 1).numpy(),
            "query_key": keys,
            "nn_score": scores.numpy(),
        },
        schema={"item_id": pl.Int64, "query_key": pl.Utf8, "nn_score": pl.Float32},
    ).write_parquet(output / "heldout.parquet", compression="zstd")
    log = {
        "rule": "exact unfiltered top-1 by inner product, fp32, ties to the lowest id",
        "device": args.device,
        "nn_score_quantiles": {
            str(q): round(float(scores.quantile(q)), 4) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "wall_clock_sec": round(time.monotonic() - t0, 1),
    }
    merge_prep_log(output, "targets", log)
    print(f"ALL DONE targets — {json.dumps(log)}", flush=True)
    return 0


def _tags(df: pl.DataFrame) -> np.ndarray:
    """``[rows, 5]``: size, similarity, watermark, unsafe buckets (domain is coded apart)."""
    side = df.select(pl.max_horizontal("original_width", "original_height")).to_series()
    return np.stack(
        [
            bucket(side.cast(pl.Float64).fill_null(np.nan).to_numpy(), SIZE_EDGES),
            bucket(
                df["similarity"].cast(pl.Float64).fill_null(np.nan).to_numpy(), SIMILARITY_EDGES
            ),
            bucket(df["pwatermark"].cast(pl.Float64).fill_null(np.nan).to_numpy(), WATERMARK_EDGES),
            bucket(df["punsafe"].cast(pl.Float64).fill_null(np.nan).to_numpy(), UNSAFE_EDGES),
        ],
        axis=1,
    )


def _narrow(domain: np.ndarray, tags: np.ndarray) -> np.ndarray:
    """``[rows, 6]`` in clause order: domain, size, similarity, domain (reverse), watermark,
    unsafe."""
    return np.stack(
        [domain, tags[:, 0], tags[:, 1], domain, tags[:, 2], tags[:, 3]], axis=1
    ).astype(np.int64)


def cmd_attrs(args: argparse.Namespace) -> int:
    import torch

    output = Path(args.output_dir).expanduser()
    t0 = time.monotonic()
    cols = ["domain", "similarity", "pwatermark", "punsafe", "original_width", "original_height"]
    items = pl.read_parquet(output / "items.parquet", columns=["item_id", *cols])
    n = items.height
    if not (items["item_id"].to_numpy() == np.arange(1, n + 1)).all():
        raise SystemExit("ERROR items.parquet is not in item-id order (re-run prep)")
    vocab = (
        items.group_by("domain")
        .len()
        .sort(["len", "domain"], descending=[True, False])
        .with_row_index("code")
        .with_columns(pl.col("code").cast(pl.Int64))
    )
    item_domain = items.join(vocab, on="domain", how="left", maintain_order="left")["code"]
    narrow = _narrow(item_domain.fill_null(-1).to_numpy(), _tags(items))
    torch.save(torch.from_numpy(narrow).unsqueeze(-1), output / "item_attrs_narrow.pt")
    torch.save(torch.tensor(CLAUSE_IS_REVERSE), output / "clause_is_reverse_narrow.pt")
    (output / "domain_vocab.json").write_text(
        json.dumps({"size": vocab.height, "domains": vocab["domain"].to_list(),
                    "counts": vocab["len"].to_list()})
    )  # fmt: skip
    (output / "bucket_edges.json").write_text(
        json.dumps({"size": SIZE_EDGES, "similarity": SIMILARITY_EDGES,
                    "watermark": WATERMARK_EDGES, "unsafe": UNSAFE_EDGES})
    )  # fmt: skip

    # Query side: every clause value is the held-out caption's own tag; a domain absent from
    # the catalog is -1 (no live clause), a missing size is -1 too.
    queries = pl.read_parquet(output / "queries.parquet", columns=["query_row", *cols])
    heldout = pl.read_parquet(output / "heldout.parquet")
    if heldout.height != queries.height:
        raise SystemExit("ERROR heldout.parquet and queries.parquet differ in rows")
    q_domain = queries.join(vocab, on="domain", how="left", maintain_order="left")["code"]
    qa = _narrow(q_domain.fill_null(-1).to_numpy(), _tags(queries))
    pl.DataFrame(
        {"target_id": heldout["item_id"], "query_attrs_narrow": qa.tolist()},
        schema={"target_id": pl.Int64, "query_attrs_narrow": pl.List(pl.Int64)},
    ).write_parquet(output / "eval_split.parquet", compression="zstd")

    domain_count = vocab["len"].to_numpy()
    active = qa >= 0
    q_pass_domain = np.where(active[:, 0], domain_count[np.maximum(qa[:, 0], 0)] / n, np.nan)
    target = heldout["item_id"].to_numpy() - 1
    log = {
        "n_items": n,
        "clauses": CLAUSE_NAMES,
        "clause_is_reverse": CLAUSE_IS_REVERSE,
        "n_domains": vocab.height,
        "bucket_shares": {
            name: (np.bincount(narrow[:, c][narrow[:, c] >= 0], minlength=5) / n).round(4).tolist()
            for c, name in ((1, "size"), (2, "similarity"), (4, "watermark"), (5, "unsafe"))
        },
        "item_tag_missing": {
            name: int((narrow[:, c] < 0).sum()) for c, name in enumerate(CLAUSE_NAMES)
        },
        "query_clause_active": {
            name: round(float(active[:, c].mean()), 4) for c, name in enumerate(CLAUSE_NAMES)
        },
        "query_domain_pass_rate_quantiles": {
            str(q): float(np.nanquantile(q_pass_domain, q)) for q in (0.1, 0.5, 0.9)
        },
        "target_passes_clause": {
            name: round(
                float(
                    ((narrow[target, c] == qa[:, c]) != CLAUSE_IS_REVERSE[c])[active[:, c]].mean()
                ),
                4,
            )
            for c, name in enumerate(CLAUSE_NAMES)
        },  # fmt: skip
        "wall_clock_sec": round(time.monotonic() - t0, 1),
    }
    merge_prep_log(output, "attrs", log)
    print(json.dumps(log, indent=2), flush=True)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_download, cmd_prep, cmd_encode_text, cmd_encode_queries, cmd_targets,
                 cmd_attrs):  # fmt: skip
        rc = step(args)
        if rc != 0:
            return rc
    return 0


def _add_encode_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--encoder", type=str, default=ENCODER)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--shard-rows", type=int, default=1_000_000)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ETL for the Re-LAION 30 M dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    default_out = str(data_root() / "laion30m")

    def _common(sp, prep: bool = False, encode: bool = False):
        sp.add_argument("--output-dir", type=str, default=default_out)
        if prep:
            sp.add_argument("--parts", type=str, default="0-1", help='part indices, e.g. "0-1"')
            sp.add_argument("--keep-items", type=int, default=30_000_000)
            sp.add_argument("--n-heldout", type=int, default=10_000)
            sp.add_argument("--seed", type=int, default=0)
        if encode:
            _add_encode_args(sp)

    sp = sub.add_parser("download", help="the parquet parts → _raw/relaion/")
    sp.add_argument("--parts", type=str, default="0-1")
    sp.set_defaults(func=cmd_download)
    sp = sub.add_parser("prep", help="parts → items.parquet + queries.parquet")
    _common(sp, prep=True)
    sp.set_defaults(func=cmd_prep)
    sp = sub.add_parser("encode_text", help="items → content_d256/text_emb_shard_*.pt")
    _common(sp, encode=True)
    sp.set_defaults(func=cmd_encode_text)
    sp = sub.add_parser("encode_queries", help="queries → content_d256/query_emb.pt")
    _common(sp, encode=True)
    sp.set_defaults(func=cmd_encode_queries)
    sp = sub.add_parser("targets", help="heldout.parquet: each query's unfiltered top-1 item")
    _common(sp)
    sp.add_argument("--device", type=str, default="cuda")
    sp.set_defaults(func=cmd_targets)
    sp = sub.add_parser("attrs", help="clause tensor + vocabs + eval_split")
    _common(sp)
    sp.set_defaults(func=cmd_attrs)
    sp = sub.add_parser("all", help="download → prep → encode → targets → attrs")
    _common(sp, prep=True, encode=True)
    sp.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["bucket", "main", "nearest_items", "registered_domain"]
