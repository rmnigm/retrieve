#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pyarrow>=15",
#   "polars>=1.0",
#   "huggingface-hub>=0.24",
#   "torch>=2.4",
#   "sentence-transformers>=3.0",
#   "tqdm>=4.66",
#   "numpy>=1.26",
# ]
# ///
"""End-to-end ETL for the arxiv filter-bench dataset.

Source: HuggingFace ``open-index/open-arxiv`` — full-corpus arxiv metadata
mirror (~2.99M papers, monthly parquet shards at ``data/YYYY/YYYY-MM.parquet``,
last refreshed 2026-03-24). Same upstream source the Kaggle Cornell-University/
arxiv snapshot was derived from; no auth required.

Subcommands::

    download       Pull all monthly parquet shards into raw/ via snapshot_download.
    convert        Merge raw/ shards into processed/arxiv_papers.parquet.
    prep           processed/ → bench-side <output-dir>/ (item_id_map, papers,
                   heldout). No interactions / no time-split — arxiv has no
                   user sequences.
    encode_text    Encode item-side text embeddings ("search_document: " prefix)
                   with nomic-embed-text-v1.5 truncated to 256-d (Matryoshka).
    encode_queries Encode held-out query-side text embeddings ("search_query: "
                   prefix). Same encoder, different prefix — nomic-specific.
    attrs          Build item_attrs_narrow / item_attrs_wide / vocabs /
                   eval_split.parquet for the filter bench.
    all            download → convert → prep → encode_text → encode_queries → attrs.

Examples::

    uv run arxiv all --output-dir data/arxiv/papers
    uv run arxiv encode_text --output-dir data/arxiv/papers --batch-size 256

Layout::

    ~/datasets/arxiv/
    ├── raw/         <- snapshot_download destination (data/YYYY/YYYY-MM.parquet)
    └── processed/   <- arxiv_papers.parquet (merged)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

from .common import sample_rare_biased_wide, synthesize_qa_narrow

ROOT = Path.home() / "datasets" / "arxiv"
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"

HF_REPO_ID = "open-index/open-arxiv"
HF_REPO_TYPE = "dataset"

# Narrow attribute layout — identical shape to goodreads' (5 clauses, A_max=4).
C_NARROW = 5
A_MAX_NARROW = 4
WIDE_BAG_SIZE = 32

# C1 license bucket vocab. Index = dense id; -1 reserved as padding.
LICENSE_BUCKETS = [
    "cc-by",
    "cc-by-sa",
    "cc-by-nc-sa",
    "cc-by-nc-nd",
    "cc-by-nc",
    "cc-by-nd",
    "cc0",
    "public-domain",
    "arxiv-default",
    "none",
    "other",
]

# Year buckets for C2.
def _year_to_bucket(y: int | None) -> int:
    """Map an integer year to one of {0..6} or -1 if missing/oor."""
    if y is None:
        return -1
    if y < 1990 or y > 2030:
        return -1
    if y < 2000:
        return 0
    if y < 2005:
        return 1
    if y < 2010:
        return 2
    if y < 2015:
        return 3
    if y < 2020:
        return 4
    if y < 2025:
        return 5
    return 6


def _versions_to_bucket(n: int | None) -> int:
    """Map version count to one of {0..3} or -1 if missing."""
    if n is None or n < 1:
        return -1
    if n == 1:
        return 0
    if n == 2:
        return 1
    if n == 3:
        return 2
    return 3


def _license_to_bucket(s: str | None) -> int:
    """Map a raw license URL/string to a bucket id, or -1 if unparseable."""
    if s is None:
        return LICENSE_BUCKETS.index("none")
    n = s.strip().lower()
    if not n or n in {"none", "null"}:
        return LICENSE_BUCKETS.index("none")
    # CC license URLs follow creativecommons.org/licenses/<spec>/<ver>/
    if "creativecommons.org/publicdomain/zero" in n or "/cc0/" in n:
        return LICENSE_BUCKETS.index("cc0")
    if "creativecommons.org/publicdomain" in n or "publicdomain" in n:
        return LICENSE_BUCKETS.index("public-domain")
    if "creativecommons.org/licenses/by-nc-sa" in n or "/by-nc-sa/" in n:
        return LICENSE_BUCKETS.index("cc-by-nc-sa")
    if "creativecommons.org/licenses/by-nc-nd" in n or "/by-nc-nd/" in n:
        return LICENSE_BUCKETS.index("cc-by-nc-nd")
    if "creativecommons.org/licenses/by-sa" in n or "/by-sa/" in n:
        return LICENSE_BUCKETS.index("cc-by-sa")
    if "creativecommons.org/licenses/by-nc" in n or "/by-nc/" in n:
        return LICENSE_BUCKETS.index("cc-by-nc")
    if "creativecommons.org/licenses/by-nd" in n or "/by-nd/" in n:
        return LICENSE_BUCKETS.index("cc-by-nd")
    if "creativecommons.org/licenses/by" in n or "/by/" in n:
        return LICENSE_BUCKETS.index("cc-by")
    if "arxiv.org/licenses/nonexclusive-distrib" in n or "arxiv.org/licenses" in n:
        return LICENSE_BUCKETS.index("arxiv-default")
    return LICENSE_BUCKETS.index("other")


def _category_to_main(leaf: str) -> str:
    """``cs.LG`` → ``cs``; ``hep-lat`` → ``hep-lat``. Top-level group prefix."""
    if "." in leaf:
        return leaf.split(".", 1)[0]
    return leaf


# ----- download --------------------------------------------------------------


def cmd_download(args) -> int:
    from huggingface_hub import snapshot_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"START snapshot_download {HF_REPO_ID} → {RAW_DIR}", flush=True)
    t0 = time.monotonic()
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        local_dir=str(RAW_DIR),
        allow_patterns=["data/*/*.parquet", "README.md"],
        max_workers=args.max_workers,
    )
    n_files = sum(1 for _ in RAW_DIR.glob("data/*/*.parquet"))
    total_bytes = sum(p.stat().st_size for p in RAW_DIR.glob("data/*/*.parquet"))
    print(
        f"DONE download {n_files} parquet shards "
        f"({total_bytes / 1024 / 1024:.0f} MiB) in {time.monotonic() - t0:.0f}s",
        flush=True,
    )
    return 0 if n_files > 0 else 1


# ----- convert ---------------------------------------------------------------


def cmd_convert(args) -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "arxiv_papers.parquet"
    if out.exists() and not args.force:
        n = pl.scan_parquet(out).select(pl.len()).collect().item()
        print(f"SKIP convert (already exists, {n:,} rows). --force to rebuild.", flush=True)
        return 0

    glob = str(RAW_DIR / "data" / "*" / "*.parquet")
    print(f"START convert: scanning {glob}", flush=True)
    t0 = time.monotonic()
    lf = pl.scan_parquet(glob).select(
        pl.col("id"),
        pl.col("title"),
        pl.col("abstract"),
        pl.col("categories"),
        pl.col("authors_parsed"),
        pl.col("license"),
        pl.col("update_date"),
        pl.col("versions"),
        pl.col("submitter"),
    )
    lf.sink_parquet(out, compression="zstd")
    n = pl.scan_parquet(out).select(pl.len()).collect().item()
    print(
        f"DONE convert {n:,} rows → {out} ({out.stat().st_size / 1024 / 1024:.0f} MiB) "
        f"in {time.monotonic() - t0:.0f}s",
        flush=True,
    )
    return 0


# ----- all -------------------------------------------------------------------


def cmd_all(args) -> int:
    rc = cmd_download(argparse.Namespace(max_workers=args.max_workers))
    if rc != 0:
        return rc
    rc = cmd_convert(argparse.Namespace(force=False))
    if rc != 0:
        return rc
    rc = cmd_prep(args)
    if rc != 0:
        return rc
    rc = cmd_encode_text(args)
    if rc != 0:
        return rc
    rc = cmd_encode_queries(args)
    if rc != 0:
        return rc
    return cmd_attrs(args)


# ----- prep ------------------------------------------------------------------
#
# Builds the bench-side dataset directory:
#
#     <output-dir>/item_id_map.json     (arxiv_id_str → 1-indexed dense int)
#     <output-dir>/papers.parquet       (one row per item_id, all source cols)
#     <output-dir>/heldout.parquet      (sampled held-out queries)
#     <output-dir>/prep_log.json        stats
#
# No interactions, no time-split — arxiv has no user sequences. The
# held-out items remain in the index; nomic's two-prefix divergence makes
# the cross-check sweep (filter_kind=none) meaningful (see plan §"Held-out").


def cmd_prep(args) -> int:
    import numpy as np

    src = PROCESSED_DIR / "arxiv_papers.parquet"
    if not src.exists():
        print(f"ERROR missing {src} (run `convert` first)", flush=True)
        return 1
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.monotonic()
    print("STEP load arxiv_papers.parquet", flush=True)
    df = pl.read_parquet(src)
    n_raw = df.height
    print(f"  raw rows: {n_raw:,}", flush=True)

    # Drop rows missing the fields the encoder will read. abstract is the
    # load-bearing one; title is sometimes empty for old papers.
    df = (
        df.with_columns(
            pl.col("title").fill_null("").str.strip_chars().alias("title"),
            pl.col("abstract").fill_null("").str.strip_chars().alias("abstract"),
        )
        .filter(pl.col("abstract").str.len_chars() > 0)
        .filter(pl.col("id").is_not_null())
        .unique(subset=["id"], keep="first", maintain_order=True)
    )
    n_filtered = df.height
    n_dropped = n_raw - n_filtered
    print(
        f"  after drop-empty-abstract + dedup: {n_filtered:,} (-{n_dropped:,})",
        flush=True,
    )

    # Sort by arxiv id for a deterministic dense remap.
    df = df.sort("id").with_row_index(name="item_id", offset=1).with_columns(
        pl.col("item_id").cast(pl.Int64)
    )
    n_items = df.height

    # ---- item_id_map.json --------------------------------------------------
    print("STEP write item_id_map.json", flush=True)
    item_id_map = dict(zip(df["id"].to_list(), df["item_id"].to_list(), strict=True))
    with open(output / "item_id_map.json", "w") as f:
        json.dump({k: int(v) for k, v in item_id_map.items()}, f)

    # ---- papers.parquet ----------------------------------------------------
    print("STEP write papers.parquet", flush=True)
    papers = df.select(
        "item_id",
        pl.col("id").alias("arxiv_id"),
        "title",
        "abstract",
        "categories",
        "authors_parsed",
        "license",
        "update_date",
        "versions",
        "submitter",
    )
    papers.write_parquet(output / "papers.parquet", compression="zstd")
    print(
        f"  wrote {papers.height:,} rows → papers.parquet "
        f"({(output / 'papers.parquet').stat().st_size / 1024 / 1024:.0f} MiB)",
        flush=True,
    )

    # ---- heldout.parquet ---------------------------------------------------
    n_heldout = min(args.n_heldout, n_items)
    rng = np.random.default_rng(args.seed)
    heldout_ids = np.sort(
        rng.choice(np.arange(1, n_items + 1, dtype=np.int64), size=n_heldout, replace=False)
    )
    heldout_df = (
        pl.DataFrame(
            {"item_id": heldout_ids.tolist()}, schema={"item_id": pl.Int64}
        )
        .join(papers.select("item_id", "arxiv_id"), on="item_id", how="inner")
    )
    heldout_df.write_parquet(output / "heldout.parquet", compression="zstd")
    print(f"  wrote {heldout_df.height:,} held-out queries → heldout.parquet", flush=True)

    log = {
        "n_raw": int(n_raw),
        "n_filtered": int(n_filtered),
        "n_dropped_empty_abstract_or_dup": int(n_dropped),
        "n_items": int(n_items),
        "n_heldout": int(heldout_df.height),
        "seed": int(args.seed),
        "wall_clock_sec": round(time.monotonic() - overall_t0, 1),
    }
    with open(output / "prep_log.json", "w") as f:
        json.dump({"prep": log}, f, indent=2)
    print(f"ALL DONE prep in {log['wall_clock_sec']:.0f}s — n_items={n_items:,}", flush=True)
    return 0


# ----- encode_text -----------------------------------------------------------


def _encode_with_prefix(
    *,
    output_dir: Path,
    prefix: str,
    out_basename: str,
    item_ids: list[int],
    n_rows_out: int,  # N_items + 1 for items, N_heldout for queries
    encoder: str,
    text_template: str,
    description_chars: int,
    batch_size: int,
    device: str,
    num_workers: int,
    max_seq_length: int,
    truncate_dim: int,
) -> dict:
    """Shared encode pass for items and queries.

    For items: ``item_ids`` is range(1, N+1), `n_rows_out = N+1`, output
    is `[N+1, D]` indexed by item_id (row 0 is zero padding).

    For queries: ``item_ids`` is the held-out item_ids in heldout.parquet
    row order, `n_rows_out = N_heldout`, output is `[N_heldout, D]`
    indexed by heldout-row.

    Returns the meta dict that the caller writes alongside the .pt.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    papers_path = output_dir / "papers.parquet"
    if not papers_path.exists():
        raise FileNotFoundError(f"missing {papers_path} (run `prep` first)")

    print(f"STEP load papers + text-build (prefix={prefix!r})", flush=True)
    papers = pl.read_parquet(papers_path, columns=["item_id", "title", "abstract"])
    # Build a lookup item_id → (title, abstract). Polars-side join would also work;
    # since we know the row-order maps to dense item_ids 1..N, we just sort.
    papers = papers.sort("item_id")
    p_item_ids = papers["item_id"].to_numpy()
    p_titles = papers["title"].to_list()
    p_abstracts = papers["abstract"].to_list()
    # Build per-item_id arrays (indexed by 0..N-1; item_id-1).
    by_iid_title = {int(i): t for i, t in zip(p_item_ids.tolist(), p_titles)}
    by_iid_abstract = {int(i): a for i, a in zip(p_item_ids.tolist(), p_abstracts)}

    texts: list[str] = []
    valid_local_idx: list[int] = []  # positions in `item_ids` we actually encode
    for k, iid in enumerate(item_ids):
        title = by_iid_title.get(int(iid), "")
        abstract = by_iid_abstract.get(int(iid), "")
        if not abstract:
            continue
        text = text_template.format(
            prefix=prefix,
            title=title.strip(),
            abstract=abstract.strip()[:description_chars],
        )
        texts.append(text)
        valid_local_idx.append(k)
    n_to_encode = len(texts)
    print(f"  encoding {n_to_encode:,} texts (skipped empty: {len(item_ids) - n_to_encode})", flush=True)

    print(f"STEP load encoder {encoder} (truncate_dim={truncate_dim})", flush=True)
    if device == "cuda":
        torch.set_float32_matmul_precision("high")  # tensor cores on the proj layer
    model = SentenceTransformer(encoder, device=device, trust_remote_code=True)
    model.max_seq_length = max_seq_length

    d_native = int(model.get_sentence_embedding_dimension())
    if truncate_dim > d_native:
        raise ValueError(
            f"truncate_dim={truncate_dim} > native dim={d_native} for {encoder}"
        )

    # Length-sort for batch packing efficiency, then unsort at the end.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]

    tokenizer = model.tokenizer

    class _TextDS(torch.utils.data.Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    def _collate(batch):
        return tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        )

    loader = torch.utils.data.DataLoader(
        _TextDS(sorted_texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=_collate,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model.to(device).eval()
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    sorted_embs = torch.zeros((len(texts), truncate_dim), dtype=torch.float16)

    pos = 0
    for batch in tqdm(loader, total=len(loader), desc=f"encode/{prefix.strip(': ')}"):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device, dtype=autocast_dtype,
                           enabled=device == "cuda"),
        ):
            features = model(batch)
            embs = features["sentence_embedding"].float()
            # Matryoshka truncation: slice then re-normalize.
            embs = embs[:, :truncate_dim]
            embs = torch.nn.functional.normalize(embs, dim=-1)
        n = embs.shape[0]
        sorted_embs[pos : pos + n] = embs.detach().to(dtype=torch.float16, device="cpu")
        pos += n

    # Unsort: row at original index `order[k]` should hold sorted_embs[k].
    order_t = torch.tensor(order, dtype=torch.long)
    text_emb_local = torch.empty_like(sorted_embs)
    text_emb_local[order_t] = sorted_embs

    # Scatter into the final output tensor.
    out = torch.zeros((n_rows_out, truncate_dim), dtype=torch.float16)
    if out_basename == "text_emb":
        # Items: scatter by item_id.
        for j, k in enumerate(valid_local_idx):
            iid = int(item_ids[k])
            out[iid] = text_emb_local[j]
    else:
        # Queries: scatter by heldout-row index.
        for j, k in enumerate(valid_local_idx):
            out[k] = text_emb_local[j]

    out_dir = output_dir / "content"
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"{out_basename}.pt"
    meta_path = out_dir / f"{out_basename}.meta.json"

    import torch as _torch
    _torch.save(out, pt_path)

    meta = {
        "encoder": encoder,
        "dim_native": d_native,
        "dim_truncated": truncate_dim,
        "normalization": "l2_post_truncate",
        "prefix": prefix,
        "text_template": text_template,
        "description_chars": description_chars,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "n_rows": int(n_rows_out),
        "n_encoded": int(n_to_encode),
        "n_skipped_empty": int(len(item_ids) - n_to_encode),
        "shape": list(out.shape),
        "dtype": "float16",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"  wrote {pt_path} {tuple(out.shape)} + {meta_path.name}",
        flush=True,
    )
    return meta


def cmd_encode_text(args) -> int:
    output = Path(args.output_dir).expanduser()
    if not (output / "papers.parquet").exists():
        print(f"ERROR missing {output / 'papers.parquet'} (run `prep` first)", flush=True)
        return 1

    with open(output / "item_id_map.json") as f:
        n_items = len(json.load(f))
    item_ids = list(range(1, n_items + 1))

    overall_t0 = time.monotonic()
    meta = _encode_with_prefix(
        output_dir=output,
        prefix="search_document: ",
        out_basename="text_emb",
        item_ids=item_ids,
        n_rows_out=n_items + 1,
        encoder=args.encoder,
        text_template="{prefix}{title}. {abstract}",
        description_chars=args.description_chars,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        max_seq_length=args.max_seq_length,
        truncate_dim=args.truncate_dim,
    )
    print(
        f"ALL DONE encode_text in {time.monotonic() - overall_t0:.0f}s — "
        f"{meta['n_encoded']:,} encoded, dim={meta['dim_truncated']}",
        flush=True,
    )
    return 0


def cmd_encode_queries(args) -> int:
    output = Path(args.output_dir).expanduser()
    heldout_path = output / "heldout.parquet"
    if not heldout_path.exists():
        print(f"ERROR missing {heldout_path} (run `prep` first)", flush=True)
        return 1

    heldout = pl.read_parquet(heldout_path)
    item_ids = heldout["item_id"].to_list()

    overall_t0 = time.monotonic()
    meta = _encode_with_prefix(
        output_dir=output,
        prefix="search_query: ",
        out_basename="query_emb",
        item_ids=item_ids,
        n_rows_out=len(item_ids),
        encoder=args.encoder,
        text_template="{prefix}{title}. {abstract}",
        description_chars=args.description_chars,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        max_seq_length=args.max_seq_length,
        truncate_dim=args.truncate_dim,
    )
    print(
        f"ALL DONE encode_queries in {time.monotonic() - overall_t0:.0f}s — "
        f"{meta['n_encoded']:,} encoded, dim={meta['dim_truncated']}",
        flush=True,
    )
    return 0


# ----- attrs -----------------------------------------------------------------


def cmd_attrs(args) -> int:
    import numpy as np
    import torch

    output = Path(args.output_dir).expanduser()
    papers_path = output / "papers.parquet"
    heldout_path = output / "heldout.parquet"
    item_id_map_path = output / "item_id_map.json"
    for p in (papers_path, heldout_path, item_id_map_path):
        if not p.exists():
            print(f"ERROR missing {p} (run `prep` first)", flush=True)
            return 1

    overall_t0 = time.monotonic()
    log: dict = {}

    print("STEP load papers + heldout", flush=True)
    with open(item_id_map_path) as f:
        n_items = len(json.load(f))
    log["n_items"] = n_items
    print(f"  n_items = {n_items:,}", flush=True)

    papers = pl.read_parquet(
        papers_path,
        columns=[
            "item_id",
            "categories",
            "authors_parsed",
            "license",
            "update_date",
            "versions",
        ],
    ).sort("item_id")

    # ---- C0 main_category_top1 + cat_main_vocab ----------------------------
    print("STEP C0 main_category", flush=True)
    # categories field is space-separated, e.g. "cs.LG cs.CL stat.ML".
    leaf0 = (
        papers.with_columns(
            pl.col("categories").fill_null("").str.split(" ").alias("cat_list"),
        )
        .with_columns(
            pl.col("cat_list").list.first().fill_null("").alias("leaf0"),
        )
        .with_columns(
            pl.col("leaf0").map_elements(_category_to_main, return_dtype=pl.Utf8).alias("main0"),
        )
        .filter(pl.col("main0") != "")
    )
    main_freq = (
        leaf0.group_by("main0").len().sort("len", descending=True)
    )
    cat_main_names = main_freq["main0"].to_list()
    cat_main_vocab = {n: i for i, n in enumerate(cat_main_names)}
    with open(output / "cat_main_vocab.json", "w") as f:
        json.dump({"size": len(cat_main_vocab), "names": cat_main_names}, f, indent=2)
    log["cat_main_vocab_size"] = len(cat_main_vocab)
    print(f"  main-cat vocab size = {len(cat_main_vocab)}; top5 = {cat_main_names[:5]}", flush=True)

    # ---- C1 license bucket -------------------------------------------------
    print("STEP C1 license", flush=True)
    with open(output / "license_vocab.json", "w") as f:
        json.dump({"buckets": LICENSE_BUCKETS}, f, indent=2)
    log["license_vocab_size"] = len(LICENSE_BUCKETS)

    # ---- C4 author dense remap (over all papers, top-2 per paper) ---------
    print("STEP C4 authors (top-2 per paper, dense by global freq)", flush=True)
    # authors_parsed is a JSON-encoded string like '[["LastName","FirstName",""], ...]'
    def _extract_top2_authors(s: str | None) -> list[str]:
        if not s:
            return []
        try:
            arr = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return []
        out: list[str] = []
        for entry in arr[:2]:
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            last = (entry[0] or "").strip()
            first = (entry[1] or "").strip()
            suffix = (entry[2] if len(entry) > 2 else "").strip()
            key = f"{last}|{first}|{suffix}".lower()
            if key.strip("|"):
                out.append(key)
        return out

    pa = papers.with_columns(
        pl.col("authors_parsed")
        .map_elements(_extract_top2_authors, return_dtype=pl.List(pl.Utf8))
        .alias("auth_top2"),
    ).select("item_id", "auth_top2")

    auth_explode = pa.explode("auth_top2").filter(
        pl.col("auth_top2").is_not_null() & (pl.col("auth_top2") != "")
    )
    auth_freq = (
        auth_explode.group_by("auth_top2").len().sort("len", descending=True)
    )
    author_keys = auth_freq["auth_top2"].to_list()
    # 0-indexed dense; -1 reserved as padding.
    author_vocab = {k: i for i, k in enumerate(author_keys)}
    # Dump the keys as a list (the dict would be ~250 MB for 1.5M authors).
    with open(output / "author_vocab.json", "w") as f:
        json.dump({"size": len(author_vocab)}, f)
    # Dense remap via join (faster than per-row dict lookup).
    auth_remap = pl.DataFrame(
        {"auth_top2": author_keys, "author_id": list(range(len(author_keys)))},
        schema={"auth_top2": pl.Utf8, "author_id": pl.Int64},
    )
    pa_authors = (
        auth_explode.join(auth_remap, on="auth_top2", how="inner")
        .group_by("item_id", maintain_order=True)
        .agg(pl.col("author_id").head(2).alias("author_ids"))
    )
    log["author_vocab_size"] = len(author_vocab)
    print(f"  author vocab size = {len(author_vocab):,}", flush=True)

    # ---- C2 year + C3 versions -------------------------------------------
    print("STEP C2 year + C3 versions buckets", flush=True)

    def _versions_count(s: str | None) -> int:
        if not s:
            return 0
        try:
            arr = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return 0
        return len(arr) if isinstance(arr, list) else 0

    pyv = papers.with_columns(
        pl.col("update_date")
        .str.slice(0, 4)
        .cast(pl.Int64, strict=False)
        .alias("year_int"),
        pl.col("versions")
        .map_elements(_versions_count, return_dtype=pl.Int64)
        .alias("vcount"),
    ).with_columns(
        pl.col("year_int").map_elements(_year_to_bucket, return_dtype=pl.Int64).alias("year_id"),
        pl.col("vcount").map_elements(_versions_to_bucket, return_dtype=pl.Int64).alias("ver_id"),
    ).select("item_id", "year_id", "ver_id")

    # ---- C1 license bucket per paper -------------------------------------
    plic = papers.with_columns(
        pl.col("license").map_elements(_license_to_bucket, return_dtype=pl.Int64).alias("lic_id"),
    ).select("item_id", "lic_id")

    # ---- per-paper main-cat id -------------------------------------------
    main_remap = pl.DataFrame(
        {"main0": cat_main_names, "main_id": list(range(len(cat_main_names)))},
        schema={"main0": pl.Utf8, "main_id": pl.Int64},
    )
    pmain = (
        leaf0.select("item_id", "main0").join(main_remap, on="main0", how="left").select(
            "item_id",
            pl.col("main_id").fill_null(-1).cast(pl.Int64),
        )
    )

    # ---- assemble item_attrs_narrow [N+1, 5, 4] --------------------------
    print("STEP assemble item_attrs_narrow", flush=True)
    narrow_t = torch.full((n_items + 1, C_NARROW, A_MAX_NARROW), -1, dtype=torch.long)

    # Fast bulk fill for single-value clauses via numpy index assignment.
    # C0 main
    main_arr = pmain["main_id"].to_numpy()
    main_iid = pmain["item_id"].to_numpy()
    narrow_t[main_iid, 0, 0] = torch.from_numpy(main_arr.astype(np.int64))
    cov0 = int((narrow_t[1:, 0, 0] != -1).sum().item())

    # C1 license
    lic_arr = plic["lic_id"].to_numpy()
    lic_iid = plic["item_id"].to_numpy()
    narrow_t[lic_iid, 1, 0] = torch.from_numpy(lic_arr.astype(np.int64))
    # license "none" is its own bucket — coverage counts as ANY non-pad value.
    cov1 = int((narrow_t[1:, 1, 0] != -1).sum().item())

    # C2 year + C3 versions
    year_arr = pyv["year_id"].to_numpy()
    ver_arr = pyv["ver_id"].to_numpy()
    yv_iid = pyv["item_id"].to_numpy()
    narrow_t[yv_iid, 2, 0] = torch.from_numpy(year_arr.astype(np.int64))
    narrow_t[yv_iid, 3, 0] = torch.from_numpy(ver_arr.astype(np.int64))
    cov2 = int((narrow_t[1:, 2, 0] != -1).sum().item())
    cov3 = int((narrow_t[1:, 3, 0] != -1).sum().item())

    # C4 authors (up to 2 per paper). Loop, but only rows that have authors.
    a_iids = pa_authors["item_id"].to_list()
    a_lists = pa_authors["author_ids"].to_list()
    cov4 = 0
    for iid, alist in zip(a_iids, a_lists, strict=True):
        if not alist:
            continue
        cov4 += 1
        for j, v in enumerate(alist[:A_MAX_NARROW]):
            narrow_t[int(iid), 4, j] = int(v)

    coverage = {
        "c0_main": round(cov0 / max(n_items, 1), 4),
        "c1_license": round(cov1 / max(n_items, 1), 4),
        "c2_year": round(cov2 / max(n_items, 1), 4),
        "c3_versions": round(cov3 / max(n_items, 1), 4),
        "c4_author": round(cov4 / max(n_items, 1), 4),
    }
    log["narrow_coverage"] = coverage
    print(f"  per-clause coverage = {coverage}", flush=True)

    torch.save(narrow_t, output / "item_attrs_narrow.pt")
    clause_is_reverse_narrow = torch.tensor(
        [False, False, False, False, False], dtype=torch.bool
    )
    torch.save(clause_is_reverse_narrow, output / "clause_is_reverse_narrow.pt")
    print(
        f"  wrote item_attrs_narrow.pt {tuple(narrow_t.shape)} + clause_is_reverse_narrow.pt",
        flush=True,
    )

    # ---- wide bag (bloom): all leaf categories per paper -----------------
    print("STEP wide bag (leaf categories)", flush=True)
    leaves_long = (
        papers.with_columns(
            pl.col("categories").fill_null("").str.split(" ").alias("cat_list"),
        )
        .select("item_id", "cat_list")
        .explode("cat_list")
        .filter(pl.col("cat_list").is_not_null() & (pl.col("cat_list") != ""))
        .rename({"cat_list": "leaf"})
    )
    leaf_freq = leaves_long.group_by("leaf").len().sort("len", descending=True)
    # Drop long tail (count < args.wide_min_count). Default 50.
    leaf_freq_kept = leaf_freq.filter(pl.col("len") >= args.wide_min_count)
    wide_names = leaf_freq_kept["leaf"].to_list()
    wide_counts = leaf_freq_kept["len"].to_list()
    wide_vocab = {n: i for i, n in enumerate(wide_names)}
    with open(output / "wide_category_vocab.json", "w") as f:
        json.dump({"size": len(wide_vocab), "names": wide_names}, f, indent=2)
    torch.save(
        torch.tensor(wide_counts, dtype=torch.long),
        output / "wide_category_global_freq.pt",
    )
    log["wide_vocab_size"] = len(wide_vocab)
    log["wide_min_count"] = int(args.wide_min_count)
    print(
        f"  wide vocab size = {len(wide_vocab):,} (after min_count={args.wide_min_count})",
        flush=True,
    )

    # Per-paper bag: dedup leaves, sort by global freq desc, take top-32.
    wide_remap = pl.DataFrame(
        {"leaf": wide_names, "shelf_id": list(range(len(wide_names)))},
        schema={"leaf": pl.Utf8, "shelf_id": pl.Int64},
    )
    paper_bag = (
        leaves_long.join(wide_remap, on="leaf", how="inner")
        .unique(subset=["item_id", "shelf_id"])
        .group_by("item_id", maintain_order=True)
        .agg(pl.col("shelf_id").sort().head(WIDE_BAG_SIZE).alias("shelf_ids"))
    )

    print("STEP assemble item_attrs_wide", flush=True)
    wide_t = torch.full((n_items + 1, 1, WIDE_BAG_SIZE), -1, dtype=torch.long)
    bag_iids = paper_bag["item_id"].to_list()
    bag_lists = paper_bag["shelf_ids"].to_list()
    wide_cov = 0
    bag_size_hist = [0] * (WIDE_BAG_SIZE + 1)
    for iid, blist in zip(bag_iids, bag_lists, strict=True):
        bag = (blist or [])[:WIDE_BAG_SIZE]
        bag_size_hist[min(len(bag), WIDE_BAG_SIZE)] += 1
        if not bag:
            continue
        wide_cov += 1
        for j, v in enumerate(bag):
            wide_t[int(iid), 0, j] = int(v)
    log["wide_coverage"] = round(wide_cov / max(n_items, 1), 4)
    log["wide_bag_size_hist"] = bag_size_hist
    avg_bag = sum(i * c for i, c in enumerate(bag_size_hist)) / max(n_items, 1)
    print(
        f"  wide coverage = {log['wide_coverage']}; avg bag size = {avg_bag:.2f}",
        flush=True,
    )

    torch.save(wide_t, output / "item_attrs_wide.pt")
    print(f"  wrote item_attrs_wide.pt {tuple(wide_t.shape)}", flush=True)

    # ---- eval_split.parquet ----------------------------------------------
    print("STEP build eval_split.parquet (rare-biased wide sampling)", flush=True)
    heldout = pl.read_parquet(heldout_path)
    target_first = heldout["item_id"].to_list()
    n_users = len(target_first)
    print(f"  heldout users: {n_users:,}", flush=True)

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

    # ---- log -------------------------------------------------------------
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
        f"ALL DONE attrs in {log['wall_clock_sec']:.0f}s — "
        f"narrow_cov={coverage} wide_cov={log['wide_coverage']} "
        f"vocabs(main/lic/auth/wide)="
        f"{log['cat_main_vocab_size']}/{log['license_vocab_size']}/"
        f"{log['author_vocab_size']}/{log['wide_vocab_size']}",
        flush=True,
    )
    return 0


# ----- main ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ETL for the arxiv filter-bench dataset (HuggingFace open-index/open-arxiv).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- common encode args (shared by encode_text / encode_queries / all)
    def _add_encode_args(p):
        p.add_argument("--encoder", type=str, default="nomic-ai/nomic-embed-text-v1.5")
        p.add_argument("--truncate-dim", type=int, default=256)
        p.add_argument("--description-chars", type=int, default=1500)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--device", type=str, default="cuda")
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--max-seq-length", type=int, default=512)

    sp_dl = sub.add_parser("download", help="snapshot_download HF parquet shards into raw/")
    sp_dl.add_argument("--max-workers", type=int, default=8)
    sp_dl.set_defaults(func=cmd_download)

    sp_cv = sub.add_parser("convert", help="raw/ shards → processed/arxiv_papers.parquet")
    sp_cv.add_argument("--force", action="store_true")
    sp_cv.set_defaults(func=cmd_convert)

    sp_pp = sub.add_parser("prep", help="processed/ → bench-side <output-dir>/ files")
    sp_pp.add_argument("--output-dir", type=str, required=True)
    sp_pp.add_argument("--n-heldout", type=int, default=10000)
    sp_pp.add_argument("--seed", type=int, default=0)
    sp_pp.set_defaults(func=cmd_prep)

    sp_et = sub.add_parser("encode_text", help="encode item-side text embeddings")
    sp_et.add_argument("--output-dir", type=str, required=True)
    _add_encode_args(sp_et)
    sp_et.set_defaults(func=cmd_encode_text)

    sp_eq = sub.add_parser("encode_queries", help="encode held-out query-side text embeddings")
    sp_eq.add_argument("--output-dir", type=str, required=True)
    _add_encode_args(sp_eq)
    sp_eq.set_defaults(func=cmd_encode_queries)

    sp_at = sub.add_parser("attrs", help="build narrow + wide attrs + eval_split")
    sp_at.add_argument("--output-dir", type=str, required=True)
    sp_at.add_argument("--seed", type=int, default=0)
    sp_at.add_argument("--wide-min-count", type=int, default=50,
                       help="drop wide-bag categories with global count below this")
    sp_at.set_defaults(func=cmd_attrs)

    sp_al = sub.add_parser(
        "all", help="download → convert → prep → encode_text → encode_queries → attrs"
    )
    sp_al.add_argument("--output-dir", type=str, required=True)
    sp_al.add_argument("--n-heldout", type=int, default=10000)
    sp_al.add_argument("--seed", type=int, default=0)
    sp_al.add_argument("--max-workers", type=int, default=8)
    sp_al.add_argument("--wide-min-count", type=int, default=50)
    _add_encode_args(sp_al)
    sp_al.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
