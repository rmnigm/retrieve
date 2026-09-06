#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "polars>=1.0",
#   "pyarrow>=15",
#   "torch>=2.4",
#   "numpy>=1.26",
# ]
# ///
"""End-to-end ETL for the YFCC-10M filtered-ANN dataset.

Source: the NeurIPS'23 Big-ANN "filtered search" track set, served without
registration from
``https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/``
(URLs fixed in ``docs/plans/dataset-candidates.md`` §3.4). Six files,
2.97 GB total:

===============================  =============  ==================================
file                             bytes          content
===============================  =============  ==================================
``base.10M.u8bin``               1,920,000,008  10M x 192 uint8 CLIP vectors
``query.public.100K.u8bin``         19,200,008  100k x 192 uint8 query vectors
``base.metadata.10M.spmat``        945,683,840  item tag bags, CSR, 200,386 tags
``query.metadata.public.100K.spmat`` 1,907,024  query tag predicates, CSR
``GT.public.ibin``                   8,000,008  filtered GT, k=10 (ids + sq-L2)
``unfiltered.GT.public.ibin``       80,000,008  unfiltered GT, k=100
===============================  =============  ==================================

**This is the one dataset whose filtered ground truth is not ours.** The
organisers ship ``GT.public.ibin``: for each of the 100k queries, the 10
squared-L2-nearest base vectors *among the items whose tag bag contains
every query tag* (conjunctive AND over 1-2 tags). We keep it verbatim as
``gt_shipped.pt`` and validate our own exact filtered oracle against it
with :mod:`eval_datasets.yfcc_check_gt` (roadmap step E1's gate).

Subcommands::

    download   Fetch the six raw files into data/_raw/yfcc10m/ (resumable).
    convert    Validate every raw header against the format spec; write
               data/_raw/yfcc10m/processed/manifest.json (sizes + sha256).
    prep       Raw -> bench layout: item_id_map, fp16 embeddings, the full
               item tag CSR, heldout.parquet, gt_shipped.pt.
    attrs      Tag vocabulary + narrow clause tensor + eval_split.parquet.
    all        download -> convert -> prep -> attrs.

Examples::

    export RETRIEVE_DATA_ROOT=/workspace/data
    uv run yfcc all --output-dir data/yfcc10m
    uv run yfcc attrs --output-dir data/yfcc10m --max-tags 32

Layout produced (under ``$RETRIEVE_DATA_ROOT``, default ``<repo>/data``)::

    data/_raw/yfcc10m/                 <- the six upstream files + download.log
    data/_raw/yfcc10m/processed/       <- manifest.json (header facts, sha256)
    data/yfcc10m/                      <- the bench layout; see docs/system/datasets.md

Three deviations from the other loaders, all forced by the upstream data
and all recorded in ``docs/system/datasets.md``:

1. **The metric is Euclidean, not inner product.** The shipped GT ranks by
   squared L2 over the raw uint8 vectors. The harness's text path
   L2-normalises item and query embeddings and scores by inner product, so
   harness cells on this dataset measure *cosine*, and the harness's own
   oracle - not the shipped GT - is what its recall is measured against.
   Base-vector norms have a coefficient of variation of ~1.1 %, so the two
   orders are close but not identical. ``yfcc_check_gt.py`` is the bridge:
   it reproduces the shipped GT under exact squared L2.
2. **fp16 embeddings are lossless here.** uint8 values 0..255 are exactly
   representable in fp16 (and every partial sum of a 192-term squared-L2
   accumulation stays under 2**24, so fp32 arithmetic over them is exact).
   We therefore ship ``text_emb.pt`` fp16 only - a separate int8/uint8 code
   file would be a bit-for-bit redundant 1.9 GB, and SilverTorch quantises
   internally at build time anyway.
3. **The narrow clause tensor is a capped approximation of the true tag
   predicate** (§3.4 gotcha (a)). Items carry 10.8 tags on average with a
   1,517-tag tail; ``ExactAttributeFilter`` needs the whole bag inside one
   clause's ``A_max`` slots, which is not affordable. ``attrs`` caps at
   ``--max-tags`` (default 32) after restricting to the 7,910 tags that the
   100k queries actually ask about, keeping the most query-frequent first,
   and records the exact fidelity loss in ``prep_log.json``. The *full*
   uncapped bags are shipped as ``item_tags_csr.pt`` so that the gate check
   - and harness v2, if it grows a sparse-set filter - can use the true
   predicate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from eval_datasets.hf_io import raw_dir

# ----- upstream facts ---------------------------------------------------------

BASE_URL = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M"

#: filename -> expected size in bytes (probed 2026-09-05, re-verified 2026-09-06).
RAW_FILES: dict[str, int] = {
    "base.10M.u8bin": 1_920_000_008,
    "query.public.100K.u8bin": 19_200_008,
    "base.metadata.10M.spmat": 945_683_840,
    "query.metadata.public.100K.spmat": 1_907_024,
    "GT.public.ibin": 8_000_008,
    "unfiltered.GT.public.ibin": 80_000_008,
}

DIM = 192
N_BASE = 10_000_000
N_QUERY = 100_000
N_TAGS = 200_386
GT_K = 10
UNFILTERED_GT_K = 100

#: Two clauses, both holding the same (capped) tag bag: a query's first tag
#: goes in clause 0, its second in clause 1, and ExactAttributeFilter ANDs
#: the clauses - which is exactly the organisers' conjunctive tag predicate.
C_NARROW = 2
DEFAULT_MAX_TAGS = 32

#: The dim-suffixed content dir. YFCC is 192-d and has no Matryoshka variants.
CONTENT_SUBDIR = "content_d192"

ROOT = raw_dir("yfcc10m")
PROCESSED_DIR = ROOT / "processed"


# ----- upstream file readers --------------------------------------------------
#
# Reimplemented from big-ann-benchmarks' ``benchmark/dataset_io.py``
# (``xbin_mmap``, ``knn_result_read``, ``read_sparse_matrix_fields``) rather
# than imported, so the parse is pinned here and unit-testable on fixtures.


def read_u8bin(path: Path, *, expect_dim: int | None = None) -> np.memmap:
    """mmap a ``.u8bin``: ``uint32 n, uint32 d`` header then ``n*d`` uint8."""
    n, d = (int(x) for x in np.fromfile(path, dtype="uint32", count=2))
    size = path.stat().st_size
    if size != 8 + n * d:
        raise ValueError(f"{path}: size {size} != 8 + {n}*{d}")
    if expect_dim is not None and d != expect_dim:
        raise ValueError(f"{path}: dim {d} != expected {expect_dim}")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(n, d))


def read_knn_result(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``.ibin`` KNN result: header, then ``n*k`` int32 ids, then ``n*k``
    float32 distances. YFCC's distances are *squared* L2 over uint8 vectors,
    so they are exact integers stored as float32."""
    n, k = (int(x) for x in np.fromfile(path, dtype="uint32", count=2))
    size = path.stat().st_size
    if size != 8 + n * k * 8:
        raise ValueError(f"{path}: size {size} != 8 + {n}*{k}*8")
    ids = np.fromfile(path, dtype="int32", offset=8, count=n * k).reshape(n, k)
    dists = np.fromfile(path, dtype="float32", offset=8 + n * k * 4, count=n * k).reshape(n, k)
    return ids, dists


def read_spmat(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Read a ``.spmat`` CSR tag matrix -> ``(indptr [nrow+1] int64,
    indices [nnz] int32, ncol)``.

    Layout: ``int64 nrow, ncol, nnz``, then ``nrow+1`` int64 indptr, then
    ``nnz`` int32 column indices, then ``nnz`` float32 values. The values are
    all 1.0 (set membership), so they are validated and dropped.
    """
    with open(path, "rb") as f:
        nrow, ncol, nnz = (int(x) for x in np.fromfile(f, dtype="int64", count=3))
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        if indptr.shape[0] != nrow + 1:
            raise ValueError(f"{path}: truncated indptr")
        if int(indptr[-1]) != nnz:
            raise ValueError(f"{path}: indptr[-1]={int(indptr[-1])} != nnz={nnz}")
        indices = np.fromfile(f, dtype="int32", count=nnz)
        if indices.shape[0] != nnz:
            raise ValueError(f"{path}: truncated indices")
        data = np.fromfile(f, dtype="float32", count=nnz)
    expected = 24 + (nrow + 1) * 8 + nnz * 8
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: size {path.stat().st_size} != {expected}")
    if nnz and (indices.min() < 0 or indices.max() >= ncol):
        raise ValueError(f"{path}: column index out of range [0, {ncol})")
    if nnz and not np.all(data == 1.0):
        raise ValueError(f"{path}: expected all-1.0 set-membership values")
    return indptr, indices, int(ncol)


# ----- tag-bag capping (the §3.4 gotcha (a) decision) -------------------------


def build_tag_rank(
    q_indices: np.ndarray, n_tags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Dense-remap the tags the queries actually use, most-frequent first.

    Returns ``(vocab, rank)`` where ``vocab[d]`` is the upstream tag id of
    dense id ``d`` (ordered by query frequency descending, ties by upstream
    id ascending) and ``rank[t]`` is the dense id of upstream tag ``t``, or
    ``-1`` for a tag no query ever asks for.

    Restricting to the query vocabulary is free of semantic cost: a tag that
    appears in no query predicate can never change whether an item passes,
    and dropping the other 192,476 tags buys 29 % more room under the cap.
    """
    qfreq = np.bincount(q_indices, minlength=n_tags)
    used = np.nonzero(qfreq)[0]
    order = used[np.argsort(-qfreq[used], kind="stable")]
    rank = np.full(n_tags, -1, dtype=np.int64)
    rank[order] = np.arange(order.shape[0], dtype=np.int64)
    return order.astype(np.int64), rank


def cap_tag_bags(
    indptr: np.ndarray,
    indices: np.ndarray,
    rank: np.ndarray,
    max_tags: int,
) -> tuple[np.ndarray, dict]:
    """Per-item tag bag -> a dense ``[N, max_tags]`` int64 array, ``-1``-padded.

    Tags outside the query vocabulary (``rank == -1``) are dropped first;
    what is left is sorted by dense id ascending (= query frequency
    descending) and truncated to ``max_tags``. Truncation is *subtractive
    only*: the capped predicate passes a subset of what the true predicate
    passes, never a superset, so a capped-attrs oracle can lose ground-truth
    items but can never invent one.

    Returns ``(bags, stats)``; ``stats`` carries the numbers that
    ``prep_log.json`` records about how much the cap threw away.
    """
    n_rows = indptr.shape[0] - 1
    counts = np.diff(indptr)
    item_of = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
    dense = rank[indices]
    keep = dense >= 0
    ie = item_of[keep]
    de = dense[keep]
    # Group by item, then by dense id ascending. One stable argsort over a
    # single composite key is ~2x faster than lexsort at 10M x 11.
    stride = int(rank.max()) + 2
    order = np.argsort(ie * stride + de, kind="stable")
    ie = ie[order]
    de = de[order]
    starts = np.searchsorted(ie, np.arange(n_rows, dtype=np.int64), side="left")
    pos = np.arange(ie.shape[0], dtype=np.int64) - starts[ie]
    sel = pos < max_tags

    bags = np.full((n_rows, max_tags), -1, dtype=np.int64)
    bags[ie[sel], pos[sel]] = de[sel]

    restricted_counts = np.bincount(ie, minlength=n_rows)
    stats = {
        "n_items": int(n_rows),
        "max_tags": int(max_tags),
        "tag_entries_total": int(indices.shape[0]),
        "tag_entries_in_query_vocab": int(keep.sum()),
        "tag_entries_kept": int(sel.sum()),
        "tags_per_item_mean": round(float(counts.mean()), 4),
        "tags_per_item_max": int(counts.max()),
        "restricted_tags_per_item_mean": round(float(restricted_counts.mean()), 4),
        "restricted_tags_per_item_max": int(restricted_counts.max()),
        "items_uncapped_frac": round(float((restricted_counts <= max_tags).mean()), 6),
        "items_with_no_tags": int((restricted_counts == 0).sum()),
    }
    return bags, stats


def bags_to_narrow(bags: np.ndarray, n_clauses: int = C_NARROW):
    """``[N, K]`` capped bags -> the ``[N, C, K]`` int64 narrow clause tensor.

    Every clause holds the *same* bag: ``ExactAttributeFilter`` matches a
    clause when the query's value for it is anywhere in that clause's slots
    and ANDs the clauses, so putting query tag ``j`` in clause ``j`` gives
    the organisers' "item bag contains all query tags" predicate exactly.
    Queries with a single tag leave clause 1 at ``-1``, which the filter
    reads as "always pass".
    """
    import torch

    t = torch.from_numpy(np.ascontiguousarray(bags))
    return t.unsqueeze(1).expand(-1, n_clauses, -1).contiguous()


def gt_survives_cap(
    bags: np.ndarray,
    gt_ids: np.ndarray,
    query_tags_dense: np.ndarray,
) -> np.ndarray:
    """Per ``(query, gt-rank)`` entry: does that GT item still pass the
    *capped* predicate? Returns a ``[n_queries, k]`` bool array.

    ``query_tags_dense`` is ``[n_queries, C]`` with ``-1`` for absent tags.
    Because capping only removes items from a pass set, a query whose whole
    GT row survives (``.all(axis=1)``) has a capped-attrs exact top-k
    identical to the shipped GT. This is the fidelity number ``attrs``
    records.
    """
    ok = np.ones(gt_ids.shape, dtype=bool)
    for c in range(query_tags_dense.shape[1]):
        tq = query_tags_dense[:, c]
        act = tq >= 0
        if not act.any():
            continue
        gathered = bags[gt_ids[act].astype(np.int64)]  # [n_act, k, K]
        ok[act] &= (gathered == tq[act][:, None, None]).any(axis=-1)
    return ok


# ----- download ---------------------------------------------------------------


def _log(log_path: Path, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def _download_one(url: str, dest: Path, expected: int, log_path: Path) -> bool:
    """Resumable single-file fetch. Returns True on a verified-size file."""
    have = dest.stat().st_size if dest.exists() else 0
    if have == expected:
        _log(log_path, f"SKIP {dest.name} (already {have} bytes)")
        return True
    if have > expected:
        _log(log_path, f"TRUNCATE {dest.name}: {have} > {expected}, restarting")
        dest.unlink()
        have = 0
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(
            dest, "ab" if have else "wb"
        ) as out:
            while chunk := resp.read(1 << 22):
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        _log(log_path, f"FAIL {dest.name}: {e}")
        return False
    got = dest.stat().st_size
    ok = got == expected
    _log(
        log_path,
        f"{'OK' if ok else 'SIZE-MISMATCH'} {dest.name}: {got} bytes "
        f"(expected {expected}) in {time.monotonic() - t0:.1f}s",
    )
    return ok


def cmd_download(args) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "download.log"
    _log(log_path, f"=== yfcc download from {BASE_URL} into {ROOT}")
    failed: list[str] = []
    for name, expected in RAW_FILES.items():
        for attempt in range(1, args.retries + 1):
            if _download_one(f"{BASE_URL}/{name}", ROOT / name, expected, log_path):
                break
            _log(log_path, f"retry {attempt}/{args.retries} for {name}")
        else:
            failed.append(name)
    if failed:
        _log(log_path, f"DONE with failures: {failed}")
        return 1
    total = sum(RAW_FILES.values())
    _log(log_path, f"DONE all {len(RAW_FILES)} files, {total / 1e9:.2f} GB")
    return 0


# ----- convert ----------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def cmd_convert(args) -> int:
    """Validate every raw header against the documented format and record a
    manifest. There is no reshaping to do at this stage - the upstream files
    are already the dense/CSR forms `prep` consumes - so `convert` exists to
    make a corrupt or partial download fail loudly and early."""
    missing = [n for n in RAW_FILES if not (ROOT / n).exists()]
    if missing:
        print(f"ERROR missing raw files {missing} (run `yfcc download` first)", flush=True)
        return 1
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"base_url": BASE_URL, "files": {}}

    for name, expected in RAW_FILES.items():
        p = ROOT / name
        got = p.stat().st_size
        if got != expected:
            print(f"ERROR {name}: {got} bytes != expected {expected}", flush=True)
            return 1
        entry: dict = {"bytes": got}
        if args.sha256:
            t0 = time.monotonic()
            entry["sha256"] = _sha256(p)
            entry["sha256_sec"] = round(time.monotonic() - t0, 1)
        manifest["files"][name] = entry
        print(f"  {name}: {got:,} bytes OK", flush=True)

    base = read_u8bin(ROOT / "base.10M.u8bin", expect_dim=DIM)
    query = read_u8bin(ROOT / "query.public.100K.u8bin", expect_dim=DIM)
    if base.shape != (N_BASE, DIM) or query.shape != (N_QUERY, DIM):
        print(f"ERROR shapes {base.shape} / {query.shape}", flush=True)
        return 1
    b_indptr, b_indices, b_ncol = read_spmat(ROOT / "base.metadata.10M.spmat")
    q_indptr, q_indices, q_ncol = read_spmat(ROOT / "query.metadata.public.100K.spmat")
    if b_ncol != N_TAGS or q_ncol != N_TAGS:
        print(f"ERROR tag vocab {b_ncol} / {q_ncol} != {N_TAGS}", flush=True)
        return 1
    gt_ids, gt_d = read_knn_result(ROOT / "GT.public.ibin")
    ugt_ids, _ = read_knn_result(ROOT / "unfiltered.GT.public.ibin")
    if gt_ids.shape != (N_QUERY, GT_K) or ugt_ids.shape != (N_QUERY, UNFILTERED_GT_K):
        print(f"ERROR gt shapes {gt_ids.shape} / {ugt_ids.shape}", flush=True)
        return 1

    q_counts = np.diff(q_indptr)
    manifest["headers"] = {
        "base": {"n": int(base.shape[0]), "d": int(base.shape[1]), "dtype": "uint8"},
        "query": {"n": int(query.shape[0]), "d": int(query.shape[1]), "dtype": "uint8"},
        "base_tags": {
            "n_rows": int(b_indptr.shape[0] - 1),
            "n_tags": b_ncol,
            "nnz": int(b_indices.shape[0]),
            "tags_per_item_mean": round(float(np.diff(b_indptr).mean()), 4),
            "tags_per_item_max": int(np.diff(b_indptr).max()),
        },
        "query_tags": {
            "n_rows": int(q_indptr.shape[0] - 1),
            "nnz": int(q_indices.shape[0]),
            "tags_per_query_hist": np.bincount(q_counts).tolist(),
            "unique_query_tags": int(np.unique(q_indices).shape[0]),
        },
        "gt_filtered": {"n": int(gt_ids.shape[0]), "k": int(gt_ids.shape[1]),
                        "metric": "squared_l2", "d_max": float(gt_d.max())},
        "gt_unfiltered": {"n": int(ugt_ids.shape[0]), "k": int(ugt_ids.shape[1])},
    }
    with open(PROCESSED_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"ALL DONE convert — wrote {PROCESSED_DIR / 'manifest.json'}", flush=True)
    return 0


# ----- prep -------------------------------------------------------------------


def cmd_prep(args) -> int:
    import polars as pl
    import torch

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    content = output / CONTENT_SUBDIR
    content.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    log: dict = {}

    print("STEP item_id_map.json (identity: base row i -> dense id i+1)", flush=True)
    # 1-indexed dense ids, per docs/system/datasets.md's shared convention.
    # YFCC has no external item keys, so the map is the identity on row order.
    with open(output / "item_id_map.json", "w") as f:
        f.write("{")
        step = 1_000_000
        for start in range(0, N_BASE, step):
            stop = min(start + step, N_BASE)
            f.write(",".join(f'"{i}":{i + 1}' for i in range(start, stop)))
            if stop < N_BASE:
                f.write(",")
        f.write("}")
    log["n_items"] = N_BASE

    print("STEP embeddings uint8 -> fp16 (lossless: 0..255 is exact in fp16)", flush=True)
    base = read_u8bin(ROOT / "base.10M.u8bin", expect_dim=DIM)
    item_emb = torch.empty((N_BASE, DIM), dtype=torch.float16)
    chunk = 1_000_000
    for start in range(0, N_BASE, chunk):
        stop = min(start + chunk, N_BASE)
        # np.array(..) not asarray(..): a memmap slice is a read-only view and
        # torch.from_numpy warns on those.
        item_emb[start:stop] = torch.from_numpy(np.array(base[start:stop])).to(torch.float16)
    torch.save(item_emb, content / "text_emb.pt")
    print(f"  text_emb.pt {tuple(item_emb.shape)} fp16", flush=True)

    query = read_u8bin(ROOT / "query.public.100K.u8bin", expect_dim=DIM)
    query_emb = torch.from_numpy(np.array(query)).to(torch.float16)
    torch.save(query_emb, content / "query_emb.pt")
    print(f"  query_emb.pt {tuple(query_emb.shape)} fp16", flush=True)

    norms = item_emb[:: max(N_BASE // 4096, 1)].float().norm(dim=-1)
    log["base_norm"] = {
        "mean": round(float(norms.mean()), 3),
        "std": round(float(norms.std()), 3),
        "cv": round(float(norms.std() / norms.mean()), 5),
        "sample": int(norms.numel()),
    }
    # NB: deliberately *not* named text_emb.meta.json / query_emb.meta.json —
    # loaders.assert_arxiv_prefixes() reads those and demands the nomic
    # "search_document: " / "search_query: " prefixes, which YFCC has no
    # concept of. With them absent the loader logs a warning and skips.
    with open(content / "emb_provenance.json", "w") as f:
        json.dump(
            {
                "source": f"{BASE_URL}/base.10M.u8bin",
                "encoder": "CLIP (Zilliz descriptors, per big-ann-benchmarks)",
                "raw_dtype": "uint8",
                "stored_dtype": "float16",
                "lossless": True,
                "dim": DIM,
                "metric_upstream": "squared_l2",
                "metric_harness": "cosine (loaders.py L2-normalises then scores IP)",
                "base_norm": log["base_norm"],
            },
            f,
            indent=2,
        )
    del item_emb, query_emb

    print("STEP item_tags_csr.pt (full, uncapped tag bags)", flush=True)
    b_indptr, b_indices, b_ncol = read_spmat(ROOT / "base.metadata.10M.spmat")
    torch.save(
        {
            "indptr": torch.from_numpy(b_indptr.astype(np.int64)),
            "indices": torch.from_numpy(b_indices.astype(np.int32)),
            "n_items": N_BASE,
            "n_tags": b_ncol,
            "note": "upstream tag ids (0..n_tags-1); row i = item_id i+1",
        },
        output / "item_tags_csr.pt",
    )
    log["tag_csr"] = {"nnz": int(b_indices.shape[0]), "n_tags": b_ncol}

    print("STEP gt_shipped.pt + heldout.parquet", flush=True)
    gt_ids, gt_d = read_knn_result(ROOT / "GT.public.ibin")
    ugt_ids, ugt_d = read_knn_result(ROOT / "unfiltered.GT.public.ibin")
    q_indptr, q_indices, _ = read_spmat(ROOT / "query.metadata.public.100K.spmat")
    torch.save(
        {
            "format": "yfcc-shipped-gt-v1",
            "ids": torch.from_numpy(gt_ids.astype(np.int64)),
            "dists": torch.from_numpy(gt_d.astype(np.float32)),
            "unfiltered_ids": torch.from_numpy(ugt_ids.astype(np.int64)),
            "unfiltered_dists": torch.from_numpy(ugt_d.astype(np.float32)),
            "k": GT_K,
            "unfiltered_k": UNFILTERED_GT_K,
            "metric": "squared_l2",
            "id_space": "0-indexed base row == item_id - 1 == item_embs row",
            "predicate": "item tag bag contains every query tag (conjunctive AND)",
            "source": f"{BASE_URL}/GT.public.ibin",
            "provenance": "shipped by the NeurIPS'23 Big-ANN organisers, NOT computed here",
        },
        output / "gt_shipped.pt",
    )
    # heldout.parquet: the harness's text path takes its `filter_kind=none`
    # targets from here. The nearest *unfiltered* neighbour is the only
    # defensible single target for an image-similarity query set.
    heldout = pl.DataFrame(
        {
            "query_row": list(range(N_QUERY)),
            # 1-indexed item ids, matching item_id_map.json.
            "item_id": (ugt_ids[:, 0].astype(np.int64) + 1).tolist(),
            "n_query_tags": np.diff(q_indptr).astype(np.int64).tolist(),
        },
        schema={"query_row": pl.Int64, "item_id": pl.Int64, "n_query_tags": pl.Int64},
    )
    heldout.write_parquet(output / "heldout.parquet", compression="zstd")
    log["n_queries"] = N_QUERY
    log["query_tag_hist"] = np.bincount(np.diff(q_indptr)).tolist()

    log["wall_clock_sec"] = round(time.monotonic() - t0, 1)
    _merge_prep_log(output, "prep", log)
    print(f"ALL DONE prep in {log['wall_clock_sec']:.0f}s", flush=True)
    return 0


# ----- attrs ------------------------------------------------------------------


def cmd_attrs(args) -> int:
    import polars as pl
    import torch

    output = Path(args.output_dir).expanduser()
    csr_path = output / "item_tags_csr.pt"
    if not csr_path.exists():
        print(f"ERROR missing {csr_path} (run `yfcc prep` first)", flush=True)
        return 1
    t0 = time.monotonic()
    log: dict = {"max_tags": int(args.max_tags), "n_clauses": C_NARROW}

    print("STEP load tag CSR + query predicates", flush=True)
    csr = torch.load(str(csr_path), map_location="cpu", weights_only=False)
    b_indptr = csr["indptr"].numpy()
    b_indices = csr["indices"].numpy()
    q_indptr, q_indices, _ = read_spmat(ROOT / "query.metadata.public.100K.spmat")

    print("STEP tag vocabulary (query-asked tags, most frequent first)", flush=True)
    vocab, rank = build_tag_rank(q_indices, N_TAGS)
    qfreq = np.bincount(q_indices, minlength=N_TAGS)
    dfreq = np.bincount(b_indices, minlength=N_TAGS)
    with open(output / "tag_vocab.json", "w") as f:
        json.dump(
            {
                "size": int(vocab.shape[0]),
                "n_tags_upstream": N_TAGS,
                "order": "query frequency descending, ties by upstream id ascending",
                "upstream_ids": vocab.tolist(),
                "query_freq": qfreq[vocab].tolist(),
                "doc_freq": dfreq[vocab].tolist(),
            },
            f,
        )
    log["tag_vocab_size"] = int(vocab.shape[0])
    print(f"  {vocab.shape[0]:,} of {N_TAGS:,} tags are ever queried", flush=True)

    print(f"STEP cap tag bags at K={args.max_tags}", flush=True)
    bags, cap_stats = cap_tag_bags(b_indptr, b_indices, rank, args.max_tags)
    log["cap"] = cap_stats
    print(
        f"  kept {cap_stats['tag_entries_kept']:,} of "
        f"{cap_stats['tag_entries_total']:,} tag entries; "
        f"{100 * cap_stats['items_uncapped_frac']:.3f}% of items fit uncapped",
        flush=True,
    )

    print("STEP query_attrs_narrow (tag j -> clause j)", flush=True)
    q_counts = np.diff(q_indptr)
    qa = np.full((N_QUERY, C_NARROW), -1, dtype=np.int64)
    for c in range(C_NARROW):
        has = q_counts > c
        qa[has, c] = rank[q_indices[q_indptr[:-1][has] + c]]
    if (qa[:, 0] < 0).any():
        raise RuntimeError("a query has no first tag — query CSR is malformed")
    log["queries_with_2_tags"] = int((qa[:, 1] >= 0).sum())

    print("STEP fidelity of the capped predicate vs the shipped GT", flush=True)
    gt = torch.load(str(output / "gt_shipped.pt"), map_location="cpu", weights_only=False)
    gt_ids = gt["ids"].numpy()
    entry_ok = gt_survives_cap(bags, gt_ids, qa)
    survives = entry_ok.all(axis=1)
    per_entry = float(entry_ok.mean())
    log["gt_fidelity"] = {
        "queries_fully_preserved": int(survives.sum()),
        "queries_fully_preserved_frac": round(float(survives.mean()), 6),
        "gt_entries_preserved_frac": round(per_entry, 6),
        "note": (
            "capping is subtractive, so a query whose whole GT row survives has "
            "a capped-attrs exact top-k identical to the shipped GT; the rest "
            "have a strictly stricter predicate. The harness builds its oracle "
            "from these same capped attrs, so its recall stays internally exact "
            "— the shipped GT is checked separately by yfcc_check_gt.py against "
            "the uncapped item_tags_csr.pt."
        ),
    }
    print(
        f"  {100 * survives.mean():.3f}% of queries keep their whole GT row; "
        f"{100 * per_entry:.3f}% of GT entries survive",
        flush=True,
    )

    print("STEP write narrow tensors", flush=True)
    narrow = bags_to_narrow(bags, C_NARROW)
    torch.save(narrow, output / "item_attrs_narrow.pt")
    clause_is_reverse = torch.zeros(C_NARROW, dtype=torch.bool)
    torch.save(clause_is_reverse, output / "clause_is_reverse_narrow.pt")
    log["item_attrs_narrow_shape"] = list(narrow.shape)
    print(
        f"  item_attrs_narrow.pt {tuple(narrow.shape)} int64 "
        f"({narrow.numel() * 8 / 1e9:.2f} GB) + clause_is_reverse_narrow.pt [2] all-False",
        flush=True,
    )
    del narrow

    print("STEP eval_split.parquet", flush=True)
    heldout = pl.read_parquet(output / "heldout.parquet")
    eval_split = pl.DataFrame(
        {
            "target_id": heldout["item_id"].to_list(),
            "query_attrs_narrow": qa.tolist(),
            "n_query_tags": heldout["n_query_tags"].to_list(),
        },
        schema={
            "target_id": pl.Int64,
            "query_attrs_narrow": pl.List(pl.Int64),
            "n_query_tags": pl.Int64,
        },
    )
    eval_split.write_parquet(output / "eval_split.parquet", compression="zstd")
    print(f"  eval_split.parquet ({eval_split.height} rows)", flush=True)

    log["wall_clock_sec"] = round(time.monotonic() - t0, 1)
    _merge_prep_log(output, "attrs", log)
    print(f"ALL DONE attrs in {log['wall_clock_sec']:.0f}s", flush=True)
    return 0


def _merge_prep_log(output: Path, key: str, payload: dict) -> None:
    path = output / "prep_log.json"
    existing: dict = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing[key] = payload
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ----- all --------------------------------------------------------------------


def cmd_all(args) -> int:
    rc = cmd_download(argparse.Namespace(retries=args.retries))
    if rc:
        return rc
    rc = cmd_convert(argparse.Namespace(sha256=args.sha256))
    if rc:
        return rc
    rc = cmd_prep(argparse.Namespace(output_dir=args.output_dir))
    if rc:
        return rc
    return cmd_attrs(
        argparse.Namespace(output_dir=args.output_dir, max_tags=args.max_tags)
    )


# ----- main -------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="yfcc",
        description="YFCC-10M (Big-ANN filtered track) ETL for the retrieval bench",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("download", help="fetch the six raw files (resumable)")
    sp.add_argument("--retries", type=int, default=3)
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("convert", help="validate raw headers, write processed/manifest.json")
    sp.add_argument("--sha256", action="store_true", help="also hash each raw file (~30 s)")
    sp.set_defaults(func=cmd_convert)

    sp = sub.add_parser("prep", help="raw → embeddings / tag CSR / gt_shipped / heldout")
    sp.add_argument("--output-dir", required=True)
    sp.set_defaults(func=cmd_prep)

    sp = sub.add_parser("attrs", help="tag vocab + narrow clause tensor + eval_split")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument(
        "--max-tags",
        type=int,
        default=DEFAULT_MAX_TAGS,
        help=f"A_max per clause (default {DEFAULT_MAX_TAGS}); see §3.4 gotcha (a)",
    )
    sp.set_defaults(func=cmd_attrs)

    sp = sub.add_parser("all", help="download → convert → prep → attrs")
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--retries", type=int, default=3)
    sp.add_argument("--sha256", action="store_true")
    sp.add_argument("--max-tags", type=int, default=DEFAULT_MAX_TAGS)
    sp.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
