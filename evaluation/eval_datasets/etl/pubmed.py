#!/usr/bin/env -S uv run --script
"""End-to-end ETL for the PubMed + MedCPT filter-bench dataset (roadmap E2).

Source: the NCBI FTP MedCPT article-embedding release
``https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/`` — 38 chunks of

* ``embeds_chunk_{i}.npy``  — ``(N_i, 768)`` **float32** (verified from the npy
  header: ``descr='<f4'``; §3.8 inferred this from the file sizes and was right),
* ``pmids_chunk_{i}.json``  — the row-aligned PMID list,
* ``pubmed_chunk_{i}.json`` — ``{pmid: {"d": date, "t": title, "a": abstract,
  "m": mesh}}``.

Sizes as served on 2026-09-16 (``plan`` re-measures them): 110.35 GB of ``.npy``,
52.89 GB of chunk JSON, 0.42 GB of PMID lists — **163.7 GB raw** for
**35,920,666 articles**. ``m`` is a ``|``-separated list of ``descriptor!qualifier``
entries where a trailing ``*`` marks a major topic, e.g.::

    "humans!|rectal neoplasms!|rectal neoplasms*|rectal neoplasms!therapy|"

Journal and language are **not** in the chunk JSON; they come from a join
against the MEDLINE baseline (``https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/``,
1334 × ``pubmed26n*.xml.gz`` ≈ 52 GB, each with a published ``.md5``).
MeSH tree-top category letters come from the MeSH descriptor file
(``xmlmesh/desc2026.gz``, 17 MB).

Subcommands::

    plan            Dry run: HEAD every remote file, compute rows, peak disk and
                    wall time for a given --keep-items / --prefetch. No download.
    download        Resumable parallel mirror of any raw source (``--what pmids``
                    is the 0.42 GB the streaming convert needs up front).
    verify          Structural check of staged shards (+ MEDLINE md5).
    medline         Stream the MEDLINE baseline → pmid/journal/language parquet.
    convert         **Streaming**: PMID lists → item_id_map.json; then shard by
                    shard (fetch → parse → fold → delete) → per-shard article
                    parquet and ``content_d768/text_emb_shard_NN.pt`` +
                    ``shard_index.json`` at the encoder's **native** 768 dims.
    attrs           Article parquet + MEDLINE join → item_attrs_narrow.pt,
                    clause_is_reverse_narrow.pt, vocab JSONs, heldout.parquet,
                    eval_split.parquet.
    queries         Build the query sets (held-out articles; NFCorpus if staged).
    encode_queries  Encode query text with ``ncbi/MedCPT-Query-Encoder``
                    (``--device cuda``) → content_d768/query_emb.pt.

**No dimensionality reduction.** Per the user decision of 2026-09-06 every
dataset is benchmarked at its encoder's native dim; there is no PCA step here
and none is planned.

**Why streaming, and what it costs (docs/system/datasets.md § pubmed).** Raw
(163.7 GB) plus the fp16 item matrix (35.9 M × 768 × 2 B = 55.2 GB) would be
219 GB before the MEDLINE join — more than the overlay can spare next to anything
else. ``convert`` therefore never holds the raw mirror whole: it fetches the
0.42 GB of PMID lists first (they fix the id map), then walks the shards one at a
time — download the next shard while parsing this one (``--prefetch``), gather the
kept rows, L2-normalise, cast to fp16, ``torch.save`` them as *their own* output
shard, and ``--delete-raw`` the input. The peak is the processed output plus
``1 + prefetch`` raw shards, never the mirror. There is no accumulator and no
second copy: the sharded layout (``shard_index.json`` +
``text_emb_shard_*.pt``) is what ``eval_datasets.layout.load_sharded`` already
reads for the synthetic arXiv catalogs.

**Slice, not the whole.** ``--keep-items N`` keeps exactly ``N`` articles chosen
by a seeded hash of the PMID (``select_pmids``): the same set whatever the shard
order, spread uniformly over 1781–2024 rather than "the oldest ``N``". The full
36 M does **not** fit the harness on one 80 GB A100 at 768-d — items are held
fp32 on the device (110 GB) — so the slice is the target, not a stopgap.

Layout (under ``$RETRIEVE_DATA_ROOT``, default ``<repo>/data``)::

    data/_raw/pubmed/                       pmids_chunk_*.json (kept), the shard in flight
    data/_raw/pubmed/medline_baseline/      pubmed26n*.xml.gz (+ .md5), streamed
    data/pubmed-medcpt/                     the bench-side output dir
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from eval_datasets.common import synthesize_qa_narrow
from eval_datasets.hub import raw_dir

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEDCPT_BASE = "https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings"
MEDLINE_BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
MESH_DESC_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.gz"

N_CHUNKS = 38  # embeds_chunk_0 .. embeds_chunk_37
N_MEDLINE_FILES = 1334  # pubmed26n0001 .. pubmed26n1334
EMB_DIM_NATIVE = 768

ROOT = raw_dir("pubmed")
MEDLINE_DIR = ROOT / "medline_baseline"
MESH_DESC_PATH = ROOT / "mesh_desc2026.xml.gz"

# Narrow attribute layout — same shape as goodreads/arxiv (5 clauses, A_max=4).
#
#   C0 mesh        multi-valued OR over the top-V MeSH descriptor vocab, K=4
#   C1 mesh_cat    MeSH tree-top category letter of C0's first (rarest) term
#   C2 year        publication-year bucket
#   C3 journal     top-`--journal-vocab` MEDLINE journal abbreviation, REVERSE
#   C4 has_abstract 0/1
#
# dataset-candidates.md §4.1 fixes this layout. `language` is parsed and stored
# in articles.parquet + lang_vocab.json but is not one of the five clauses.
C_NARROW = 5
A_MAX_NARROW = 4
CLAUSE_NAMES = ["mesh", "mesh_cat", "year", "journal", "has_abstract"]
CLAUSE_IS_REVERSE = [False, False, False, True, False]

# C2 year buckets. PubMed spans 1781..now; the mass is post-1975.
YEAR_BUCKET_EDGES = [1975, 1990, 2000, 2010, 2015, 2020]
YEAR_BUCKET_NAMES = [
    "<1975",
    "1975-1989",
    "1990-1999",
    "2000-2009",
    "2010-2014",
    "2015-2019",
    ">=2020",
]

_QUERY_SCHEMA = {"query_id": pl.Utf8, "text": pl.Utf8, "target_id": pl.Int64}

# MeSH tree-top category letters (the first character of a tree number).
MESH_CATEGORIES = list("ABCDEFGHIJKLMNVZ")

# `plan`'s per-row allowance for the zstd article parquet (pmid, year, has_abstract,
# mesh list, title). Measured 55.0 B/row on the real chunk 37 (20,947,498 B for 380,761
# rows, 2026-09-16, the E2 record in docs/plans/dataset-candidates.md); 60 keeps a margin.
ARTICLE_PARQUET_BYTES_PER_ROW = 60
# `plan`'s allowance for the MEDLINE join output (`medline/*.parquet`, pmid + journal +
# language for every citation), an upper bound at ~30 B/row zstd over 40 M citations.
MEDLINE_PARQUET_BYTES = 1_200_000_000


# ---------------------------------------------------------------------------
# Small parsing helpers (pure functions — these are what tests pin down)
# ---------------------------------------------------------------------------


def parse_mesh_field(m: str | None) -> list[str]:
    """``"humans!|rectal neoplasms*|rectal neoplasms!therapy|"`` → descriptors.

    Entries are ``|``-separated ``descriptor!qualifier`` pairs; a trailing ``*``
    marks a major topic. We keep the descriptor only, lower-cased and
    de-duplicated, in first-seen order.
    """
    if not m:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for entry in m.split("|"):
        e = entry.strip()
        if not e:
            continue
        desc = e.split("!", 1)[0]
        desc = desc.rstrip("*").strip().lower()
        if desc and desc not in seen:
            seen.add(desc)
            out.append(desc)
    return out


def parse_year(d: str | None) -> int | None:
    """``"20071115"`` → ``2007``. Returns None for anything unparseable."""
    if not d:
        return None
    s = str(d).strip()
    if len(s) < 4 or not s[:4].isdigit():
        return None
    y = int(s[:4])
    return y if 1500 <= y <= 2100 else None


def year_to_bucket(y: int | None) -> int:
    """Map a year to ``0..len(YEAR_BUCKET_EDGES)`` or ``-1`` if missing."""
    if y is None:
        return -1
    for i, edge in enumerate(YEAR_BUCKET_EDGES):
        if y < edge:
            return i
    return len(YEAR_BUCKET_EDGES)


def cap_mesh_by_rarity(
    descriptors: list[str],
    vocab: dict[str, int],
    freq: np.ndarray,
    k: int = A_MAX_NARROW,
) -> list[int]:
    """Cap an article's MeSH bag to the ``k`` **globally rarest** descriptors.

    dataset-candidates.md §4.1 asks for a K=4 multi-valued OR clause over a
    top-V vocabulary with a MeSH-heading pass rate in the 0.01–5 % range. The
    query-side clause value is
    :func:`eval_datasets.common.synthesize_qa_narrow`'s *first non-pad* entry,
    so the order inside the row decides selectivity: sorting by **ascending**
    global frequency puts the rarest heading first and keeps the filter
    selective, where a frequency-descending order would always hand the query
    "humans" (a ~40 % pass rate). Ties break on the vocab id so the result is
    deterministic. Descriptors outside ``vocab`` are dropped.
    """
    ids = [vocab[d] for d in descriptors if d in vocab]
    if not ids:
        return []
    ids = sorted(set(ids), key=lambda i: (int(freq[i]), i))
    return ids[:k]


def mesh_category_of(tree_numbers: list[str]) -> int:
    """First tree number's leading letter → an index into MESH_CATEGORIES."""
    for tn in tree_numbers:
        c = tn.strip()[:1].upper()
        if c in MESH_CATEGORIES:
            return MESH_CATEGORIES.index(c)
    return -1


def pmid_hash(pmids: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64's finaliser over the PMIDs, keyed by ``seed`` — a fixed, order-free
    pseudo-random rank per article (uint64). Used by ``select_pmids``."""
    with np.errstate(over="ignore"):
        z = np.asarray(pmids, dtype=np.uint64) + np.uint64(seed + 1) * np.uint64(
            0x9E3779B97F4A7C15
        )
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def select_pmids(pmids: np.ndarray, keep_items: int | None, seed: int = 0) -> np.ndarray:
    """``[len(pmids)]`` bool: the ``keep_items`` articles with the smallest
    ``pmid_hash`` (ties on the PMID itself), or everything when ``keep_items`` is
    ``None`` / not smaller than the catalog. The choice depends only on the PMID set
    and the seed — not on shard order, chunking or which shards have been fetched —
    so a resumed or re-sharded run keeps the same articles, and the slice is spread
    uniformly over the whole 1781–2024 catalog instead of being its oldest prefix."""
    n = int(pmids.shape[0])
    if keep_items is None or keep_items >= n:
        return np.ones(n, dtype=bool)
    if keep_items <= 0:
        raise ValueError("keep_items must be positive")
    order = np.lexsort((pmids, pmid_hash(pmids, seed)))
    keep = np.zeros(n, dtype=bool)
    keep[order[:keep_items]] = True
    return keep


# ---------------------------------------------------------------------------
# download — resumable, parallel, checksum-verified where NCBI publishes one
# ---------------------------------------------------------------------------


def _remote_size(url: str) -> int:
    """``Content-Length`` of a HEAD, or 0 when the server does not say."""
    try:
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, timeout=60) as r:
            return int(r.headers.get("Content-Length", "0"))
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return 0


_RE_NPY_SHAPE = re.compile(rb"'shape':\s*\((\d+),\s*(\d+)\)")


def _remote_npy_shape(url: str) -> tuple[int, int] | None:
    """``(rows, cols)`` from the first 256 bytes of a remote ``.npy`` (a Range GET)."""
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-255"})
        with urllib.request.urlopen(req, timeout=60) as r:
            head = r.read(256)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    m = _RE_NPY_SHAPE.search(head)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _fetch_one(url: str, dest: Path, *, attempts: int = 8, log: Path | None = None) -> bool:
    """Resumable single-file GET. Returns True when ``dest`` holds the full file.

    Uses a ``Range`` header against whatever is already on disk, so a killed run
    picks up where it stopped. The remote length comes from a HEAD; a local file
    already at that length is a no-op.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n"
        if log is not None:
            with open(log, "a") as f:
                f.write(line)
        print(line.rstrip(), flush=True)

    total = _remote_size(url)
    if total == 0:
        _log(f"HEADFAIL {dest.name}")

    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if total and have == total:
            _log(f"DONE {dest.name} {have}")
            return True
        if total and have > total:  # corrupt / stale — start over
            dest.unlink()
            have = 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab") as f:
                while True:
                    buf = r.read(8 << 20)
                    if not buf:
                        break
                    f.write(buf)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            _log(f"RETRY {dest.name} attempt={attempt} {type(e).__name__}: {e}")
            time.sleep(min(60, 5 * attempt))
            continue
        have = dest.stat().st_size if dest.exists() else 0
        if not total or have == total:
            _log(f"OK {dest.name} {have}")
            return True
        _log(f"SHORT {dest.name} {have}/{total} attempt={attempt}")
        time.sleep(min(60, 5 * attempt))
    _log(f"FAIL {dest.name}")
    return False


def _md5(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 — NCBI publishes md5, not our choice
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_shards(spec: str | None) -> list[int]:
    """``"0-3,7"`` → ``[0, 1, 2, 3, 7]``; ``None`` → every chunk."""
    if not spec:
        return list(range(N_CHUNKS))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _shard_paths(i: int) -> dict[str, tuple[str, Path]]:
    """``{"pmids" | "content" | "embeds": (url, local path)}`` of chunk ``i``."""
    names = {
        "pmids": f"pmids_chunk_{i}.json",
        "content": f"pubmed_chunk_{i}.json",
        "embeds": f"embeds_chunk_{i}.npy",
    }
    return {k: (f"{MEDCPT_BASE}/{n}", ROOT / n) for k, n in names.items()}


def _medcpt_urls(shards: list[int], what: str) -> list[tuple[str, Path]]:
    parts = {
        "all": ("pmids", "content", "embeds"),
        "meta": ("pmids", "content"),
        "pmids": ("pmids",),
        "embeds": ("embeds",),
    }[what]
    return [_shard_paths(i)[p] for i in shards for p in parts]


def cmd_download(args) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    log = ROOT / "download.log"
    shards = _parse_shards(args.shards)
    jobs: list[tuple[str, Path]] = []

    if args.what in ("all", "embeds", "meta", "pmids"):
        jobs += _medcpt_urls(shards, args.what)
    if args.what in ("all", "mesh"):
        jobs.append((MESH_DESC_URL, MESH_DESC_PATH))
    if args.what in ("all", "medline"):
        n = args.medline_files or N_MEDLINE_FILES
        for i in range(1, n + 1):
            name = f"pubmed26n{i:04d}.xml.gz"
            jobs.append((f"{MEDLINE_BASE}/{name}", MEDLINE_DIR / name))
            jobs.append((f"{MEDLINE_BASE}/{name}.md5", MEDLINE_DIR / f"{name}.md5"))

    print(f"START download what={args.what} files={len(jobs)} parallel={args.parallel}", flush=True)
    t0 = time.monotonic()
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(_fetch_one, u, p, log=log): p for u, p in jobs}
        for fut in as_completed(futs):
            n_ok += bool(fut.result())
    total_bytes = sum(p.stat().st_size for _, p in jobs if p.exists())
    print(
        f"DONE download {n_ok}/{len(jobs)} files, {total_bytes / 1e9:.1f} GB "
        f"in {time.monotonic() - t0:.0f}s",
        flush=True,
    )
    return 0 if n_ok == len(jobs) else 1


def cmd_verify(args) -> int:
    """Structural + checksum verification of the raw mirror.

    NCBI publishes **no** checksums for the MedCPT embedding directory, so the
    shards are verified structurally instead: the ``.npy`` header must parse,
    its dtype must be float32, its width 768, and its row count must equal
    ``len(pmids_chunk_i.json)``. That catches every truncation we can produce.
    The MEDLINE baseline *does* publish ``.md5`` and those are checked.
    """
    shards = _parse_shards(args.shards)
    bad: list[str] = []
    for i in shards:
        npy = ROOT / f"embeds_chunk_{i}.npy"
        pm = ROOT / f"pmids_chunk_{i}.json"
        if not npy.exists() or not pm.exists():
            continue
        try:
            arr = np.load(npy, mmap_mode="r")
        except (ValueError, OSError) as e:
            bad.append(f"chunk {i}: npy unreadable ({e})")
            continue
        n_pmids = len(json.loads(pm.read_text()))
        if arr.dtype != np.float32:
            bad.append(f"chunk {i}: dtype {arr.dtype} != float32")
        if arr.shape[1] != EMB_DIM_NATIVE:
            bad.append(f"chunk {i}: width {arr.shape[1]} != {EMB_DIM_NATIVE}")
        if arr.shape[0] != n_pmids:
            bad.append(f"chunk {i}: {arr.shape[0]} rows != {n_pmids} pmids")
        print(f"  chunk {i}: {arr.shape} {arr.dtype} pmids={n_pmids}", flush=True)

    if args.medline and MEDLINE_DIR.exists():
        for md5f in sorted(MEDLINE_DIR.glob("*.md5")):
            gz = md5f.with_suffix("")
            if not gz.exists():
                continue
            want = md5f.read_text().strip().rsplit("=", 1)[-1].strip()
            got = _md5(gz)
            if want != got:
                bad.append(f"{gz.name}: md5 {got} != {want}")

    for b in bad:
        print(f"BAD {b}", flush=True)
    print(f"DONE verify — {len(bad)} problem(s)", flush=True)
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# plan — the disk and wall-time arithmetic, from the server's own numbers
# ---------------------------------------------------------------------------


def plan_budget(
    sizes: dict[int, dict[str, int]],
    rows: dict[int, int],
    *,
    keep_items: int | None,
    prefetch: int,
    mbps: float,
    medline_bytes: int = 0,
    parse_s_per_shard: float = 30.0,
) -> dict:
    """Peak disk and wall time of a streaming ``convert`` — pure arithmetic over the
    remote sizes ``{shard: {"pmids", "content", "embeds"}}`` and the per-shard row
    counts, so it is testable without a network and re-derivable by anyone.

    Peak disk = the finished fp16 shards + the article parquets + the PMID lists that
    stay behind + ``1 + prefetch`` raw shards in flight (the largest ones, an upper
    bound) + the MEDLINE parquet when the baseline is included. Wall time = every raw
    byte over ``mbps`` (each shard must be scanned even for a slice: the slice is a
    hash of the PMID, not a prefix) + the JSON parses, which overlap the next shard's
    download only when ``prefetch > 0``."""
    n_rows = int(sum(rows.values()))
    kept = n_rows if keep_items is None else min(int(keep_items), n_rows)
    raw = {k: int(sum(s[k] for s in sizes.values())) for k in ("pmids", "content", "embeds")}
    shard_bytes = [s["content"] + s["embeds"] for s in sizes.values()]
    in_flight = int(sum(sorted(shard_bytes, reverse=True)[: 1 + prefetch]))
    fp16 = kept * EMB_DIM_NATIVE * 2
    parquet = kept * ARTICLE_PARQUET_BYTES_PER_ROW
    medline_parquet = MEDLINE_PARQUET_BYTES if medline_bytes else 0
    peak = fp16 + parquet + raw["pmids"] + in_flight + medline_parquet
    raw_total = sum(raw.values())
    download_s = (raw_total + medline_bytes) / (mbps * 1e6)
    parse_s = parse_s_per_shard * len(sizes)
    wall_s = download_s + (parse_s_per_shard if prefetch else parse_s)
    return {
        "n_shards": len(sizes),
        "n_rows": n_rows,
        "keep_items": kept,
        "raw_bytes": {**raw, "medline": int(medline_bytes), "total": int(raw_total + medline_bytes)},  # noqa: E501
        "processed_bytes": {
            "text_emb_fp16": int(fp16),
            "article_parquet_est": int(parquet),
            "medline_parquet_est": int(medline_parquet),
        },
        "in_flight_raw_bytes": in_flight,
        "prefetch": int(prefetch),
        "peak_disk_bytes": int(peak),
        "mbps": float(mbps),
        "download_s": round(download_s),
        "parse_s_serial": round(parse_s),
        "wall_s_est": round(wall_s),
        "fp32_on_device_bytes": int(kept * EMB_DIM_NATIVE * 4),
    }  # fmt: skip


def _probe_mbps(url: str, n_bytes: int = 64 << 20) -> float:
    """Download the first ``n_bytes`` of ``url`` and report MB/s (0.0 on failure)."""
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n_bytes - 1}"})
    t0 = time.monotonic()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            while True:
                buf = r.read(4 << 20)
                if not buf:
                    break
                got += len(buf)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0.0
    return got / 1e6 / max(time.monotonic() - t0, 1e-6)


def cmd_plan(args) -> int:
    shards = _parse_shards(args.shards)
    print(f"STEP HEAD {3 * len(shards)} MedCPT files + npy headers", flush=True)
    sizes: dict[int, dict[str, int]] = {}
    rows: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        size_f = {
            (i, k): ex.submit(_remote_size, url)
            for i in shards
            for k, (url, _) in _shard_paths(i).items()
        }
        shape_f = {i: ex.submit(_remote_npy_shape, _shard_paths(i)["embeds"][0]) for i in shards}
        for i in shards:
            sizes[i] = {k: size_f[i, k].result() for k in ("pmids", "content", "embeds")}
            shape = shape_f[i].result()
            if shape is None or shape[1] != EMB_DIM_NATIVE:
                print(f"ERROR chunk {i}: npy header {shape} unreadable or not {EMB_DIM_NATIVE}-d")
                return 1
            rows[i] = shape[0]
            if any(v == 0 for v in sizes[i].values()):
                print(f"ERROR chunk {i}: a HEAD returned no Content-Length: {sizes[i]}")
                return 1
    medline_bytes = 0
    if args.medline:
        print(f"STEP HEAD {N_MEDLINE_FILES} MEDLINE baseline files", flush=True)
        urls = [f"{MEDLINE_BASE}/pubmed26n{i:04d}.xml.gz" for i in range(1, N_MEDLINE_FILES + 1)]
        with ThreadPoolExecutor(max_workers=16) as ex:
            medline_bytes = sum(ex.map(_remote_size, urls))
    mbps = args.mbps or _probe_mbps(_shard_paths(shards[0])["embeds"][0])
    if mbps <= 0:
        print("ERROR could not measure the download rate; pass --mbps", flush=True)
        return 1
    budget = plan_budget(
        sizes, rows, keep_items=args.keep_items, prefetch=args.prefetch, mbps=mbps,
        medline_bytes=medline_bytes,
    )  # fmt: skip
    budget["per_shard"] = {str(i): {**sizes[i], "rows": rows[i]} for i in shards}
    gb = 1e9
    rb, pb = budget["raw_bytes"], budget["processed_bytes"]
    lines = [
        f"rows {budget['n_rows']:,} over {budget['n_shards']} shards; "
        f"keep {budget['keep_items']:,}",
        f"raw: npy {rb['embeds'] / gb:.2f} GB + json {rb['content'] / gb:.2f} GB + pmids "
        f"{rb['pmids'] / gb:.2f} GB + medline {medline_bytes / gb:.2f} GB = "
        f"{rb['total'] / gb:.2f} GB",
        f"processed: fp16 shards {pb['text_emb_fp16'] / gb:.2f} GB, article parquet "
        f"~{pb['article_parquet_est'] / gb:.2f} GB, medline parquet "
        f"~{pb['medline_parquet_est'] / gb:.2f} GB",
        f"in flight: {1 + args.prefetch} raw shard(s) = "
        f"{budget['in_flight_raw_bytes'] / gb:.2f} GB",
        f"PEAK DISK ~{budget['peak_disk_bytes'] / gb:.1f} GB; "
        f"WALL ~{budget['wall_s_est'] / 3600:.2f} h at {mbps:.1f} MB/s "
        f"(download {budget['download_s'] / 3600:.2f} h)",
        f"harness: items fp32 on device = {budget['fp32_on_device_bytes'] / gb:.1f} GB",
    ]
    print("\n".join(lines), flush=True)
    if args.report:
        Path(args.report).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).expanduser().write_text(json.dumps(budget, indent=2))
        print(f"wrote {args.report}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# medline — stream the baseline into a pmid → (journal, language) parquet
# ---------------------------------------------------------------------------

_RE_PMID = re.compile(rb"<PMID[^>]*>(\d+)</PMID>")
_RE_TA = re.compile(rb"<MedlineTA>(.*?)</MedlineTA>", re.S)
_RE_LANG = re.compile(rb"<Language>(.*?)</Language>")


def parse_medline_gz(path: Path) -> tuple[list[int], list[str], list[str]]:
    """Extract ``(pmid, journal_abbrev, language)`` from one baseline file.

    Splits on ``</PubmedArticle>`` and takes the *first* ``<PMID>``,
    ``<MedlineTA>`` and ``<Language>`` in each record — the first PMID of a
    record is the article's own (later ones live inside ``<CommentsCorrections>``
    / ``<ReferenceList>``), and MedlineTA is the journal title abbreviation.
    Regex rather than ElementTree: we need three scalars out of 52 GB of XML
    and iterparse is ~4× slower for that.
    """
    pmids: list[int] = []
    journals: list[str] = []
    langs: list[str] = []
    with gzip.open(path, "rb") as f:
        buf = f.read()
    for rec in buf.split(b"</PubmedArticle>"):
        m = _RE_PMID.search(rec)
        if not m:
            continue
        ta = _RE_TA.search(rec)
        lg = _RE_LANG.search(rec)
        pmids.append(int(m.group(1)))
        journals.append(ta.group(1).decode("utf-8", "replace").strip() if ta else "")
        langs.append(lg.group(1).decode("utf-8", "replace").strip() if lg else "")
    return pmids, journals, langs


def _medline_worker(args_tuple) -> tuple[str, int]:
    path, out_dir, delete_raw = args_tuple
    path = Path(path)
    out = Path(out_dir) / f"{path.name.split('.')[0]}.parquet"
    if out.exists():
        return path.name, -1
    pmids, journals, langs = parse_medline_gz(path)
    pl.DataFrame(
        {"pmid": pmids, "journal": journals, "language": langs},
        schema={"pmid": pl.Int64, "journal": pl.Utf8, "language": pl.Utf8},
    ).write_parquet(out, compression="zstd")
    if delete_raw:
        path.unlink()
        md5 = path.with_suffix(path.suffix + ".md5")
        if md5.exists():
            md5.unlink()
    return path.name, len(pmids)


def cmd_medline(args) -> int:
    """MEDLINE baseline ``.xml.gz`` → ``medline/<file>.parquet`` shards.

    With ``--stream`` each file is downloaded, md5-checked, parsed and deleted
    one at a time, so the 52 GB baseline never has to fit on disk at once.
    """
    out_dir = Path(args.output_dir).expanduser() / "medline"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = args.medline_files or N_MEDLINE_FILES
    t0 = time.monotonic()
    n_rows = 0
    n_done = 0

    def _one(i: int) -> int:
        name = f"pubmed26n{i:04d}.xml.gz"
        gz = MEDLINE_DIR / name
        if args.stream and not gz.exists():
            if not _fetch_one(f"{MEDLINE_BASE}/{name}", gz, log=ROOT / "download.log"):
                return 0
            _fetch_one(f"{MEDLINE_BASE}/{name}.md5", MEDLINE_DIR / f"{name}.md5", attempts=2)
        if not gz.exists():
            return 0
        md5f = MEDLINE_DIR / f"{name}.md5"
        if md5f.exists():
            want = md5f.read_text().strip().rsplit("=", 1)[-1].strip()
            if _md5(gz) != want:
                print(f"BAD md5 {name} — redownloading", flush=True)
                gz.unlink()
                if not _fetch_one(f"{MEDLINE_BASE}/{name}", gz, log=ROOT / "download.log"):
                    return 0
        _, n = _medline_worker((str(gz), str(out_dir), args.stream or args.delete_raw))
        return max(n, 0)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n in ex.map(_one, range(1, n_files + 1)):
            n_rows += n
            n_done += 1
            if n_done % 50 == 0:
                print(f"  medline {n_done}/{n_files} files, {n_rows:,} rows", flush=True)

    print(
        f"DONE medline {n_done} files, {n_rows:,} rows in {time.monotonic() - t0:.0f}s "
        f"→ {out_dir}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# convert — streaming: pmids → id map; per shard: fetch → parse → fold → delete
# ---------------------------------------------------------------------------


def _shard_name(i: int) -> str:
    return f"text_emb_shard_{i:02d}.pt"


def _parse_shard_content(content_path: Path, pmids: np.ndarray) -> pl.DataFrame:
    """One ``pubmed_chunk_{i}.json`` → the article rows of ``pmids``, **in the order of
    ``pmids``** (which ``cmd_convert`` passes in item-id order). A PMID present in the
    embedding matrix but absent from the content dump keeps its row (the vector is real)
    with empty attributes. The title is kept so ``queries`` can build item-as-query
    text after ``--delete-raw`` has removed the JSON."""
    content = json.loads(content_path.read_text())
    years: list[int] = []
    has_abs: list[bool] = []
    mesh: list[list[str]] = []
    titles: list[str] = []
    for p in pmids.tolist():
        rec = content.get(str(p))
        if rec is None:
            years.append(-1)
            has_abs.append(False)
            mesh.append([])
            titles.append("")
            continue
        y = parse_year(rec.get("d"))
        years.append(y if y is not None else -1)
        has_abs.append(bool((rec.get("a") or "").strip()))
        mesh.append(parse_mesh_field(rec.get("m")))
        titles.append((rec.get("t") or "").strip())
    del content
    return pl.DataFrame(
        {"pmid": pmids.tolist(), "year": years, "has_abstract": has_abs, "mesh": mesh,
         "title": titles},
        schema={
            "pmid": pl.Int64,
            "year": pl.Int32,
            "has_abstract": pl.Boolean,
            "mesh": pl.List(pl.Utf8),
            "title": pl.Utf8,
        },
    )  # fmt: skip


def _fold_shard_embeddings(npy: Path, positions: np.ndarray, out: Path, batch_rows: int) -> None:
    """Rows ``positions`` of the shard's ``(N_i, 768)`` fp32 matrix, in that order →
    L2-normalised fp16 ``[len(positions), 768]`` saved at ``out``. The source is mmapped
    and gathered in ascending-position batches, so a 3 GB shard costs one pass and
    ``batch_rows × 768 × 4`` bytes of RAM, not the shard."""
    import torch
    import torch.nn.functional as F

    from eval_datasets.layout import atomic_write

    arr = np.load(npy, mmap_mode="r")
    if arr.ndim != 2 or arr.shape[1] != EMB_DIM_NATIVE or arr.dtype != np.float32:
        raise ValueError(
            f"{npy}: expected (N, {EMB_DIM_NATIVE}) float32, got {arr.shape} {arr.dtype}"
        )
    dest = torch.empty((positions.shape[0], EMB_DIM_NATIVE), dtype=torch.float16)
    # Gather in source order (sequential reads) and scatter to the item order.
    order = np.argsort(positions, kind="stable")
    src_sorted = positions[order]
    for s in range(0, src_sorted.shape[0], batch_rows):
        e = min(s + batch_rows, src_sorted.shape[0])
        block = torch.from_numpy(np.array(arr[src_sorted[s:e]], dtype=np.float32))
        dest[torch.from_numpy(order[s:e])] = F.normalize(block, dim=-1).to(torch.float16)
    del arr
    atomic_write(out, lambda fh: torch.save(dest, fh))


def _write_shard_index(content_dir: Path, n_items: int, entries: list[dict]) -> None:
    payload = {
        "n_items": int(n_items),
        "dim": EMB_DIM_NATIVE,
        "dtype": "float16",
        "n_shards": len(entries),
        "shards": entries,
        "source": f"{MEDCPT_BASE} (embeds_chunk_*.npy, fp32 → L2-normalised fp16)",
        "order": "shard order, ascending PMID within a shard (= item_id order)",
    }
    (content_dir / "shard_index.json").write_text(json.dumps(payload, indent=2))


def cmd_convert(args) -> int:
    """Streaming build of ``item_id_map.json``, ``staging/articles_chunk_*.parquet`` and
    ``content_d768/{text_emb_shard_NN.pt, shard_index.json, text_emb.meta.json}``.

    Item ids are **1-indexed dense** in *(shard order, ascending PMID within the shard)*
    — the chunks partition the PMID space by million (chunk *i* holds PMIDs
    ``i,000,000..i,999,999``), so this is ascending PMID order too; ``prep_log.json`` records
    ``pmid_ranges_disjoint`` so the claim is checked, not assumed. With ``--keep-items``
    the map covers the selected articles only (``select_pmids``). Every output shard is
    a contiguous ``[start_id, start_id + n_rows)`` block, which is what
    ``eval_datasets.layout.load_sharded`` reassembles.

    Resumable: a shard whose parquet and ``.pt`` exist with the index's row count is
    skipped; ``--fetch`` pulls missing raw files (``--prefetch`` shards ahead) and
    ``--delete-raw`` removes a shard's JSON + npy once folded (the PMID list stays).
    """
    output = Path(args.output_dir).expanduser()
    staging = output / "staging"
    content_dir = output / f"content_d{EMB_DIM_NATIVE}"
    staging.mkdir(parents=True, exist_ok=True)
    content_dir.mkdir(parents=True, exist_ok=True)
    shards = _parse_shards(args.shards)
    log_path = ROOT / "download.log"
    t0 = time.monotonic()
    log: dict = {"shards": shards, "raw_root": str(ROOT), "keep_items": args.keep_items,
                 "seed": args.seed}  # fmt: skip

    # ---- phase 1: pmid lists → the id map ----------------------------------
    print("STEP pmid lists → item_id_map.json", flush=True)
    shard_pmids: dict[int, np.ndarray] = {}
    for i in shards:
        url, p = _shard_paths(i)["pmids"]
        if not p.exists() and args.fetch:
            _fetch_one(url, p, log=log_path)
        if not p.exists():
            print(f"  WARN missing {p.name}, skipping shard {i}", flush=True)
            continue
        shard_pmids[i] = np.asarray(json.loads(p.read_text()), dtype=np.int64)
    if not shard_pmids:
        print("ERROR no pmids_chunk_*.json found (run `download --what pmids`)", flush=True)
        return 1
    order = sorted(shard_pmids)
    all_pmids = np.concatenate([shard_pmids[i] for i in order])
    _, first_idx = np.unique(all_pmids, return_index=True)
    first = np.zeros(all_pmids.shape[0], dtype=bool)
    first[first_idx] = True
    n_dup = int((~first).sum())
    if n_dup:
        print(f"  WARN {n_dup:,} duplicate PMIDs — the first occurrence (shard order) wins")
    # Select over the de-duplicated set so `--keep-items` is exact.
    keep = np.zeros_like(first)
    keep[first] = select_pmids(all_pmids[first], args.keep_items, args.seed)
    lo: list[int] = []
    hi: list[int] = []
    kept_per_shard: dict[int, np.ndarray] = {}  # shard → kept row positions, PMID-ascending
    starts: dict[int, int] = {}
    offset = 0
    with open(output / "item_id_map.json", "w") as f:
        f.write("{")
        first_entry = True
        for i in order:
            pm = shard_pmids[i]
            k = keep[offset : offset + pm.shape[0]]
            offset += pm.shape[0]
            pos = np.nonzero(k)[0]
            pos = pos[np.argsort(pm[pos], kind="stable")]
            kept_per_shard[i] = pos
            starts[i] = sum(len(v) for j, v in kept_per_shard.items() if j < i)
            if pos.size:
                lo.append(int(pm[pos[0]]))
                hi.append(int(pm[pos[-1]]))
            base = starts[i] + 1
            chunk = ",".join(f'"{int(p)}":{base + j}' for j, p in enumerate(pm[pos]))
            if chunk:
                f.write(("" if first_entry else ",") + chunk)
                first_entry = False
        f.write("}")
    n_items = int(keep.sum())
    disjoint = all(hi[j] < lo[j + 1] for j in range(len(lo) - 1))
    log.update(
        n_pmids_listed=int(all_pmids.shape[0]), n_duplicates=n_dup, n_items=n_items,
        pmid_ranges_disjoint=disjoint,
    )  # fmt: skip
    print(
        f"  {all_pmids.shape[0]:,} PMIDs listed, {n_items:,} kept → item_id_map.json "
        f"(shard PMID ranges disjoint: {disjoint})",
        flush=True,
    )
    del all_pmids, keep, first

    # ---- phase 2: shard by shard ------------------------------------------
    index_path = content_dir / "shard_index.json"
    done: dict[str, dict] = {}
    if index_path.exists():
        done = {e["filename"]: e for e in json.loads(index_path.read_text())["shards"]}
    entries: list[dict] = []
    fetcher = ThreadPoolExecutor(max_workers=max(1, args.prefetch)) if args.fetch else None
    pending: dict[int, list] = {}

    def _submit(i: int) -> None:
        if fetcher is None or i in pending:
            return
        paths = _shard_paths(i)
        need = ["content"] + ([] if args.skip_embeds else ["embeds"])
        pending[i] = [
            fetcher.submit(_fetch_one, *paths[k], log=log_path)
            for k in need
            if not paths[k][1].exists()
        ]

    for j, i in enumerate(order):
        for ahead in order[j : j + 1 + args.prefetch]:
            _submit(ahead)
        if i in pending and not all(fut.result() for fut in pending.pop(i)):
            print(f"ERROR shard {i}: download failed", flush=True)
            return 1
        pos = kept_per_shard[i]
        pm = shard_pmids[i][pos]
        n_rows = int(pos.shape[0])
        paths = _shard_paths(i)
        parquet = staging / f"articles_chunk_{i}.parquet"
        shard_pt = content_dir / _shard_name(i)
        entry = {"filename": _shard_name(i), "start_id": starts[i], "n_rows": n_rows,
                 "source_shard": i}  # fmt: skip
        if n_rows == 0:
            print(f"  shard {i}: no kept rows", flush=True)
            continue
        if not parquet.exists():
            src = paths["content"][1]
            if not src.exists():
                print(f"  WARN shard {i}: missing {src.name}, no article parquet", flush=True)
            else:
                _parse_shard_content(src, pm).write_parquet(parquet, compression="zstd")
        if not args.skip_embeds:
            prev = done.get(entry["filename"])
            if shard_pt.exists() and prev == entry:
                print(f"  shard {i}: {n_rows:,} rows already folded", flush=True)
            else:
                npy = paths["embeds"][1]
                if not npy.exists():
                    print(f"ERROR shard {i}: missing {npy.name} (pass --fetch)", flush=True)
                    return 1
                if np.load(npy, mmap_mode="r").shape[0] != shard_pmids[i].shape[0]:
                    print(f"ERROR shard {i}: npy rows != pmid list length", flush=True)
                    return 1
                _fold_shard_embeddings(npy, pos, shard_pt, args.batch_rows)
                print(f"  shard {i}: {n_rows:,} rows folded → {shard_pt.name}", flush=True)
            entries.append(entry)
            _write_shard_index(content_dir, n_items, entries)
        if args.delete_raw:
            for k in ("content", "embeds"):
                paths[k][1].unlink(missing_ok=True)
    if fetcher is not None:
        fetcher.shutdown(wait=True)

    if not args.skip_embeds:
        (content_dir / "text_emb.meta.json").write_text(
            json.dumps(
                {
                    "prefix": None,
                    "encoder": "ncbi/MedCPT-Article-Encoder (precomputed, NCBI FTP)",
                    "dim": EMB_DIM_NATIVE,
                    "reduction": "none",
                    "normalization": "l2",
                    "n_rows": n_items,
                    "shape": [n_items, EMB_DIM_NATIVE],
                    "dtype": "float16",
                    "layout": "sharded (shard_index.json)",
                    "keep_items": args.keep_items,
                    "seed": args.seed,
                },
                indent=2,
            )
        )
        log["text_emb"] = str(index_path)
    else:
        print("SKIP embeds (--skip-embeds): id map + article parquets only", flush=True)

    log["wall_clock_sec"] = round(time.monotonic() - t0, 1)
    _merge_log(output, "convert", log)
    print(f"ALL DONE convert in {log['wall_clock_sec']:.0f}s — n_items={n_items:,}", flush=True)
    return 0


def _merge_log(output: Path, key: str, payload: dict) -> None:
    path = output / "prep_log.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[key] = payload
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# attrs — narrow clause tensor, vocabs, heldout, eval_split
# ---------------------------------------------------------------------------


def load_mesh_tree_tops(path: Path) -> dict[str, int]:
    """``desc2026.gz`` → ``{lower-cased descriptor name: category index}``."""
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    name: str | None = None
    trees: list[str] = []
    in_name = False
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("<DescriptorName>"):
                in_name = True
                continue
            if s.startswith("</DescriptorName>"):
                in_name = False
                continue
            if in_name and s.startswith("<String>"):
                name = s[len("<String>") : -len("</String>")].strip().lower()
                continue
            if s.startswith("<TreeNumber>"):
                trees.append(s[len("<TreeNumber>") : -len("</TreeNumber>")])
                continue
            if s.startswith("</DescriptorRecord>"):
                if name:
                    out[name] = mesh_category_of(trees)
                name, trees = None, []
    return out


def cmd_attrs(args) -> int:
    import torch

    output = Path(args.output_dir).expanduser()
    staging = output / "staging"
    id_map_path = output / "item_id_map.json"
    if not id_map_path.exists():
        print(f"ERROR missing {id_map_path} (run `convert` first)", flush=True)
        return 1

    t0 = time.monotonic()
    log: dict = {}
    print("STEP load item_id_map + article parquets", flush=True)
    id_map: dict[str, int] = json.loads(id_map_path.read_text())
    n_items = len(id_map)
    log["n_items"] = n_items

    parts = sorted(staging.glob("articles_chunk_*.parquet"))
    if not parts:
        print(f"ERROR no article parquets in {staging} (run `convert` first)", flush=True)
        return 1
    arts = pl.concat(
        [pl.read_parquet(p).select("pmid", "year", "has_abstract", "mesh") for p in parts],
        how="vertical",
    )
    # 1-indexed item_id per the shared convention; rows land at item_id - 1.
    arts = arts.with_columns(
        pl.col("pmid")
        .cast(pl.Utf8)
        .replace_strict(id_map, default=0, return_dtype=pl.Int64)
        .alias("item_id")
    ).filter(pl.col("item_id") > 0)
    print(f"  {arts.height:,} articles mapped onto {n_items:,} item ids", flush=True)

    # ---- C0 MeSH vocab ------------------------------------------------------
    print("STEP C0 MeSH vocab", flush=True)
    counts = Counter()
    for bag in arts["mesh"].to_list():
        counts.update(bag or [])
    kept = [
        (d, c)
        for d, c in counts.most_common(args.mesh_vocab)
        if c >= args.mesh_min_count
    ]
    mesh_names = [d for d, _ in kept]
    mesh_freq = np.asarray([c for _, c in kept], dtype=np.int64)
    mesh_vocab = {d: i for i, d in enumerate(mesh_names)}
    with open(output / "mesh_vocab.json", "w") as f:
        json.dump({"size": len(mesh_names), "names": mesh_names}, f)
    torch.save(torch.from_numpy(mesh_freq), output / "mesh_global_freq.pt")
    log["mesh_vocab_size"] = len(mesh_names)
    log["mesh_cap_k"] = A_MAX_NARROW
    log["mesh_cap_rule"] = "keep the K globally-rarest in-vocab descriptors, rarest first"
    print(
        f"  mesh vocab = {len(mesh_names):,} (cap {args.mesh_vocab}, "
        f"min_count {args.mesh_min_count}); top3 = {mesh_names[:3]}",
        flush=True,
    )

    # ---- C1 MeSH tree-top category -----------------------------------------
    print("STEP C1 MeSH tree-top category", flush=True)
    tree_tops = load_mesh_tree_tops(Path(args.mesh_desc or MESH_DESC_PATH))
    if not tree_tops:
        print(
            f"  WARN no MeSH descriptor file at {args.mesh_desc or MESH_DESC_PATH} — C1 all -1",
            flush=True,
        )
    with open(output / "mesh_cat_vocab.json", "w") as f:
        json.dump({"size": len(MESH_CATEGORIES), "names": MESH_CATEGORIES}, f, indent=2)
    log["mesh_cat_vocab_size"] = len(MESH_CATEGORIES)
    log["mesh_tree_tops_known"] = len(tree_tops)

    # ---- C2 year vocab ------------------------------------------------------
    with open(output / "year_vocab.json", "w") as f:
        json.dump({"size": len(YEAR_BUCKET_NAMES), "names": YEAR_BUCKET_NAMES}, f, indent=2)
    log["year_vocab_size"] = len(YEAR_BUCKET_NAMES)

    # ---- C3 journal + language, from the MEDLINE join ----------------------
    print("STEP C3 journal + language (MEDLINE join)", flush=True)
    med_parts = sorted((output / "medline").glob("*.parquet"))
    journal_names: list[str] = []
    lang_names: list[str] = []
    jl = None
    if med_parts:
        med = pl.concat([pl.read_parquet(p) for p in med_parts], how="vertical").unique(
            subset=["pmid"], keep="first"
        )
        jfreq = (
            med.filter(pl.col("journal") != "")
            .group_by("journal")
            .len()
            .sort("len", descending=True)
            .head(args.journal_vocab)
        )
        journal_names = jfreq["journal"].to_list()
        lfreq = (
            med.filter(pl.col("language") != "")
            .group_by("language")
            .len()
            .sort("len", descending=True)
        )
        lang_names = lfreq["language"].to_list()
        jl = med.select(
            "pmid",
            pl.col("journal")
            .replace_strict(
                {n: i for i, n in enumerate(journal_names)}, default=-1, return_dtype=pl.Int64
            )
            .alias("journal_id"),
            pl.col("language")
            .replace_strict(
                {n: i for i, n in enumerate(lang_names)}, default=-1, return_dtype=pl.Int64
            )
            .alias("lang_id"),
        )
        arts = arts.join(jl, on="pmid", how="left")
        print(
            f"  MEDLINE rows {med.height:,}; journal vocab {len(journal_names):,}; "
            f"languages {len(lang_names)}",
            flush=True,
        )
    else:
        print("  WARN no medline/*.parquet — C3 journal all -1 (run `medline` first)", flush=True)
        arts = arts.with_columns(
            pl.lit(-1, dtype=pl.Int64).alias("journal_id"),
            pl.lit(-1, dtype=pl.Int64).alias("lang_id"),
        )
    with open(output / "journal_vocab.json", "w") as f:
        json.dump({"size": len(journal_names), "names": journal_names}, f)
    with open(output / "lang_vocab.json", "w") as f:
        json.dump({"size": len(lang_names), "names": lang_names}, f, indent=2)
    log["journal_vocab_size"] = len(journal_names)
    log["lang_vocab_size"] = len(lang_names)

    # ---- assemble item_attrs_narrow [N, 5, 4] ------------------------------
    print("STEP assemble item_attrs_narrow", flush=True)
    narrow = torch.full((n_items, C_NARROW, A_MAX_NARROW), -1, dtype=torch.long)
    narrow_np = narrow.numpy()

    iids = arts["item_id"].to_numpy() - 1
    # C2 year bucket (vectorised).
    years = arts["year"].to_numpy()
    ybuck = np.full(years.shape, -1, dtype=np.int64)
    valid = years > 0
    ybuck[valid] = np.searchsorted(np.asarray(YEAR_BUCKET_EDGES), years[valid], side="right")
    narrow_np[iids, 2, 0] = ybuck
    # C3 journal (reverse clause).
    narrow_np[iids, 3, 0] = arts["journal_id"].to_numpy()
    # C4 has-abstract flag.
    narrow_np[iids, 4, 0] = arts["has_abstract"].to_numpy().astype(np.int64)

    # C0/C1 need per-row work: cap the MeSH bag by rarity, then read C1 off the
    # first (rarest) kept descriptor.
    mesh_lists = arts["mesh"].to_list()
    cat_of_id = np.full(len(mesh_names), -1, dtype=np.int64)
    for d, i in mesh_vocab.items():
        cat_of_id[i] = tree_tops.get(d, -1)
    n_c0 = 0
    for row, bag in zip(iids.tolist(), mesh_lists, strict=True):
        ids = cap_mesh_by_rarity(bag or [], mesh_vocab, mesh_freq, k=A_MAX_NARROW)
        if not ids:
            continue
        n_c0 += 1
        for j, v in enumerate(ids):
            narrow_np[row, 0, j] = v
        narrow_np[row, 1, 0] = cat_of_id[ids[0]]

    coverage = {
        f"c{c}_{CLAUSE_NAMES[c]}": round(float((narrow[:, c, 0] != -1).float().mean()), 4)
        for c in range(C_NARROW)
    }
    log["narrow_coverage"] = coverage
    print(f"  per-clause coverage = {coverage}", flush=True)

    torch.save(narrow, output / "item_attrs_narrow.pt")
    torch.save(
        torch.tensor(CLAUSE_IS_REVERSE, dtype=torch.bool),
        output / "clause_is_reverse_narrow.pt",
    )
    print(
        f"  wrote item_attrs_narrow.pt {tuple(narrow.shape)} + "
        f"clause_is_reverse_narrow.pt {CLAUSE_IS_REVERSE}",
        flush=True,
    )

    # ---- articles.parquet (compact, for query building) --------------------
    arts.select(
        "item_id", "pmid", "year", "has_abstract", "journal_id", "lang_id"
    ).sort("item_id").write_parquet(output / "articles.parquet", compression="zstd")

    # ---- heldout + eval_split ----------------------------------------------
    print("STEP heldout + eval_split", flush=True)
    rng = np.random.default_rng(args.seed)
    n_heldout = min(args.n_heldout, n_items)
    heldout_ids = np.sort(
        rng.choice(np.arange(1, n_items + 1, dtype=np.int64), size=n_heldout, replace=False)
    )
    pmid_of_item = {v: k for k, v in id_map.items()}
    pl.DataFrame(
        {
            "item_id": heldout_ids.tolist(),
            "pmid": [pmid_of_item[int(i)] for i in heldout_ids],
        },
        schema={"item_id": pl.Int64, "pmid": pl.Utf8},
    ).write_parquet(output / "heldout.parquet", compression="zstd")

    qa = synthesize_qa_narrow(heldout_ids.tolist(), narrow, C_NARROW)
    pl.DataFrame(
        {"target_id": heldout_ids.tolist(), "query_attrs_narrow": qa.tolist()},
        schema={"target_id": pl.Int64, "query_attrs_narrow": pl.List(pl.Int64)},
    ).write_parquet(output / "eval_split.parquet", compression="zstd")
    log["n_heldout"] = int(n_heldout)
    print(f"  wrote heldout.parquet + eval_split.parquet ({n_heldout:,} rows)", flush=True)

    log["wall_clock_sec"] = round(time.monotonic() - t0, 1)
    _merge_log(output, "attrs", log)
    print(f"ALL DONE attrs in {log['wall_clock_sec']:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
# queries — build the query sets
# ---------------------------------------------------------------------------


def cmd_queries(args) -> int:
    """Write ``queries.parquet`` (``query_id, text, target_id``).

    Two sources, per dataset-candidates.md §3.8/§4.1:

    * ``heldout`` — item-as-query: the held-out article's title is the query and
      the article itself is the single relevant item. Always available; the
      titles come from the article parquets, so ``--delete-raw`` costs nothing.
      **Every held-out row is kept** — a title-less article gets an empty query
      string rather than being dropped — so ``query_emb.pt`` stays row-aligned
      with ``heldout.parquet`` / ``eval_split.parquet`` (the layout contract;
      the 2026-09-06 review's finding on this loader).
    * ``nfcorpus`` — the NFCorpus (BEIR) biomedical query set. NFCorpus document
      ids *are* PMIDs, so its qrels map straight onto our item ids. Needs
      ``queries.jsonl`` + ``qrels/test.tsv`` staged under
      ``--nfcorpus-dir``; if they are absent the set is skipped with a warning
      rather than failing (BEIR asks that its corpus not be redistributed, so
      the harness never downloads it automatically). NFCorpus rows are written
      to ``queries_nfcorpus.parquet``, *not* appended to the held-out set: the
      harness's text path reads one query set aligned with ``heldout.parquet``.
    """
    output = Path(args.output_dir).expanduser()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    log: dict = {"sources": sources}
    wrote = False

    if "heldout" in sources:
        heldout = output / "heldout.parquet"
        if not heldout.exists():
            print(f"ERROR missing {heldout} (run `attrs` first)", flush=True)
            return 1
        ho = pl.read_parquet(heldout)
        titles = _titles_for_pmids(output, {int(p) for p in ho["pmid"].to_list()})
        rows = [
            {"query_id": f"heldout:{p}", "text": titles.get(int(p), ""), "target_id": int(t)}
            for p, t in zip(ho["pmid"].to_list(), ho["item_id"].to_list(), strict=True)
        ]
        n_empty = sum(1 for r in rows if not r["text"])
        pl.DataFrame(rows, schema=_QUERY_SCHEMA).write_parquet(
            output / "queries.parquet", compression="zstd"
        )
        log["n_heldout"] = len(rows)
        log["n_heldout_empty_title"] = n_empty
        print(f"  heldout queries: {len(rows):,} ({n_empty:,} with no title)", flush=True)
        wrote = True

    if "nfcorpus" in sources:
        nf = Path(args.nfcorpus_dir).expanduser() if args.nfcorpus_dir else ROOT / "nfcorpus"
        qf, rf = nf / "queries.jsonl", nf / "qrels" / "test.tsv"
        if not (qf.exists() and rf.exists()):
            print(f"  WARN nfcorpus not staged at {nf} — skipping (see docstring)", flush=True)
        else:
            id_map = json.loads((output / "item_id_map.json").read_text())
            qtext = {}
            for line in qf.read_text().splitlines():
                if line.strip():
                    o = json.loads(line)
                    qtext[str(o["_id"])] = o.get("text", "")
            rows = []
            for line in rf.read_text().splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) < 3 or int(float(parts[2])) <= 0:
                    continue
                qid, did = parts[0], parts[1]
                tgt = id_map.get(str(did))
                if tgt is None or qid not in qtext:
                    continue
                rows.append(
                    {"query_id": f"nfcorpus:{qid}", "text": qtext[qid], "target_id": int(tgt)}
                )
            pl.DataFrame(rows, schema=_QUERY_SCHEMA).write_parquet(
                output / "queries_nfcorpus.parquet", compression="zstd"
            )
            log["n_nfcorpus"] = len(rows)
            print(f"  nfcorpus (query, relevant-PMID) pairs in-catalog: {len(rows):,}", flush=True)
            wrote = True

    if not wrote:
        print("ERROR no query sets built", flush=True)
        return 1
    _merge_log(output, "queries", log)
    print("ALL DONE queries", flush=True)
    return 0


def _titles_for_pmids(output: Path, want: set[int]) -> dict[int, str]:
    """Titles for a PMID set, out of ``staging/articles_chunk_*.parquet``."""
    out: dict[int, str] = {}
    wanted = list(want)
    for p in sorted((output / "staging").glob("articles_chunk_*.parquet")):
        df = pl.read_parquet(p, columns=["pmid", "title"]).filter(pl.col("pmid").is_in(wanted))
        out.update(zip(df["pmid"].to_list(), df["title"].to_list(), strict=True))
    return out


# ---------------------------------------------------------------------------
# encode_queries — MedCPT-Query-Encoder, native 768-d
# ---------------------------------------------------------------------------


def cmd_encode_queries(args) -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    output = Path(args.output_dir).expanduser()
    qpath = output / "queries.parquet"
    if not qpath.exists():
        print(f"ERROR missing {qpath} (run `queries` first)", flush=True)
        return 1
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("ERROR --device cuda but no CUDA device visible", flush=True)
        return 1

    q = pl.read_parquet(qpath)
    texts = q["text"].to_list()
    print(f"STEP encode {len(texts):,} queries with {args.encoder} on {device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder).to(device).eval()

    embs = torch.empty((len(texts), EMB_DIM_NATIVE), dtype=torch.float32)
    for s in range(0, len(texts), args.batch_size):
        batch = texts[s : s + args.batch_size]
        enc = tok(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=args.max_seq_length,
        ).to(device)
        with torch.inference_mode():
            # MedCPT reads the [CLS] vector of the last hidden state.
            out = model(**enc).last_hidden_state[:, 0, :]
        embs[s : s + len(batch)] = out.float().cpu()

    # Native dim, no reduction. L2-normalised so the harness's inner-product
    # top-k is cosine on both sides (the item matrix is normalised the same way
    # in `convert`).
    q16 = F.normalize(embs, dim=-1).to(torch.float16)
    sub = output / f"content_d{EMB_DIM_NATIVE}"
    sub.mkdir(parents=True, exist_ok=True)
    torch.save(q16, sub / "query_emb.pt")
    with open(sub / "query_emb.meta.json", "w") as f:
        json.dump(
            {
                "prefix": None,
                "encoder": args.encoder,
                "pooling": "cls",
                "dim": EMB_DIM_NATIVE,
                "reduction": "none",
                "normalization": "l2",
                "max_seq_length": args.max_seq_length,
                "n_rows": int(q16.shape[0]),
                "shape": list(q16.shape),
                "dtype": "float16",
            },
            f,
            indent=2,
        )
    print(f"  wrote {sub}/query_emb.pt {tuple(q16.shape)}", flush=True)
    print("ALL DONE encode_queries", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ETL for the PubMed + MedCPT filter-bench dataset.")
    sub = p.add_subparsers(dest="cmd", required=True)

    default_out = str(Path(os.environ.get("RETRIEVE_DATA_ROOT", "data")) / "pubmed-medcpt")

    sp = sub.add_parser("plan", help="dry run: remote sizes → rows, peak disk, wall time")
    sp.add_argument("--shards", type=str, default=None, help='e.g. "0-9,37" (default: all 38)')
    sp.add_argument("--keep-items", type=int, default=None, help="slice size (default: all)")
    sp.add_argument("--prefetch", type=int, default=1)
    sp.add_argument("--medline", action="store_true", help="also HEAD the 1334 MEDLINE files")
    sp.add_argument("--mbps", type=float, default=0.0, help="0 = measure with a 64 MB probe")
    sp.add_argument("--report", type=str, default="", help="write the JSON budget here")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("download", help="resumable mirror of the NCBI raw files")
    sp.add_argument(
        "--what", choices=["all", "embeds", "meta", "pmids", "medline", "mesh"], default="all"
    )
    sp.add_argument("--shards", type=str, default=None, help='e.g. "0-9,37" (default: all 38)')
    sp.add_argument("--parallel", type=int, default=12)
    sp.add_argument("--medline-files", type=int, default=0, help="0 = all 1334")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("verify", help="structural + md5 verification of the raw mirror")
    sp.add_argument("--shards", type=str, default=None)
    sp.add_argument("--medline", action="store_true")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("medline", help="MEDLINE baseline → pmid/journal/language parquet")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--workers", type=int, default=16)
    sp.add_argument("--medline-files", type=int, default=0)
    sp.add_argument(
        "--stream", action="store_true", help="download+parse+delete one file at a time"
    )
    sp.add_argument("--delete-raw", action="store_true")
    sp.set_defaults(func=cmd_medline)

    sp = sub.add_parser("convert", help="streaming: pmids → id map; shards → parquet + fp16")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--shards", type=str, default=None)
    sp.add_argument("--keep-items", type=int, default=None,
                    help="keep exactly N articles, chosen by a seeded PMID hash (default: all)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--batch-rows", type=int, default=100_000)
    sp.add_argument("--fetch", action="store_true", help="download missing raw files as needed")
    sp.add_argument("--prefetch", type=int, default=1, help="shards downloaded ahead (--fetch)")
    sp.add_argument("--skip-embeds", action="store_true", help="id map + attrs only")
    sp.add_argument("--delete-raw", action="store_true", help="drop each shard once folded in")
    sp.set_defaults(func=cmd_convert)

    sp = sub.add_parser("attrs", help="narrow clause tensor + vocabs + eval_split")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--mesh-vocab", type=int, default=30_000)
    sp.add_argument("--mesh-min-count", type=int, default=20)
    sp.add_argument("--journal-vocab", type=int, default=5_000)
    sp.add_argument("--mesh-desc", type=str, default=None)
    sp.add_argument("--n-heldout", type=int, default=10_000)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_attrs)

    sp = sub.add_parser("queries", help="build queries.parquet (heldout / nfcorpus)")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--sources", type=str, default="heldout")
    sp.add_argument("--nfcorpus-dir", type=str, default=None)
    sp.set_defaults(func=cmd_queries)

    sp = sub.add_parser("encode_queries", help="MedCPT-Query-Encoder → content_d768/query_emb.pt")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--encoder", type=str, default="ncbi/MedCPT-Query-Encoder")
    sp.add_argument("--device", type=str, default="cuda", help="cpu only for tests")
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--max-seq-length", type=int, default=64)
    sp.set_defaults(func=cmd_encode_queries)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "cap_mesh_by_rarity",
    "load_mesh_tree_tops",
    "main",
    "mesh_category_of",
    "parse_medline_gz",
    "parse_mesh_field",
    "parse_year",
    "plan_budget",
    "pmid_hash",
    "select_pmids",
    "year_to_bucket",
]
