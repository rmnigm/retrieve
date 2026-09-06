#!/usr/bin/env -S uv run --script
"""End-to-end ETL for the PubMed + MedCPT filter-bench dataset (roadmap E2).

Source: the NCBI FTP MedCPT article-embedding release
``https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/`` — 38 chunks of

* ``embeds_chunk_{i}.npy``  — ``(N_i, 768)`` **float32** (verified from the npy
  header: ``descr='<f4'``; §3.8 inferred this from the file sizes and was right),
* ``pmids_chunk_{i}.json``  — the row-aligned PMID list,
* ``pubmed_chunk_{i}.json`` — ``{pmid: {"d": date, "t": title, "a": abstract,
  "m": mesh}}``.

``m`` is a ``|``-separated list of ``descriptor!qualifier`` entries where a
trailing ``*`` marks a major topic, e.g.::

    "humans!|rectal neoplasms!|rectal neoplasms*|rectal neoplasms!therapy|"

Journal and language are **not** in the chunk JSON; they come from a join
against the MEDLINE baseline (``https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/``,
1334 × ``pubmed26n*.xml.gz`` ≈ 52 GB, each with a published ``.md5``).
MeSH tree-top category letters come from the MeSH descriptor file
(``xmlmesh/desc2026.gz``, 17 MB).

Subcommands::

    download        Resumable parallel mirror of any of the four raw sources.
    medline         Stream the MEDLINE baseline → pmid/journal/language parquet.
    convert         Chunk JSON + npy → item_id_map.json, per-shard article
                    parquet, and the fp16 ``content_d768/text_emb.pt`` item
                    matrix at the encoder's **native** 768 dims.
    attrs           Article parquet + MEDLINE join → item_attrs_narrow.pt,
                    clause_is_reverse_narrow.pt, vocab JSONs, heldout.parquet,
                    eval_split.parquet.
    queries         Build the query sets (held-out articles; NFCorpus if staged).
    encode_queries  Encode query text with ``ncbi/MedCPT-Query-Encoder``
                    (``--device cuda``) → content_d768/query_emb.pt.

**No dimensionality reduction.** Per the user decision of 2026-09-06 every
dataset is benchmarked at its encoder's native dim; there is no PCA step here
and none is planned. MedCPT is 768-d, so the item matrix is 36M × 768 fp16
≈ 55 GB and the raw mirror is ~198 GB (102 GB embeddings + 44 GB chunk JSON
+ 52 GB MEDLINE baseline). ``convert --delete-raw`` folds each shard in and
drops it so the raw mirror need not be held whole; the 55 GB output still
needs a volume that can hold it. See docs/system/datasets.md § pubmed.

Layout (under ``$RETRIEVE_DATA_ROOT``, default ``<repo>/data``)::

    data/_raw/pubmed/                       chunk npy/json, mesh_desc*.xml.gz
    data/_raw/pubmed/medline_baseline/      pubmed26n*.xml.gz (+ .md5)
    data/pubmed-medcpt/                     the bench-side output dir
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from eval_datasets.hf_io import raw_dir

from .common import synthesize_qa_narrow

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


# ---------------------------------------------------------------------------
# download — resumable, parallel, checksum-verified where NCBI publishes one
# ---------------------------------------------------------------------------


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

    try:
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, timeout=60) as r:
            total = int(r.headers.get("Content-Length", "0"))
    except (urllib.error.URLError, ValueError, TimeoutError) as e:
        _log(f"HEADFAIL {dest.name} {e}")
        total = 0

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


def _medcpt_urls(shards: list[int], what: str) -> list[tuple[str, Path]]:
    jobs: list[tuple[str, Path]] = []
    for i in shards:
        if what in ("all", "meta"):
            jobs.append((f"{MEDCPT_BASE}/pmids_chunk_{i}.json", ROOT / f"pmids_chunk_{i}.json"))
            jobs.append((f"{MEDCPT_BASE}/pubmed_chunk_{i}.json", ROOT / f"pubmed_chunk_{i}.json"))
        if what in ("all", "embeds"):
            jobs.append((f"{MEDCPT_BASE}/embeds_chunk_{i}.npy", ROOT / f"embeds_chunk_{i}.npy"))
    return jobs


def cmd_download(args) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    log = ROOT / "download.log"
    shards = _parse_shards(args.shards)
    jobs: list[tuple[str, Path]] = []

    if args.what in ("all", "embeds", "meta"):
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
# convert — chunk JSON + npy → item_id_map.json, article parquet, fp16 matrix
# ---------------------------------------------------------------------------


def _convert_shard_attrs(args_tuple) -> tuple[int, int]:
    """Parse one ``pubmed_chunk_{i}.json`` into a compact article parquet.

    ``args_tuple`` is ``(shard, content_root, pmids_root, staging)`` — the two
    roots differ under `stream`, where the 10 MB pmids lists stay in the raw
    mirror but the 1.5 GB content JSON is staged on local scratch and deleted.
    """
    i, content_root, pmids_root, staging = args_tuple
    content_root, pmids_root, staging = Path(content_root), Path(pmids_root), Path(staging)
    out = staging / f"articles_chunk_{i}.parquet"
    if out.exists():
        return i, -1
    src = content_root / f"pubmed_chunk_{i}.json"
    pmid_src = pmids_root / f"pmids_chunk_{i}.json"
    if not src.exists() or not pmid_src.exists():
        return i, 0
    order = json.loads(pmid_src.read_text())  # row order inside the shard
    content = json.loads(src.read_text())

    pmids: list[int] = []
    years: list[int] = []
    has_abs: list[bool] = []
    mesh: list[list[str]] = []
    for p in order:
        rec = content.get(p)
        if rec is None:
            # Row present in the embedding matrix but absent from the content
            # dump — keep the row (the vector is real) with empty attributes.
            pmids.append(int(p))
            years.append(-1)
            has_abs.append(False)
            mesh.append([])
            continue
        pmids.append(int(p))
        y = parse_year(rec.get("d"))
        years.append(y if y is not None else -1)
        has_abs.append(bool((rec.get("a") or "").strip()))
        mesh.append(parse_mesh_field(rec.get("m")))

    pl.DataFrame(
        {"pmid": pmids, "year": years, "has_abstract": has_abs, "mesh": mesh},
        schema={
            "pmid": pl.Int64,
            "year": pl.Int32,
            "has_abstract": pl.Boolean,
            "mesh": pl.List(pl.Utf8),
        },
    ).write_parquet(out, compression="zstd")
    return i, len(pmids)


def cmd_convert(args) -> int:
    """Build ``item_id_map.json``, the per-shard article parquets, and (unless
    ``--skip-embeds``) the single memory-mapped fp16 ``item_emb_768.f16``.

    Item ids are **1-indexed dense** over the PMIDs present in the chunks,
    assigned in ascending numeric PMID order (chunk *i* already holds PMIDs
    ``i,000,000..i,999,999``, so this is chunk order), matching the
    ``item_id_map.json`` convention in docs/system/datasets.md.
    """
    output = Path(args.output_dir).expanduser()
    staging = output / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    shards = _parse_shards(args.shards)
    t0 = time.monotonic()
    log: dict = {"shards": shards, "raw_root": str(ROOT)}

    # ---- item_id_map.json ---------------------------------------------------
    print("STEP scan pmids → item_id_map.json", flush=True)
    shard_pmids: dict[int, list[int]] = {}
    for i in shards:
        p = ROOT / f"pmids_chunk_{i}.json"
        if not p.exists():
            print(f"  WARN missing {p.name}, skipping shard {i}", flush=True)
            continue
        shard_pmids[i] = [int(x) for x in json.loads(p.read_text())]
    if not shard_pmids:
        print("ERROR no pmids_chunk_*.json found (run `download --what meta`)", flush=True)
        return 1

    all_pmids = np.concatenate(
        [np.asarray(shard_pmids[i], dtype=np.int64) for i in sorted(shard_pmids)]
    )
    order = np.argsort(all_pmids, kind="stable")
    sorted_pmids = all_pmids[order]
    uniq, first_idx = np.unique(sorted_pmids, return_index=True)
    if len(uniq) != len(all_pmids):
        print(f"  WARN {len(all_pmids) - len(uniq):,} duplicate PMIDs dropped", flush=True)
    n_items = len(uniq)
    # item_id = rank in ascending PMID order, 1-indexed.
    with open(output / "item_id_map.json", "w") as f:
        json.dump({str(int(p)): i + 1 for i, p in enumerate(uniq)}, f)
    log["n_items"] = int(n_items)
    print(f"  n_items = {n_items:,} → item_id_map.json", flush=True)

    # pmid → row index (0-indexed = item_id - 1), as a sorted-array lookup.
    def _rows_for(pmids: np.ndarray) -> np.ndarray:
        return np.searchsorted(uniq, pmids)

    # ---- per-shard article parquet -----------------------------------------
    print(f"STEP parse pubmed_chunk_*.json ({args.workers} workers)", flush=True)
    jobs = [(i, str(ROOT), str(ROOT), str(staging)) for i in sorted(shard_pmids)]
    n_parsed = 0
    # A pool only pays off across many 1.5 GB shards, and it must be a *spawn*
    # pool: forking a process that has already initialised polars' rayon thread
    # pool deadlocks the child inside `write_parquet`. A single worker (the test
    # path) runs inline and forks nothing at all.
    if args.workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            results = list(ex.map(_convert_shard_attrs, jobs))
    else:
        results = [_convert_shard_attrs(j) for j in jobs]
    for i, n in results:
        if n >= 0:
            n_parsed += n
            print(f"  shard {i}: {n:,} articles", flush=True)
    log["n_articles_parsed"] = int(n_parsed)

    # ---- fp16 [N, 768] item matrix -----------------------------------------
    #
    # Written through a `.npy` memmap accumulator on `--emb-scratch` (crash-safe
    # and resumable across reruns) and finalized once into the harness's
    # `content_d768/text_emb.pt`. No dimensionality reduction: MedCPT's native
    # 768 dims are the benchmark dims (user decision 2026-09-06).
    if args.skip_embeds:
        print("SKIP embeds (--skip-embeds): id map + article parquets only", flush=True)
    else:
        import torch
        import torch.nn.functional as F

        scratch = (
            Path(args.emb_scratch).expanduser() if args.emb_scratch else output / "_staging_emb"
        )
        scratch.mkdir(parents=True, exist_ok=True)
        acc_path = scratch / f"text_emb_d{EMB_DIM_NATIVE}.npy"
        print(
            f"STEP accumulate [{n_items:,}, {EMB_DIM_NATIVE}] fp16 "
            f"({n_items * EMB_DIM_NATIVE * 2 / 1e9:.1f} GB) → {acc_path}",
            flush=True,
        )
        mm = np.lib.format.open_memmap(
            acc_path,
            mode="r+" if acc_path.exists() else "w+",
            dtype=np.float16,
            shape=(n_items, EMB_DIM_NATIVE),
        )
        for i in sorted(shard_pmids):
            npy = ROOT / f"embeds_chunk_{i}.npy"
            if not npy.exists():
                print(f"  WARN missing {npy.name}", flush=True)
                continue
            arr = np.load(npy, mmap_mode="r")
            rows = _rows_for(np.asarray(shard_pmids[i], dtype=np.int64))
            for s in range(0, arr.shape[0], args.batch_rows):
                e = min(s + args.batch_rows, arr.shape[0])
                block = torch.from_numpy(np.array(arr[s:e], dtype=np.float32))
                mm[rows[s:e]] = F.normalize(block, dim=-1).to(torch.float16).numpy()
            del arr
            print(f"  shard {i}: {len(rows):,} rows folded in", flush=True)
            if args.delete_raw:
                npy.unlink()
                (ROOT / f"pubmed_chunk_{i}.json").unlink(missing_ok=True)
        mm.flush()
        del mm

        sub_dir = output / f"content_d{EMB_DIM_NATIVE}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        # np.array (not ascontiguousarray) forces a writable copy: torch needs
        # one, and torch.save materialises the whole tensor anyway.
        acc = np.load(acc_path, mmap_mode="r")
        torch.save(torch.from_numpy(np.array(acc)), sub_dir / "text_emb.pt")
        del acc
        with open(sub_dir / "text_emb.meta.json", "w") as f:
            json.dump(
                {
                    "encoder": "ncbi/MedCPT-Article-Encoder (precomputed, NCBI FTP)",
                    "dim": EMB_DIM_NATIVE,
                    "reduction": "none",
                    "normalization": "l2",
                    "n_rows": int(n_items),
                    "shape": [int(n_items), EMB_DIM_NATIVE],
                    "dtype": "float16",
                },
                f,
                indent=2,
            )
        acc_path.unlink()
        log["text_emb"] = str(sub_dir / "text_emb.pt")
        print(f"  wrote {sub_dir}/text_emb.pt (accumulator deleted)", flush=True)

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
    arts = pl.concat([pl.read_parquet(p) for p in parts], how="vertical")
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
      the article itself is the single relevant item. Always available.
    * ``nfcorpus`` — the NFCorpus (BEIR) biomedical query set. NFCorpus document
      ids *are* PMIDs, so its qrels map straight onto our item ids. Needs
      ``queries.jsonl`` + ``qrels/test.tsv`` staged under
      ``--nfcorpus-dir``; if they are absent the set is skipped with a warning
      rather than failing (BEIR asks that its corpus not be redistributed, so
      the harness never downloads it automatically).
    """
    output = Path(args.output_dir).expanduser()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    frames: list[pl.DataFrame] = []
    log: dict = {"sources": sources}

    if "heldout" in sources:
        heldout = output / "heldout.parquet"
        if not heldout.exists():
            print(f"ERROR missing {heldout} (run `attrs` first)", flush=True)
            return 1
        ho = pl.read_parquet(heldout)
        titles = _titles_for_pmids(set(ho["pmid"].to_list()), args.shards)
        rows = [
            {"query_id": f"heldout:{p}", "text": titles.get(p, ""), "target_id": int(t)}
            for p, t in zip(ho["pmid"].to_list(), ho["item_id"].to_list(), strict=True)
        ]
        rows = [r for r in rows if r["text"]]
        frames.append(pl.DataFrame(rows, schema=_QUERY_SCHEMA))
        log["n_heldout"] = len(rows)
        print(f"  heldout queries: {len(rows):,}", flush=True)

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
            frames.append(pl.DataFrame(rows, schema=_QUERY_SCHEMA))
            log["n_nfcorpus"] = len(rows)
            print(f"  nfcorpus (query, relevant-PMID) pairs in-catalog: {len(rows):,}", flush=True)

    if not frames:
        print("ERROR no query sets built", flush=True)
        return 1
    out = pl.concat(frames, how="vertical")
    out.write_parquet(output / "queries.parquet", compression="zstd")
    log["n_queries"] = out.height
    _merge_log(output, "queries", log)
    print(f"ALL DONE queries — {out.height:,} rows → queries.parquet", flush=True)
    return 0


def _titles_for_pmids(want: set[str], shards_spec: str | None) -> dict[str, str]:
    """Pull titles for a PMID set out of the raw ``pubmed_chunk_*.json``."""
    out: dict[str, str] = {}
    for i in _parse_shards(shards_spec):
        src = ROOT / f"pubmed_chunk_{i}.json"
        if not src.exists():
            continue
        content = json.loads(src.read_text())
        for p in want:
            rec = content.get(p)
            if rec is not None:
                out[p] = (rec.get("t") or "").strip()
        del content
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


def main() -> int:
    p = argparse.ArgumentParser(description="ETL for the PubMed + MedCPT filter-bench dataset.")
    sub = p.add_subparsers(dest="cmd", required=True)

    default_out = str(Path(os.environ.get("RETRIEVE_DATA_ROOT", "data")) / "pubmed-medcpt")

    sp = sub.add_parser("download", help="resumable mirror of the NCBI raw files")
    sp.add_argument("--what", choices=["all", "embeds", "meta", "medline", "mesh"], default="all")
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

    sp = sub.add_parser("convert", help="chunk json/npy → id map, article parquet, fp16 matrix")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--shards", type=str, default=None)
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--batch-rows", type=int, default=100_000)
    sp.add_argument("--skip-embeds", action="store_true", help="id map + attrs only")
    sp.add_argument("--emb-scratch", type=str, default=None,
                    help="dir for the fp16 accumulator (put it on a disk that can hold 55 GB)")
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
    sp.add_argument("--shards", type=str, default=None)
    sp.add_argument("--nfcorpus-dir", type=str, default=None)
    sp.set_defaults(func=cmd_queries)

    sp = sub.add_parser("encode_queries", help="MedCPT-Query-Encoder → content_d768/query_emb.pt")
    sp.add_argument("--output-dir", type=str, default=default_out)
    sp.add_argument("--encoder", type=str, default="ncbi/MedCPT-Query-Encoder")
    sp.add_argument("--device", type=str, default="cuda", help="cpu only for tests")
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--max-seq-length", type=int, default=64)
    sp.set_defaults(func=cmd_encode_queries)

    args = p.parse_args()
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
    "year_to_bucket",
]
