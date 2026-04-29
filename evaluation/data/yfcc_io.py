"""IO helpers for the YFCC-5M pipeline.

Pulls the BigANN'23 ID sample list, the canonical YFCC100M metadata + hash +
autotags dumps from the multimedia-commons AWS Open Data bucket, and provides
a streaming producer for the embedding pipeline. All network endpoints are
public; multimedia-commons is read with anonymous (UNSIGNED) boto3.

Files this module pulls (cached under ``<output-dir>/raw/``):

* ``yfcc100m_id_sampled_10m.txt`` — BigANN'23 ID subsample list (HTTPS).
* ``yfcc100m_dataset.bz2``       — YFCC100M canonical per-photo metadata.
* ``yfcc100m_hash.bz2``          — md5 hashes used to derive S3 image keys.
* ``yfcc100m_autotags-v1.gz``    — visual-concept autotags + probabilities.
"""

from __future__ import annotations

import bz2
import gzip
import io
import time
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from urllib.parse import unquote_plus

# Public endpoints
ID_LIST_URL = (
    "https://comp21storage.z5.web.core.windows.net/yfcc100m_images/"
    "yfcc100m_id_sampled_10m.txt"
)
MULTIMEDIA_COMMONS_BUCKET = "multimedia-commons"
METADATA_KEY = "yfcc100m_dataset.bz2"
HASH_KEY = "yfcc100m_hash.bz2"
AUTOTAGS_KEY = "metadata/autotags/yfcc100m_autotags-v1.gz"

# YFCC100M canonical metadata schema (column order from Yahoo's release notes).
METADATA_COLS = [
    "photo_id",
    "owner_id",
    "owner_name",
    "date_taken",
    "date_uploaded",
    "capture_device",
    "title",
    "description",
    "user_tags",
    "machine_tags",
    "lon",
    "lat",
    "accuracy",
    "page_url",
    "download_url",
    "license_name",
    "license_url",
    "server_id",
    "farm_id",
    "secret",
    "secret_original",
    "extension",
    "marker",
]
NUM_METADATA_COLS = len(METADATA_COLS)

# OpenCLIP B/32 input size; resize to this before feeding the image tower.
IMAGE_RESIZE = 224
EMBEDDING_DIM = 512


# ----- HTTPS / S3 download primitives -----------------------------------------


def _http_download(url: str, dest: Path, *, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def _s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config

    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, max_pool_connections=512),
    )


def _s3_download(key: str, dest: Path, *, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    client = _s3_client()
    with open(tmp, "wb") as f:
        client.download_fileobj(MULTIMEDIA_COMMONS_BUCKET, key, f)
    tmp.rename(dest)
    return dest


def download_id_list(cache_dir: Path) -> Path:
    return _http_download(ID_LIST_URL, cache_dir / "yfcc100m_id_sampled_10m.txt")


def download_metadata_dump(cache_dir: Path) -> Path:
    return _s3_download(METADATA_KEY, cache_dir / "yfcc100m_dataset.bz2")


def download_hash_dump(cache_dir: Path) -> Path:
    return _s3_download(HASH_KEY, cache_dir / "yfcc100m_hash.bz2")


def download_autotags_dump(cache_dir: Path) -> Path:
    return _s3_download(AUTOTAGS_KEY, cache_dir / "yfcc100m_autotags-v1.gz")


# ----- ID list ---------------------------------------------------------------


def load_ids(id_list_path: Path, *, limit: int | None = None) -> list[str]:
    """Read line-delimited photo ids; deterministic order."""
    ids: list[str] = []
    with open(id_list_path) as f:
        for line in f:
            tok = line.strip()
            if not tok:
                continue
            ids.append(tok)
            if limit is not None and len(ids) >= limit:
                break
    return ids


# ----- Per-row metadata ------------------------------------------------------


@dataclass
class MetadataRow:
    photo_id: str
    owner_id: str
    title: str
    description: str
    user_tags: list[str]
    machine_tags: list[str]
    lat: float | None
    lon: float | None
    date_taken: str
    download_url: str
    license_name: str
    extension: str
    marker: int


def _parse_metadata_line(line: str) -> MetadataRow | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < NUM_METADATA_COLS:
        return None
    rec = dict(zip(METADATA_COLS, parts, strict=False))
    title = unquote_plus(rec["title"])
    description = unquote_plus(rec["description"])
    user_tags = [t for t in unquote_plus(rec["user_tags"]).split(",") if t]
    machine_tags = [t for t in unquote_plus(rec["machine_tags"]).split(",") if t]
    try:
        lat = float(rec["lat"]) if rec["lat"] else None
        lon = float(rec["lon"]) if rec["lon"] else None
    except ValueError:
        lat = lon = None
    try:
        marker = int(rec["marker"])
    except ValueError:
        marker = 0
    return MetadataRow(
        photo_id=rec["photo_id"],
        owner_id=rec["owner_id"],
        title=title,
        description=description,
        user_tags=user_tags,
        machine_tags=machine_tags,
        lat=lat,
        lon=lon,
        date_taken=rec["date_taken"],
        download_url=rec["download_url"],
        license_name=rec["license_name"],
        extension=rec["extension"],
        marker=marker,
    )


def stream_metadata_for_ids(
    metadata_path: Path,
    ids: set[str],
) -> Iterator[MetadataRow]:
    """Decompress yfcc100m_dataset.bz2 lazily and yield rows whose id is in ``ids``."""
    with bz2.open(metadata_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            row = _parse_metadata_line(line)
            if row is None:
                continue
            if row.photo_id in ids:
                yield row


def stream_hashes_for_ids(
    hash_path: Path,
    ids: set[str],
) -> Iterator[tuple[str, str]]:
    """Yield (photo_id, md5_hash) pairs from yfcc100m_hash.bz2."""
    with bz2.open(hash_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pid, h = parts[0], parts[1]
            if pid in ids:
                yield pid, h


def stream_autotags_for_ids(
    autotags_path: Path,
    ids: set[str],
    *,
    top_k: int = 8,
) -> Iterator[tuple[str, list[tuple[str, float]]]]:
    """Yield (photo_id, [(concept, prob), …]) sorted by prob desc, capped at top_k.

    Format per line: ``<photo_id>\\t<concept>:<prob>,<concept>:<prob>,…``.
    """
    with gzip.open(autotags_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            pid = parts[0]
            if pid not in ids:
                continue
            tag_str = parts[1]
            tags: list[tuple[str, float]] = []
            for token in tag_str.split(","):
                if ":" not in token:
                    continue
                concept, prob_s = token.rsplit(":", 1)
                try:
                    prob = float(prob_s)
                except ValueError:
                    continue
                tags.append((concept, prob))
            tags.sort(key=lambda kv: kv[1], reverse=True)
            yield pid, tags[:top_k]


# ----- S3 image fetch --------------------------------------------------------


def s3_key_for_hash(photo_hash: str) -> str:
    """Multimedia-commons key for an MD5-of-download-url photo hash."""
    return f"data/images/{photo_hash[0:3]}/{photo_hash[3:6]}/{photo_hash}.jpg"


def s3_head_check(
    keys: list[str],
    *,
    max_workers: int = 256,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[bool]:
    """Concurrent HEAD checks; returns a parallel ``[True/False]`` list."""
    client = _s3_client()
    out = [False] * len(keys)

    def _check(i: int) -> tuple[int, bool]:
        try:
            client.head_object(Bucket=MULTIMEDIA_COMMONS_BUCKET, Key=keys[i])
            return i, True
        except Exception:
            return i, False

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_check, i) for i in range(len(keys))]
        for fut in as_completed(futures):
            i, ok = fut.result()
            out[i] = ok
            done += 1
            if progress_cb is not None and done % 1024 == 0:
                progress_cb(done, len(keys))
    if progress_cb is not None:
        progress_cb(len(keys), len(keys))
    return out


def s3_get_image(key: str, *, client=None) -> bytes:
    if client is None:
        client = _s3_client()
    buf = io.BytesIO()
    client.download_fileobj(MULTIMEDIA_COMMONS_BUCKET, key, buf)
    return buf.getvalue()


# ----- Streaming image producer for the embed pipeline -----------------------


@dataclass
class FetchResult:
    """Output of a producer worker.

    ``tensor`` is the OpenCLIP-preprocessed image (shape ``[3, 224, 224]``,
    float32) on success, ``None`` on any decode/network failure.
    """

    local_idx: int
    photo_id: str
    tensor: object | None  # torch.Tensor or None — typed as object to keep this file torch-free


def image_producer(
    rows: list[tuple[int, str, str]],  # (local_idx, photo_id, s3_key)
    preprocess: Callable[[object], object],
    *,
    max_workers: int = 32,
    queue_size: int = 256,
) -> tuple[Queue, Event]:
    """Spin up a thread pool that fetches + decodes + preprocesses images.

    Returns ``(out_queue, done_event)``. Items in the queue are
    ``FetchResult`` objects; one final ``None`` sentinel is pushed when all
    rows are processed.
    """
    from PIL import Image  # local import keeps module load light when smoking

    out: Queue = Queue(maxsize=queue_size)
    done = Event()
    client = _s3_client()  # one shared client; boto3 clients are thread-safe

    def _worker(local_idx: int, photo_id: str, key: str) -> None:
        try:
            data = s3_get_image(key, client=client)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            tensor = preprocess(img)
        except Exception:
            tensor = None
        out.put(FetchResult(local_idx=local_idx, photo_id=photo_id, tensor=tensor))

    def _drive() -> None:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_worker, local_idx, pid, key) for local_idx, pid, key in rows
            ]
            for fut in as_completed(futures):
                # _worker already pushed; just drain exceptions if any.
                _ = fut.result()
        out.put(None)
        done.set()

    import threading

    threading.Thread(target=_drive, daemon=True).start()
    return out, done


def drain_queue_with_timeout(q: Queue, timeout: float = 1.0):
    """Yield items from ``q`` until a ``None`` sentinel arrives. Used by the GPU loop."""
    while True:
        try:
            item = q.get(timeout=timeout)
        except Empty:
            continue
        if item is None:
            return
        yield item


# ----- Throughput probe ------------------------------------------------------


def s3_throughput_probe(
    sample_keys: list[str],
    *,
    max_workers: int = 32,
    target_bytes: int = 1 << 30,
) -> dict:
    """Pull objects from ``sample_keys`` (skipping failures) until ``target_bytes``
    have arrived, measuring sustained MB/s.

    Returns ``{"mb_per_sec": float, "bytes_read": int, "wall_seconds": float,
    "objects_read": int, "errors": int}``.
    """
    client = _s3_client()
    bytes_read = 0
    objects_read = 0
    errors = 0
    stop = Event()
    t0 = time.monotonic()

    def _pull(key: str) -> int:
        if stop.is_set():
            return 0
        try:
            data = s3_get_image(key, client=client)
            return len(data)
        except Exception:
            return -1

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_pull, k) for k in sample_keys]
        for fut in as_completed(futures):
            n = fut.result()
            if n < 0:
                errors += 1
            else:
                bytes_read += n
                objects_read += 1
            if bytes_read >= target_bytes:
                stop.set()
                # cancel pending; they'll early-out on stop.is_set()
                for f in futures:
                    f.cancel()
                break
    elapsed = time.monotonic() - t0
    mbps = (bytes_read / max(1e-9, elapsed)) / (1 << 20)
    return {
        "mb_per_sec": mbps,
        "bytes_read": bytes_read,
        "wall_seconds": elapsed,
        "objects_read": objects_read,
        "errors": errors,
    }
