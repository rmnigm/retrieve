"""YFCC-5M data pipeline: metadata pull + CLIP labeling + HF Hub upload.

Three subcommands implementing the narrowed scope from
`docs/plans/yfcc-dataset.md`:

* ``prep``   — pull the BigANN'23 sample id list, the YFCC100M metadata +
               hash + autotags dumps, S3-HEAD the candidate image keys,
               drop link-rot, drop empty-text, write ``metadata.parquet``.
* ``embed``  — stream images from multimedia-commons, run OpenCLIP B/32,
               persist per-shard then concatenated image / text / fused
               embedding tensors.
* ``upload`` — push artifacts to a HuggingFace Hub dataset repo.

CLI examples::

    python -m data.yfcc prep   --slice-size 5_000_000 --output-dir data/yfcc/5m
    python -m data.yfcc embed  --output-dir data/yfcc/5m \\
                               --encoder hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K
    python -m data.yfcc upload --output-dir data/yfcc/5m --owner my-hf-user

The ``--smoke`` / ``--smoke-n`` flags route every step into ``<output>/smoke/``
and reduce N to 100 by default — see ``scripts/yfcc_smoke.py``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import polars as pl
from loguru import logger

from data import yfcc_io as yio

DEFAULT_ENCODER = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
DEFAULT_SLICE_SIZE = 5_000_000
DEFAULT_OVERSAMPLE = 1.2
DEFAULT_SHARD_SIZE = 100_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_TEXT_BATCH_SIZE = 1024
DEFAULT_PRODUCER_WORKERS = 32
DEFAULT_HEAD_WORKERS = 256


# ----- prep_log helpers ------------------------------------------------------


def _read_log(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _write_log(path: Path, log: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, indent=2, default=str))


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


# ----- prep ------------------------------------------------------------------


def _prep_step(
    *,
    output_dir: Path,
    slice_size: int,
    oversample: float,
    head_workers: int,
    autotag_top_k: int,
) -> dict[str, Any]:
    """Stage 1 (metadata pull). Idempotent: cached files are reused."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "prep_log.json"
    log = _read_log(log_path)
    t0 = time.monotonic()

    log["stage1_started_at"] = _now_iso()
    log["requested_slice_size"] = slice_size
    log["oversample"] = oversample
    _write_log(log_path, log)

    # 1. id list
    logger.info("Fetching BigANN'23 id sample list…")
    id_list_path = yio.download_id_list(raw_dir)
    working_n = math.ceil(slice_size * oversample)
    ids = yio.load_ids(id_list_path, limit=working_n)
    logger.info("Working set: {} ids (slice_size={}, oversample={:.2f})",
                len(ids), slice_size, oversample)
    log["working_set_size"] = len(ids)
    _write_log(log_path, log)
    id_set = set(ids)

    # 2. metadata dump (~12 GB compressed) — the big one. Cached.
    logger.info("Fetching YFCC100M metadata dump (cached={})…",
                (raw_dir / "yfcc100m_dataset.bz2").exists())
    md_path = yio.download_metadata_dump(raw_dir)

    logger.info("Streaming metadata for {} candidate ids…", len(ids))
    md_rows: dict[str, yio.MetadataRow] = {}
    for row in yio.stream_metadata_for_ids(md_path, id_set):
        md_rows[row.photo_id] = row
    logger.info("Found metadata for {} / {} candidates", len(md_rows), len(ids))
    log["metadata_source"] = "multimedia-commons:yfcc100m_dataset.bz2"
    log["metadata_match_count"] = len(md_rows)

    # 3. hash dump
    logger.info("Fetching YFCC100M hash dump…")
    h_path = yio.download_hash_dump(raw_dir)
    logger.info("Streaming hashes…")
    hash_for_id: dict[str, str] = {}
    for pid, h in yio.stream_hashes_for_ids(h_path, id_set):
        hash_for_id[pid] = h
    logger.info("Found hashes for {} / {} candidates", len(hash_for_id), len(ids))
    log["hash_match_count"] = len(hash_for_id)

    # 4. autotags
    logger.info("Fetching YFCC100M autotags…")
    at_path = yio.download_autotags_dump(raw_dir)
    logger.info("Streaming autotags (top {} per item)…", autotag_top_k)
    autotags_for_id: dict[str, list[tuple[str, float]]] = {}
    for pid, tags in yio.stream_autotags_for_ids(at_path, id_set, top_k=autotag_top_k):
        autotags_for_id[pid] = tags
    logger.info("Found autotags for {} / {} candidates", len(autotags_for_id), len(ids))
    log["autotag_match_count"] = len(autotags_for_id)
    _write_log(log_path, log)

    # 5. join + filter empty-text + filter videos + S3 head check
    keep_records: list[dict[str, Any]] = []
    s3_keys_to_check: list[str] = []
    for pid in ids:
        if pid not in md_rows or pid not in hash_for_id:
            continue
        m = md_rows[pid]
        if m.marker != 0:  # videos out of scope
            continue
        if not (m.title or m.description or m.machine_tags):
            continue
        h = hash_for_id[pid]
        s3_keys_to_check.append(yio.s3_key_for_hash(h))
        autotag_pairs = autotags_for_id.get(pid, [])
        keep_records.append({
            "photo_id": pid,
            "photo_hash": h,
            "s3_key": yio.s3_key_for_hash(h),
            "owner_id": m.owner_id,
            "title": m.title,
            "description": m.description,
            "user_tags": m.user_tags,
            "machine_tags": m.machine_tags,
            "autotag_concepts": [c for c, _p in autotag_pairs],
            "autotag_probs": [p for _c, p in autotag_pairs],
            "lat": m.lat,
            "lon": m.lon,
            "date_taken": m.date_taken,
            "download_url": m.download_url,
            "license_name": m.license_name,
            "extension": m.extension,
        })
    logger.info("Pre-HEAD candidates after join + text filter: {}", len(keep_records))
    log["dropped_text_or_join"] = len(ids) - len(keep_records)

    logger.info("S3 HEAD-checking {} keys with {} workers…",
                len(s3_keys_to_check), head_workers)

    def _progress(done: int, total: int) -> None:
        if done % (max(1, total // 50)) == 0 or done == total:
            logger.debug("HEAD progress: {}/{}", done, total)

    presence = yio.s3_head_check(
        s3_keys_to_check, max_workers=head_workers, progress_cb=_progress
    )
    survived = [r for r, ok in zip(keep_records, presence, strict=True) if ok]
    log["dropped_link_rot"] = len(keep_records) - len(survived)
    logger.info("Survived link rot: {} / {}", len(survived), len(keep_records))

    # 6. take first slice_size and densify
    survived = survived[:slice_size]
    for dense_id, rec in enumerate(survived):
        rec["dense_id"] = dense_id
    log["final_n"] = len(survived)

    # 7. write parquet
    df = pl.DataFrame(survived)
    md_out = output_dir / "metadata.parquet"
    df.write_parquet(md_out, compression="snappy")
    logger.info("Wrote {} rows → {}", df.height, md_out)

    # 8. kept ids
    kept_ids_path = output_dir / "kept_ids.txt"
    kept_ids_path.write_text("\n".join(rec["photo_id"] for rec in survived) + "\n")

    log["stage1_wall_seconds"] = round(time.monotonic() - t0, 2)
    log["stage1_finished_at"] = _now_iso()
    _write_log(log_path, log)
    return log


@click.group()
def cli() -> None:
    """YFCC-5M dataset prep / embed / upload."""


@cli.command("prep")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--slice-size", type=int, default=DEFAULT_SLICE_SIZE,
              help=f"Target N after link-rot filter (default {DEFAULT_SLICE_SIZE}).")
@click.option("--oversample", type=float, default=DEFAULT_OVERSAMPLE,
              help="Oversample multiplier on the BigANN list to absorb rot.")
@click.option("--head-workers", type=int, default=DEFAULT_HEAD_WORKERS)
@click.option("--autotag-top-k", type=int, default=8)
@click.option("--smoke/--no-smoke", default=False,
              help="Smoke mode: write under <output-dir>/smoke/ and use --slice-size as-is.")
@click.option("--smoke-n", type=int, default=100,
              help="If --smoke and --slice-size is unset, use this many ids.")
def prep_cmd(
    output_dir: Path,
    slice_size: int,
    oversample: float,
    head_workers: int,
    autotag_top_k: int,
    smoke: bool,
    smoke_n: int,
) -> None:
    """Stage 1: pull metadata + S3 HEAD-check + write metadata.parquet."""
    if smoke:
        output_dir = output_dir / "smoke"
        if slice_size == DEFAULT_SLICE_SIZE:
            slice_size = smoke_n
        oversample = max(oversample, 1.0)
    _prep_step(
        output_dir=output_dir,
        slice_size=slice_size,
        oversample=oversample,
        head_workers=head_workers,
        autotag_top_k=autotag_top_k,
    )


# ----- embed -----------------------------------------------------------------


def _load_clip(encoder: str, device: str):
    """Load OpenCLIP model, preprocess transform, and tokenizer.

    Accepts either a plain OpenCLIP name like ``ViT-B-32`` (then ``pretrained``
    is resolved to ``laion2b_s34b_b79k``) or a ``hf-hub:<repo>`` reference.
    """
    import open_clip
    import torch

    if encoder.startswith("hf-hub:"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            encoder, device=device
        )
        tokenizer = open_clip.get_tokenizer(encoder)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            encoder, pretrained="laion2b_s34b_b79k", device=device
        )
        tokenizer = open_clip.get_tokenizer(encoder)
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.half()
    return model, preprocess, tokenizer


def _encode_image_shard(
    rows: list[dict[str, Any]],
    model,
    preprocess,
    *,
    device: str,
    batch_size: int,
    producer_workers: int,
):
    """Stream images for ``rows`` and produce ``[len(rows), 512]`` fp16, L2-normalized.

    Failed fetches are filled with NaN; the post-concat step drops those rows.
    """
    import torch
    import torch.nn.functional as F

    out = torch.full(
        (len(rows), yio.EMBEDDING_DIM), float("nan"), dtype=torch.float16
    )
    fetch_jobs = [(i, r["photo_id"], r["s3_key"]) for i, r in enumerate(rows)]
    q, _done = yio.image_producer(
        fetch_jobs, preprocess, max_workers=producer_workers, queue_size=4 * batch_size
    )

    pending: list[tuple[int, torch.Tensor]] = []
    n_failed = 0

    def _flush() -> None:
        nonlocal pending
        if not pending:
            return
        idxs = [p[0] for p in pending]
        batch = torch.stack([p[1] for p in pending], dim=0).to(device)
        if device == "cuda":
            batch = batch.half()
        with torch.inference_mode():
            feats = model.encode_image(batch)
        feats = F.normalize(feats.float(), dim=-1).to(torch.float16).cpu()
        for j, i in enumerate(idxs):
            out[i] = feats[j]
        pending = []

    for item in yio.drain_queue_with_timeout(q):
        if item.tensor is None:
            n_failed += 1
            continue
        pending.append((item.local_idx, item.tensor))
        if len(pending) >= batch_size:
            _flush()
    _flush()
    return out, n_failed


def _encode_text_shard(
    rows: list[dict[str, Any]],
    model,
    tokenizer,
    *,
    device: str,
    batch_size: int,
):
    """Build ``title || description || machine_tags`` strings, encode → fp16 L2-norm."""
    import torch
    import torch.nn.functional as F

    out = torch.zeros((len(rows), yio.EMBEDDING_DIM), dtype=torch.float16)

    def _build_text(r: dict[str, Any]) -> str:
        parts: list[str] = []
        if r.get("title"):
            parts.append(str(r["title"]))
        if r.get("description"):
            parts.append(str(r["description"]))
        mt = r.get("machine_tags") or []
        if mt:
            parts.append(" ".join(str(t) for t in mt))
        return " ".join(parts).strip() or " "

    texts = [_build_text(r) for r in rows]

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(batch_texts).to(device)
        with torch.inference_mode():
            feats = model.encode_text(tokens)
        feats = F.normalize(feats.float(), dim=-1).to(torch.float16).cpu()
        out[start : start + feats.shape[0]] = feats
    return out


def _embed_step(
    *,
    output_dir: Path,
    encoder: str,
    device: str,
    shard_size: int,
    batch_size: int,
    producer_workers: int,
    keep_shards: bool,
    do_throughput_probe: bool,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    md_path = output_dir / "metadata.parquet"
    if not md_path.exists():
        raise click.ClickException(
            f"Missing {md_path}. Run `python -m data.yfcc prep --output-dir {output_dir}` first."
        )

    log_path = output_dir / "prep_log.json"
    log = _read_log(log_path)
    log["stage2_started_at"] = _now_iso()
    log["encoder"] = encoder
    log["device"] = device
    _write_log(log_path, log)

    df = pl.read_parquet(md_path)
    rows = df.to_dicts()
    n = len(rows)
    logger.info("Embedding {} rows; encoder={}, device={}, shard_size={}",
                n, encoder, device, shard_size)

    logger.info("Loading OpenCLIP model…")
    model, preprocess, tokenizer = _load_clip(encoder, device)

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Optional throughput probe
    if do_throughput_probe:
        sample_keys = [r["s3_key"] for r in rows[: min(2048, n)]]
        logger.info("Running 1 GB S3 throughput probe…")
        probe = yio.s3_throughput_probe(sample_keys, max_workers=producer_workers)
        log["throughput_probe"] = probe
        logger.info("Probe: {:.1f} MB/s sustained over {} objects ({:.1f} s)",
                    probe["mb_per_sec"], probe["objects_read"], probe["wall_seconds"])
        _write_log(log_path, log)

    n_shards = math.ceil(n / shard_size)
    shard_failures: list[int] = []
    t0 = time.monotonic()

    for shard_idx in range(n_shards):
        img_path = shards_dir / f"item_embs_image_{shard_idx:04d}.pt"
        txt_path = shards_dir / f"item_embs_text_{shard_idx:04d}.pt"
        if img_path.exists() and txt_path.exists():
            logger.info("Shard {}/{}: cached, skipping", shard_idx + 1, n_shards)
            continue
        start = shard_idx * shard_size
        end = min(start + shard_size, n)
        shard_rows = rows[start:end]
        logger.info("Shard {}/{}: rows [{}, {}) ({} items)",
                    shard_idx + 1, n_shards, start, end, len(shard_rows))

        t_shard = time.monotonic()
        img_emb, n_failed = _encode_image_shard(
            shard_rows, model, preprocess,
            device=device, batch_size=batch_size,
            producer_workers=producer_workers,
        )
        shard_failures.append(n_failed)
        torch.save(img_emb, img_path)
        logger.info("  image side: {:.1f}s, {} failed", time.monotonic() - t_shard, n_failed)

        t_text = time.monotonic()
        txt_emb = _encode_text_shard(
            shard_rows, model, tokenizer,
            device=device, batch_size=DEFAULT_TEXT_BATCH_SIZE,
        )
        torch.save(txt_emb, txt_path)
        logger.info("  text side: {:.1f}s", time.monotonic() - t_text)

    log["shard_failures_per_shard"] = shard_failures
    log["total_image_failures"] = sum(shard_failures)
    log["stage2_wall_seconds"] = round(time.monotonic() - t0, 2)
    _write_log(log_path, log)

    # Concatenate
    logger.info("Concatenating {} shards…", n_shards)
    img_chunks = [torch.load(shards_dir / f"item_embs_image_{i:04d}.pt") for i in range(n_shards)]
    txt_chunks = [torch.load(shards_dir / f"item_embs_text_{i:04d}.pt") for i in range(n_shards)]
    img = torch.cat(img_chunks, dim=0)
    txt = torch.cat(txt_chunks, dim=0)
    assert img.shape == txt.shape == (n, yio.EMBEDDING_DIM), (
        f"shape mismatch: img={tuple(img.shape)}, txt={tuple(txt.shape)}, expected=({n}, 512)"
    )

    # Identify NaN rows (image-side fetch failures) and densify
    nan_mask = img.float().isnan().any(dim=1)
    survived = (~nan_mask).nonzero(as_tuple=True)[0]
    n_survived = int(survived.numel())
    logger.info("Post-embed: {} / {} rows survived (image fetch + decode succeeded)",
                n_survived, n)
    log["final_n_post_embed"] = n_survived
    log["dropped_image_fetch_or_decode"] = int(n - n_survived)

    img = img[survived]
    txt = txt[survived]
    fused = F.normalize(img.float() + txt.float(), dim=-1).to(torch.float16)

    torch.save(img, output_dir / "item_embs_image.pt")
    torch.save(txt, output_dir / "item_embs_text.pt")
    torch.save(fused, output_dir / "item_embs_fused.pt")
    logger.info("Wrote item_embs_image.pt / _text.pt / _fused.pt → {}", output_dir)

    # Re-densify metadata to align with embedding rows
    keep_idx = survived.tolist()
    df_out = df[keep_idx].with_columns(pl.Series("dense_id", list(range(n_survived))))
    df_out.write_parquet(md_path, compression="snappy")
    logger.info("Updated metadata.parquet ({} rows)", df_out.height)

    if not keep_shards:
        logger.info("Removing shard tmp dir {}", shards_dir)
        shutil.rmtree(shards_dir)

    log["embedding_dim"] = yio.EMBEDDING_DIM
    log["embedding_dtype"] = "float16"
    log["stage2_finished_at"] = _now_iso()
    _write_log(log_path, log)
    return log


@cli.command("embed")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--encoder", type=str, default=DEFAULT_ENCODER)
@click.option("--device", type=str, default="cuda")
@click.option("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
@click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
@click.option("--producer-workers", type=int, default=DEFAULT_PRODUCER_WORKERS)
@click.option("--keep-shards/--no-keep-shards", default=False)
@click.option("--throughput-probe/--no-throughput-probe", default=True)
@click.option("--smoke/--no-smoke", default=False,
              help="Smoke mode: read <output-dir>/smoke/, force device=cpu, smaller batches.")
@click.option("--cpu-threads", type=int, default=0,
              help="If > 0, torch.set_num_threads(N) before loading the model. "
                   "Use this when sharing a CPU host with a training run — "
                   "OpenCLIP inference otherwise saturates every core.")
def embed_cmd(
    output_dir: Path,
    encoder: str,
    device: str,
    shard_size: int,
    batch_size: int,
    producer_workers: int,
    keep_shards: bool,
    throughput_probe: bool,
    smoke: bool,
    cpu_threads: int,
) -> None:
    """Stage 2: stream images, run OpenCLIP, persist embedding tensors."""
    import torch as _torch
    if smoke:
        output_dir = output_dir / "smoke"
        device = "cpu"
        shard_size = min(shard_size, 100)
        batch_size = min(batch_size, 8)
        producer_workers = min(producer_workers, 8)
        throughput_probe = False
        if cpu_threads == 0:
            cpu_threads = 2  # never starve a co-tenant in smoke mode
    if cpu_threads > 0:
        _torch.set_num_threads(cpu_threads)
    _embed_step(
        output_dir=output_dir,
        encoder=encoder,
        device=device,
        shard_size=shard_size,
        batch_size=batch_size,
        producer_workers=producer_workers,
        keep_shards=keep_shards,
        do_throughput_probe=throughput_probe,
    )


# ----- upload ----------------------------------------------------------------


REQUIRED_UPLOAD_FILES = (
    "item_embs_image.pt",
    "item_embs_text.pt",
    "item_embs_fused.pt",
    "metadata.parquet",
    "prep_log.json",
)
UPLOAD_IGNORE = ["raw/*", "shards/*", "smoke/*"]


def _build_dataset_card(output_dir: Path, repo_id: str) -> str:
    log = _read_log(output_dir / "prep_log.json")
    parts: list[str] = [
        f"# {repo_id}",
        "",
        "Re-embedded YFCC100M subsample for filtered-ANN benchmarking.",
        "",
        "## Run metadata",
        "",
    ]
    for k in (
        "encoder", "embedding_dim", "embedding_dtype", "final_n_post_embed",
        "requested_slice_size", "metadata_source",
        "dropped_link_rot", "dropped_text_or_join", "dropped_image_fetch_or_decode",
        "stage1_wall_seconds", "stage2_wall_seconds",
        "stage1_finished_at", "stage2_finished_at",
    ):
        if k in log:
            parts.append(f"- **{k}**: `{log[k]}`")
    parts.append("")
    parts.append("## Files")
    parts.append("")
    for fname in REQUIRED_UPLOAD_FILES:
        p = output_dir / fname
        if p.exists():
            mb = p.stat().st_size / (1024 * 1024)
            parts.append(f"- `{fname}` ({mb:.1f} MB)")
    parts.append("")
    parts.append("## Loading")
    parts.append("")
    parts.append("```python")
    parts.append("from huggingface_hub import snapshot_download")
    parts.append(
        f'snapshot_download(repo_id="{repo_id}", repo_type="dataset",'
        ' local_dir="data/yfcc/5m")'
    )
    parts.append("```")
    parts.append("")
    return "\n".join(parts)


def _upload_step(
    *,
    output_dir: Path,
    owner: str,
    repo_name: str,
    private: bool,
    dry_run: bool,
    write_card: bool,
    token: str | None,
) -> None:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    repo_id = f"{owner}/{repo_name}"

    missing = [f for f in REQUIRED_UPLOAD_FILES if not (output_dir / f).exists()]
    if missing:
        raise click.ClickException(
            f"Missing required files in {output_dir}: {missing}. "
            "Run `prep` and `embed` first."
        )

    files = sorted(output_dir.iterdir())
    visible = [
        p for p in files
        if p.is_file() and not any(
            p.match(pat) or p.name.startswith(pat.split("/")[0] + "/")
            for pat in UPLOAD_IGNORE
        )
    ]
    size_mb = sum(p.stat().st_size for p in visible) / (1024 * 1024)
    logger.info("Uploading {} ({:.1f} MB, {} top-level files) → {} (private={})",
                output_dir, size_mb, len(visible), repo_id, private)

    if dry_run:
        for p in visible:
            mb = p.stat().st_size / (1024 * 1024)
            logger.info("  would upload: {} ({:.1f} MB)", p.name, mb)
        # Honour the ignore patterns visibly
        for pat in UPLOAD_IGNORE:
            sub = output_dir / pat.split("/")[0]
            if sub.exists():
                logger.info("  would ignore: {} (matches '{}')", sub, pat)
        return

    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
        )
    except HfHubHTTPError as e:
        raise click.ClickException(f"Failed to create/access {repo_id}: {e}") from e

    readme_path = output_dir / "README.md"
    wrote_card = False
    if write_card and not readme_path.exists():
        readme_path.write_text(_build_dataset_card(output_dir, repo_id))
        wrote_card = True
        logger.info("Wrote dataset card: {}", readme_path)

    try:
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {output_dir.name}",
            ignore_patterns=UPLOAD_IGNORE,
        )
    finally:
        if wrote_card:
            readme_path.unlink(missing_ok=True)
    logger.success("Uploaded https://huggingface.co/datasets/{}", repo_id)


@cli.command("upload")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--owner", required=True, help="HF user or org.")
@click.option("--repo-name", default="yfcc-5m-clipB32",
              help="Dataset repo name (default: yfcc-5m-clipB32).")
@click.option("--private/--public", default=True)
@click.option("--dry-run", is_flag=True)
@click.option("--write-card/--no-write-card", default=True)
@click.option("--token", default=None, envvar="HF_TOKEN")
def upload_cmd(
    output_dir: Path,
    owner: str,
    repo_name: str,
    private: bool,
    dry_run: bool,
    write_card: bool,
    token: str | None,
) -> None:
    """Stage 3: upload artifacts to a HuggingFace Hub dataset repo."""
    _upload_step(
        output_dir=output_dir,
        owner=owner,
        repo_name=repo_name,
        private=private,
        dry_run=dry_run,
        write_card=write_card,
        token=token,
    )


if __name__ == "__main__":
    # `os` and `Any` are imported for type-annotation use elsewhere; reference here
    # to keep ruff quiet about the unused name in CLI-only invocation.
    _ = os.environ.get  # noqa: B018
    _ = Any
    cli()
