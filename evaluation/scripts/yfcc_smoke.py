"""End-to-end smoke test for the YFCC-5M data pipeline.

Runs ``prep --smoke`` → ``embed --smoke`` → ``upload --dry-run`` on the first
~100 ids of the BigANN'23 sample list, validates tensor shapes / dtypes /
norms, and exits non-zero on any invariant violation. CPU-only.

The smoke test pulls real data over the network. The dominant cost is the
first-run download of three multimedia-commons artifacts (~12 GB metadata,
~few GB hash, ~few GB autotags). They are cached under
``<output-dir>/smoke/raw/`` so repeat runs are network-free.

Usage::

    python -m scripts.yfcc_smoke --output-dir /tmp/yfcc-smoke
    python -m scripts.yfcc_smoke --output-dir /tmp/yfcc-smoke --owner my-hf-user
    # skip the network-heavy prep step (assumes you've already run it once):
    python -m scripts.yfcc_smoke --output-dir /tmp/yfcc-smoke --skip-prep
"""

from __future__ import annotations

from pathlib import Path

import click
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger

from data import yfcc as yfcc_cli
from data import yfcc_io as yio

REQUIRED_META_COLS = {
    "photo_id", "photo_hash", "s3_key", "title", "description",
    "user_tags", "machine_tags", "autotag_concepts", "autotag_probs",
    "lat", "lon", "date_taken", "dense_id",
}


def _check_embeddings(smoke_dir: Path) -> int:
    img = torch.load(smoke_dir / "item_embs_image.pt", weights_only=True)
    txt = torch.load(smoke_dir / "item_embs_text.pt", weights_only=True)
    fused = torch.load(smoke_dir / "item_embs_fused.pt", weights_only=True)

    assert img.shape == txt.shape == fused.shape, (
        f"shape mismatch: img={tuple(img.shape)}, "
        f"txt={tuple(txt.shape)}, fused={tuple(fused.shape)}"
    )
    n, d = img.shape
    assert d == yio.EMBEDDING_DIM, f"expected D={yio.EMBEDDING_DIM}, got {d}"
    assert img.dtype == torch.float16, f"expected fp16, got {img.dtype}"
    assert txt.dtype == torch.float16, f"expected fp16, got {txt.dtype}"
    assert fused.dtype == torch.float16, f"expected fp16, got {fused.dtype}"
    assert n > 0, "no rows survived the embed pipeline"

    img_norms = img.float().norm(dim=1)
    txt_norms = txt.float().norm(dim=1)
    fused_norms = fused.float().norm(dim=1)
    ones = torch.ones(n)
    assert torch.allclose(img_norms, ones, atol=1e-2), (
        f"image embeddings not L2-normed: max-err={(img_norms - 1).abs().max().item():.4f}"
    )
    assert torch.allclose(txt_norms, ones, atol=1e-2), (
        f"text embeddings not L2-normed: max-err={(txt_norms - 1).abs().max().item():.4f}"
    )
    assert torch.allclose(fused_norms, ones, atol=1e-2), (
        f"fused embeddings not L2-normed: max-err={(fused_norms - 1).abs().max().item():.4f}"
    )

    expected_fused = F.normalize(img.float() + txt.float(), dim=-1)
    diff = (fused.float() - expected_fused).abs().max().item()
    assert diff < 5e-3, f"fused != normalize(img + txt); max-err={diff:.4f}"

    logger.success(
        "Embeddings OK: N={}, D={}, all L2-normed, fused = normalize(img+txt) within 5e-3",
        n, d,
    )
    return n


def _check_metadata_alignment(smoke_dir: Path, n_emb: int) -> None:
    meta = pl.read_parquet(smoke_dir / "metadata.parquet")
    assert meta.height == n_emb, (
        f"metadata.parquet has {meta.height} rows but embeddings have {n_emb}"
    )
    missing = REQUIRED_META_COLS - set(meta.columns)
    assert not missing, f"metadata.parquet missing columns: {missing}"
    dense_ids = meta["dense_id"].to_list()
    assert dense_ids == list(range(n_emb)), "dense_id is not a contiguous 0..N range"
    logger.success(
        "Metadata aligned: {} rows, dense_id contiguous, all required cols present",
        n_emb,
    )


@click.command()
@click.option("--output-dir", type=click.Path(path_type=Path), required=True,
              help="Smoke artifacts will be written under <output-dir>/smoke/.")
@click.option("--smoke-n", type=int, default=100, help="Number of ids to test.")
@click.option("--owner", default=None,
              help="Optional HF owner; if set, exercises upload --dry-run.")
@click.option("--skip-prep", is_flag=True,
              help="Skip Stage 1 (assumes <output-dir>/smoke/metadata.parquet exists).")
@click.option("--skip-embed", is_flag=True,
              help="Skip Stage 2 (only re-runs the validation checks).")
@click.option("--cpu-threads", type=int, default=2,
              help="torch.set_num_threads cap during the embed step. Default 2 — "
                   "OpenCLIP B/32 inference is multi-threaded and will otherwise "
                   "saturate every core, starving any training run sharing the host.")
def main(
    output_dir: Path,
    smoke_n: int,
    owner: str | None,
    skip_prep: bool,
    skip_embed: bool,
    cpu_threads: int,
) -> None:
    smoke_dir = output_dir / "smoke"
    torch.set_num_threads(max(1, cpu_threads))

    if not skip_prep:
        logger.info("== Stage 1 (prep, smoke) ==")
        yfcc_cli._prep_step(
            output_dir=smoke_dir,
            slice_size=smoke_n,
            oversample=1.0,
            head_workers=64,
            autotag_top_k=8,
        )
    else:
        logger.info("Skipping prep (assumes {} already exists)", smoke_dir / "metadata.parquet")

    if not skip_embed:
        logger.info("== Stage 2 (embed, smoke, CPU) ==")
        yfcc_cli._embed_step(
            output_dir=smoke_dir,
            encoder=yfcc_cli.DEFAULT_ENCODER,
            device="cpu",
            shard_size=max(1, smoke_n),
            batch_size=8,
            producer_workers=8,
            keep_shards=False,
            do_throughput_probe=False,
        )
    else:
        logger.info("Skipping embed (running validation only)")

    logger.info("== Validation ==")
    n = _check_embeddings(smoke_dir)
    _check_metadata_alignment(smoke_dir, n)

    if owner is not None:
        logger.info("== Stage 3 (upload, dry-run) ==")
        yfcc_cli._upload_step(
            output_dir=smoke_dir,
            owner=owner,
            repo_name="yfcc-smoke-test",
            private=True,
            dry_run=True,
            write_card=True,
            token=None,
        )

    logger.success("✓ smoke OK ({} items)", n)


if __name__ == "__main__":
    main()
