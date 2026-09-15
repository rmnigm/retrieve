"""``train upload-checkpoint`` — push a local checkpoint directory (or every one under
``data/<dataset>/checkpoints/`` with ``--ckpt-id all``) to ``<repo>/checkpoints/<ckpt-id>/``
of the dataset's HF eval repo, through ``eval_datasets.hub.upload_checkpoint``. Needs a
write-scoped HF token (``hf auth login`` or ``HF_TOKEN``).

    uv run train upload-checkpoint --dataset goodreads-work-id --ckpt-id all --dry-run
"""

from __future__ import annotations

import click
from loguru import logger

from eval_datasets import hub


def _resolve_ckpt_ids(dataset: str, selector: str) -> list[str]:
    root = hub.eval_dir(dataset) / "checkpoints"
    if not root.exists():
        raise click.ClickException(f"Checkpoint root not found: {root}")
    all_ids = sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.iterdir()))
    if selector == "all":
        if not all_ids:
            raise click.ClickException(f"No non-empty checkpoint subdirs under {root}")
        return all_ids
    if selector not in all_ids:
        avail = ", ".join(all_ids) or "(none)"
        raise click.ClickException(f"Checkpoint not found: {selector}\nAvailable: {avail}")
    return [selector]


@click.command()
@click.option(
    "--dataset",
    required=True,
    type=click.Choice(sorted(hub.EVAL_REPOS)),
    help="Logical dataset name (key into EVAL_REPOS).",
)
@click.option(
    "--ckpt-id",
    required=True,
    help="Subdir name under data/<dataset>/checkpoints/, or 'all'.",
)
@click.option("--private/--public", default=True, help="Repo visibility (default: private).")
@click.option("--dry-run", is_flag=True, help="List files without uploading.")
@click.option("--write-card/--no-write-card", default=True, help="Generate README.md if absent.")
@click.option("--include-epoch-snapshots", is_flag=True, help="Keep gsasrec-ep*.pt snapshots.")
def upload_checkpoint(
    dataset: str,
    ckpt_id: str,
    private: bool,
    dry_run: bool,
    write_card: bool,
    include_epoch_snapshots: bool,
) -> None:
    """Upload checkpoint(s) to the dataset's HF eval repo."""
    ids = _resolve_ckpt_ids(dataset, ckpt_id)
    for cid in ids:
        src = hub.ckpt_dir(dataset, cid)
        logger.info("=== {} ===", src)
        hub.upload_checkpoint(
            dataset,
            cid,
            source=src,
            private=private,
            include_epoch_snapshots=include_epoch_snapshots,
            write_card=write_card,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    upload_checkpoint()


__all__ = ["upload_checkpoint"]
