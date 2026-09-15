"""``bench upload`` — mirror ``results/`` to a HuggingFace dataset repo (H §3.1: a mirror, no
staging layout). ``_logs/`` and ``_parity/`` (campaign scratch) are skipped; everything else
under ``--results`` goes up as-is, so the JSONL + samples sidecars keep their names."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import click

EVAL_DIR = Path(__file__).resolve().parents[1]
IGNORE = ["_*", "_*/**", "**/_*", "**/_*/**"]


def files(results: Path) -> list[Path]:
    return sorted(
        p
        for p in results.rglob("*")
        if p.is_file() and not any(part.startswith("_") for part in p.relative_to(results).parts)
    )


@click.command()
@click.option("--repo-id", required=True, help="target HF dataset repo, e.g. user/retrieval-evals")
@click.option("--results", type=click.Path(path_type=Path), default=EVAL_DIR / "results")
@click.option("--private", is_flag=True, help="create the repo as private")
@click.option("--dry-run", is_flag=True, help="list what would be uploaded")
def upload(repo_id: str, results: Path, private: bool, dry_run: bool) -> None:
    """Mirror the results directory to a HuggingFace dataset repo."""
    results = results if results.is_absolute() else EVAL_DIR / results
    listing = files(results)
    total = sum(p.stat().st_size for p in listing)
    for p in listing:
        click.echo(f"  {p.relative_to(results)} ({p.stat().st_size:,} bytes)")
    click.echo(f"{len(listing)} files, {total:,} bytes -> {repo_id} (private={private})")
    if dry_run:
        return
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(results),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=IGNORE,
        commit_message=f"results mirror {dt.date.today().isoformat()}",
    )
    click.echo(f"done: https://huggingface.co/datasets/{repo_id}")


__all__ = ["upload"]
