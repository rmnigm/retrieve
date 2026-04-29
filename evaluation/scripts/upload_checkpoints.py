"""Upload local checkpoint directories to the Hugging Face Hub.

Each checkpoint subdir under ``evaluation/checkpoints/`` becomes its own model
repo on the Hub (one repo per run, easy to download just one).

Auth: requires an HF token with write access — either ``hf auth login`` once,
or set ``HF_TOKEN`` in the environment.

Examples
--------
Upload one run to a private repo:

    uv run python -m scripts.upload_checkpoints \\
        --owner my-hf-username \\
        --checkpoint gsasrec-500m-listens-d128-drop0.5 \\
        --private

Upload all runs (one repo each) and dry-run first:

    uv run python -m scripts.upload_checkpoints --owner my-hf-username --checkpoint all --dry-run
    uv run python -m scripts.upload_checkpoints --owner my-hf-username --checkpoint all
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import click
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / "checkpoints"

# `gsasrec-ep{N}-ndcg10{X}.pt` is saved at the best val epoch and then copied
# to `best_model.pt` at training end — same bytes, twice the storage. Skip the
# epoch-tagged copy by default; pass --include-epoch-snapshots to keep it.
EPOCH_SNAPSHOT_PATTERN = "gsasrec-ep*-ndcg*.pt"


def _filter_files(ckpt_dir: Path, ignore_patterns: list[str]) -> list[Path]:
    return [
        p for p in sorted(ckpt_dir.iterdir())
        if p.is_file() and not any(fnmatch.fnmatch(p.name, pat) for pat in ignore_patterns)
    ]


def _build_model_card(ckpt_dir: Path, repo_id: str, files: list[Path]) -> str:
    """Generate a minimal model-card README from local metadata files."""
    parts: list[str] = [f"# {ckpt_dir.name}\n"]

    eval_path = ckpt_dir / "eval_quality.json"
    if eval_path.exists():
        meta = json.loads(eval_path.read_text())
        metrics = meta.get("metrics", {})
        parts.append("## Test metrics\n")
        for k, v in metrics.items():
            parts.append(f"- **{k}**: {v:.4f}")
        if (target := meta.get("paper_target")):
            parts.append("\n## Paper target\n")
            for k, v in target.items():
                parts.append(f"- {k}: {v:.4f}")
        parts.append("")

    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        parts.append("## Training config\n")
        parts.append("```json")
        parts.append(config_path.read_text().rstrip())
        parts.append("```\n")

    parts.append("## Files\n")
    for p in files:
        size_mb = p.stat().st_size / (1024 * 1024)
        parts.append(f"- `{p.name}` ({size_mb:.1f} MB)")

    parts.append(
        "\n## Loading\n"
        "See [`docs/system/checkpoints.md`]"
        "(https://github.com/) in the source repo for the full loading recipe.\n"
        f"\nDownload locally with:\n"
        "```python\n"
        "from huggingface_hub import snapshot_download\n"
        f'snapshot_download(repo_id="{repo_id}", local_dir="checkpoints/{ckpt_dir.name}")\n'
        "```\n"
    )
    return "\n".join(parts)


def _resolve_checkpoints(selector: str) -> list[Path]:
    if not CHECKPOINT_ROOT.exists():
        raise click.ClickException(f"Checkpoint root not found: {CHECKPOINT_ROOT}")

    def _has_files(d: Path) -> bool:
        return any(p.is_file() for p in d.iterdir())

    all_dirs = sorted(p for p in CHECKPOINT_ROOT.iterdir() if p.is_dir())
    if selector == "all":
        non_empty = [d for d in all_dirs if _has_files(d)]
        skipped = [d.name for d in all_dirs if not _has_files(d)]
        if skipped:
            logger.warning("Skipping empty checkpoint dirs: {}", ", ".join(skipped))
        if not non_empty:
            raise click.ClickException(f"No non-empty subdirs under {CHECKPOINT_ROOT}")
        return non_empty

    target = CHECKPOINT_ROOT / selector
    if not target.is_dir():
        available = ", ".join(p.name for p in all_dirs) or "(none)"
        raise click.ClickException(
            f"Checkpoint dir not found: {target}\nAvailable: {available}"
        )
    if not _has_files(target):
        raise click.ClickException(f"Checkpoint dir is empty: {target}")
    return [target]


@click.command()
@click.option("--owner", required=True, help="HF user or org that will own the repo(s).")
@click.option(
    "--checkpoint",
    required=True,
    help="Subdir name under evaluation/checkpoints/, or 'all'.",
)
@click.option(
    "--repo-name",
    default=None,
    help="Override repo name (only valid when uploading a single checkpoint).",
)
@click.option("--private/--public", default=True, help="Repo visibility (default: private).")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print what would be uploaded without creating repos or pushing files.",
)
@click.option(
    "--write-card/--no-write-card",
    default=True,
    help="Generate README.md from eval_quality.json/config.json if absent.",
)
@click.option(
    "--include-epoch-snapshots",
    is_flag=True,
    help=(
        "Also upload the epoch-tagged `gsasrec-ep*-ndcg*.pt` snapshot. "
        "By default it is skipped because best_model.pt is the same bytes."
    ),
)
@click.option(
    "--token",
    default=None,
    envvar="HF_TOKEN",
    help="HF token (defaults to HF_TOKEN env var or cached login).",
)
def main(
    owner: str,
    checkpoint: str,
    repo_name: str | None,
    private: bool,
    dry_run: bool,
    write_card: bool,
    include_epoch_snapshots: bool,
    token: str | None,
) -> None:
    """Upload checkpoint directories to the Hugging Face Hub."""
    ckpt_dirs = _resolve_checkpoints(checkpoint)
    if repo_name and len(ckpt_dirs) != 1:
        raise click.ClickException("--repo-name only valid with a single --checkpoint")

    ignore_patterns: list[str] = [] if include_epoch_snapshots else [EPOCH_SNAPSHOT_PATTERN]

    api = HfApi(token=token)

    for ckpt_dir in ckpt_dirs:
        target_name = repo_name or ckpt_dir.name
        repo_id = f"{owner}/{target_name}"
        files = _filter_files(ckpt_dir, ignore_patterns)
        skipped = [p.name for p in sorted(ckpt_dir.iterdir())
                   if p.is_file() and p not in files]
        if not files:
            logger.warning(
                "{} has no files after filtering (skipped: {}) — not uploading.",
                ckpt_dir.name, ", ".join(skipped) or "none",
            )
            continue
        size_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
        logger.info(
            "Uploading {} ({:.1f} MB, {} files) -> {} (private={})",
            ckpt_dir.name, size_mb, len(files), repo_id, private,
        )
        if skipped:
            logger.info("  skipping: {}", ", ".join(skipped))

        if dry_run:
            for p in files:
                logger.info("  would upload: {}", p.name)
            continue

        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        except HfHubHTTPError as e:
            raise click.ClickException(f"Failed to create/access repo {repo_id}: {e}") from e

        readme_path = ckpt_dir / "README.md"
        wrote_card = False
        if write_card and not readme_path.exists():
            readme_path.write_text(_build_model_card(ckpt_dir, repo_id, files))
            wrote_card = True
            files = files + [readme_path]
            logger.info("  generated model card: {}", readme_path)

        try:
            api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {ckpt_dir.name}",
                ignore_patterns=ignore_patterns or None,
            )
        finally:
            if wrote_card:
                # Don't leave the auto-generated card in the working tree —
                # the source of truth is docs/system/checkpoints.md (formerly evaluation/CHECKPOINTS.md).
                readme_path.unlink(missing_ok=True)

        logger.success("Uploaded https://huggingface.co/{}", repo_id)


if __name__ == "__main__":
    main()
