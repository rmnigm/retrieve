"""Single source of truth for HuggingFace I/O in the evaluation harness.

Replaces the three independent HF call sites (`data/arxiv.py`, `data/yambda.py`,
`training/upload_checkpoints.py`) with one registry + one set of download/upload
helpers. All local paths resolve under a single `data_root()`.

Layout:

    $RETRIEVE_DATA_ROOT/                    (default <repo_root>/data)
    ├── <dataset>/                          eval inputs from pinkmeme/eval-<dataset>
    │   ├── content/, content_d64/, ...
    │   ├── item_id_map.json, eval_split.parquet, item_attrs_narrow.pt, ...
    │   ├── checkpoints/<ckpt_id>/          model checkpoints under the same repo
    │   └── gt/                             local-only oracle cache (recomputed)
    └── _raw/<source>/                      upstream raw HF reads
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EVAL_REPOS: dict[str, str] = {
    "arxiv-papers":      "pinkmeme/eval-arxiv-papers",
    "yambda-500m":       "pinkmeme/eval-yambda-500m",
    "yambda-5b":         "pinkmeme/eval-yambda-5b",
    "goodreads-work-id": "pinkmeme/eval-goodreads-work-id",
}

# Upstream raw repos kept here only so their local target dirs are centralized.
RAW_REPOS: dict[str, tuple[str, str]] = {
    # source -> (repo_id, repo_type)
    "arxiv":  ("open-index/open-arxiv", "dataset"),
    "yambda": ("yandex/yambda",         "dataset"),
}

# What we *don't* want in the per-dataset eval repo. Excludes training inputs,
# intermediate ETL artifacts, backup files, recomputable caches, and the
# `checkpoints/` subtree (which is uploaded separately by upload_checkpoint()).
EVAL_IGNORE_PATTERNS: list[str] = [
    "train.parquet",
    "val.parquet",
    "papers.parquet",        # arxiv ETL artifact (raw text, used only at encode time)
    "book_to_work.parquet",  # goodreads ETL artifact
    "item_attrs_wide.pt",
    "wide_*",
    "*.bak",
    "*.bak.*",
    "gt/**",
    "gt_*/**",               # dim-suffixed oracle caches (gt_d64, gt_d128, ...)
    "checkpoints/**",
    "evaluate*.json",        # run logs
    "eval_arxiv_retrieval.json",
    "train_metrics.json",
    "prep_log.json",
    "*.tmp",
    "__pycache__/**",
    ".cache/**",             # HF download cache from prior snapshot_download calls
    ".gitattributes",
    ".git/**",
]

# `gsasrec-ep{N}-{metric}{X}.pt` is saved at the best val epoch and then copied
# to `best_model.pt` at training end — same bytes, twice the storage. Skip the
# epoch-tagged copy by default; pass `include_epoch_snapshots=True` to keep it.
EPOCH_SNAPSHOT_PATTERN = "gsasrec-ep*.pt"

# Always-skip files in checkpoint dirs (run output, not training meta).
CKPT_ALWAYS_IGNORE: list[str] = [
    "evaluate.json",
    "benchmark.json",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    # evaluation/data/hf_io.py -> evaluation/
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Root for all dataset/checkpoint/raw artifacts. Override with RETRIEVE_DATA_ROOT."""
    env = os.environ.get("RETRIEVE_DATA_ROOT")
    return Path(env).expanduser().resolve() if env else (_repo_root() / "data").resolve()


def eval_dir(dataset: str) -> Path:
    return data_root() / dataset


def ckpt_dir(dataset: str, ckpt_id: str) -> Path:
    return eval_dir(dataset) / "checkpoints" / ckpt_id


def raw_dir(source: str) -> Path:
    return data_root() / "_raw" / source


def _require_dataset(dataset: str) -> str:
    try:
        return EVAL_REPOS[dataset]
    except KeyError as e:
        avail = ", ".join(sorted(EVAL_REPOS)) or "(none)"
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {avail}") from e


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def download_eval_dataset(
    dataset: str,
    *,
    dims: list[str] | None = None,
    include_checkpoints: bool = False,
    target: Path | None = None,
) -> Path:
    """Pull the eval-input files for `dataset` from HF.

    `dims=["d64", "d128"]` restricts content/* downloads to those dims (and the
    default `content/` is always included). `include_checkpoints=True` also
    pulls everything under `checkpoints/`.
    """
    repo_id = _require_dataset(dataset)
    local = (target or eval_dir(dataset)).resolve()
    local.mkdir(parents=True, exist_ok=True)

    allow: list[str] | None
    if dims is None and include_checkpoints:
        allow = None
    else:
        allow = [
            "*.json", "*.parquet", "*.pt",
            "README.md", "manifest.json",
            "content/*",
        ]
        if dims:
            for d in dims:
                allow.append(f"content_{d}/*")
        else:
            allow.append("content_*/*")
        if include_checkpoints:
            allow.append("checkpoints/**")

    logger.info("snapshot_download {} -> {} (allow={})", repo_id, local, allow)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local),
        allow_patterns=allow,
    )
    return local


def download_checkpoint(dataset: str, ckpt_id: str, target: Path | None = None) -> Path:
    """Pull a single checkpoint directory from a per-dataset eval repo."""
    repo_id = _require_dataset(dataset)
    local = (target or eval_dir(dataset)).resolve()
    local.mkdir(parents=True, exist_ok=True)

    logger.info("snapshot_download {} (checkpoints/{}) -> {}", repo_id, ckpt_id, local)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local),
        allow_patterns=[f"checkpoints/{ckpt_id}/*"],
    )
    return local / "checkpoints" / ckpt_id


def download_raw(
    source: str,
    *,
    allow_patterns: list[str] | None = None,
    max_workers: int | None = None,
) -> Path:
    """Pull an upstream raw HF dataset into the unified `_raw/<source>/` dir."""
    try:
        repo_id, repo_type = RAW_REPOS[source]
    except KeyError as e:
        avail = ", ".join(sorted(RAW_REPOS)) or "(none)"
        raise ValueError(f"Unknown raw source {source!r}. Known: {avail}") from e

    local = raw_dir(source).resolve()
    local.mkdir(parents=True, exist_ok=True)

    logger.info("snapshot_download {} -> {} (allow={})", repo_id, local, allow_patterns)
    kwargs: dict = dict(repo_id=repo_id, repo_type=repo_type, local_dir=str(local),
                        allow_patterns=allow_patterns)
    if max_workers is not None:
        kwargs["max_workers"] = max_workers
    snapshot_download(**kwargs)
    return local


def download_raw_file(source: str, filename: str) -> Path:
    """Pull a single file from an upstream raw HF dataset. Returns the local path."""
    from huggingface_hub import hf_hub_download

    try:
        repo_id, repo_type = RAW_REPOS[source]
    except KeyError as e:
        avail = ", ".join(sorted(RAW_REPOS)) or "(none)"
        raise ValueError(f"Unknown raw source {source!r}. Known: {avail}") from e

    local = raw_dir(source).resolve()
    local.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename, local_dir=str(local))
    )


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------

def _ignored(rel: str, name: str, patterns: list[str]) -> bool:
    """Match `rel` (posix relative path) against fnmatch patterns. Handles
    `prefix/**` as a directory-prefix exclusion since fnmatch itself doesn't
    support `**`. Also matches plain filename patterns against `name`."""
    for pat in patterns:
        if pat.endswith("/**"):
            prefix = pat[:-3]
            if rel == prefix or rel.startswith(prefix + "/"):
                return True
            continue
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat):
            return True
    return False


def _list_files_filtered(root: Path, ignore: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if _ignored(rel, p.name, ignore):
            continue
        out.append(p)
    return out


def upload_eval_dataset(
    dataset: str,
    source: Path | None = None,
    *,
    private: bool = True,
    dry_run: bool = False,
) -> None:
    """Create-or-update the per-dataset eval repo and upload eval inputs."""
    repo_id = _require_dataset(dataset)
    src = (source or eval_dir(dataset)).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"{src} does not exist")

    files = _list_files_filtered(src, EVAL_IGNORE_PATTERNS)
    if not files:
        raise RuntimeError(f"{src} has no files to upload after applying ignore patterns")
    total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
    logger.info(
        "Uploading eval dataset {} ({} files, {:.1f} MB) -> {} (private={})",
        dataset, len(files), total_mb, repo_id, private,
    )
    for p in files:
        logger.info("  + {}", p.relative_to(src))

    if dry_run:
        return

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    except HfHubHTTPError as e:
        raise RuntimeError(f"Failed to create/access repo {repo_id}: {e}") from e

    api.upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload eval inputs for {dataset}",
        ignore_patterns=EVAL_IGNORE_PATTERNS,
    )
    logger.success("Uploaded https://huggingface.co/datasets/{}", repo_id)


def _build_model_card(ckpt_path: Path, repo_id: str, ckpt_id: str, files: list[Path]) -> str:
    parts: list[str] = [f"# {ckpt_id}\n"]

    eval_path = ckpt_path / "eval_quality.json"
    if eval_path.exists():
        meta = json.loads(eval_path.read_text())
        metrics = meta.get("metrics", {})
        if metrics:
            parts.append("## Test metrics\n")
            for k, v in metrics.items():
                parts.append(f"- **{k}**: {v:.4f}")
        if target := meta.get("paper_target"):
            parts.append("\n## Paper target\n")
            for k, v in target.items():
                parts.append(f"- {k}: {v:.4f}")
        parts.append("")

    config_path = ckpt_path / "config.json"
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
        "```python\n"
        "from eval_datasets.hf_io import download_checkpoint\n"
        f'download_checkpoint({repo_id.split("/")[-1].removeprefix("eval-")!r}, {ckpt_id!r})\n'
        "```\n"
    )
    return "\n".join(parts)


def upload_checkpoint(
    dataset: str,
    ckpt_id: str,
    source: Path | None = None,
    *,
    private: bool = True,
    include_epoch_snapshots: bool = False,
    write_card: bool = True,
    dry_run: bool = False,
) -> None:
    """Upload a single checkpoint directory to `checkpoints/<ckpt_id>/` in the
    per-dataset eval repo. Repo is created if it doesn't exist."""
    repo_id = _require_dataset(dataset)
    src = (source or ckpt_dir(dataset, ckpt_id)).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"{src} does not exist")

    ignore: list[str] = list(CKPT_ALWAYS_IGNORE)
    if not include_epoch_snapshots:
        ignore.append(EPOCH_SNAPSHOT_PATTERN)
    files = _list_files_filtered(src, ignore)
    if not files:
        raise RuntimeError(f"{src} has no files to upload after filter (ignore={ignore})")

    total_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
    logger.info(
        "Uploading checkpoint {} -> {}/checkpoints/{} ({} files, {:.1f} MB)",
        src, repo_id, ckpt_id, len(files), total_mb,
    )
    for p in files:
        logger.info("  + {}", p.relative_to(src))

    if dry_run:
        return

    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    except HfHubHTTPError as e:
        raise RuntimeError(f"Failed to create/access repo {repo_id}: {e}") from e

    readme_path = src / "README.md"
    wrote_card = False
    if write_card and not readme_path.exists():
        readme_path.write_text(_build_model_card(src, repo_id, ckpt_id, files))
        wrote_card = True
        logger.info("  generated model card: {}", readme_path)

    try:
        api.upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=f"checkpoints/{ckpt_id}",
            commit_message=f"Upload checkpoint {ckpt_id}",
            ignore_patterns=ignore or None,
        )
    finally:
        if wrote_card:
            readme_path.unlink(missing_ok=True)

    logger.success("Uploaded https://huggingface.co/datasets/{}/tree/main/checkpoints/{}", repo_id, ckpt_id)


# ---------------------------------------------------------------------------
# CLI entry points (used by pyproject.toml [project.scripts])
# ---------------------------------------------------------------------------

def fetch_cli() -> None:
    import click

    @click.command()
    @click.argument("dataset")
    @click.option("--dims", "dims_str", default=None, help="Comma-separated dim suffixes, e.g. d64,d128")
    @click.option("--include-checkpoints", is_flag=True)
    @click.option("--target", type=click.Path(file_okay=False, path_type=Path), default=None)
    def _cmd(dataset: str, dims_str: str | None, include_checkpoints: bool, target: Path | None) -> None:
        dims = [s.strip() for s in dims_str.split(",")] if dims_str else None
        out = download_eval_dataset(
            dataset, dims=dims, include_checkpoints=include_checkpoints, target=target,
        )
        click.echo(str(out))

    _cmd()


def publish_cli() -> None:
    import click

    @click.command()
    @click.argument("dataset")
    @click.option("--source", type=click.Path(file_okay=False, path_type=Path), default=None)
    @click.option("--private/--public", default=True)
    @click.option("--dry-run", is_flag=True)
    def _cmd(dataset: str, source: Path | None, private: bool, dry_run: bool) -> None:
        upload_eval_dataset(dataset, source=source, private=private, dry_run=dry_run)

    _cmd()


def publish_checkpoint_cli() -> None:
    import click

    @click.command()
    @click.argument("dataset")
    @click.argument("ckpt_id")
    @click.option("--source", type=click.Path(file_okay=False, path_type=Path), default=None)
    @click.option("--private/--public", default=True)
    @click.option("--include-epoch-snapshots", is_flag=True)
    @click.option("--write-card/--no-write-card", default=True)
    @click.option("--dry-run", is_flag=True)
    def _cmd(
        dataset: str,
        ckpt_id: str,
        source: Path | None,
        private: bool,
        include_epoch_snapshots: bool,
        write_card: bool,
        dry_run: bool,
    ) -> None:
        upload_checkpoint(
            dataset,
            ckpt_id,
            source=source,
            private=private,
            include_epoch_snapshots=include_epoch_snapshots,
            write_card=write_card,
            dry_run=dry_run,
        )

    _cmd()


__all__ = [
    "EVAL_REPOS",
    "RAW_REPOS",
    "EVAL_IGNORE_PATTERNS",
    "EPOCH_SNAPSHOT_PATTERN",
    "data_root",
    "eval_dir",
    "ckpt_dir",
    "raw_dir",
    "download_eval_dataset",
    "download_checkpoint",
    "download_raw",
    "download_raw_file",
    "upload_eval_dataset",
    "upload_checkpoint",
    "fetch_cli",
    "publish_cli",
    "publish_checkpoint_cli",
]
