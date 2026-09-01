"""Per-algo retrieval benchmark CLI.

Runs exactly one algorithm's cells against the dataset's queries (cached
on disk via ``queries_cache.py`` after the first build) and writes its
rows to a single JSON file. The driver ``retrieval.cli.run_evaluation``
(``uv run run-evaluation``) loops over algos from the YAML config and
writes per-algo JSONs into the ``cfg.output`` directory; downstream
analysis reads them back via ``retrieval.results_io.load_results``.

Each invocation is a fresh Python process, which is the point — torch
compile / Triton autotune / CUDA-graph private pools that survive
``torch._dynamo.reset()`` get cleared between algos by the OS.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from loguru import logger

from retrieval.config import load_eval_config
from retrieval.context import SweepContext
from retrieval.loaders import apply_users_limit, load_query_attrs
from retrieval.queries_cache import load_or_cache_queries
from retrieval.sweep import run_sweep


@click.command()
@click.option("--config", "config_path", type=str, required=True)
@click.option("--algo", type=str, required=True)
@click.option("--output", "output_path", type=str, required=True)
@click.option("--filter-kind", "filter_kinds", multiple=True, type=str, default=())
@click.option(
    "--backend",
    "backend_override",
    multiple=True,
    type=click.Choice(["triton", "torch", "cuda"]),
    default=(),
)
@click.option("--sweep", "sweep_filter", type=str, default=None)
@click.option("--skip-quality", is_flag=True, default=False)
def main(
    config_path: str,
    algo: str,
    output_path: str,
    filter_kinds: tuple[str, ...],
    backend_override: tuple[str, ...],
    sweep_filter: str | None,
    skip_quality: bool,
) -> None:
    cfg = load_eval_config(Path(config_path))

    import torch._dynamo  # noqa: PLC0415

    torch._dynamo.config.recompile_limit = 64

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    data_path = Path(cfg.data_dir)
    dev = torch.device(cfg.device)
    item_embs, queries, targets, n_targets, _ = load_or_cache_queries(
        cfg, data_path, dev
    )

    qa_narrow_all: torch.Tensor | None = None
    if cfg.filters is not None:
        qa_narrow_all = load_query_attrs(
            data_path / "eval_split.parquet", queries.shape[0]
        )

    # queries_cache already trims checkpoint-path tensors to users_limit
    # (no-op guard there); this is the real trim on the arxiv path.
    queries, targets, n_targets, qa_narrow_all = apply_users_limit(
        cfg, queries, targets, n_targets, qa_narrow_all
    )

    backends = backend_override or tuple(cfg.backends) or ("triton",)

    ctx = SweepContext(
        cfg=cfg,
        algorithms=(algo,),
        item_embs=item_embs,
        queries=queries,
        targets=targets,
        n_targets=n_targets,
        qa_narrow_all=qa_narrow_all,
        device=dev,
        gt_dir=data_path / cfg.gt_subdir,
        k_gt=max(cfg.ks),
        suite="yambda" if cfg.filters is None else "filter",
        backends=backends,  # type: ignore[arg-type]
        skip_quality=skip_quality,
        sweep_filter=sweep_filter,
        filter_kinds=filter_kinds,
    )

    rows = run_sweep(ctx)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("wrote {} rows to {}", len(rows), out)


if __name__ == "__main__":
    main()


__all__ = ["main"]
