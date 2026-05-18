"""Unified retrieval benchmark driver.

One config-driven entry point spanning yambda, goodreads, and arxiv. Dispatch
is implicit: config shape decides whether to encode queries from a SASRec
checkpoint vs load pre-encoded text embeddings, and whether to run the filter
sweep loop vs a single unfiltered cell.

| Config shape                              | Mode                                   |
|-------------------------------------------|----------------------------------------|
| `checkpoint` set, `query_emb_path` unset  | Encode queries via SASRec (yambda/gr)  |
| `query_emb_path` set, `checkpoint` unset  | Load pre-encoded text embs (arxiv)     |
| `filters: null`                           | Quality run, single unfiltered cell    |
| `filters: {clause, bloom}`                | Filter run, per-kind sweeps            |

Usage::

    uv run evaluate --config conf/500m/d128-quality.yaml
    uv run evaluate --config conf/goodreads/d128-quality.yaml
    uv run evaluate --config conf/goodreads/d128-filter.yaml
    uv run evaluate --config conf/arxiv/d256-filter.yaml

Implementation lives in sibling modules:

- ``loaders.py``  — disk I/O (embeddings, query attrs, filter assets)
- ``oracle.py``   — filtered-FullScan ground truth and disk cache
- ``sweep.py``    — nested per-(filter_kind, sweep, algo, k, bs) loop
- ``bench_tools.py`` — perf timing primitives, query encoding, ckpt load
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from loguru import logger

from retrieval.config import EvalConfig, load_eval_config
from retrieval.loaders import load_item_and_queries, load_query_attrs
from retrieval.sweep import run_sweep


@click.command()
@click.option("--config", "config_path", type=str, required=True)
@click.option("--algorithms", "algos_override", multiple=True, type=str, default=())
@click.option(
    "--filter-kind",
    "filter_kinds",
    multiple=True,
    type=str,
    default=(),
    help="Restrict to one or more filter_kinds (repeat the flag); empty = run all.",
)
@click.option(
    "--backend",
    "backend_override",
    multiple=True,
    type=click.Choice(["triton", "torch"]),
    default=(),
    help="Restrict to one or more backends (repeat the flag); empty = use cfg.backends.",
)
@click.option("--sweep", "sweep_filter", type=str, default=None)
@click.option("--output", "output_override", type=str, default=None)
@click.option(
    "--skip-quality",
    is_flag=True,
    default=False,
    help="Skip the bs=1 quality stream and report recall=ndcg=NaN. "
    "Perf timing rows are still emitted. Useful for fast latency/memory sweeps.",
)
def main(
    config_path: str,
    algos_override: tuple[str, ...],
    filter_kinds: tuple[str, ...],
    backend_override: tuple[str, ...],
    sweep_filter: str | None,
    output_override: str | None,
    skip_quality: bool,
) -> None:
    cfg = load_eval_config(Path(config_path))
    if algos_override:
        cfg.algorithms = list(algos_override)
    if output_override:
        cfg.output = output_override

    # Dynamo's default recompile_limit (8) is too low for our sweep — each
    # `(k, bs)` combo and each grad-context (inference_mode on/off across the
    # autotune-prewarm / quality / perf passes) is a fresh trace. Hitting the
    # limit silently falls back to eager and erases the torch-backend
    # `torch.compile` benefit. 64 gives every shape × dispatch-key variant
    # in the worst-case cell its own compiled path with margin.
    import torch._dynamo  # noqa: PLC0415

    torch._dynamo.config.recompile_limit = 64

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        # Disable TF32 so the oracle (cuBLAS `q @ E_t`) and the algos
        # (per-impl Triton GEMM, generally fp32) compute scores in the
        # same precision. Otherwise exact-mask algos like
        # `linr_v1_filter_mask` show ~1e-3 recall drift vs the oracle
        # on narrow filters where top-K boundaries land on items
        # within TF32's 10-bit mantissa noise band. ``run_sweep`` also
        # calls ``pin_precision_globals`` — keep this here as a safety
        # net for direct callers that bypass ``run_sweep``.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    data_path = Path(cfg.data_dir)
    dev = torch.device(cfg.device)

    item_embs, queries, targets, n_targets, ckpt_path = load_item_and_queries(
        cfg, data_path, dev
    )
    out_path = _resolve_output_path(cfg, ckpt_path, data_path)

    qa_narrow_all: torch.Tensor | None = None
    if cfg.filters is not None:
        qa_narrow_all = load_query_attrs(
            data_path / "eval_split.parquet", queries.shape[0]
        )

    rows = run_sweep(
        cfg,
        item_embs,
        queries,
        targets,
        n_targets,
        qa_narrow_all,
        data_path=data_path,
        device=dev,
        filter_kinds=filter_kinds,
        backends=backend_override,  # type: ignore[arg-type]
        sweep_filter=sweep_filter,
        skip_quality=skip_quality,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("wrote {} rows to {}", len(rows), out_path)


def _resolve_output_path(cfg: EvalConfig, ckpt_path: Path | None, data_path: Path) -> Path:
    """Pick the output JSON path.

    Explicit ``cfg.output`` wins. Otherwise default next to the SASRec
    checkpoint (so each ckpt has a colocated ``evaluate.json``); arxiv
    runs without a checkpoint fall back to ``<data_dir>/evaluate.json``.
    """
    if cfg.output:
        return Path(cfg.output)
    if ckpt_path is not None:
        return ckpt_path.parent / "evaluate.json"
    return data_path / "evaluate.json"


if __name__ == "__main__":
    main()


__all__ = ["main"]
