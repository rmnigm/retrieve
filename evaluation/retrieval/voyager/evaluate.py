"""CPU quality-baseline driver: voyager HNSW × params × k.

Standalone CLI for the only CPU algo in the bench. The unified
``evaluate`` driver sweeps GPU algos × backends × filter kinds and is
shaped around CUDA-event timing, dynamo recompile budgets, and the
oracle-cache for filter ground truth — none of which apply to voyager.
Reuses ``EvalConfig`` / ``load_item_and_queries`` / ``quality_pass_cached``
so the embedding-loading and recall/ndcg math stays in one place.

Usage::

    uv run evaluate-voyager --config conf/voyager/yambda-500m-d128.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from loguru import logger

from retrieval.bench_tools import quality_pass_cached
from retrieval.config import EvalConfig, load_eval_config
from retrieval.loaders import load_item_and_queries
from retrieval.voyager.baseline import VoyagerHNSW


@click.command()
@click.option("--config", "config_path", type=str, required=True)
@click.option("--output", "output_override", type=str, default=None)
def main(config_path: str, output_override: str | None) -> None:
    cfg = load_eval_config(Path(config_path))
    if output_override:
        cfg.output = output_override

    torch.manual_seed(cfg.seed)
    data_path = Path(cfg.data_dir)
    dev = torch.device(cfg.device)

    item_embs, queries, targets, n_targets, ckpt_path = load_item_and_queries(
        cfg, data_path, dev
    )
    if cfg.users_limit is not None and cfg.users_limit < queries.shape[0]:
        n_keep = int(cfg.users_limit)
        logger.info(
            "users_limit={}: subsampling {}→{} users",
            n_keep, queries.shape[0], n_keep,
        )
        queries = queries[:n_keep].contiguous()
        targets = targets[:n_keep].contiguous()
        n_targets = n_targets[:n_keep].contiguous()

    rows: list[dict] = []
    param_combos = cfg.algo_params.get("voyager_hnsw") or [{}]
    for params in param_combos:
        for k in cfg.ks:
            idx = VoyagerHNSW(
                k=k,
                m=int(params.get("m", 16)),
                ef_construction=int(params.get("ef_construction", 200)),
                ef_query=int(params["ef_query"]) if "ef_query" in params else None,
                num_threads=int(params.get("num_threads", -1)),
                seed=int(params.get("seed", 0)),
            )
            idx.register_index(item_embs)
            desc = f"voyager_hnsw k={k}"
            recall, ndcg = quality_pass_cached(
                idx, queries, targets, n_targets,
                k=k, device=dev, desc=desc,
            )
            rows.append({
                "suite": "voyager",
                "impl": "voyager_hnsw",
                "device": "cpu",
                "seed": cfg.seed,
                "k": k,
                "n_users": int(queries.shape[0]),
                f"recall@{k}": recall,
                f"ndcg@{k}": ndcg,
                "extra": {"params": {str(pk): str(pv) for pk, pv in params.items()}},
            })
            logger.info(
                "voyager_hnsw{} k={} recall={:.4f} ndcg={:.4f}",
                " " + ",".join(f"{pk}={pv}" for pk, pv in params.items()) if params else "",
                k, recall, ndcg,
            )

    out_path = _resolve_output_path(cfg, ckpt_path, data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("wrote {} rows to {}", len(rows), out_path)


def _resolve_output_path(cfg: EvalConfig, ckpt_path: Path | None, data_path: Path) -> Path:
    if cfg.output:
        return Path(cfg.output)
    if ckpt_path is not None:
        return ckpt_path.parent / "evaluate_voyager.json"
    return data_path / "evaluate_voyager.json"


if __name__ == "__main__":
    main()


__all__ = ["main"]
