"""Yambda retrieval benchmark: quality + perf across {1, 8, 16} batch sizes.

Loads a trained GSASRec checkpoint, encodes the full test set into a query
cache once, then benchmarks each retrieval method by streaming the cached
queries — no transformer re-runs across algorithms. Writes one row per
``(algorithm, K, batch_size)`` cell to JSON: recall@K, ndcg@K, latency
(median/p20/p80 ms), peak GPU memory (MiB).

Configuration lives in YAML; see ``conf/smoke.yaml`` and ``conf/500m-*.yaml``.
The CLI keeps three flags only:

  --config <path>           required; YAML config path
  --algorithms <name> ...   optional; *replaces* (does not merge into)
                            the YAML's algorithms list
  --output <path>           optional; overrides the YAML's output path

Run from ``/workspace/evaluation/``:

    uv run benchmark --config conf/500m-d128.yaml
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import click
import torch
from loguru import logger

from retrieval._bench_primitives import (
    cuda_allocated_mib,
    encode_queries,
    load_model_for_eval as load_model,
    perf_pass_cached,
    quality_pass_cached,
)
from retrieval.config import load_eval_config
from retrieval.registry import build_algorithm


# ---------- param sweep -------------------------------------------------------


def expand_param_combos(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """List-valued entries become sweep axes (cartesian product); scalars stay fixed.

    Empty dict yields ``[{}]`` so callers can iterate uniformly.
    """
    swept_keys = [k for k, v in raw.items() if isinstance(v, list)]
    fixed = {k: v for k, v in raw.items() if not isinstance(v, list)}
    if not swept_keys:
        return [dict(fixed)]
    swept_vals = [raw[k] for k in swept_keys]
    return [{**fixed, **dict(zip(swept_keys, combo, strict=True))}
            for combo in itertools.product(*swept_vals)]


def is_valid_combo(algo: str, params: dict[str, Any]) -> bool:
    """Skip combos the underlying algo would assert on, so one bad cell does
    not abort the whole sweep. SilverTorch enforces ``n_probe <= n_lists`` at
    register time; pre-filter here."""
    if algo == "silvertorch":
        n_lists = params.get("n_lists")
        n_probe = params.get("n_probe")
        if n_lists is not None and n_probe is not None and n_probe > n_lists:
            return False
    return True


# ---------- driver ------------------------------------------------------------


@click.command()
@click.option(
    "--config",
    "config_path",
    type=str,
    required=True,
    help="YAML config path (see conf/smoke.yaml, conf/500m-*.yaml).",
)
@click.option(
    "--algorithms",
    "algos_override",
    multiple=True,
    type=str,
    default=(),
    help="Replace (do not merge into) the YAML's algorithms list.",
)
@click.option(
    "--output", "output_override", type=str, default=None, help="Override the YAML's output path."
)
def main(
    config_path: str,
    algos_override: tuple[str, ...],
    output_override: str | None,
) -> None:
    cfg = load_eval_config(Path(config_path))
    if algos_override:
        cfg.algorithms = list(algos_override)
    if output_override:
        cfg.output = output_override

    # Lock torch.topk tie-order and any default-RNG draws.
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    ckpt_path = Path(cfg.checkpoint)
    data_path = Path(cfg.data_dir)
    dev = torch.device(cfg.device)

    with open(data_path / "item_id_map.json") as f:
        num_items = len(json.load(f))
    logger.info("num_items={}", num_items)

    model = load_model(ckpt_path, num_items=num_items, device=dev)
    item_embs = model.get_output_embeddings().weight.detach().to(dev).contiguous()
    item_embs[0] = 0.0
    logger.info("item_embs shape={} dtype={}", tuple(item_embs.shape), item_embs.dtype)

    eval_parquet = data_path / f"{cfg.split}.parquet"
    out_path = Path(cfg.output) if cfg.output else ckpt_path.parent / "benchmark.json"

    queries, targets, n_targets = encode_queries(
        model,
        eval_parquet,
        max_length=cfg.encode.max_seq_length,
        encode_batch_size=cfg.encode.batch_size,
        num_workers=cfg.encode.num_workers,
        device=dev,
    )
    logger.info("encoded queries: {} users, dim={}", queries.shape[0], queries.shape[1])
    # Free the transformer; only item_embs is needed downstream.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: list[dict] = []
    for algo in cfg.algorithms:
        raw_params = cfg.algo_params.get(algo, {})
        combos = expand_param_combos(raw_params)
        for params in combos:
            if not is_valid_combo(algo, params):
                logger.warning("skipping invalid combo {}: {}", algo, params)
                continue
            for k in cfg.ks:
                logger.info("=== {} k={} params={} ===", algo, k, params)

                # Memory snapshot order: drain previous cell's residue first, then
                # take baseline. Otherwise kmeans/topk transients from the prior
                # build pollute the index-memory delta.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                mem_before = cuda_allocated_mib()

                forward, modules, is_cpu = build_algorithm(
                    algo,
                    item_embs,
                    k=k,
                    params=params,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                index_mem = 0.0 if is_cpu else cuda_allocated_mib() - mem_before

                recall, ndcg = quality_pass_cached(
                    forward,
                    queries,
                    targets,
                    n_targets,
                    k=k,
                    device=dev,
                    desc=f"{algo} k={k}",
                )
                for bs in cfg.batch_sizes:
                    med, p20, p80, peak, scratch = perf_pass_cached(
                        forward,
                        queries,
                        batch_size=bs,
                        device=dev,
                        is_cpu=is_cpu,
                        seed=cfg.seed,
                    )
                    row = {
                        "suite": f"yambda_{cfg.split}",
                        "cell": f"bs{bs}_k{k}",
                        "impl": algo,
                        "device": "cpu" if is_cpu else "cuda",
                        "seed": cfg.seed,
                        "batch_size": bs,
                        "k": k,
                        "median_ms": med,
                        "p20_ms": p20,
                        "p80_ms": p80,
                        "peak_mem_mib": peak,
                        "index_mem_mib": index_mem,
                        "fwd_scratch_mib": scratch,
                        f"recall@{k}": recall,
                        f"ndcg@{k}": ndcg,
                        "extra": {
                            "params": {str(pk): str(pv) for pk, pv in params.items()},
                        },
                    }
                    rows.append(row)
                    logger.info(
                        "{} k={} bs={} median={:.3f}ms p20={:.3f} p80={:.3f} "
                        "peak={:.1f}MiB recall={:.4f} ndcg={:.4f}",
                        algo,
                        k,
                        bs,
                        med,
                        p20,
                        p80,
                        peak,
                        recall,
                        ndcg,
                    )
                # `for m in modules: del m` only drops the loop var — the list
                # itself still holds refs, leaking the index into the next cell's
                # mem_before snapshot. Drop the bindings explicitly.
                modules.clear()
                del forward, modules
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("wrote {} rows to {}", len(rows), out_path)


if __name__ == "__main__":
    main()
