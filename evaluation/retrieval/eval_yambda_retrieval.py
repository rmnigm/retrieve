"""Yambda retrieval benchmark: quality + perf at bs=1.

Loads a trained GSASRec checkpoint, encodes the full test set into a query
cache once, then benchmarks each retrieval method by streaming the cached
queries — no transformer re-runs across algorithms. Writes one row per
``(algorithm, K)`` cell to JSON: recall@K, ndcg@K, latency (median/p20/p80
ms), peak GPU memory (MiB).

Configuration lives in YAML; see ``conf/smoke.yaml`` and ``conf/500m.yaml``.
The CLI keeps three flags only:

  --config <path>           required; YAML config path
  --algorithms <name> ...   optional; *replaces* (does not merge into)
                            the YAML's algorithms list
  --output <path>           optional; overrides the YAML's output path

Run from ``/workspace/evaluation/`` so ``-m retrieval.…`` resolves:

    uv run python -m retrieval.eval_yambda_retrieval --config conf/500m.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
import triton.testing as ttesting
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from retrieval.algorithms import ForwardFn, build_algorithm
from retrieval.config import load_eval_config
from retrieval.metrics import accumulate_metrics, finalize_metrics
from training.evaluate import EvalDataset, collate_eval
from training.model import GSASRec


# ---------- perf primitives (mirrors retrieve/tests/bench/conftest.py) -------

def measure(
    fn,
    *,
    rep_ms: float = 200.0,
    quantiles: tuple[float, float, float] = (0.5, 0.2, 0.8),
) -> tuple[float, float, float]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    med, p20, p80 = ttesting.do_bench(
        fn, quantiles=list(quantiles), rep=rep_ms, warmup=200
    )
    return float(med), float(p20), float(p80)


def forward_memory_mib(fn) -> tuple[float, float]:
    """Returns ``(peak_mib, scratch_mib)`` for a single forward call.

    ``peak_mib`` is total resident at the high-water mark (shared item_embs +
    index buffers + transient scratch). ``scratch_mib`` is peak minus the
    baseline allocated *before* the call — i.e. only the tensors the forward
    pass allocates and (typically) frees within the call. Computed via
    ``torch.cuda.memory_allocated`` deltas around ``reset_peak_memory_stats``;
    no profiler hooks. For per-allocation attribution, set
    ``CUDA_MEM_DUMP=<path>`` to dump a ``torch.cuda.memory._snapshot()`` after
    the perf pass — view with ``torch.cuda.memory._dump_snapshot``."""
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return peak / (1024 * 1024), (peak - baseline) / (1024 * 1024)


def cuda_allocated_mib() -> float:
    """Currently-allocated CUDA memory in MiB. Deltas around ``build_algorithm``
    isolate the index's marginal cost (excludes shared ``item_embs`` and any
    transient build-time scratch the caching allocator has since released)."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


# ---------- model load --------------------------------------------------------

_D128_DROP05 = {
    "max_seq_length": 200, "embedding_dim": 128, "num_heads": 2, "num_blocks": 2,
    "ffn_hidden_dim": 512, "dropout": 0.5, "reuse_item_embeddings": False,
}


def load_model(checkpoint_path: Path, num_items: int, device: torch.device) -> GSASRec:
    """Read sibling config.json when present (smoke ckpt has it); fall back to
    the d128-drop0.5 hyperparams from docs/checkpoints.md (formerly CHECKPOINTS.md;
    500M ckpts don't ship one).
    """
    cfg_path = checkpoint_path.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        params = {k: cfg[k] for k in _D128_DROP05 if k in cfg}
    else:
        params = dict(_D128_DROP05)
    logger.info("model params: {}", params)
    model = GSASRec(num_items=num_items, **params).to(device).eval()
    state = torch.load(
        str(checkpoint_path), map_location=str(device), weights_only=True
    )
    model.load_state_dict(state)
    return model


# ---------- query cache (encode the test set once) ---------------------------

@torch.inference_mode()
def encode_queries(
    model: GSASRec,
    data_path: Path,
    *,
    max_length: int,
    encode_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run ``model.predict_last`` over the entire eval split once.

    Returns ``(queries [N, D], targets [N, T_max], num_targets [N])`` on CPU.
    Encoding batch size is independent of the per-algo perf batch (always 1).
    """
    dataset = EvalDataset(str(data_path), max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=encode_batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    q_chunks: list[torch.Tensor] = []
    t_lists: list[list[int]] = []
    for item_seqs, targets, num_targets in tqdm(loader, desc="encode queries"):
        item_seqs = item_seqs.to(device, non_blocking=True)
        q = model.predict_last(item_seqs).detach().cpu()
        q_chunks.append(q)
        # Reconstruct ragged target lists from the padded tensor (-1 marks pad).
        for row, n in zip(targets, num_targets, strict=True):
            t_lists.append(row[:n].tolist())

    queries = torch.cat(q_chunks, dim=0)
    n_users = queries.shape[0]
    n_targets = torch.tensor([len(t) for t in t_lists], dtype=torch.long)
    t_max = max(int(n_targets.max().item()), 1)
    targets = torch.full((n_users, t_max), -1, dtype=torch.long)
    for i, t in enumerate(t_lists):
        if t:
            targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)
    return queries, targets, n_targets


# ---------- passes ------------------------------------------------------------

@torch.inference_mode()
def quality_pass_cached(
    forward: ForwardFn,
    queries: torch.Tensor,         # [N, D] on cpu
    targets: torch.Tensor,         # [N, T] on cpu
    num_targets: torch.Tensor,     # [N]   on cpu
    *,
    k: int,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    """Stream cached queries through the index at bs=1 and accumulate metrics."""
    accum = None
    n = queries.shape[0]
    for i in tqdm(range(n), desc=desc, leave=False):
        q = queries[i : i + 1].to(device, non_blocking=True)
        t = targets[i : i + 1].to(device, non_blocking=True)
        nt = num_targets[i : i + 1].to(device, non_blocking=True)
        topk_ids, _ = forward(q)
        accum = accumulate_metrics(topk_ids, t, nt, [k], accum)
    metrics = finalize_metrics(accum) if accum is not None else {}
    return metrics[f"recall@{k}"], metrics[f"ndcg@{k}"]


def perf_pass_cached(
    forward: ForwardFn,
    queries: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """Time the index forward at bs=1 on a fixed query.

    Returns ``(median_ms, p20_ms, p80_ms, peak_mib, scratch_mib)``."""
    query = queries[:1].to(device).clone()

    def perf_fn() -> None:
        with torch.inference_mode():
            forward(query)

    med, p20, p80 = measure(perf_fn, rep_ms=200.0)
    peak, scratch = forward_memory_mib(perf_fn)
    return med, p20, p80, peak, scratch


# ---------- driver ------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", type=str, required=True,
              help="YAML config path (see conf/smoke.yaml, conf/500m.yaml).")
@click.option("--algorithms", "algos_override", multiple=True, type=str, default=(),
              help="Replace (do not merge into) the YAML's algorithms list.")
@click.option("--output", "output_override", type=str, default=None,
              help="Override the YAML's output path.")
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
    out_path = (
        Path(cfg.output) if cfg.output
        else ckpt_path.parent / "eval_yambda_retrieval.json"
    )

    queries, targets, n_targets = encode_queries(
        model, eval_parquet,
        max_length=cfg.encode.max_seq_length,
        encode_batch_size=cfg.encode.batch_size,
        num_workers=cfg.encode.num_workers,
        device=dev,
    )
    logger.info("encoded queries: {} users, dim={}", queries.shape[0], queries.shape[1])
    # Free the transformer; only item_embs is needed downstream.
    del model
    torch.cuda.empty_cache()

    rows: list[dict] = []
    for algo in cfg.algorithms:
        algo_params = cfg.algo_params.get(algo, {})
        for k in cfg.ks:
            logger.info("=== {} k={} ===", algo, k)
            mem_before = cuda_allocated_mib()
            forward, modules = build_algorithm(
                algo, item_embs, k=k, params=algo_params,
            )
            index_mem = cuda_allocated_mib() - mem_before
            recall, ndcg = quality_pass_cached(
                forward, queries, targets, n_targets,
                k=k, device=dev, desc=f"{algo} k={k}",
            )
            med, p20, p80, peak, scratch = perf_pass_cached(
                forward, queries, device=dev,
            )
            row = {
                "suite": f"yambda_{cfg.split}",
                "cell": f"bs1_k{k}",
                "impl": algo,
                "median_ms": med,
                "p20_ms": p20,
                "p80_ms": p80,
                "peak_mem_mib": peak,
                "index_mem_mib": index_mem,
                "fwd_scratch_mib": scratch,
                f"recall@{k}": recall,
                f"ndcg@{k}": ndcg,
                "extra": {
                    "batch_size": "1",
                    "k": str(k),
                    "params": {str(pk): str(pv) for pk, pv in algo_params.items()},
                },
            }
            rows.append(row)
            logger.info(
                "recall@{}={:.4f} ndcg@{}={:.4f} median={:.3f}ms "
                "peak={:.1f}MiB index={:.1f}MiB fwd_scratch={:.1f}MiB",
                k, recall, k, ndcg, med, peak, index_mem, scratch,
            )
            # `for m in modules: del m` only drops the loop var — the list
            # itself still holds refs, leaking the index into the next cell's
            # mem_before snapshot. Drop the bindings explicitly.
            modules.clear()
            del forward, modules
            torch.cuda.empty_cache()

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("wrote {} rows to {}", len(rows), out_path)


if __name__ == "__main__":
    main()
