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

import json
import time
from pathlib import Path

import click
import torch
import triton.testing as ttesting
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from retrieval.registry import ForwardFn, build_algorithm
from retrieval.config import load_eval_config
from retrieval.metrics import accumulate_metrics, finalize_metrics
from training.evaluate import EvalDataset, collate_eval
from training.model import GSASRec

# Long enough to settle triton autotune, short enough to keep cells <60 s of overhead.
WARMUP_ITERS = 20
DEFAULT_REP_MS = 200.0


# ---------- perf primitives ---------------------------------------------------


def _allocated() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_allocated())


def _peak() -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())


def measure_forward_cuda(
    fn,
    *,
    rep_ms: float = DEFAULT_REP_MS,
    warmup_iters: int = WARMUP_ITERS,
    mem_reps: int = 5,
) -> tuple[float, float, float, float, float]:
    """Warmup, capture transient peak in a clean window, then time via do_bench.

    The peak window does NOT use ``do_bench`` because do_bench allocates a ~256 MiB L2 cache-buster
    each call, which would dominate the reported transient peak for any small
    kernel. We measure memory in isolation, then time separately.

    Returns ``(median_ms, p20_ms, p80_ms, peak_mib, transient_mib)``.
    """
    # Warmup must happen BEFORE the peak counter reset, otherwise autotune
    # compile-window peak pollutes the metric.
    for _ in range(warmup_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline = _allocated()
    for _ in range(mem_reps):
        out = fn()
        del out
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak = _peak()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    median, p20, p80 = ttesting.do_bench(fn, quantiles=[0.5, 0.2, 0.8], rep=rep_ms, warmup=50)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    peak_mib = peak / (1024 * 1024)
    transient_mib = max(0, peak - baseline) / (1024 * 1024)
    return float(median), float(p20), float(p80), peak_mib, transient_mib


def measure_forward_cpu(
    fn,
    *,
    rep_ms: float = DEFAULT_REP_MS,
    warmup_iters: int = 3,
) -> tuple[float, float, float, float, float]:
    """Time a CPU forward via wall-clock samples within a ``rep_ms`` budget.

    Returns ``(median_ms, p20_ms, p80_ms, 0.0, 0.0)`` — peak/transient memory
    are GPU-only and reported as 0 for CPU rows.
    """
    for _ in range(warmup_iters):
        fn()
    times: list[float] = []
    deadline = time.perf_counter() + rep_ms / 1000.0
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    if not times:
        # Pathological case: rep_ms < single-call latency.
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    n = len(times)
    median = times[n // 2]
    p20 = times[max(0, (n * 20) // 100)]
    p80 = times[min(n - 1, (n * 80) // 100)]
    return float(median), float(p20), float(p80), 0.0, 0.0


def cuda_allocated_mib() -> float:
    """Currently-allocated CUDA memory in MiB. Deltas around ``build_algorithm``
    isolate the index's marginal cost (excludes shared ``item_embs`` and any
    transient build-time scratch the caching allocator has since released)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 * 1024)


# ---------- model load --------------------------------------------------------

_D128_DROP05 = {
    "max_seq_length": 200,
    "embedding_dim": 128,
    "num_heads": 2,
    "num_blocks": 2,
    "ffn_hidden_dim": 512,
    "dropout": 0.5,
    "reuse_item_embeddings": False,
}


def load_model(checkpoint_path: Path, num_items: int, device: torch.device) -> GSASRec:
    """Read sibling config.json when present (smoke / 5B / freshly-trained
    ckpts have it via ``GSASRecConfig.save``); fall back to the d128-drop0.5
    hyperparams for the legacy 500M ckpts that don't ship one.
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
    state = torch.load(str(checkpoint_path), map_location=str(device), weights_only=True)
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
    Encoding batch size is independent of the per-algo perf batch.
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
    queries: torch.Tensor,  # [N, D] on cpu
    targets: torch.Tensor,  # [N, T] on cpu
    num_targets: torch.Tensor,  # [N]   on cpu
    *,
    k: int,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    """Stream cached queries through the index at bs=1 and accumulate metrics.

    Quality is invariant to perf batch size (same scoring math), so we pay
    the bs=1 stream once per ``(algo, k)`` cell and attach the result to every
    bs row of that cell.
    """
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
    batch_size: int,
    device: torch.device,
    is_cpu: bool,
    seed: int,
    n_pool: int = 64,
) -> tuple[float, float, float, float, float]:
    """Time the index forward at the given ``batch_size`` over a query pool.

    Single fixed query collapses p20/p80 to one cluster's traversal cost for
    IVF-style algorithms. Sample ``n_pool`` query batches with a fixed seed
    and round-robin through them inside the timing loop so each iteration
    sees a different cluster.

    Returns ``(median_ms, p20_ms, p80_ms, peak_mib, transient_mib)``.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = queries.shape[0]
    rows = torch.randint(0, n, (n_pool, batch_size), generator=g)
    pool = queries[rows.reshape(-1)].reshape(n_pool, batch_size, -1).to(device).contiguous()

    counter = {"i": 0}

    def perf_fn() -> None:
        q = pool[counter["i"] % n_pool]
        counter["i"] += 1
        with torch.inference_mode():
            forward(q)

    if is_cpu:
        return measure_forward_cpu(perf_fn)
    return measure_forward_cuda(perf_fn)


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
        algo_params = cfg.algo_params.get(algo, {})
        for k in cfg.ks:
            logger.info("=== {} k={} ===", algo, k)

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
                params=algo_params,
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
                        "params": {str(pk): str(pv) for pk, pv in algo_params.items()},
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
