from __future__ import annotations

import json
import time
from pathlib import Path

import click
import torch
from loguru import logger
from torch.export import load

from retrieve import (
    BloomIndex,
    ClauseIndex,
    IVF_INT8_ANN,
    SilverTorch,
)


def _module_bytes(module: torch.nn.Module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


def _time_cuda(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _time_cpu(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def _stats(values: list[float]) -> dict[str, float]:
    xs = sorted(values)
    n = len(xs)
    return {
        "mean": sum(xs) / n,
        "p50": xs[n // 2],
        "p95": xs[min(int(n * 0.95), n - 1)],
        "min": xs[0],
        "max": xs[-1],
    }


def _bloom_metrics(
    item_attrs: torch.Tensor,
    query_attrs: torch.Tensor,
    m_bits: int,
    k_hash: int,
    device: torch.device,
) -> dict[str, float]:
    bi = BloomIndex().to(device)
    bi.register_index(item_attrs, m_bits=m_bits, k_hash=k_hash)
    ci = ClauseIndex().to(device)
    ci.register_index(item_attrs)

    bloom = bi.evaluate(query_attrs)
    clause = ci.evaluate_mask(query_attrs)

    bloom_pass = int(bloom.sum().item())
    clause_pass = int(clause.sum().item())
    total = int(bloom.numel())
    false_positives = int((bloom & ~clause).sum().item())
    return {
        "bloom_pass_rate": bloom_pass / total,
        "clause_pass_rate": clause_pass / total,
        "fpr": false_positives / max(1, total - clause_pass),
    }


def _run_single(
    checkpoint_dir: str,
    index_name: str,
    batch_size: int,
    warmup: int,
    iters: int,
    device: str,
) -> None:
    ckpt = Path(checkpoint_dir)
    with open(ckpt / "config.json") as f:
        cfg = json.load(f)
    max_len = cfg["max_seq_length"]
    emb_dim = cfg["embedding_dim"]

    with open(ckpt / "stats.json") as f:
        num_items = json.load(f)["num_items"]

    dev = torch.device(device)
    is_cuda = dev.type == "cuda"

    encoder = load(ckpt / "encoder.pt2").module().to(dev).eval()
    idx = load(ckpt / f"index_{index_name}.pt2").module().to(dev).eval()

    item_seq = torch.randint(1, num_items + 1, (batch_size, max_len), device=dev)
    query_input = torch.randn(batch_size, emb_dim, device=dev)

    timer = _time_cuda if is_cuda else _time_cpu

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    with torch.inference_mode():
        enc_times = timer(lambda: encoder(item_seq), warmup, iters)
    enc_peak = int(torch.cuda.max_memory_allocated(dev)) if is_cuda else 0

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    with torch.inference_mode():
        idx_times = timer(lambda: idx(query_input), warmup, iters)
    idx_peak = int(torch.cuda.max_memory_allocated(dev)) if is_cuda else 0

    out = {
        "device": str(dev),
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "encoder": {
            "latency_ms": _stats(enc_times),
            "mem_bytes": {"params": _module_bytes(encoder), "peak": enc_peak},
        },
        "index": {
            "name": index_name,
            "latency_ms": _stats(idx_times),
            "mem_bytes": {"params": _module_bytes(idx), "peak": idx_peak},
        },
    }
    logger.info("Perf: {}", json.dumps(out, indent=2))
    with open(ckpt / "eval_perf.json", "w") as f:
        json.dump(out, f, indent=2)


def _build_random_queries(
    item_attrs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Synthesize per-query attrs by sampling each clause's first non-pad value
    from a random item (skipping pad item 0)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    n = item_attrs.shape[0]
    c = item_attrs.shape[1]
    items = torch.randint(1, n, (batch_size,), generator=g)
    q = torch.full((batch_size, c), -1, dtype=torch.long)
    for i, item_id in enumerate(items.tolist()):
        for ci in range(c):
            vals = item_attrs[item_id, ci]
            positive = vals[vals >= 0]
            if positive.numel() > 0:
                q[i, ci] = positive[0]
    return q.to(device)


def _run_sweep(
    checkpoint_dir: str,
    attrs_path: str,
    batch_sizes: tuple[int, ...],
    n_lists_grid: tuple[int, ...],
    n_probe_grid: tuple[int, ...],
    m_bits_grid: tuple[int, ...],
    k_hash_grid: tuple[int, ...],
    k: int,
    warmup: int,
    iters: int,
    device: str,
) -> None:
    ckpt = Path(checkpoint_dir)
    dev = torch.device(device)
    is_cuda = dev.type == "cuda"
    timer = _time_cuda if is_cuda else _time_cpu

    item_embs = torch.load(ckpt / "item_embs.pt", map_location=dev, weights_only=True)
    item_attrs = torch.load(attrs_path, map_location=dev, weights_only=True)

    results: list[dict] = []

    for n_lists in n_lists_grid:
        for n_probe in n_probe_grid:
            if n_probe > n_lists:
                continue
            for m_bits in m_bits_grid:
                for k_hash in k_hash_grid:
                    st = SilverTorch(
                        k=k, n_lists=n_lists, n_probe=n_probe,
                        m_bits=m_bits, k_hash=k_hash, n_iter=10,
                    ).to(dev)
                    st.register_index(item_embs, item_attrs)

                    ivf = IVF_INT8_ANN(
                        k=k, n_lists=n_lists, n_probe=n_probe, n_iter=10
                    ).to(dev)
                    ivf.register_index(item_embs)

                    for batch_size in batch_sizes:
                        q = torch.randn(batch_size, item_embs.shape[1], device=dev)
                        q_attrs = _build_random_queries(item_attrs, batch_size, dev)

                        bm = _bloom_metrics(
                            item_attrs, q_attrs, m_bits, k_hash, dev
                        )

                        if is_cuda:
                            torch.cuda.reset_peak_memory_stats(dev)
                        with torch.inference_mode():
                            st_times = timer(lambda: st(q, q_attrs), warmup, iters)
                        st_peak = (
                            int(torch.cuda.max_memory_allocated(dev)) if is_cuda else 0
                        )

                        if is_cuda:
                            torch.cuda.reset_peak_memory_stats(dev)
                        with torch.inference_mode():
                            ivf_times = timer(lambda: ivf(q), warmup, iters)
                        ivf_peak = (
                            int(torch.cuda.max_memory_allocated(dev)) if is_cuda else 0
                        )

                        results.append(
                            {
                                "n_lists": n_lists,
                                "n_probe": n_probe,
                                "m_bits": m_bits,
                                "k_hash": k_hash,
                                "batch_size": batch_size,
                                "bloom_metrics": bm,
                                "silvertorch": {
                                    "latency_ms": _stats(st_times),
                                    "peak_bytes": st_peak,
                                    "params_bytes": _module_bytes(st),
                                },
                                "ivf_int8": {
                                    "latency_ms": _stats(ivf_times),
                                    "peak_bytes": ivf_peak,
                                    "params_bytes": _module_bytes(ivf),
                                },
                            }
                        )
                        logger.info(
                            "n_lists={} n_probe={} m_bits={} k_hash={} bs={} "
                            "st_p95={:.3f}ms ivf_p95={:.3f}ms fpr={:.4f} pass={:.4f}",
                            n_lists, n_probe, m_bits, k_hash, batch_size,
                            _stats(st_times)["p95"],
                            _stats(ivf_times)["p95"],
                            bm["fpr"], bm["bloom_pass_rate"],
                        )

    out = {
        "device": str(dev),
        "warmup": warmup,
        "iters": iters,
        "k": k,
        "n_items": int(item_embs.shape[0]),
        "embedding_dim": int(item_embs.shape[1]),
        "results": results,
    }
    with open(ckpt / "eval_perf_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote sweep with {} configs to eval_perf_sweep.json", len(results))


@click.command()
@click.option("--checkpoint-dir", type=str, required=True)
@click.option("--index", type=str, default="fullscan")
@click.option("--batch-size", type=int, default=256)
@click.option("--warmup", type=int, default=20)
@click.option("--iters", type=int, default=200)
@click.option("--device", type=str, default="cuda")
@click.option("--sweep/--no-sweep", default=False)
@click.option("--attrs-path", type=str, default=None)
@click.option("--k", type=int, default=100)
@click.option(
    "--sweep-batches",
    type=str,
    default="32,128",
    help="Comma-separated batch sizes for sweep mode.",
)
@click.option(
    "--sweep-n-lists",
    type=str,
    default="256",
    help="Comma-separated n_lists values for sweep.",
)
@click.option(
    "--sweep-n-probe",
    type=str,
    default="16,64,256",
    help="Comma-separated n_probe values for sweep.",
)
@click.option(
    "--sweep-m-bits",
    type=str,
    default="512,1024,2048",
    help="Comma-separated m_bits values for sweep.",
)
@click.option(
    "--sweep-k-hash",
    type=str,
    default="3,5,7",
    help="Comma-separated k_hash values for sweep.",
)
def main(
    checkpoint_dir: str,
    index: str,
    batch_size: int,
    warmup: int,
    iters: int,
    device: str,
    sweep: bool,
    attrs_path: str | None,
    k: int,
    sweep_batches: str,
    sweep_n_lists: str,
    sweep_n_probe: str,
    sweep_m_bits: str,
    sweep_k_hash: str,
) -> None:
    if sweep:
        if attrs_path is None:
            raise click.UsageError("--attrs-path is required in sweep mode.")
        _run_sweep(
            checkpoint_dir=checkpoint_dir,
            attrs_path=attrs_path,
            batch_sizes=tuple(int(x) for x in sweep_batches.split(",")),
            n_lists_grid=tuple(int(x) for x in sweep_n_lists.split(",")),
            n_probe_grid=tuple(int(x) for x in sweep_n_probe.split(",")),
            m_bits_grid=tuple(int(x) for x in sweep_m_bits.split(",")),
            k_hash_grid=tuple(int(x) for x in sweep_k_hash.split(",")),
            k=k,
            warmup=warmup,
            iters=iters,
            device=device,
        )
    else:
        _run_single(
            checkpoint_dir=checkpoint_dir,
            index_name=index,
            batch_size=batch_size,
            warmup=warmup,
            iters=iters,
            device=device,
        )


if __name__ == "__main__":
    main()
