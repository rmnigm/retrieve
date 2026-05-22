"""Quality bench: SilverTorch int8-storage/fp32-compute vs paper-faithful int8 matmul.

Temporary bench — exploratory, not part of CI. Compares three scoring paths
on the same item index + query batch and reports recall@K against an fp32
oracle:

- ``A_current``: per-item fp32 scale, ``codes.float() @ query`` (fp32 sum).
  Matches the math in ``layers/silvertorch/main.py`` and the Triton
  ``codesigned_probe_score`` kernel.
- ``B_hybrid``: per-item fp32 scale + int8-quantized query (per-row scale) +
  int8 × int8 → int32 matmul, dequantized after. Differs from ``A_current``
  by query quantization + int8 dot, and from ``C_paper`` only by the item
  scale scheme — so (C - B) isolates the global-scale cost.
- ``C_paper``: single global per-tensor scale + int8-quantized query +
  int8 × int8 → int32 matmul. Matches the SilverTorch paper's Int8 ANN
  description (Section 4.2).

Why ``torch._int_mm`` and not ``int_a @ int_b``: plain ``@`` on CUDA fails
for every integer dtype (``addmm_cuda`` is float/complex only). ``_int_mm``
wraps the CUTLASS int8 IMMA GEMM — the same int8 tensor-core path the
paper's dp4a reference exercises. It requires M >= 16 (IMMA tile size),
so the bench asserts ``B >= 16``.

Usage (from ``evaluation/``):

    uv run python scripts/bench_silvertorch_int8mm_quality.py
    uv run python scripts/bench_silvertorch_int8mm_quality.py --normalized
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from torch import Tensor

from retrieve.layers.utils.quantize import quantize_int8


def quantize_int8_global(embs: Tensor) -> tuple[Tensor, float]:
    """Symmetric per-tensor INT8 quantization — one scale for the whole index.

    Paper's scheme (Section 4.2 of the SilverTorch paper): single global
    abs-max, integer codes scaled to ``[-128, 127]``. Reconstruction:
    ``embs ~= codes.float() * scale``.
    """
    abs_max = embs.abs().amax().clamp_min(1e-8)
    scale = float((abs_max / 127.0).item())
    codes = (embs / abs_max * 127.0).round().clamp(-128, 127).to(torch.int8)
    return codes, scale


def score_oracle(query: Tensor, embs: Tensor) -> Tensor:
    return query @ embs.t()


def score_A_current(query: Tensor, codes: Tensor, item_scales: Tensor) -> Tensor:
    """Per-item scale, fp32 dot from int8 codes. Mirrors the current kernel."""
    return (query @ codes.float().t()) * item_scales.unsqueeze(0)


def score_B_hybrid(query: Tensor, codes_t: Tensor, item_scales: Tensor) -> Tensor:
    """Per-item scale + int8 query + int8 matmul.

    Expects pre-transposed codes ``[D, N]`` contiguous. In production the
    index would store this layout directly — including the transpose in the
    timed region would charge a per-call 64 MB memcopy that a Triton kernel
    avoids entirely (per-tile loads need no contiguous transpose).
    """
    q_codes, q_scales = quantize_int8(query)
    acc = torch._int_mm(q_codes, codes_t).to(torch.float32)
    return acc * q_scales.unsqueeze(1) * item_scales.unsqueeze(0)


def score_C_paper(query: Tensor, codes_g_t: Tensor, global_scale: float) -> Tensor:
    """Global scale + int8 query + int8 matmul. Paper-faithful.

    Same ``[D, N]`` pre-transposed contract as ``score_B_hybrid``.
    """
    q_codes, q_scales = quantize_int8(query)
    acc = torch._int_mm(q_codes, codes_g_t).to(torch.float32)
    return acc * q_scales.unsqueeze(1) * global_scale


def recall_at_k(approx_ids: Tensor, exact_ids: Tensor) -> float:
    """Mean per-row id-set overlap divided by K. Matches conftest's helper."""
    b, k = approx_ids.shape
    return sum(
        len(set(approx_ids[i].tolist()) & set(exact_ids[i].tolist())) for i in range(b)
    ) / (b * k)


def topk_score_mae(
    approx_scores: Tensor,
    oracle_scores: Tensor,
    oracle_ids: Tensor,
) -> tuple[float, float]:
    """MAE and mean-rel-error of approx vs oracle at oracle's top-K positions.

    Picks scores at oracle's top-K ids in both tensors — ``approx`` is read
    at those same ids so we measure ranking faithfulness on the items the
    oracle considered top, not on whichever items ``approx`` happened to
    rank highly.
    """
    approx_at = approx_scores.gather(1, oracle_ids)
    oracle_at = oracle_scores.gather(1, oracle_ids)
    abs_err = (approx_at - oracle_at).abs()
    mae = float(abs_err.mean().item())
    denom = oracle_at.abs().clamp_min(1e-8)
    mre = float((abs_err / denom).mean().item())
    return mae, mre


def per_row_corr(approx: Tensor, oracle: Tensor) -> float:
    """Mean Pearson correlation between approx and oracle row-by-row."""
    a = approx - approx.mean(dim=1, keepdim=True)
    o = oracle - oracle.mean(dim=1, keepdim=True)
    denom = (a.norm(dim=1) * o.norm(dim=1)).clamp_min(1e-8)
    return float(((a * o).sum(dim=1) / denom).mean().item())


def _make_data(n: int, d: int, b: int, seed: int, normalized: bool, device: torch.device):
    g_e = torch.Generator(device=device).manual_seed(seed)
    g_q = torch.Generator(device=device).manual_seed(seed + 1)
    embs = torch.randn(n, d, generator=g_e, device=device)
    query = torch.randn(b, d, generator=g_q, device=device)
    if normalized:
        embs = embs / embs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        query = query / query.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return embs, query


def _print_table(rows: list[dict], k_list: list[int]) -> None:
    headers = (
        ["variant"]
        + [f"R@{k}" for k in k_list]
        + ["score_mae", "score_mre", "corr", "us/call"]
    )
    widths = [max(len(h), 12) for h in headers]
    sep = "  ".join("-" * w for w in widths)
    click.echo("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    click.echo(sep)
    for row in rows:
        cells = [row["variant"].ljust(widths[0])]
        for i, k in enumerate(k_list):
            cells.append(f"{row[f'recall@{k}']:.4f}".ljust(widths[i + 1]))
        cells.append(f"{row['score_mae']:.4e}".ljust(widths[-4]))
        cells.append(f"{row['score_mre']:.4e}".ljust(widths[-3]))
        cells.append(f"{row['corr']:.4f}".ljust(widths[-2]))
        us = row.get("us_per_call")
        cells.append((f"{us:.1f}" if us is not None else "-").ljust(widths[-1]))
        click.echo("  ".join(cells))


def _time_us(fn, *, warmup: int = 10, iters: int = 50) -> float:
    """Median microseconds per call. Score-only timing — no top-K."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ms[len(ms) // 2] * 1000.0


@click.command()
@click.option("--n", default=50_000, show_default=True, help="Index size.")
@click.option("--d", default=128, show_default=True, help="Embedding dim.")
@click.option("--b", default=64, show_default=True, help="Query batch size (>=16 for _int_mm).")
@click.option(
    "--k-list",
    default="10,100,1024,2048",
    show_default=True,
    help="Comma-separated K values for recall@K.",
)
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--normalized/--no-normalized",
    default=False,
    show_default=True,
    help="L2-normalize embs+query. Paper item embs aren't unit-norm; default off so "
    "the per-item-scale advantage is visible.",
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("../_agent_scratch/bench_results/silvertorch_int8mm_quality.json"),
    show_default=True,
    help="Where to dump per-variant results (relative to evaluation/).",
)
def main(
    n: int,
    d: int,
    b: int,
    k_list: str,
    seed: int,
    normalized: bool,
    device: str,
    json_out: Path,
) -> None:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA required.")
    if b < 16:
        raise click.ClickException(f"B must be >= 16 for torch._int_mm, got {b}.")
    dev = torch.device(device)
    ks = [int(x) for x in k_list.split(",") if x.strip()]
    k_max = max(ks)
    if k_max > n:
        raise click.ClickException(f"max K ({k_max}) exceeds N ({n}).")

    click.echo(
        f"# bench: N={n} D={d} B={b} K={ks} normalized={normalized} seed={seed} device={dev}",
        err=True,
    )

    embs, query = _make_data(n, d, b, seed, normalized, dev)

    # One quantization per variant — done once, reused across K's.
    item_codes_pi, item_scales_pi = quantize_int8(embs)
    item_codes_g, global_scale = quantize_int8_global(embs)
    # Pre-transposed [D, N] codes for _int_mm — a production index would
    # store this layout (or both) at build time. Charging the transpose
    # per-call would dominate timing with a 64 MB int8 memcopy at N=500k.
    item_codes_pi_t = item_codes_pi.t().contiguous()
    item_codes_g_t = item_codes_g.t().contiguous()

    variants = {
        "A_current": lambda: score_A_current(query, item_codes_pi, item_scales_pi),
        "B_hybrid": lambda: score_B_hybrid(query, item_codes_pi_t, item_scales_pi),
        "C_paper": lambda: score_C_paper(query, item_codes_g_t, global_scale),
    }

    # Oracle once.
    oracle_scores = score_oracle(query, embs)
    # Top K_max once; slice per K when computing recall.
    _, oracle_topk = torch.topk(oracle_scores, k_max, dim=1)

    # Include the fp32 oracle in the perf table so the int8 paths have a
    # fair fp32 reference too (oracle has no recall row — that's the baseline).
    variants_with_oracle = {"oracle_fp32": lambda: score_oracle(query, embs), **variants}

    rows: list[dict] = []
    for name, fn in variants_with_oracle.items():
        scores = fn()
        row: dict[str, float | str | None] = {"variant": name}
        if name == "oracle_fp32":
            for k in ks:
                row[f"recall@{k}"] = 1.0
            row["score_mae"] = 0.0
            row["score_mre"] = 0.0
            row["corr"] = 1.0
        else:
            _, approx_topk = torch.topk(scores, k_max, dim=1)
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(approx_topk[:, :k], oracle_topk[:, :k])
            mae, mre = topk_score_mae(scores, oracle_scores, oracle_topk[:, : ks[0]])
            row["score_mae"] = mae
            row["score_mre"] = mre
            row["corr"] = per_row_corr(scores, oracle_scores)
        row["us_per_call"] = _time_us(fn)
        rows.append(row)

    _print_table(rows, ks)

    # Sanity checks — surface assumption violations loudly.
    by_name = {r["variant"]: r for r in rows}
    a_r = by_name["A_current"][f"recall@{ks[-1]}"]
    if a_r < 0.99:
        click.echo(
            f"!! sanity: A_current recall@{ks[-1]} = {a_r:.4f} < 0.99 — "
            "per-item scale with fp32 sum should be near-exact at this D.",
            err=True,
        )
    a_top = by_name["A_current"][f"recall@{ks[0]}"]
    b_top = by_name["B_hybrid"][f"recall@{ks[0]}"]
    if abs(a_top - b_top) > 0.02:
        click.echo(
            f"!! sanity: A_current vs B_hybrid diverged at recall@{ks[0]} "
            f"({a_top:.4f} vs {b_top:.4f}) — query int8-quant should cost <2%.",
            err=True,
        )

    if json_out is not None:
        out_path = json_out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "n": n,
                "d": d,
                "b": b,
                "k_list": ks,
                "seed": seed,
                "normalized": normalized,
                "device": str(dev),
            },
            "global_scale": global_scale,
            "results": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        click.echo(f"# JSON dumped to {out_path}", err=True)


if __name__ == "__main__":
    main()
