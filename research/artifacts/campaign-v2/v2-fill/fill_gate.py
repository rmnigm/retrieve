"""V2-FILL gates for `fused_masked_knn_topk` (LiNR V2's scorer), one (N, D) per process (from v2-highp/fmkt_gate.py).
Arms: `before` = campaign-v2.4 (`retrieve_v24`, package from st-ids/make_pkg.sh), `after` = this tree.

Inputs as V2 feeds them: fp16 items [N, D] (Gaussian, seed 0), fp16 queries, and per row a compacted candidate
list [B, N] (`clause_compact`'s shape: the row's passing ids, here an independent Bernoulli(p) mask per row, then
garbage past `counts`), so P = N.

  exact  ids + scores `torch.equal`, before vs after, for the public op and the bucketed `_impl`:
         p {0, 0.001, 0.01, 0.1, 1} + the skewed cell × bs {1, 16} × k {100, 1000}.
  time   the scorer kernel alone (the public op's launch, no top-k): CUDA graphs of LAUNCHES over POOL query
         batches, ABAB windows, `--pairs` pairs, after / before with a 95 % t-interval;
         p {0.001, 0.01, 0.1, 1} × bs {1, 16}, plus a skewed cell (2 rows at p 0.12, 14 at 0.00002). SM clock
         sampled after every window.

    PYTHONPATH=<pkgs>:retrieve/src python fmkt_gate.py {exact|time} N D out.json [--pairs 10]
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch
import triton

DEV = torch.device("cuda")
POOL, LAUNCHES = 4, 8
SKEW_BASE = 0.00002  # the skewed cell: rows 0-1 at p 0.12, the other 14 at this rate
T975 = {7: 2.365, 9: 2.262, 11: 2.201, 15: 2.131, 23: 2.069}
PKGS = {"before": "retrieve_v24", "after": "retrieve"}
MODS = {
    a: importlib.import_module(f"{p}.ops.triton.fused_masked_knn_topk")
    for a, p in PKGS.items()
}


def sm_mhz() -> int:
    q = [
        "nvidia-smi",
        "--query-gpu=clocks.sm",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    return int(subprocess.check_output(q, text=True).strip().splitlines()[0])


def candidates(
    n: int, b: int, p: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = torch.randint(0, n, (b, n), device=DEV, generator=g)  # garbage past counts
    counts = torch.empty(b, dtype=torch.long, device=DEV)
    for r in range(b):
        ids = (torch.rand(n, device=DEV, generator=g) < p).nonzero().squeeze(1)
        out[r, : ids.numel()] = ids
        counts[r] = ids.numel()
    return out, counts


def launch_of(arm, q, items, cand, counts):
    mod = MODS[arm]
    # The arm's own config rule, as its public op uses it (only this tree has config_for_width).
    cfg = getattr(mod, "config_for_width", lambda d: mod.DEFAULT_CONFIG)(q.shape[1])
    la = mod._fmkt_prep(q, items, cand, counts, cfg, bucket=False)
    return mod._fused_masked_knn_topk_kernel, la


def graph_of(ls):
    def run():
        for i in range(LAUNCHES):
            kern, la = ls[i % len(ls)]
            kern[la.grid](**la.kwargs)

    run()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            run()
    torch.cuda.synchronize()
    return g


def window_ms(g) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / LAUNCHES


def ratio_ci(num, den):
    logs = [math.log(a / b) for a, b in zip(num, den, strict=True)]
    mu, sd = statistics.mean(logs), statistics.stdev(logs)
    half = T975[len(logs) - 1] * sd / math.sqrt(len(logs))
    return math.exp(mu), math.exp(mu - half), math.exp(mu + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["exact", "time"])
    ap.add_argument("n", type=int)
    ap.add_argument("d", type=int)
    ap.add_argument("out")
    ap.add_argument("--pairs", type=int, default=10)
    args = ap.parse_args()
    g = torch.Generator(device=DEV).manual_seed(0)
    items = torch.randn(args.n, args.d, device=DEV, generator=g).to(torch.float16)
    queries = torch.randn(POOL, 16, args.d, device=DEV, generator=g).to(torch.float16)
    res = {"what": args.what, "n": args.n, "d": args.d, "device": torch.cuda.get_device_name(),
           "torch": torch.__version__, "triton": triton.__version__, "rows": []}  # fmt: skip
    rates = (
        (0.0, 0.001, 0.01, 0.1, 1.0)
        if args.what == "exact"
        else (0.001, 0.01, 0.1, 1.0)
    )
    rates = (*rates, SKEW_BASE)
    for p in rates:
        cand16, counts16 = candidates(args.n, 16, p, seed=1)
        if (
            p == SKEW_BASE
        ):  # two heavy rows at p 0.12, as PubMed all5's (median 142, p90 1.2 M)
            hi, hi_counts = candidates(args.n, 2, 0.12, seed=2)
            cand16[:2], counts16[:2] = hi, hi_counts
        for bs in (1, 16):
            cand, counts = cand16[:bs].contiguous(), counts16[:bs].contiguous()
            if args.what == "exact":
                for k in (100, 1000):
                    same = True
                    for i in range(POOL):
                        q = queries[i, :bs].contiguous()
                        for fn in (
                            "fused_masked_knn_topk",
                            "_fused_masked_knn_topk_impl",
                        ):
                            ib, sb = getattr(MODS["before"], fn)(
                                q, items, cand, counts, k
                            )
                            ia, sa = getattr(MODS["after"], fn)(
                                q, items, cand, counts, k
                            )
                            same &= torch.equal(ib, ia) and torch.equal(sb, sa)
                    row = {"p": p, "bs": bs, "k": k, "equal": same,
                           "mean_count": float(counts.float().mean())}  # fmt: skip
                    res["rows"].append(row)
                    print(f"  n={args.n} d={args.d} {row}", flush=True)
                continue
            ls = {a: [launch_of(a, queries[i, :bs].contiguous(), items, cand, counts) for i in range(POOL)]
                  for a in MODS}  # fmt: skip
            same = True
            for (kb, lb), (ka, la_) in zip(ls["before"], ls["after"], strict=True):
                kb[lb.grid](**lb.kwargs)
                ka[la_.grid](**la_.kwargs)
                same &= torch.equal(lb.all_scores, la_.all_scores)
            graphs = {a: graph_of(v) for a, v in ls.items()}
            for _ in range(2):
                for gr in graphs.values():
                    window_ms(gr)
            t = {a: [] for a in graphs}
            mhz = []
            for _ in range(args.pairs):
                for a in ("before", "after"):
                    t[a].append(window_ms(graphs[a]))
                mhz.append(sm_mhz())
            r, lo, hi = ratio_ci(t["after"], t["before"])
            row = {"p": p, "bs": bs, "before_ms": statistics.median(t["before"]),
                   "after_ms": statistics.median(t["after"]), "after_over_before": [r, lo, hi],
                   "scores_equal": same, "programs": {a: v[0][1].grid for a, v in ls.items()},
                   "sm_mhz": [min(mhz), max(mhz)], "unstable": max(mhz) - min(mhz) > 50}  # fmt: skip
            res["rows"].append(row)
            print(f"  n={args.n} d={args.d} p={p} bs={bs}: before {row['before_ms']:.3f} after "
                  f"{row['after_ms']:.3f} ms; after/before {r:.3f} [{lo:.3f}, {hi:.3f}] equal {same} sm {row['sm_mhz']}"
                  f"{' UNSTABLE' if row['unstable'] else ''}", flush=True)  # fmt: skip
            del graphs, ls
            torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
