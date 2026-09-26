"""Kernel timings for the kernel-opt pass: one process times every public op at fixed, seeded
shapes. `--src` puts another `retrieve` tree first on sys.path, so `interleave.sh` can alternate
the baseline and the working tree process by process.

Method (docs/artifacts/q3/roofline.py): 10 warm-up calls, then WINDOWS CUDA-event windows of
`iters` calls each (iters sized to ~20 ms), the per-call median over windows; the SM clock is
sampled with nvidia-smi right after each window's sync (under load). `unstable` when the
window spread (max-min)/median exceeds 5 % or the sampled clock moves by more than 50 MHz.

    python bench_kernels.py out.json [--src /path/to/retrieve/src] [--cases a,b,...]
"""

import argparse
import json
import statistics
import subprocess
import sys

WINDOWS = 15


def sm_mhz() -> int:
    q = ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]
    return int(subprocess.check_output(q, text=True).strip())


def time_fn(torch, fn) -> dict:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    iters = max(3, min(2000, int(20.0 / max(a.elapsed_time(b), 1e-3))))
    per_call, mhz = [], []
    for _ in range(WINDOWS):
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        per_call.append(a.elapsed_time(b) / iters * 1e3)  # µs
        mhz.append(sm_mhz())
    med = statistics.median(per_call)
    spread = (max(per_call) - min(per_call)) / med
    return {
        "us_median": med,
        "us_min": min(per_call),
        "spread": spread,
        "iters": iters,
        "sm_mhz": [min(mhz), max(mhz)],
        "unstable": spread > 0.05 or max(mhz) - min(mhz) > 50,
    }


def build_cases(torch, names):
    from retrieve.indexing.bloom_hash import build_query_signatures, build_signatures, generate_seeds
    from retrieve.ops import triton as T

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    n, b, c, a, w = 3_000_000, 16, 2, 4, 16
    attrs = torch.randint(0, 20, (n, c, a), device=dev, generator=g)
    attrs[torch.rand(n, c, a, device=dev, generator=g) < 0.3] = -1
    rev = torch.zeros(c, dtype=torch.bool, device=dev)
    qa = torch.randint(0, 20, (b, c), device=dev, generator=g)
    seeds = generate_seeds(5, dev)
    sigs = build_signatures(attrs, seeds, 1024, 5, w)
    qb = build_query_signatures(qa.unsqueeze(-1), seeds, 1024, 5, w)
    embs = torch.randn(n, 128, device=dev, generator=g).half()
    q = torch.randn(b, 128, device=dev, generator=g)
    bits = torch.randint(-(2**62), 2**62, (n, 2), device=dev, generator=g)
    qbits = torch.randint(-(2**62), 2**62, (b, 2), device=dev, generator=g)
    pos, counts = T.clause_compact(attrs, rev, qa)

    # A padded IVF probe pool shaped like goodreads (skewed: one 25k cluster, the rest ~780).
    n_lists, n_probe = 1024, 24
    sizes = torch.full((n_lists,), n // n_lists, device=dev)
    sizes[0] = 25_000
    max_size = int(sizes.max())
    padded = torch.full((n_lists, max_size), -1, dtype=torch.long, device=dev)
    lane = torch.arange(max_size, device=dev)
    ids = torch.randint(0, n, (n_lists, max_size), device=dev, generator=g)
    padded = torch.where(lane[None, :] < sizes[:, None], ids, padded)
    probe = torch.stack([torch.randperm(n_lists, device=dev, generator=g)[:n_probe] for _ in range(b)])
    probe[:, 0] = 0
    flat = padded[probe].reshape(b, -1)
    codes = torch.randint(-127, 128, (n, 128), dtype=torch.int8, device=dev, generator=g)

    cases = {
        "clause_mask_b16": lambda: T.clause_mask(attrs, rev, qa),
        "clause_mask_b1": lambda: T.clause_mask(attrs, rev, qa[:1]),
        "clause_compact_b16": lambda: T.clause_compact(attrs, rev, qa),
        "clause_compact_b1": lambda: T.clause_compact(attrs, rev, qa[:1]),
        "bloom_match_b16": lambda: T.bloom_match(qb, sigs),
        "bloom_compact_b16": lambda: T.bloom_compact(qb, sigs),
        "bloom_compact_b1": lambda: T.bloom_compact(qb[:1], sigs),
        "fmkt_b16_k100": lambda: T.fused_masked_knn_topk(q.half(), embs, pos, counts, 100),
        "oporp_full_b16_k5000": lambda: T.oporp_1bit_match_topk_full(qbits, bits, 5000),
        "oporp_indirect_b16_k5000": lambda: T.oporp_1bit_match_topk_indirect(
            qbits, bits, 5000, pos, counts
        ),
        "cps_none_b16": lambda: T.codesigned_probe_score(q, flat, codes, 0.01, 100),
        "cps_bloom_b16": lambda: T.codesigned_probe_score_bloom(q, flat, codes, qb, sigs, 0.01, 100),
        "cps_exact_b16": lambda: T.codesigned_probe_score_exact(
            q, flat, codes, attrs, rev, qa, 0.01, 100
        ),
    }
    return {k: v for k, v in cases.items() if not names or k in names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--src")
    ap.add_argument("--cases", default="")
    args = ap.parse_args()
    if args.src:
        sys.path.insert(0, args.src)
    import torch
    import triton

    import retrieve

    torch.manual_seed(0)
    cases = build_cases(torch, [c for c in args.cases.split(",") if c])
    out = {
        "retrieve": retrieve.__file__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "cases": {},
    }
    for name, fn in cases.items():
        out["cases"][name] = time_fn(torch, fn)
        r = out["cases"][name]
        print(f"{name:28s} {r['us_median']:10.2f} us  spread {r['spread']:.3f}  "
              f"sm {r['sm_mhz']}  {'UNSTABLE' if r['unstable'] else ''}", flush=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
