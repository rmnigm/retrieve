"""L4 timing gate: every padded op at D1's power-of-two widths (D = 128, OPORP W = 2, bloom
W = 16), goodreads-scale N. Timing method and `unstable` rule are kernel-opt's
(docs/artifacts/kernel-opt/bench_kernels.py, imported from there). `--src` puts another
`retrieve` tree first on sys.path so interleave.sh can alternate staging and L4 process by
process; the fixtures come from this worktree's retrieve/tests (data builders only).

    PYTHONPATH=retrieve:docs/artifacts/kernel-opt python bench_d128.py out.json [--src SRC]
"""

import argparse
import json
import sys


def build_cases(torch):
    from retrieve.indexing.quantize import quantize_int8_global
    from retrieve.ops import triton as T
    from tests.parity.conftest import make_bloom, make_exact, make_probe_family

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    b, d = 16, 128
    lay = make_probe_family(b, 1024, 5000, 24)
    n = lay.n
    codes, gs = quantize_int8_global(torch.randn(n, d, device=dev, generator=g))
    q = torch.randn(b, d, device=dev, generator=g)
    qpos, bt, sigs, qb = make_bloom(n, b, m_bits=1024)
    attrs, rev, qa = make_exact(n, b)
    pos, counts = T.clause_compact(attrs, rev, qa)
    embs = torch.randn(n, d, device=dev, generator=g).half()
    bits = torch.randint(-(2**62), 2**62, (n, 2), device=dev, generator=g)
    qbits = torch.randint(-(2**62), 2**62, (b, 2), device=dev, generator=g)
    csr = (lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm)
    k, w = 100, lay.width
    return {
        "cps_none_b16": lambda: T.codesigned_probe_score(q, *csr, gs, k, w),
        "cps_bloom_b16": lambda: T.codesigned_probe_score_bloom(
            q, *csr, qpos, bt, gs, k, w
        ),
        "cps_exact_b16": lambda: T.codesigned_probe_score_exact(
            q, *csr, attrs, rev, qa, gs, k, w
        ),
        "fmkt_b16_k100": lambda: T.fused_masked_knn_topk(
            q.half(), embs, pos, counts, k
        ),
        "oporp_full_b16_k5000": lambda: T.oporp_1bit_match_topk_full(qbits, bits, 5000),
        "oporp_indirect_b16_k5000": lambda: T.oporp_1bit_match_topk_indirect(
            qbits, bits, 5000, pos, counts
        ),
        "bloom_match_b16": lambda: T.bloom_match(qb, sigs),
        "bloom_compact_b16": lambda: T.bloom_compact(qb, sigs),
        "bloom_compact_b1": lambda: T.bloom_compact(qb[:1], sigs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--src")
    args = ap.parse_args()
    if args.src:
        sys.path.insert(0, args.src)
    import torch

    import retrieve
    from bench_kernels import time_fn

    torch.manual_seed(0)
    out = {"retrieve": retrieve.__file__, "cases": {}}
    for name, fn in build_cases(torch).items():
        r = out["cases"][name] = time_fn(torch, fn)
        print(
            f"{name:28s} {r['us_median']:10.2f} us  spread {r['spread']:.3f}  "
            f"sm {r['sm_mhz'][0]}-{r['sm_mhz'][-1]}  {'UNSTABLE' if r['unstable'] else ''}",
            flush=True,
        )
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
