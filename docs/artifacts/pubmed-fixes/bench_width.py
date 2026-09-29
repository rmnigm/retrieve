"""Fix 1 timing gate: the three probe-scorer ops at one width, the L4 gate's layout and timing
method (docs/artifacts/l4-pow2-pad/bench_d128.py, kernel-opt's time_fn). `--src` puts another
`retrieve` tree first on sys.path so interleave.sh can alternate base and new process by process.

    PYTHONPATH=retrieve:docs/artifacts/kernel-opt python bench_width.py out.json --d 192 [--src SRC]
"""

import argparse
import json
import sys


def build_cases(torch, d):
    from retrieve.indexing.quantize import quantize_int8_global
    from retrieve.ops import triton as T
    from tests.parity.conftest import make_bloom, make_exact, make_probe_family

    dev = torch.device("cuda")
    cases = {}
    for b in (1, 16):
        g = torch.Generator(device=dev).manual_seed(0)
        lay = make_probe_family(b, 1024, 5000, 24)
        codes, gs = quantize_int8_global(torch.randn(lay.n, d, device=dev, generator=g))
        q = torch.randn(b, d, device=dev, generator=g)
        qpos, bt, _, _ = make_bloom(lay.n, b, m_bits=1024)
        attrs, rev, qa = make_exact(lay.n, b)
        csr = (lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm)
        k, w = 100, lay.width
        cases[f"cps_none_b{b}"] = lambda q=q, csr=csr, gs=gs, w=w: T.codesigned_probe_score(
            q, *csr, gs, k, w
        )
        cases[f"cps_bloom_b{b}"] = lambda q=q, csr=csr, gs=gs, w=w, qpos=qpos, bt=bt: (
            T.codesigned_probe_score_bloom(q, *csr, qpos, bt, gs, k, w)
        )
        cases[f"cps_exact_b{b}"] = lambda q=q, csr=csr, gs=gs, w=w, a=attrs, r=rev, qa=qa: (
            T.codesigned_probe_score_exact(q, *csr, a, r, qa, gs, k, w)
        )
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--src")
    args = ap.parse_args()
    if args.src:
        sys.path.insert(0, args.src)
    import torch

    import retrieve
    from bench_kernels import time_fn

    torch.manual_seed(0)
    out = {"retrieve": retrieve.__file__, "d": args.d, "cases": {}}
    for name, fn in build_cases(torch, args.d).items():
        r = out["cases"][name] = time_fn(torch, fn)
        print(
            f"{name:18s} {r['us_median']:10.2f} us  spread {r['spread']:.3f}  "
            f"sm {r['sm_mhz'][0]}-{r['sm_mhz'][-1]}  {'UNSTABLE' if r['unstable'] else ''}",
            flush=True,
        )
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
