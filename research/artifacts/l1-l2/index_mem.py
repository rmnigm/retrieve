"""L1: item-table memory of the exact LiNR modules at real catalog shapes, one process per side.

Per shape: `index_mib` as the harness computes it (Σ buffer bytes, `bench.measure.index_bytes`),
the allocation `register_index` adds on top of the caller's fp32 table, and the transient peak of
one masked forward (B=16, k=1000) above the steady state. `--src` puts another `retrieve` tree
first on sys.path (the pre-L1 baseline). `fp32-table` is the rejected alternative: a `[D, N]` fp32
buffer, measured the same way.

    python index_mem.py out.json [--src /path/to/retrieve/src]
"""

import argparse
import json
import sys

SHAPES = {
    "arxiv": (2_988_997, 128),
    "yfcc10m": (10_000_000, 192),
    "pubmed": (10_000_000, 768),
}
B, K = 16, 1000
MiB = 2**20


def measure(torch, make, n, d):
    dev = torch.device("cuda")
    items = torch.randn(n, d, device=dev)
    q = torch.randn(B, d, device=dev)
    mask = torch.rand(B, n, device=dev) < 0.5
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    m = make(items)
    torch.cuda.synchronize()
    steady = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    m(q, mask)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    out = {
        "index_mib": sum(b.numel() * b.element_size() for b in m.buffers()) / MiB,
        "register_mib": (steady - base) / MiB,
        "forward_transient_mib": (peak - steady) / MiB,
    }
    del m, items, q, mask
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--src")
    args = ap.parse_args()
    if args.src:
        sys.path.insert(0, args.src)
    import torch

    import retrieve
    from retrieve.functional import masked_topk
    from retrieve.modules.knn import PostfilterKNN

    torch.backends.cuda.matmul.allow_tf32 = False

    class Fp32Table(torch.nn.Module):
        def register_index(self, items):
            self.register_buffer("item_embs_t", items.t().contiguous())

        def forward(self, q, mask):
            return masked_topk(q @ self.item_embs_t, K, valid=mask)

    def postfilter(items):
        m = PostfilterKNN(k=K)
        m.register_index(items)
        return m

    def fp32_table(items):
        m = Fp32Table()
        m.register_index(items)
        return m

    res = {"retrieve": retrieve.__file__, "shapes": {}}
    for name, (n, d) in SHAPES.items():
        res["shapes"][name] = {
            "PostfilterKNN": measure(torch, postfilter, n, d),
            "fp32-table": measure(torch, fp32_table, n, d),
        }
        print(name, json.dumps(res["shapes"][name]), flush=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
