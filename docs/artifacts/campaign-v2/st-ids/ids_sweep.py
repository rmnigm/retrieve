"""ST-IDS: the id epilogue's tile against v2.2's, kernel-only, to locate the k 100 × n_probe 1024 slowdown of the
first cut (one program per row looping over k and probes). Variants: `loop` (that first cut, defined here) and
`grid` (the shipped library kernel: each k chunk its own program on grid axis 1, the probe loop inside), BLOCK_K /
BLOCK_N / num_warps overridden. Inputs: real top-k slots from the after scorer (dloop_gate's index, D 128). ABAB graphs, pairs.

    PYTHONPATH=<pkgs>:<v21 pkg>:retrieve/src:retrieve:docs/artifacts/campaign-v2/st-dloop:. python ids_sweep.py out.json
"""

import json
import statistics
import sys

import torch
import triton
import triton.language as tl
from dloop_gate import (
    DEV,
    POOL,
    build_index,
    graph_of,
    probe_width,
    ratio_ci,
    sm_mhz,
    window_us,
)
from ids_gate import COMMON, HOSTS, scores_and_slots


@triton.jit
def ids_loop_kernel(slots_ptr, scores_ptr, probe_ids_ptr, offsets_ptr, sort_perm_ptr, ids_ptr, n_probe, k,
                    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):  # fmt: skip
    """The first cut: one program per row, looping over k chunks and, inside, over probe chunks."""
    bid = tl.program_id(0)
    for k0 in tl.range(0, k, BLOCK_K):
        r = k0 + tl.arange(0, BLOCK_K)
        in_k = r < k
        slot = tl.load(slots_ptr + bid * k + r, mask=in_k, other=0)
        score = tl.load(scores_ptr + bid * k + r, mask=in_k, other=float("-inf"))
        shift = tl.zeros([BLOCK_K], dtype=tl.int64)
        base = tl.zeros([], dtype=tl.int64)
        for n0 in tl.range(0, n_probe, BLOCK_N):
            i = n0 + tl.arange(0, BLOCK_N)
            live = i < n_probe
            c = tl.load(probe_ids_ptr + bid * n_probe + i, mask=live, other=0)
            lo = tl.load(offsets_ptr + c, mask=live, other=0)
            size = tl.load(offsets_ptr + c + 1, mask=live, other=0) - lo
            end = base + tl.cumsum(size, 0)
            hit = (slot[:, None] >= (end - size)[None, :]) & (
                slot[:, None] < end[None, :]
            )
            shift += tl.sum(tl.where(hit, (lo - (end - size))[None, :], 0), axis=1)
            base += tl.sum(size)
        ids = tl.load(
            sort_perm_ptr + shift + slot, mask=in_k & (score > float("-inf")), other=-1
        )
        tl.store(ids_ptr + bid * k + r, ids, mask=in_k)


CELLS = ((100, 24), (100, 256), (100, 1024), (1000, 24), (1000, 1024))
VARIANTS = [
    ("loop", bk, bn, nw) for bk in (64, 128) for bn in (128, 256, 512) for nw in (4, 8)
]
VARIANTS += [
    ("grid", bk, bn, nw) for bk in (16, 32) for bn in (128, 256, 512) for nw in (4, 8)
]

base = build_index(2_000_000, 128)
g = torch.Generator(device=DEV).manual_seed(1)
pool16 = [torch.randn(16, 128, device=DEV, generator=g) for _ in range(POOL)]
res = []
for k, n_probe in CELLS:
    width = probe_width(base, n_probe)
    for bs in (1, 16):
        fins = []
        for q16 in pool16:
            q = q16[:bs].contiguous()
            probe = torch.topk(q @ base.centroids.t(), n_probe, dim=1).indices
            la = scores_and_slots(base, "none", q, probe, None, k, width)
            fins.append(
                HOSTS["before"].probe_topk(
                    la, k, probe, base.cluster_offsets, base.sort_perm
                )
            )
        before = [(COMMON["before"].probe_ids_kernel, f) for f in fins]
        gb = graph_of(before)
        ref = [f.ids.clone() for _, f in before]
        for kind, bk, bn, nw in VARIANTS:
            bk_, bn_ = (
                min(triton.next_power_of_2(k), bk),
                min(triton.next_power_of_2(n_probe), bn),
            )
            ls = []
            for f in fins:
                kw = {
                    key: v
                    for key, v in f.kwargs.items()
                    if key not in ("NPP", "KP", "num_warps")
                }
                ids = torch.empty_like(f.ids)
                kw.update(ids_ptr=ids, BLOCK_K=bk_, BLOCK_N=bn_, num_warps=nw)
                if kind == "loop":
                    ls.append(
                        (ids_loop_kernel, type(f)((f.grid[0],), kw, ids, f.scores))
                    )
                else:
                    grid = (f.grid[0], triton.cdiv(k, bk_))
                    ls.append(
                        (
                            COMMON["after"].probe_ids_kernel,
                            type(f)(grid, kw, ids, f.scores),
                        )
                    )
            for kern, f in ls:
                kern[f.grid](**f.kwargs)
            same = all(torch.equal(f.ids, r) for (_, f), r in zip(ls, ref, strict=True))
            ga = graph_of(ls)
            for _ in range(2):
                window_us(gb), window_us(ga)
            tb, ta, mhz = [], [], []
            for _ in range(8):
                tb.append(window_us(gb))
                ta.append(window_us(ga))
                mhz.append(sm_mhz())
            r, lo, hi = ratio_ci(tb, ta)
            row = {"k": k, "n_probe": n_probe, "bs": bs, "kind": kind, "block_k": bk, "block_n": bn,
                   "num_warps": nw, "before_us": statistics.median(tb), "after_us": statistics.median(ta),
                   "ratio": r, "ci": [lo, hi], "equal": same, "sm_mhz": [min(mhz), max(mhz)]}  # fmt: skip
            res.append(row)
            print(f"k={k:4d} np={n_probe:4d} bs={bs:2d} {kind} bk={bk:3d} bn={bn:3d} w={nw} before "
                  f"{row['before_us']:8.2f} after {row['after_us']:8.2f} ratio {r:.3f} [{lo:.3f}, {hi:.3f}] "
                  f"eq={same}", flush=True)  # fmt: skip
            del ga
        del gb
json.dump(res, open(sys.argv[1], "w"), indent=1)
