"""Why `linr_v1_filter_mask` fails the exact-algo gate on YFCC-10M (E1, 2026-09-16).

Run from evaluation/ with PYTHONPATH set, on the CPU, after a `bench run --dataset yfcc10m
--suite filter --algo linr_v1_filter_mask --backend torch --filter-kind clause` whose parity
spill (`<out>/_parity/*.npz`) and oracle blob (`data/yfcc10m/gt_d192/oracle_v4_tags_and_*.pt`)
are still on disk. Two passes:

1. split the module's misses against the oracle's top-1000 into "tied at the module's 1000th
   score" and "strictly better than it", with the largest score gap;
2. for a sample of rows, rebuild the capped-predicate pass set from item_attrs_narrow.pt, take
   the fp64-exact top-1000 over it, and score the oracle blob, the module and a plain fp32
   top-k against that reference.

Output of the 2026-09-16 run is next to this file (fp16-gate-diagnostic.log).
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

OUT = sys.argv[1] if len(sys.argv) > 1 else "/scratch/campaigns/e1-yfcc-cpu"
DATA = "data/yfcc10m"
torch.set_num_threads(16)

blob = torch.load(glob.glob(f"{DATA}/gt_d192/oracle_v4_tags_and_*.pt")[0], map_location="cpu",
                  weights_only=True)  # fmt: skip
topk = blob["topk"]
spills = {p: np.load(p) for p in glob.glob(f"{OUT}/_parity/*.npz")}
emb = torch.load(f"{DATA}/content_d192/text_emb.pt", map_location="cpu", weights_only=True)
emb32 = F.normalize(emb.float(), dim=-1)
del emb
n_q = topk.shape[0]
q = F.normalize(
    torch.load(f"{DATA}/content_d192/query_emb.pt", map_location="cpu", weights_only=True)[:n_q].float(),
    dim=-1,
)  # fmt: skip


def recall(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a[a >= 0].numpy(), b[b >= 0].numpy()
    return np.intersect1d(a, b).size / max(a.size, 1)


# ---- pass 1: ties vs real misses, per spill --------------------------------------------
for p, z in spills.items():
    ids, sc = torch.from_numpy(z["ids"]).long(), torch.from_numpy(z["scores"])
    hit = tot = tie = real = 0
    worst = []
    for r in range(min(ids.shape[0], n_q)):
        o, m = topk[r][topk[r] >= 0], ids[r][ids[r] >= 0]
        if o.numel() == 0:
            continue
        hit += np.intersect1d(o.numpy(), m.numpy()).size
        tot += o.numel()
        miss = torch.from_numpy(np.setdiff1d(o.numpy(), m.numpy()))
        if miss.numel():
            s_min = sc[r][ids[r] >= 0].min()
            x = emb32[miss] @ q[r]
            t = int((x <= s_min + 1e-6).sum())
            tie += t
            real += miss.numel() - t
            if miss.numel() - t:
                worst.append((r, miss.numel() - t, float((x - s_min).max())))
    print(
        f"{p} [{z['backend']}]: recall {hit / tot:.4f}, misses {tot - hit}: "
        f"ties at the module's 1000th score {tie}, strictly better {real}; "
        f"worst rows (row, real misses, max gap) {sorted(worst, key=lambda t: -t[1])[:5]}"
    )

# ---- pass 2: fp64 reference over the true capped pass set --------------------------------
filter_spill = max(spills.values(), key=lambda z: len(np.setdiff1d(z["ids"][0], topk[0].numpy())) * -1)
mids = torch.from_numpy(filter_spill["ids"]).long()
qa = torch.tensor(pl.read_parquet(f"{DATA}/eval_split.parquet")["query_attrs_narrow"].to_list()[:n_q])
narrow = torch.load(f"{DATA}/item_attrs_narrow.pt", map_location="cpu", weights_only=True)
rng = np.random.default_rng(0)
rows = sorted(set([422, 963, 221, 268, 91] + rng.choice(n_q, 60, replace=False).tolist()))
r_o = r_m = r_32 = 0.0
n = 0
span = []
for r in rows:
    m = (narrow[:, 0, :] == qa[r, 0]).any(-1)
    if qa[r, 1] >= 0:
        m &= (narrow[:, 1, :] == qa[r, 1]).any(-1)
    cand = m.nonzero().reshape(-1)
    if cand.numel() < 1000:
        continue
    x = emb32[cand]
    s64, s32 = x.double() @ q[r].double(), x @ q[r]
    k64, k32 = cand[torch.topk(s64, 1000).indices], cand[torch.topk(s32, 1000).indices]
    r_o += recall(k64, topk[r])
    r_m += recall(k64, mids[r])
    r_32 += recall(k64, k32)
    n += 1
    ss = torch.sort(s64, descending=True).values
    span.append(float(ss[0] - ss[999]))
print(
    f"rows {n}: recall vs fp64-exact — oracle blob {r_o / n:.4f}, module {r_m / n:.4f}, "
    f"plain fp32 topk {r_32 / n:.4f}; score span rank1→rank1000 median {np.median(span):.5f} "
    f"min {min(span):.5f} (fp16 spacing in [0.5, 1) is 2^-11 = {2**-11:.5f})"
)
u = torch.unique(emb32[:2_000_000], dim=0).shape[0]
print(f"distinct vectors in the first 2M rows: {u} ({100 * (1 - u / 2e6):.2f}% exact duplicates)")
