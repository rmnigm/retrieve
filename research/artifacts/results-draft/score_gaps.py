"""Why YFCC loses recall to fp16 / int8: the score gap at rank k against the rounding error of the stored items.

Per dataset, 500 oracle rows (unfiltered-equivalent sweep where one exists): fp32 scores of the oracle's top-101 over the
harness's normalised items and queries; gap = s100 - s101, spread = s1 - s100; fp16 / int8 error = |score change| when
the normalised item is stored fp16 / symmetric per-row int8. CPU only. usage: score_gaps.py OUT.json
"""

import glob
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SETS = {  # name: (data_dir, content_dir, oracle sweep, pad row possible)
    "yfcc10m-synth d192": ("/data/yfcc10m", "content_d192", "gt_d192", "p1"),
    "pubmed d768": ("/data/pubmed-medcpt", "content_d768", "gt_d768", "c0_mesh"),
    "laion10m-synth d256": ("/data/laion10m", "content_d256", "gt_d256", "p1"),
    "arxiv-synth d128": ("/data/arxiv-papers", "content_d128", "gt_d128", "p1"),
}
N_Q = 500


def items(content):
    si = content / "shard_index.json"
    if si.exists():
        idx = json.load(open(si))
        parts = [
            torch.load(str(content / s["filename"]), map_location="cpu", mmap=True)
            for s in idx["shards"]
        ]
        return torch.cat(parts)
    t = torch.load(str(content / "text_emb.pt"), map_location="cpu", mmap=True)
    return t[1:] if bool((t[0] == 0).all()) else t


def oracle(gt, sweep):
    best = None
    for f in glob.glob(f"{gt}/oracle_v4_{sweep}_*.pt"):
        o = torch.load(f, map_location="cpu", weights_only=False)
        if best is None or o["k_gt"] > best["k_gt"]:
            best = o
    return best


def med(x):
    return float(x.median())


out = {}
only = sys.argv[2:]
for name, (data, content, gt, sweep) in SETS.items():
    if only and not any(o_ in name for o_ in only):
        continue
    data = Path(data)
    E = items(data / content)
    o = oracle(data / gt, sweep)
    q = F.normalize(
        torch.load(str(data / content / "query_emb.pt"), map_location="cpu").float()[
            : o["topk"].shape[0]
        ],
        dim=-1,
    )
    K = min(100, o["k_gt"] - 1)
    rows = torch.nonzero(o["topk"][:, K] >= 0).flatten()
    rows = rows[
        torch.randperm(len(rows), generator=torch.Generator().manual_seed(0))[:N_Q]
    ]
    ids = o["topk"][rows, : K + 1]
    x = F.normalize(E[ids.flatten()].float(), dim=-1).view(len(rows), K + 1, -1)
    qq = q[rows].unsqueeze(-1)
    s = (x @ qq).squeeze(-1)
    s16 = (x.half().float() @ qq).squeeze(-1)
    scale = x.abs().amax(-1, keepdim=True) / 127
    s8 = ((torch.round(x / scale) * scale) @ qq).squeeze(-1)
    gap = s[:, K - 1] - s[:, K]
    e16 = (s16 - s).abs()
    e8 = (s8 - s).abs()
    ok_sorted = float(((s[:, :-1] - s[:, 1:]) >= -1e-6).all(1).float().mean())
    rnd = F.normalize(
        E[
            torch.randint(
                0, E.shape[0], (20000,), generator=torch.Generator().manual_seed(1)
            )
        ].float(),
        dim=-1,
    )
    out[name] = dict(
        oracle=f"{sweep} k_gt {o['k_gt']}",
        sorted_check=ok_sorted,
        s1=med(s[:, 0]),
        s100=med(s[:, K - 1]),
        spread_s1_s100=med(s[:, 0] - s[:, K - 1]),
        gap_at_k=med(gap),
        k=K,
        err_fp16=med(e16),
        err_int8_row=med(e8),
        frac_gap_below_fp16_err=float(
            (gap < 2 * e16[:, K - 1 : K + 1].amax(1)).float().mean()
        ),
        frac_gap_below_int8_err=float(
            (gap < 2 * e8[:, K - 1 : K + 1].amax(1)).float().mean()
        ),
        cone_norm_of_mean_item=float(rnd.mean(0).norm()),
        random_pair_cos=float((rnd[:10000] * rnd[10000:]).sum(-1).mean()),
    )
    print(name, json.dumps(out[name]), flush=True)
    del E
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
