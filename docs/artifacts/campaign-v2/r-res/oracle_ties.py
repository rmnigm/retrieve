"""R-RES, CPU: arXiv `c0_maincat` ground-truth ties at the oracle's rank-k boundary.

    python oracle_ties.py FREEZE_DIR

Rescores the cached v4 oracle's top-1000 (k_gt 1000 blob, the one the freeze's per-query
recall matches) in fp64 from the staged fp16 tables, and per k counts the rows whose oracle
rank k and k+1 items tie exactly (fp64, so on the inputs themselves) or nearly (|gap| < 1e-6,
inside fp32 reduction-order noise at |s| ~ 1). On such a row the oracle's member choice is
arbitrary; it changes SilverTorch's hits only where the frozen run's top-k holds some but not
all of the tied class. Those rows, with the hits a different oracle choice could move, are
listed."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from eval_datasets import layout

DATA = Path("/data/arxiv-papers")
SPILL = "_parity/arxiv-d128_silvertorch/a2c1e9801c948aa88ee5.npz"  # n_probe 24, seed 0
ORACLE = DATA / "gt_d128/oracle_v4_c0_maincat_4e83f24768de46f1.pt"
NEAR = 1e-6

root = Path(sys.argv[1])
items = layout.load_text_items(DATA / "content_d128", torch.device("cpu")).double()
queries, _, _ = layout.load_text_queries(DATA, DATA / "content_d128", items.shape[1])
gt = torch.load(ORACLE, weights_only=False)["topk"]
st = np.load(root / SPILL)["ids"]
q = queries[: gt.shape[0]].double()
valid = gt >= 0
s = torch.cat(
    [
        torch.einsum("ud,ukd->uk", q[i : i + 256], items[gt[i : i + 256].clamp_min(0)])
        for i in range(0, gt.shape[0], 256)
    ]
)
s[~valid] = float("-inf")
for k in (100, 500):
    a, b = s[:, k - 1], s[:, k]
    exact = (a == b) & valid[:, k]
    near = ((a - b).abs() < NEAR) & valid[:, k]
    print(
        f"k={k}: rows with an exact tie at the oracle boundary {int(exact.sum())}, near (<{NEAR}) {int(near.sum())}"
    )
    for i in near.nonzero().reshape(-1).tolist():
        cls = ((s[i] - a[i]).abs() < NEAR).nonzero().reshape(-1)
        ids = gt[i, cls].numpy()
        in_st = np.isin(ids, st[i, :k])
        n_gt_in = int((cls < k).sum())
        if 0 < in_st.sum() < len(ids):
            print(
                json.dumps(
                    {
                        "row": i,
                        "oracle_ranks": (cls + 1).tolist(),
                        "ids": ids.tolist(),
                        "gap": float(a[i] - b[i]),
                        "in_silvertorch_top_k": in_st.tolist(),
                        "oracle_slots_inside_k": n_gt_in,
                    }
                )
            )
