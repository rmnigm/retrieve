"""R-RES, GPU: does the arXiv `c0_maincat` oracle depend on how it is built? Rebuilds it
the old harness's way — one ``q @ E^T`` over all N items per 64-query batch, one
``torch.topk`` — via ``oracle.compute(item_chunk=N)``, compares it with the cached item-chunked
v4 blob, and rescores the freeze's SilverTorch spill (n_probe 24, seed 0) against both.

    cd evaluation && python ../docs/artifacts/campaign-v2/r-res/oracle_oneshot.py FREEZE_DIR
"""

import sys
from pathlib import Path

import numpy as np
import torch

from bench import config, inputs, measure, oracle

GOLDEN_R100 = 0.8840420681593009
BLOB = "data/arxiv-papers/gt_d128/oracle_v4_c0_maincat_4e83f24768de46f1.pt"
SPILL = "_parity/arxiv-d128_silvertorch/a2c1e9801c948aa88ee5.npz"


def recall(ids: np.ndarray, gt: np.ndarray, k: int) -> float:
    t = gt[:, :k]
    hits = [
        np.isin(a[:k], b[b >= 0]).sum() / max(min((b >= 0).sum(), k), 1)
        for a, b in zip(ids, t)
    ]
    return float(np.mean(hits))


measure.setup(0)
dev = torch.device("cuda")
ds = config.load_dataset(Path("config/arxiv.yaml"), 128)
inp = inputs.load_inputs(ds, dev)
qa_s, skip = inputs.sweep_qa(inp["qa"], (0,))
filters = inputs.build_filters(
    "clause", inp, ["triton"], bloom={"m_bits": 0, "k_hash": 0}
)  # unused on clause
fm = inputs.exact_filter("clause", filters, inp, "triton")
n = inp["n_items"]
one = oracle.compute(
    inp["item_embs"], inp["queries"], qa_s, skip, fm, 1000, item_chunk=n, device=dev
)["topk"]
chunked = torch.load(BLOB, weights_only=False)["topk"]
diff = (one != chunked).any(dim=1)
print(f"one-shot vs chunked blob: rows differing {int(diff.sum())} / {one.shape[0]}")
for k in (100, 500, 1000):
    d = (one[:, :k].sort(dim=1).values != chunked[:, :k].sort(dim=1).values).any(dim=1)
    print(
        f"  top-{k} as sets differ on {int(d.sum())} rows: {d.nonzero().reshape(-1).tolist()[:20]}"
    )
ids = np.load(Path(sys.argv[1]) / SPILL)["ids"]
for name, gt in (("chunked", chunked), ("one-shot", one)):
    r = recall(ids, gt.numpy(), 100)
    print(
        f"silvertorch recall@100 vs {name} oracle {r:.10f} (golden {GOLDEN_R100:.10f}, diff {r - GOLDEN_R100:+.1e})"
    )
