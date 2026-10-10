"""PubMed diagnostic (a), final classes: exact_miss.py scores V2's misses with V1's cuBLAS scorer. This rescores each (missed item m,
intruder a) pair of V2 with V2's own Triton kernel (`fused_masked_knn_topk` on the two-candidate list [m, a]), then classifies every
miss of both arms with its own arm's scores; `oracle_tie` also takes pairs within two fp32 ulps (the oracle's chunked fp32 matmul and
the single-vector rescoring here round differently). Writes OUT.json (counts per sweep / arm) and OUT.csv (`arm_m`, `arm_a`, `class2`).

    python v2_rescore.py MISS_CSV CONFIG_DIR OUT_STEM   (from evaluation/, PYTHONPATH to the tree under test)
"""

import collections
import csv
import json
import sys
from pathlib import Path

import torch
from bench import inputs, measure
from bench.config import load_matrix
from retrieve.interfaces import ops_for

miss_csv, cfg, stem = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
K = 100
device = torch.device("cuda")
measure.setup(0)
job = load_matrix(cfg / "pubmed.yaml", cfg / "suites.yaml", "filter", algos=["linr_v2"], backends=["triton"],
                  filter_kinds=["clause"], sweeps=["c0_mesh"], seeds=[0])[0]  # fmt: skip
inp = inputs.load_inputs(job.data, device, with_filters=False)
items16 = inp["item_embs"].to(torch.float16)
queries16 = inp["queries"].to(device).to(torch.float16)
del inp
allrows = list(csv.DictReader(open(miss_csv)))
rows = [r for r in allrows if r["algo"] == "linr_v2"]
op = ops_for("triton").fused_masked_knn_topk
for s in range(0, len(rows), 256):
    chunk = rows[s : s + 256]
    q = queries16[torch.tensor([int(r["user_row"]) for r in chunk], device=device)]
    cand = torch.full((len(chunk), K), -1, dtype=torch.long, device=device)
    cand[:, 0] = torch.tensor([int(r["missed_item"]) for r in chunk], device=device)
    cand[:, 1] = torch.tensor([int(r["arm_kth_item"]) for r in chunk], device=device)
    ids, scores = op(q, items16, cand, torch.full((len(chunk),), 2, dtype=torch.long, device=device), K)
    for i, r in enumerate(chunk):
        got = dict(zip(ids[i, :2].tolist(), scores[i, :2].tolist()))
        r["arm_m"], r["arm_a"] = got[int(r["missed_item"])], got[int(r["arm_kth_item"])]
for r in allrows:
    if r["algo"] != "linr_v2":
        r["arm_m"], r["arm_a"] = float(r["arm_score_m"]), float(r["arm_score_a"])
counts = collections.defaultdict(collections.Counter)
for r in allrows:
    m, a, f_m, f_a = r["arm_m"], r["arm_a"], float(r["fp32_m"]), float(r["fp32_a"])
    if m == a:
        c = "arm_tie"
    elif m < a and f_m > f_a:
        c = "fp16_order"
    elif f_m == f_a or (m < a and abs(f_m - f_a) <= 2.0**-22 * abs(f_a)):
        c = "oracle_tie"  # equal, or within two fp32 ulps: the oracle's chunked matmul and this rescoring round differently
    else:
        c = "other"
    r["class2"] = c
    counts[f"{r['sweep']}/{r['algo']}"][c] += 1
Path(f"{stem}.json").write_text(json.dumps({k: dict(v) for k, v in counts.items()}, indent=1))
with open(f"{stem}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
    w.writeheader()
    w.writerows(allrows)
print(json.dumps({k: dict(v) for k, v in counts.items()}))
