"""L4 §2 third pass: the rows where the fp32-accumulating probe kernel and `torch` still disagree.
If every swapped pair there sits within one fp16 ulp of the boundary score, the residual is
`torch`'s fp16 *output* rounding (cuBLAS bmm accumulates fp32, then rounds the score to fp16),
measured rather than assumed.

    cd /workspace/wt/l4/retrieve && flock /workspace/gpu.lock uv run --no-sync python \
        ../docs/plans/linr-v2-backend-parity-artifacts/torch_side_rounding.py \
        --out ../docs/plans/linr-v2-backend-parity-artifacts/torch_side_rounding.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from boundary_and_fp32_variant import fp32_topk  # noqa: E402
from probe_v2_parity import CHUNK, K_MAX, build, fp16_ulp, jaccard, load  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    item_embs, queries, attrs, reverse, qa, keep = load(device)
    rows = keep.nonzero().reshape(-1)
    mt = build("torch", item_embs, attrs, reverse)
    mr = build("triton", item_embs, attrs, reverse)
    item16 = mr.idx.item_embs
    ids_t, ids_f = [], []
    for s in range(0, rows.numel(), CHUNK):
        sel = rows[s : s + CHUNK]
        q, qs = queries[sel].to(device), qa[sel].to(device)
        ids_t.append(mt(q, qs)[0])
        cand, counts = mr.filter.evaluate_indices(qs)
        ids_f.append(fp32_topk(q.to(torch.float16), item16, cand, counts, K_MAX)[0])
    ids_t, ids_f = torch.cat(ids_t), torch.cat(ids_f)
    bad = (jaccard(ids_t, ids_f, 100) < 1).nonzero().reshape(-1).tolist()
    pairs = []
    for r in bad:
        sel_r = rows[r : r + 1]
        q16 = queries[sel_r].to(device).to(torch.float16)
        cand, counts = mr.filter.evaluate_indices(qa[sel_r].to(device))
        c = cand[0, : int(counts[0])]
        e = item16[c]
        truth = e.double() @ q16[0].double()
        tor = torch.bmm(q16.unsqueeze(1), e.unsqueeze(0).transpose(1, 2)).squeeze()  # fp16, the torch path's scores
        s, _ = truth.sort(descending=True)
        ulp = fp16_ulp(s[99]).item()
        a = set(ids_t[r, :100].tolist()) - {-1}
        b = set(ids_f[r, :100].tolist()) - {-1}
        for g_t, g_f in zip(sorted(a - b), sorted(b - a)):
            i_t, i_f = int((c == g_t).nonzero()[0]), int((c == g_f).nonzero()[0])
            pairs.append({
                "row": int(sel_r[0]),
                "truth_gap_pair": truth[i_t].item() - truth[i_f].item(),
                "gap_in_fp16_ulps_at_boundary": abs(truth[i_t].item() - truth[i_f].item()) / ulp,
                "torch_fp16_scores_tie": bool(tor[i_t] == tor[i_f]),
                "torch_orders_pair_as_returned": bool(tor[i_t] >= tor[i_f]),
                "dist_torch_item_from_boundary_ulps": abs(truth[i_t].item() - s[99].item()) / ulp,
                "dist_fp32_item_from_boundary_ulps": abs(truth[i_f].item() - s[99].item()) / ulp,
            })
    g = torch.tensor([p["gap_in_fp16_ulps_at_boundary"] for p in pairs])
    out = {
        "rows_top100_differ_fp32probe_vs_torch": len(bad),
        "swapped_pairs": len(pairs),
        "pair_gap_in_fp16_ulps_at_boundary": {"max": g.max().item(), "median": g.median().item()},
        "pairs_within_one_ulp": int((g <= 1).sum()),
        "torch_fp16_scores_tie": sum(p["torch_fp16_scores_tie"] for p in pairs),
        "torch_orders_pair_as_returned": sum(p["torch_orders_pair_as_returned"] for p in pairs),
        "max_dist_from_boundary_ulps": max(max(p["dist_torch_item_from_boundary_ulps"], p["dist_fp32_item_from_boundary_ulps"]) for p in pairs),
        "first_pairs": pairs[:10],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "first_pairs"}, indent=1))
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
