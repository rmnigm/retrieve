"""L4 §2 — do the `torch` and `triton` LiNR V2 backends score the same candidate set, and if so
where do their top-k lists diverge? Reproduces the C4 goodreads-d128 `c0_genre` cell's inputs
(SASRec cache, first 10 000 users, clause 0 only, chunks of 16, k_max = 1000) outside the harness.

    cd /workspace/wt/l4/retrieve && flock /workspace/gpu.lock uv run --no-sync python \
        ../docs/plans/linr-v2-backend-parity-artifacts/probe_v2_parity.py \
        --out ../docs/plans/linr-v2-backend-parity-artifacts/probe_v2_parity.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time

import polars as pl
import torch
from retrieve.modules import ExactAttributeFilter, LiNRV2
from retrieve.ops.triton.fused_masked_knn_topk import (
    DEFAULT_CONFIG,
    _fmkt_prep,
    _fused_masked_knn_topk_kernel,
)

DATA = "/workspace/data/goodreads-work-id"
USERS = 10_000
CHUNK = 16
K_MAX = 1000
KS = (100, 500, 1000)


def canon(ids: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    valid = torch.arange(ids.shape[1], device=ids.device)[None, :] < counts[:, None]
    return torch.where(valid, ids, torch.full_like(ids, -1))


def jaccard(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    a, b = a[:, :k], b[:, :k]
    hits = ((a[:, :, None] == b[:, None, :]) & (a[:, :, None] != -1)).any(dim=2).sum(dim=1)
    union = (a != -1).sum(dim=1) + (b != -1).sum(dim=1) - hits
    return torch.where(union > 0, hits.double() / union.clamp(min=1), torch.ones_like(hits).double())


def fp16_ulp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(x.abs())) - 10)


def load(device):
    blob = torch.load(
        f"{DATA}/checkpoints/gsasrec-d128-drop0.5-id/encoded_queries_v2.pt",
        map_location="cpu",
        weights_only=True,
    )
    item_embs = blob["item_embs"].to(device).contiguous()
    queries = blob["queries"][:USERS].contiguous()
    attrs = torch.load(f"{DATA}/item_attrs_narrow.pt", map_location=device, weights_only=True)
    assert attrs.shape[0] == item_embs.shape[0] + 1 and bool((attrs[0] == -1).all())
    attrs = attrs[1:].contiguous()
    reverse = torch.load(f"{DATA}/clause_is_reverse_narrow.pt", map_location=device, weights_only=True)
    qa = torch.tensor(pl.read_parquet(f"{DATA}/eval_split.parquet")["query_attrs_narrow"].to_list())
    qa = qa[:USERS].clone()
    qa[:, 1:] = -1
    keep = (qa != -1).any(dim=1)
    return item_embs, queries, attrs, reverse, qa, keep


def build(backend, item_embs, attrs, reverse):
    f = ExactAttributeFilter(backend=backend)
    f.register_index(attrs, clause_is_reverse=reverse)
    m = LiNRV2(k=K_MAX, filter=f, backend=backend)
    m.register_index(item_embs)
    return m


def kernel_scores(query16, item16, cand, counts):
    launch = _fmkt_prep(query16, item16, cand, counts, DEFAULT_CONFIG, bucket=False)
    _fused_masked_knn_topk_kernel[launch.grid](**launch.kwargs)
    return launch.all_scores


def boundary(q16: torch.Tensor, e16: torch.Tensor, ids_a: torch.Tensor, ids_b: torch.Tensor, k: int):
    """One row: exact fp64 dots of the fp16 inputs over the candidate ids, the rank-k boundary
    gap, how many candidates sit within one fp16 ulp of the boundary, and how far (in ulps of
    the boundary score) every item that only one backend returned sits from that boundary."""
    truth = (e16.double() @ q16.double())  # [P]
    s, order = truth.sort(descending=True)
    ulp = fp16_ulp(s[k - 1]).item()
    gap = (s[k - 1] - s[k]).item()
    within = int(((s - s[k - 1]).abs() <= ulp).sum().item())
    return s, order, ulp, gap, within


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ptx-out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda")
    t0 = time.time()
    item_embs, queries, attrs, reverse, qa, keep = load(device)
    n = item_embs.shape[0]
    rows = keep.nonzero().reshape(-1)
    print(f"items {n} users {USERS} kept {rows.numel()}  load {time.time() - t0:.1f}s", flush=True)

    mt = build("torch", item_embs, attrs, reverse)
    mr = build("triton", item_embs, attrs, reverse)
    item16 = mt.idx.item_embs
    assert torch.equal(item16, mr.idx.item_embs)

    cand_rows_differ = 0
    counts_rows_differ = 0
    first_cand_diff = None
    ids_t, sc_t, ids_r, sc_r = [], [], [], []
    cand_info = []  # (row, count)
    nondet = {"triton": 0, "torch": 0}

    for s in range(0, rows.numel(), CHUNK):
        sel = rows[s : s + CHUNK]
        q = queries[sel].to(device)
        qs = qa[sel].to(device)
        ct, cnt_t = mt.filter.evaluate_indices(qs)
        cr, cnt_r = mr.filter.evaluate_indices(qs)
        if not torch.equal(cnt_t, cnt_r):
            counts_rows_differ += int((cnt_t != cnt_r).sum())
        eq = (canon(ct, cnt_t) == canon(cr, cnt_r)).all(dim=1)
        if not bool(eq.all()):
            cand_rows_differ += int((~eq).sum())
            if first_cand_diff is None:
                i = int((~eq).nonzero()[0])
                first_cand_diff = {
                    "row": int(sel[i]),
                    "count_torch": int(cnt_t[i]),
                    "count_triton": int(cnt_r[i]),
                    "first_positions": (canon(ct, cnt_t)[i] != canon(cr, cnt_r)[i]).nonzero()[:8].reshape(-1).tolist(),
                }
        it, st = mt(q, qs)
        ir, sr = mr(q, qs)
        if s == 0:
            it2, st2 = mt(q, qs)
            ir2, sr2 = mr(q, qs)
            nondet["torch"] = int(not (torch.equal(it, it2) and torch.equal(st, st2)))
            nondet["triton"] = int(not (torch.equal(ir, ir2) and torch.equal(sr, sr2)))
        ids_t.append(it)
        sc_t.append(st.float())
        ids_r.append(ir)
        sc_r.append(sr)
        cand_info.extend((int(r), int(c)) for r, c in zip(sel.tolist(), cnt_t.tolist()))
        if s % (CHUNK * 100) == 0:
            print(f"  chunk {s // CHUNK}  cand rows differ so far {cand_rows_differ}", flush=True)

    ids_t, sc_t, ids_r, sc_r = map(torch.cat, (ids_t, sc_t, ids_r, sc_r))
    out = {
        "n_items": n,
        "n_kept_rows": int(rows.numel()),
        "candidate_set": {
            "rows_with_different_counts": counts_rows_differ,
            "rows_with_different_ids": cand_rows_differ,
            "first_difference": first_cand_diff,
        },
        "nondeterministic_repeat_on_chunk0": nondet,
        "k_max": K_MAX,
    }
    both = torch.isfinite(sc_t) & torch.isfinite(sc_r)
    diff = (sc_t - sc_r).abs()[both]
    out["harness_metrics"] = {
        **{f"jaccard_vs_first@{k}": jaccard(ids_t, ids_r, k).mean().item() for k in KS},
        "score_max_abs_diff": diff.max().item(),
        "score_mean_abs_diff": diff.mean().item(),
    }
    j100 = jaccard(ids_t, ids_r, 100)
    bad = (j100 < 1).nonzero().reshape(-1)
    out["rows_top100_differ"] = int(bad.numel())
    print(json.dumps(out, indent=1), flush=True)

    # --- what each backend's arithmetic is, on chunk 0 -----------------------------------------
    sel = rows[:CHUNK]
    q = queries[sel].to(device)
    qs = qa[sel].to(device)
    q16 = q.to(torch.float16)
    cand, counts = mt.filter.evaluate_indices(qs)
    p = int(counts.max())
    cand = cand[:, :p].contiguous()
    valid = torch.arange(p, device=device)[None, :] < counts[:, None]
    tri = kernel_scores(q16, item16, cand, counts)[:, :p]
    e = item16[cand.clamp_min(0)]  # [B, P, D] fp16
    tor = torch.bmm(q16.unsqueeze(1), e.transpose(1, 2)).squeeze(1)  # fp16 out, cuBLAS
    d64 = torch.einsum("bpd,bd->bp", e.double(), q16.double())
    d32_seq = torch.zeros_like(d64, dtype=torch.float32)
    d16_seq = torch.zeros(d64.shape, dtype=torch.float16, device=device)
    for j in range(e.shape[2]):
        d32_seq += e[:, :, j].float() * q16[:, None, j].float()
        d16_seq = d16_seq + e[:, :, j] * q16[:, None, j]
    m = valid
    ulps = fp16_ulp(d64).float()

    def err(x):
        dd = (x.double() - d64).abs()[m]
        return {"max_abs": dd.max().item(), "max_in_fp16_ulps": (dd / ulps.double()[m]).max().item(), "mean_abs": dd.mean().item()}

    out["arithmetic_chunk0"] = {
        "P": p,
        "score_magnitude": {"max": d64[m].abs().max().item(), "median": d64[m].abs().median().item()},
        "triton_vs_fp64": err(tri),
        "torch_bmm_fp16out_vs_fp64": err(tor),
        "fp16_round_of_fp64_vs_fp64": err(d64.half()),
        "fp32_sequential_vs_fp64": err(d32_seq),
        "fp16_sequential_vs_fp64": err(d16_seq),
        "triton_scores_all_fp16_representable": bool(torch.equal(tri[m].half().float(), tri[m])),
        "torch_equals_fp16_round_of_fp64": bool(torch.equal(tor[m], d64.half()[m])),
        "torch_equals_fp16_round_of_fp32seq": bool(torch.equal(tor[m], d32_seq.half()[m])),
        "torch_vs_fp16_round_of_fp64_max_ulps": ((tor.double() - d64.half().double()).abs()[m] / ulps.double()[m]).max().item(),
        "triton_vs_fp16_sequential_max_ulps": ((tri.double() - d16_seq.double()).abs()[m] / ulps.double()[m]).max().item(),
        "triton_output_dtype": str(tri.dtype),
        "torch_output_dtype": str(tor.dtype),
    }

    ptx = None
    for cache in _fused_masked_knn_topk_kernel.device_caches.values():
        for kern in cache[0].values():
            ptx = kern.asm["ptx"]
    if ptx is not None:
        ops = {}
        for pat in ("add.rn.f16", "add.f16", "fma.rn.f16", "mul.rn.f16", "mul.f16", "add.rn.f32", "add.f32", "fma.rn.f32", "mul.rn.f32", "mul.f32", "cvt.f32.f16", "cvt.rn.f16.f32"):
            ops[pat] = len(re.findall(r"\b" + re.escape(pat) + r"\b", ptx))
        out["ptx_op_counts"] = ops
        if args.ptx_out:
            with open(args.ptx_out, "w") as fh:
                fh.write(ptx)

    # --- boundary structure on every row whose top-100 differs --------------------------------
    detail = []
    max_ulps_from_boundary = 0.0
    beyond_one_ulp = 0
    items_differ_total = 0
    for r in bad.tolist()[:2000]:
        sel_r = rows[r : r + 1]
        q16r = queries[sel_r].to(device).to(torch.float16)[0]
        cand_r, cnt_r = mt.filter.evaluate_indices(qa[sel_r].to(device))
        c = cand_r[0, : int(cnt_r[0])]
        e16r = item16[c]
        s, order, ulp, gap, within = boundary(q16r, e16r, ids_t[r], ids_r[r], 100)
        gid = c[order]  # global ids in truth order
        a = set(ids_t[r, :100].tolist()) - {-1}
        b = set(ids_r[r, :100].tolist()) - {-1}
        sym = a ^ b
        items_differ_total += len(sym)
        pos = {int(g): i for i, g in enumerate(gid[: max(400, 100 + 300)].tolist())}
        dist = []
        for g in sym:
            i = pos.get(g)
            if i is None:
                i = int((gid == g).nonzero()[0])
            dist.append(((s[i] - s[99]).abs().item() / ulp, i))
        worst = max(d for d, _ in dist)
        max_ulps_from_boundary = max(max_ulps_from_boundary, worst)
        beyond_one_ulp += sum(d > 1.0 for d, _ in dist)
        if len(detail) < 40:
            detail.append({
                "row": int(sel_r[0]),
                "count": int(cnt_r[0]),
                "score_rank100": s[99].item(),
                "fp16_ulp_at_rank100": ulp,
                "gap_rank100_to_101": gap,
                "gap_in_ulps": gap / ulp,
                "candidates_within_one_ulp_of_rank100": within,
                "n_items_only_in_torch": len(a - b),
                "n_items_only_in_triton": len(b - a),
                "differing_items_[dist_from_boundary_in_ulps, truth_rank]": sorted(dist),
            })
    out["boundary"] = {
        "rows_examined": min(int(bad.numel()), 2000),
        "differing_items_total": items_differ_total,
        "differing_items_beyond_one_fp16_ulp_of_boundary": beyond_one_ulp,
        "max_distance_from_boundary_in_fp16_ulps": max_ulps_from_boundary,
        "detail_first_rows": detail,
    }
    # gap statistics over all rows, not just the differing ones: a sample of 500 rows
    gaps, withins = [], []
    for r in range(0, rows.numel(), rows.numel() // 500):
        sel_r = rows[r : r + 1]
        q16r = queries[sel_r].to(device).to(torch.float16)[0]
        cand_r, cnt_r = mt.filter.evaluate_indices(qa[sel_r].to(device))
        c = cand_r[0, : int(cnt_r[0])]
        s, order, ulp, gap, within = boundary(q16r, item16[c], None, None, 100)
        gaps.append(gap / ulp)
        withins.append(within)
    gaps_t = torch.tensor(gaps)
    out["boundary_sample_500_rows"] = {
        "gap_rank100_in_fp16_ulps": {"median": gaps_t.median().item(), "p10": gaps_t.quantile(0.1).item(), "min": gaps_t.min().item(), "frac_below_1ulp": (gaps_t < 1).float().mean().item()},
        "candidates_within_one_ulp_of_rank100": {"median": float(torch.tensor(withins).float().median()), "max": max(withins)},
    }
    out["elapsed_s"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "boundary"}, indent=1))
    print(json.dumps({k: v for k, v in out["boundary"].items() if k != "detail_first_rows"}, indent=1))
    print(json.dumps(out["boundary"]["detail_first_rows"][:3], indent=1))
    print(f"done {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
