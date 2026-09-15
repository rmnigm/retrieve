"""L4 §2 second pass, after `probe_v2_parity.py` showed identical candidate sets and an fp16-
accumulating Triton kernel.

A. Every row whose top-100 differs between the backends: the exact (fp64) rank-100 boundary
   gap in absolute score units, the truth rank and boundary distance of each differing item,
   and whether the Triton kernel's *own* scores order the swapped pair the way it returned it.
B. A probe-only copy of the kernel that accumulates in fp32 (the shipped kernel is untouched):
   parity against `torch` and against the shipped kernel over the same 9 859 rows.
C. `do_bench` timings of the shipped and the fp32-accumulating launch on the cell's real shapes.

    cd /workspace/wt/l4/retrieve && flock /workspace/gpu.lock uv run --no-sync python \
        ../docs/plans/linr-v2-backend-parity-artifacts/boundary_and_fp32_variant.py \
        --out ../docs/plans/linr-v2-backend-parity-artifacts/boundary_and_fp32_variant.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).parent))
from probe_v2_parity import CHUNK, K_MAX, KS, build, jaccard, load  # noqa: E402

from retrieve.ops.triton.fused_masked_knn_topk import (  # noqa: E402
    DEFAULT_CONFIG,
    _fmkt_finish,
    _fmkt_prep,
    _fused_masked_knn_topk_kernel,
)
from retrieve.ops.tune import _bench  # noqa: E402


@triton.jit
def _kernel_fp32_acc(
    query_ptr, item_embs_ptr, pos_indices_ptr, counts_ptr, out_scores_ptr,
    P: tl.constexpr, D: tl.constexpr,
    stride_qb, stride_qd, stride_in, stride_id, stride_pb, stride_pp, stride_sb, stride_sp,
    BLOCK_N: tl.constexpr,
):  # fmt: skip
    tile_id = tl.program_id(0)
    bid = tl.program_id(1)
    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, D)
    p_valid = n_offsets < P
    count = tl.load(counts_ptr + bid)
    in_count = n_offsets < count
    q = tl.load(query_ptr + bid * stride_qb + d_offsets * stride_qd).to(tl.float32)
    item_ids = tl.load(pos_indices_ptr + bid * stride_pb + n_offsets * stride_pp, mask=in_count, other=0)
    emb_rows = tl.load(
        item_embs_ptr + item_ids[:, None] * stride_in + d_offsets[None, :] * stride_id,
        mask=in_count[:, None],
        other=0.0,
    ).to(tl.float32)
    dots = tl.sum(emb_rows * q[None, :], axis=1)
    dots = tl.where(in_count, dots, float("-inf"))
    tl.store(out_scores_ptr + bid * stride_sb + n_offsets * stride_sp, dots, mask=p_valid)


def fp32_topk(query16, item16, cand, counts, k):
    launch = _fmkt_prep(query16, item16, cand, counts, DEFAULT_CONFIG, bucket=False)
    _kernel_fp32_acc[launch.grid](**launch.kwargs)
    return _fmkt_finish(launch, k, pad_to_k=False)


def sm_mhz() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout  # fmt: skip
    return int(out.strip().splitlines()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    t0 = time.time()
    item_embs, queries, attrs, reverse, qa, keep = load(device)
    rows = keep.nonzero().reshape(-1)
    mt = build("torch", item_embs, attrs, reverse)
    mr = build("triton", item_embs, attrs, reverse)
    item16 = mr.idx.item_embs

    ids_t, sc_t, ids_r, sc_r, ids_f, sc_f = [], [], [], [], [], []
    for s in range(0, rows.numel(), CHUNK):
        sel = rows[s : s + CHUNK]
        q = queries[sel].to(device)
        qs = qa[sel].to(device)
        it, st = mt(q, qs)
        ir, sr = mr(q, qs)
        cand, counts = mr.filter.evaluate_indices(qs)
        i32, s32 = fp32_topk(q.to(torch.float16), item16, cand, counts, K_MAX)
        ids_t.append(it); sc_t.append(st.float()); ids_r.append(ir); sc_r.append(sr)  # noqa: E702
        ids_f.append(i32); sc_f.append(s32)  # noqa: E702
    ids_t, sc_t, ids_r, sc_r, ids_f, sc_f = map(torch.cat, (ids_t, sc_t, ids_r, sc_r, ids_f, sc_f))

    def parity(ia, sa, ib, sb):
        both = torch.isfinite(sa) & torch.isfinite(sb)
        d = (sa - sb).abs()[both]
        return {
            **{f"jaccard@{k}": jaccard(ia, ib, k).mean().item() for k in KS},
            "rows_top100_differ": int((jaccard(ia, ib, 100) < 1).sum()),
            "score_max_abs_diff": d.max().item(),
            "ids_torch_equal": bool(torch.equal(ia, ib)),
        }

    out = {
        "n_kept_rows": int(rows.numel()),
        "parity": {
            "shipped_triton_vs_torch": parity(ids_t, sc_t, ids_r, sc_r),
            "fp32acc_probe_vs_torch": parity(ids_t, sc_t, ids_f, sc_f),
            "fp32acc_probe_vs_shipped_triton": parity(ids_r, sc_r, ids_f, sc_f),
        },
    }
    print(json.dumps(out, indent=1), flush=True)

    # --- A. boundary in absolute units over every differing row --------------------------------
    bad = (jaccard(ids_t, ids_r, 100) < 1).nonzero().reshape(-1).tolist()
    swaps = []
    gaps_all = []
    consistent = 0
    torch_item_is_true_top100 = 0
    triton_item_is_true_top100 = 0
    for r in range(rows.numel()):
        sel_r = rows[r : r + 1]
        qs = qa[sel_r].to(device)
        q16 = queries[sel_r].to(device).to(torch.float16)
        cand, counts = mr.filter.evaluate_indices(qs)
        c = cand[0, : int(counts[0])]
        truth = item16[c].double() @ q16[0].double()
        s, order = truth.sort(descending=True)
        gaps_all.append((s[99] - s[100]).item())
        if r not in bad:
            continue
        gid = c[order]
        rank = {int(g): i for i, g in enumerate(gid[:2000].tolist())}
        a = set(ids_t[r, :100].tolist()) - {-1}
        b = set(ids_r[r, :100].tolist()) - {-1}
        only_t, only_r = sorted(a - b), sorted(b - a)
        launch = _fmkt_prep(q16, item16, cand[:, : int(counts[0])].contiguous(), counts, DEFAULT_CONFIG, bucket=False)
        _fused_masked_knn_topk_kernel[launch.grid](**launch.kwargs)
        kern = launch.all_scores[0]
        rec = {"row": int(sel_r[0]), "count": int(counts[0]), "boundary_score": s[99].item(), "gap": gaps_all[-1], "pairs": []}
        for g_t, g_r in zip(only_t, only_r):
            rt, rr = rank.get(g_t, -1), rank.get(g_r, -1)
            torch_item_is_true_top100 += rt != -1 and rt < 100
            triton_item_is_true_top100 += rr != -1 and rr < 100
            i_t = int((c == g_t).nonzero()[0])
            i_r = int((c == g_r).nonzero()[0])
            kt, kr = kern[i_t].item(), kern[i_r].item()
            consistent += kr >= kt  # the kernel's own scores rank the item it returned at least as high
            rec["pairs"].append({
                "torch_only": {"id": g_t, "truth_rank": rt, "truth": truth[i_t].item(), "kernel": kt, "kernel_err": kt - truth[i_t].item()},
                "triton_only": {"id": g_r, "truth_rank": rr, "truth": truth[i_r].item(), "kernel": kr, "kernel_err": kr - truth[i_r].item()},
                "truth_gap_between_pair": truth[i_t].item() - truth[i_r].item(),
            })
        swaps.append(rec)
    gaps_t = torch.tensor(gaps_all)
    pair_gaps = torch.tensor([p["truth_gap_between_pair"] for rec in swaps for p in rec["pairs"]])
    kern_errs = torch.tensor([abs(p[side]["kernel_err"]) for rec in swaps for p in rec["pairs"] for side in ("torch_only", "triton_only")])
    n_pairs = int(pair_gaps.numel())
    out["boundary"] = {
        "rows_top100_differ": len(bad),
        "swapped_pairs": n_pairs,
        "rows_with_exactly_one_swap": sum(len(rec["pairs"]) == 1 for rec in swaps),
        "torch_only_item_is_in_true_top100": torch_item_is_true_top100,
        "triton_only_item_is_in_true_top100": triton_item_is_true_top100,
        "kernel_scores_consistent_with_its_own_choice": consistent,
        "truth_gap_between_swapped_pair": {"max": pair_gaps.max().item(), "median": pair_gaps.median().item(), "min": pair_gaps.min().item()},
        "kernel_abs_error_on_swapped_items": {"max": kern_errs.max().item(), "median": kern_errs.median().item()},
        "rank100_gap_all_rows": {
            "median": gaps_t.median().item(), "p10": gaps_t.quantile(0.1).item(), "p1": gaps_t.quantile(0.01).item(),
            "min": gaps_t.min().item(),
            "frac_below_0.03": (gaps_t < 0.03).float().mean().item(),
            "frac_below_0.01": (gaps_t < 0.01).float().mean().item(),
            "frac_below_0.001": (gaps_t < 0.001).float().mean().item(),
        },
        "first_rows": swaps[:12],
    }
    print(json.dumps({k: v for k, v in out["boundary"].items() if k != "first_rows"}, indent=1), flush=True)

    # --- C. cost of fp32 accumulation on the cell's shapes -------------------------------------
    timing = {}
    for bsz in (1, 16):
        sel = rows[:bsz]
        q16 = queries[sel].to(device).to(torch.float16)
        qs = qa[sel].to(device)
        cand, counts = mr.filter.evaluate_indices(qs)
        cand = cand[:, : int(counts.max())].contiguous()
        for name, kern in (("shipped_fp16_acc", _fused_masked_knn_topk_kernel), ("probe_fp32_acc", _kernel_fp32_acc)):
            launch = _fmkt_prep(q16, item16, cand, counts, DEFAULT_CONFIG, bucket=False)
            fn = lambda: kern[launch.grid](**launch.kwargs)  # noqa: E731
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            m0 = sm_mhz()
            ms = _bench(fn)
            timing[f"kernel_B{bsz}_P{cand.shape[1]}_{name}"] = {"ms": ms, "sm_mhz": [m0, sm_mhz()]}
        q = queries[sel].to(device)
        for name, fn in (
            ("linr_v2_triton_eager_shipped", lambda: mr(q, qs)),
            ("linr_v2_torch_eager", lambda: mt(q, qs)),
        ):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            m0 = sm_mhz()
            ms = _bench(fn)
            timing[f"forward_B{bsz}_{name}"] = {"ms": ms, "sm_mhz": [m0, sm_mhz()]}
    out["timing"] = timing
    print(json.dumps(timing, indent=1), flush=True)
    out["elapsed_s"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
