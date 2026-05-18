"""Diagnose why linr_v1_filter_mask recall < 1.0 on arxiv clause sweeps.

Loads arxiv d256, builds the same ExactAttributeFilter the eval uses, runs
the oracle's matmul path and the algo's matmul path on a small batch, and
reports where they disagree (scores, masks, topk ids).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "retrieve" / "src"))

from retrieve import SimilarityMasking  # noqa: E402
from retrieval.algos.filter import build_filter, make_mask  # noqa: E402

DATA = ROOT / "data" / "arxiv-papers"
CONTENT = DATA / "content"         # d256 — has cached oracle in gt/
GT_DIR = DATA / "gt"
SWEEP = "all4"                     # strictest: all 4 clauses active
ACTIVE_CLAUSES = [0, 1, 2, 3]
K = 100
N_USERS_DEBUG = 256                # full sweep is 10k users; 256 fits per-batch
COMPARE_VS_CACHED_GT = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_printoptions(precision=8, sci_mode=False)

dev = torch.device("cuda")

# ---- load embeddings + queries (mirror load_pre_encoded_arxiv) ---------------
item_embs = torch.load(str(CONTENT / "text_emb.pt"), map_location=dev)
item_embs[0] = 0.0
item_embs = item_embs.float().contiguous()
item_embs = F.normalize(item_embs, dim=-1)
print(f"item_embs: shape={tuple(item_embs.shape)} dtype={item_embs.dtype} "
      f"contig={item_embs.is_contiguous()} device={item_embs.device}")

queries_all = torch.load(str(CONTENT / "query_emb.pt"), map_location="cpu")
queries_all = F.normalize(queries_all.float(), dim=-1)
print(f"queries: shape={tuple(queries_all.shape)} dtype={queries_all.dtype}")

# ---- load attrs + build filter (clause / exact) ------------------------------
item_attrs = torch.load(str(DATA / "item_attrs_narrow.pt"), map_location=dev)
clause_rev = torch.load(str(DATA / "clause_is_reverse_narrow.pt"), map_location=dev)
print(f"item_attrs_narrow: {tuple(item_attrs.shape)} dtype={item_attrs.dtype}")

# Project to active clauses (mirroring build_sweep_qa for clause kind: keep only active)
# For clause kind, build_sweep_qa does NOT slice columns — it zeros out non-active
# clauses in qa, so the filter still sees full-width attrs. Keep that logic faithful.
import polars as pl
heldout = pl.read_parquet(DATA / "heldout.parquet")
n_users = heldout.height
queries_all = queries_all[:n_users]

# qa_narrow_all is [n_users, C] int64 from eval_split.parquet.query_attrs_narrow
eval_split = pl.read_parquet(DATA / "eval_split.parquet")
qa_narrow_all = torch.tensor(eval_split["query_attrs_narrow"].to_list(), dtype=torch.long)
print(f"qa_narrow_all: shape={tuple(qa_narrow_all.shape)} dtype={qa_narrow_all.dtype}")

n_clauses = qa_narrow_all.shape[1]
qa_n_sweep = qa_narrow_all.clone()
inactive_mask = torch.ones(n_clauses, dtype=torch.bool)
for c in ACTIVE_CLAUSES:
    inactive_mask[c] = False
qa_n_sweep[:, inactive_mask] = -1
skip_mask = (qa_n_sweep == -1).all(dim=1)
print(f"sweep '{SWEEP}': active_clauses={ACTIVE_CLAUSES}  skipped={int(skip_mask.sum())}/{n_users}")

filter_mod = build_filter(
    "clause",
    item_attrs_narrow=item_attrs,
    clause_is_reverse=clause_rev,
    device=dev,
)

# ---- pick a debug slice of kept users ---------------------------------------
keep_idx = (~skip_mask).nonzero(as_tuple=False).reshape(-1)
debug_idx = keep_idx[:N_USERS_DEBUG]
print(f"debug slice: {debug_idx.numel()} kept users")

q = queries_all[debug_idx].to(dev)
qa_n = qa_n_sweep[debug_idx].to(dev)
print(f"q: shape={tuple(q.shape)} dtype={q.dtype} contig={q.is_contiguous()}")

# ---- ORACLE path: q @ item_embs.t().contiguous(), exact mask, topk ----------
item_embs_t = item_embs.t().contiguous()
print(f"item_embs_t (oracle): shape={tuple(item_embs_t.shape)} "
      f"strides={item_embs_t.stride()} contig={item_embs_t.is_contiguous()}")

scores_oracle = q @ item_embs_t                       # [B, N]
mask_oracle = filter_mod.evaluate_mask(qa_n)          # [B, N] bool
mask_oracle = mask_oracle.clone()
# Oracle does not flip col 0 in the mask — it sets scores[:, 0] = -inf instead.
scores_oracle_masked = scores_oracle.masked_fill(~mask_oracle, float("-inf")).clone()
scores_oracle_masked[:, 0] = float("-inf")
topk_oracle = torch.topk(scores_oracle_masked, K, dim=1)
# Mirror compute_filtered_oracle's -1 post-processing so the diagnostic
# reflects what the real eval compares against.
topk_oracle_ids_post = torch.where(
    torch.isfinite(topk_oracle.values),
    topk_oracle.indices,
    torch.full_like(topk_oracle.indices, -1),
)

# ---- ALGO path: SimilarityMasking(backend="triton") with our new contiguous-T buffer ----
algo = SimilarityMasking(k=K, backend="triton").to(dev)
algo.register_index(item_embs)
print(f"algo.item_embs_t: shape={tuple(algo.item_embs_t.shape)} "
      f"strides={algo.item_embs_t.stride()} contig={algo.item_embs_t.is_contiguous()}")

mask_algo = make_mask(filter_mod, qa_n)               # [B, N] bool, with col 0 = False
algo_topk_ids, algo_topk_scores = algo(q, mask=mask_algo)

# ---- Diagnostic 1: are the algo's pre-mask scores bit-identical to oracle? --
scores_algo_raw = q @ algo.item_embs_t
diff = (scores_oracle - scores_algo_raw).abs()
print()
print("=== raw matmul score comparison (oracle q@E_t vs algo q@E_t) ===")
print(f"  shape={tuple(diff.shape)}  max_abs_diff={diff.max().item():.3e}  "
      f"mean_abs_diff={diff.mean().item():.3e}")
print(f"  bit_exact_pct={float((diff == 0).float().mean()) * 100:.4f}%")

# ---- Diagnostic 2: are the masks identical? --------------------------------
# Note: oracle mask has col 0 from filter (could be True/False), algo mask forces False.
mask_oracle_for_compare = mask_oracle.clone()
mask_oracle_for_compare[:, 0] = False
mask_diff = (mask_oracle_for_compare ^ mask_algo)
print()
print("=== mask comparison (after forcing col 0 = False on both) ===")
print(f"  identical={bool(mask_diff.sum() == 0)}  diff_cells={int(mask_diff.sum())}")

# ---- Diagnostic 3: top-K id agreement, per-row recall -----------------------
oracle_ids = topk_oracle_ids_post.cpu()
algo_ids = algo_topk_ids.cpu()
def _row_recall(oracle_row, algo_row):
    o = set(oracle_row.tolist()) - {-1}
    a = set(algo_row.tolist()) - {-1}
    if not o:
        return 1.0  # vacuous: no relevant items
    return len(o & a) / len(o)

row_recalls = torch.tensor([
    _row_recall(oracle_ids[i], algo_ids[i])
    for i in range(oracle_ids.shape[0])
])
print()
print(f"=== top-{K} id-set recall vs oracle on {oracle_ids.shape[0]} kept users ===")
print(f"  mean recall@{K} = {row_recalls.mean().item():.6f}")
print(f"  perfect rows: {int((row_recalls == 1.0).sum())}/{oracle_ids.shape[0]}")
print(f"  worst recall: {row_recalls.min().item():.4f}")

# ---- Diagnostic 4: for first N disagreement rows, dump K-boundary -----------
# ---- Diagnostic 5: compare algo vs CACHED oracle (gt_topk_*.pt) -------------
if COMPARE_VS_CACHED_GT:
    gt_path = GT_DIR / f"gt_topk_{SWEEP}.pt"
    if gt_path.exists():
        cached = torch.load(str(gt_path), map_location="cpu")
        cached_slice = cached[debug_idx][:, :K]
        cached_row_recalls = torch.tensor([
            _row_recall(cached_slice[i], algo_ids[i])
            for i in range(algo_ids.shape[0])
        ])
        print()
        print(f"=== algo top-{K} vs CACHED gt {gt_path.name} ===")
        print(f"  mean recall@{K} = {cached_row_recalls.mean().item():.6f}")
        print(f"  perfect rows: {int((cached_row_recalls == 1.0).sum())}/{algo_ids.shape[0]}")
        print(f"  worst recall: {cached_row_recalls.min().item():.4f}")
        # Compare cached vs fresh oracle to see if cache is stale
        fresh_oracle_ids = oracle_ids
        cache_vs_fresh = torch.tensor([
            _row_recall(cached_slice[i], fresh_oracle_ids[i])
            for i in range(algo_ids.shape[0])
        ])
        print(f"  cached gt vs fresh oracle: mean={cache_vs_fresh.mean().item():.6f}  "
              f"perfect={int((cache_vs_fresh == 1.0).sum())}/{algo_ids.shape[0]}")

        # Drill into worst-disagreement rows to see what kind of items differ
        worst = cache_vs_fresh.argsort()[:3]
        for i in worst.tolist():
            o_set = set(fresh_oracle_ids[i].tolist())
            c_set = set(cached_slice[i].tolist())
            only_fresh = sorted(o_set - c_set)
            only_cached = sorted(c_set - o_set)
            print(f"\n  worst row {i}: recall_vs_cached={cache_vs_fresh[i].item():.4f}  "
                  f"|only_fresh|={len(only_fresh)} |only_cached|={len(only_cached)}")
            # Score the disputed items in fresh oracle
            disp_ids = torch.tensor(only_fresh + only_cached, dtype=torch.long, device=dev)
            disp_scores = (q[i:i+1] @ algo.item_embs_t).squeeze(0)[disp_ids].cpu()
            print(f"    fresh-K-th score: {topk_oracle.values[i, -1].item():.10f}")
            print(f"    only_fresh  ids[:5]={only_fresh[:5]}  scores={disp_scores[:5].tolist()}")
            print(f"    only_cached ids[:5]={only_cached[:5]}  "
                  f"scores={disp_scores[len(only_fresh):len(only_fresh)+5].tolist()}")
    else:
        print(f"\n(no cached gt at {gt_path}, skipping cache check)")

bad = (row_recalls < 1.0).nonzero(as_tuple=False).reshape(-1)
print(f"  rows w/ fresh-oracle recall<1: {bad.numel()}")
for i in bad[:5].tolist():
    o_set = set(oracle_ids[i].tolist())
    a_set = set(algo_ids[i].tolist())
    only_oracle = sorted(o_set - a_set)
    only_algo = sorted(a_set - o_set)
    o_scores_for_only_oracle = scores_oracle_masked[i, torch.tensor(only_oracle, dtype=torch.long)].cpu()
    a_scores_for_only_algo_rows = (q[i:i+1] @ algo.item_embs_t).squeeze(0)
    a_scores_for_only_algo_rows = a_scores_for_only_algo_rows.masked_fill(
        ~mask_algo[i], float("-inf")
    )
    a_scores_for_only_algo = a_scores_for_only_algo_rows[torch.tensor(only_algo, dtype=torch.long)].cpu()

    # K-th boundary scores
    o_boundary = topk_oracle.values[i, -1].item()
    a_boundary = algo_topk_scores[i, -1].item()
    print(f"  row {i}: |only_oracle|={len(only_oracle)} |only_algo|={len(only_algo)} "
          f"oracle_K-th={o_boundary:.10f}  algo_K-th={a_boundary:.10f}  "
          f"|Δ_K-th|={abs(o_boundary - a_boundary):.3e}")
    print(f"     only_oracle ids[:6]={only_oracle[:6]}  scores={o_scores_for_only_oracle[:6].tolist()}")
    print(f"     only_algo    ids[:6]={only_algo[:6]}  scores={a_scores_for_only_algo[:6].tolist()}")
