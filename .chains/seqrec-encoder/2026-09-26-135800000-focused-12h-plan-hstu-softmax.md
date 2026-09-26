---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-134900000-user-amendment-opus-only-two-agents.md"
nextStep: "Wait for W1 (pane w1:p3, agent w1-encoder) and W2 (pane w1:p4, agent w2-etl); when either reports, close its pane, start an opus reviewer (deslop + python-review) on that branch in the freed slot, then merge into dev/hstu and push. Then run E1-E3 below as one opus run worker, one GPU job at a time."
created: "2026-09-26T10:58:00Z"
---

# Focused 12 h plan: HSTU-style block with softmax + sampled softmax

## 1. Primary request and intent (user, mid-session 2026-09-26)
"~12 hours of worktime, pick the best set of experiments. Do not try to check everything,
the best combo you most believe in (I believe in HSTU, but don't believe in attention
without softmax)." Plus the earlier amendment: opus only, at most two herdr agents,
agents may not launch subagents (enforced by `--disallowedTools Agent Workflow` and in briefs).

## 2. Decisions and rationale
- **The bet:** `encoder=hstu` = HSTU block with **softmax kept** (SiLU U/V/Q/K projections,
  U-gating, norm(AV), rel-position + log time-bucket bias as a float SDPA mask), wider body
  (H=256, 4 blocks, 4 heads, projected to D), + `sampled_softmax` on L2-normalized embeddings,
  tau 0.05, M in-batch + K uniform negatives, no logQ (HSTU's own recipe has none).
  Evidence: HSTU Table 5 shows the bias carries a real part of the gain (softmax+no-bias 4.024 vs
  full 3.978); loss upgrade is the best-evidenced lever ("Dross into Gold"); user rejects pointwise.
- **Dropped from scope:** LlamaBlock, pointwise-SiLU attention, logQ, row-wise Adagrad, bf16 table,
  FuXi/SCE. KuaiRand is out of reach in 12 h unless E1-E3 finish early and win clearly.
- **yambda-500m first:** trainer inputs exist, gSASRec d64 trained in 47 min on the A100 (and hit
  its 100-epoch cap still improving); goodreads d64 took 3 h and its inputs are still being built (W2).
- Gate B moved to yambda d64 (recorded in the W1 brief, A7).

## 3. Experiment set (one GPU job at a time, after Gate B)
| id | dataset | D | encoder | loss | purpose |
|---|---|---|---|---|---|
| E1 | yambda-500m | 64 | sasrec (published body) | sampled_softmax | loss effect alone |
| E2 | yambda-500m | 64 | hstu H=256 x4, use_time | sampled_softmax | the bet |
| E3 | goodreads-work-id | 64 | winner of E1/E2 | same recipe | second dataset, success bar |
| E4 (if time) | winner on the dataset where it is weakest at D=128 | 128 | | | same-D check at 128 |
Bars (test, same D): yambda d64 0.0813 / 0.1489; goodreads d64 0.0350 / 0.1486.
Shared recipe: B=256, L=200, lr 1e-3, wd 0, warmup 1000, dropout 0.2, K=8192 uniform + in-batch,
patience 10 on val ndcg@10; epochs capped so each run is <= 2.5 h on the H100.

## 4. Current work
- dev/hstu = origin/staging 8c493f2 (roadmap-only commit on top of ae0290d), pushed.
- W1 opus `w1-encoder` pane w1:p3, worktree /scratch/wt/encoder, branch dev/hstu-encoder;
  briefs /scratch/briefs/W1-encoder.md + W1-amendment-1.md.
- W2 opus `w2-etl` pane w1:p4, worktree /scratch/wt/etl, branch dev/hstu-etl; brief
  /scratch/briefs/W2-etl.md; writes /data/goodreads-work-id/trainer and /data/yambda-500m/trainer.new.
- Venvs /venvs/wt-encoder, /venvs/wt-etl (UV_PROJECT_ENVIRONMENT; the default /venvs/retrieve is the main checkout's).

## 5. Unresolved
- The container cgroup limit is ~251 GB (RUNPOD_MEM_GB), not the ~2 TB `free` shows; the goodreads prep peaks ~129 GB.
- `.chains/evaluation-cleanup` backlog overlaps training/ dead code (GSASRecConfig.num_items, wandb fallback):
  W1's rewrite removes them; tell that chain when it is dispatched.
