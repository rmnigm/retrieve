---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-27-013919889-user-stop-kuairand-at-d64.md"
nextStep: "None scheduled. The user decides whether to merge dev/hstu into staging (roadmap: 'Sequential encoder follow-ups'). If resumed, start from the follow-ups list below; KuaiRand d128 needs a memory fix first."
created: "2026-09-26T22:52:08Z"
---

# Final: dev/hstu seqrec-encoder line: what passed, what was skipped, what is unverified

## 1. Primary request and intent
Replace gSASRec with a better, bigger, generic encoder (HSTU and beyond) with more efficient training; goodreads and
yambda-500m first, KuaiRand on success. The user refined this during the run:
- opus only, at most two herdr agents, no subagents;
- about 12 h, then relaxed;
- HSTU only with softmax;
- logQ sampled softmax;
- "stick with the E1c recipe";
- d64/d128/d256 on yambda and goodreads;
- KuaiRand d64 (d128 only if it fits; it did not);
- the KuaiRand refit on train + val;
- keep gBCE;
- drop HSTU;
- W&B logging (entity pinkmeme);
- briefs and reports as chain branches;
- the orchestrator dispatches all work;
- never print env files.

## 3. Work completed: dev/hstu at 6662191 on origin; never merged into staging
**The trainer** (`evaluation/training/`):
- one gSASRec-body `Encoder`;
- `loss: gbce | sampled_softmax`. The default is the **E1c recipe**: cosine, T 0.05, uniform 8192 + in-batch 4096, and
  **logQ = log(M·p + K/N)** with the positive uncorrected;
- GPU-resident batches, compile, warmup;
- `train run FIELD=VALUE` with the loss-override fix;
- `resume_every` with crash-safe snapshot retention;
- `train_on_val`, the final fit on train + val with the loss masked to val-day positions.

HSTU was tried (E2a/E2c: softmax HSTU, worse than the plain body with the same loss, 3.6× slower) and removed. The
ETLs write a `timestamps` column (unused by the trainer).

**Every change passed a review gate** (R1-R4, R6, R8, R10-R12; deslop + python-review). The refactor proof was
bit-identical 50-step losses before and after the cleanup, plus Gate A.

## Results (test, full catalog, H100; not yet validated, not citable; docs/validation.md § Seqrec encoder)
| model | ndcg@10 / R@100 | vs published gSASRec, same D, same test file |
|---|---|---|
| yambda d64 | 0.0945 / 0.1619 | +0.0099 / +0.0056 |
| yambda d128 | 0.1006 / 0.1662 | +0.0195 / +0.0176 |
| yambda d256 | 0.0966 / 0.1481 | +0.0152 / +0.0083 |
| goodreads d64 | 0.0381 / 0.1447 | +0.0031 / **−0.0039** |
| goodreads d128 | 0.0410 / 0.1518 | +0.0049 / +0.0038 |
| goodreads d256 | 0.0418 / 0.1530 | +0.0064 / +0.0058 |
| KuaiRand d64 (refit on train + val) | 0.0276 / 0.0063 | no bar; val-day most-popular 0.0314 / 0.0073 |

- **5 of 6 yambda/goodreads models beat the bar on both metrics.**
- **Attribution (yambda d64):** the loss (logQ) gives +0.028 ndcg@10; the HSTU body gives −0.006.
- **KuaiRand is limited by temporal drift.** The train-only model scored 0.0046, and the refit gave a 6× gain.
  Details: `2026-09-26-234034101-e-k64-diagnosis.md`.

## 4. Decisions (all in docs/decisions.md § Sequential encoder)
- The bars are same-file re-scores of the published checkpoints. The stored yambda numbers came from a split that is
  no longer on disk.
- gBCE needs per-position negatives (a shared vector collapses).
- The logQ normalization is the expected count (R3 F1).
- The user's choices listed in §1.

## 6. Skipped / unverified
- **KuaiRand d128 was not run.** OOM in backward even with one table (~78 of 79 GiB); the user chose to stop.
- Nothing is citable; no gate turns these results green.
- Unverified:
  - `--resume` for `train_on_val` runs on the GPU;
  - the atomic `_resume.pt` write (R10 F1, deferred);
  - 25% of heavy users' val-day clicks are not trained;
  - launch overhead is unprofiled (HSTU showed CPU-bound steps; KuaiRand showed ~25% of each epoch in checkpoint I/O,
    now addressed by `resume_every`);
  - `ruff format` fails on 7 files that predate this work (evaluation-cleanup Batch 4).

## 7. Follow-ups (docs/roadmap.md "Sequential encoder follow-ups", not scheduled)
- the KuaiRand d128 memory fix (single gather or a sparse optimizer);
- R10 F1;
- val-day windowing;
- CUDA graphs / launch overhead;
- goodreads d64 R@100;
- merging dev/hstu into staging (the user's call);
- Hub publishing of the new checkpoints (not done; only with the user).

## 8. Current state
- **No agents running and the GPU is idle.**
- **Checkpoints:** `/scratch/ckpt/*` and `/data/*/checkpoints` (the published ones, untouched).
- **Worktrees:** `/scratch/wt/*` (12, all merged; the branches are on origin only where pushed: dev/hstu, dev/hstu-runs).
- **Venvs:** `/venvs/wt-*`.
- **Chains snapshot:** committed on dev/hstu at the user's request.
