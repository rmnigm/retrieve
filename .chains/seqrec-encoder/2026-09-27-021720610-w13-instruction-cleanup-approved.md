---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-021344219-w13-hf-inventory.md"
nextStep: "w13: after the KuaiRand squash and uploads, run proposals 2-6 from the inventory (user approved all), then squash the touched repos, verify, and write a cleanup report."
created: "2026-09-26T23:17:20Z"
---

# W13 instruction: the user approved ALL cleanup proposals 2-6

User, 2026-09-27, verbatim: **"do all of them"** (proposals 2-6 in `2026-09-27-021344219-w13-hf-inventory.md`, shown
to the user with the risks, including #4). Run this **after** the current KuaiRand squash, upload and docs work
finishes.

1. **#2:** delete the 8 exact-duplicate dirs, one commit per repo:
   - `goodreads-gsasrec` d64/d128/d256;
   - `yambda-500m-gsasrec` d64/d128/d256;
   - `yambda-5b-gsasrec` d64-v4/d128-v5.

   Re-verify the sha256 equality right before deleting.
2. **#3, move first:**
   - Copy `arxiv-retrieval/eval_arxiv_retrieval.json` into `pinkmeme/eval-results` (path `legacy/arxiv-retrieval/`) and
     verify that the bytes are equal.
   - Then delete everything else in `arxiv-retrieval`. Keep the (now nearly empty) repo; do not delete repos.
3. **#4:** delete `yambda-500m-gsasrec/v1`.
4. **#5:** delete `eval-goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id/encoded_queries_{test,v2}.pt`.
5. **#6:** delete `gt_d128/` in `eval-goodreads-work-id` and in `eval-arxiv-papers`.
6. **Squash the history** (`super_squash_history`) of exactly the repos touched above, so the quota is freed:
   `arxiv-retrieval`, `yambda-500m-gsasrec`, `eval-goodreads-work-id` and `eval-arxiv-papers`, plus
   `goodreads-gsasrec` and `yambda-5b-gsasrec` (harmless). Do not squash any other repo.
   - Before each squash, list the current files.
   - After each, verify that they are unchanged and record `used_storage` before and after.
7. **Do not touch** class A, B, K or F items, or any eval/trainer data outside the paths above. Nothing becomes public.

**If the harness blocks a delete or squash,** do not retry or work around it. Record the exact call (repo, path, API),
continue with the rest, and list the blocked calls in the report so the user can run them.

**Report note:** `$(/scratch/briefs/chain-ts)-w13-cleanup-report.md`. Per action: done or blocked, the commit id, and the
bytes. Also the storage total before and after, and the final repo table. If any docs referenced a deleted path (they
should not), fix them on dev/hstu-hub and commit.
