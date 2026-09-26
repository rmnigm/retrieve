---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-021344218-w13-final-report.md"
nextStep: "w13: squash eval-kuairand history, confirm the freed quota, upload the KuaiRand eval inputs then the checkpoint (verified), update the docs, write the report; then finish the inventory note."
created: "2026-09-26T23:15:53Z"
---

# W13 instruction: squash eval-kuairand, then upload the KuaiRand eval inputs and checkpoint (user approved)

User, 2026-09-27, chose "Yes, squash + upload".
1. **Pre-check:** list `pinkmeme/eval-kuairand` current files. They must be only `.gitattributes` and `trainer/` (3 files).
   If anything else is there, stop and report.
2. **Squash:** `HfApi.super_squash_history(repo_id="pinkmeme/eval-kuairand", repo_type="dataset")`, on that repo only.
   Then re-read `used_storage` (it may take a moment) and record before and after. Verify the `trainer/` files are
   still present with the same sizes.
3. **Upload** `eval-data publish kuairand --private` and verify it. Then upload the `sasrec-ssm-logq-d64-trainval`
   checkpoint and verify it. On a 403, stop and report.
4. **Docs, on dev/hstu-hub:** the KuaiRand row in checkpoints.md and "KuaiRand eval inputs on the Hub (private)" in
   datasets.md, only for what landed. The link checker must be at zero. Commit.
5. Write a report note. Then do the inventory note from the earlier instruction.
   - The duplicate model-repo folders stay proposed, not deleted: the harness blocked the mass delete and they cost
     almost no quota.
   - Include in it the per-repo `used_storage`, and flag any other repo where deleted blobs linger in history.
