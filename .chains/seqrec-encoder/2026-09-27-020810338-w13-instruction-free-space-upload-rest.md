---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-020628587-w13-report.md"
nextStep: "w13: delete eval-kuairand/checkpoints/gsasrec-d128-shared, then upload in order (trainer inputs, KuaiRand eval inputs, KuaiRand checkpoint), verify each, commit the script and docs on dev/hstu-hub, write the w13 final report."
created: "2026-09-26T23:08:10Z"
---

# W13 instruction: the user approved freeing space; upload the rest

User, 2026-09-27, when asked about the full private HF storage: **delete
`pinkmeme/eval-kuairand/checkpoints/gsasrec-d128-shared`** (33.4 GB, the old A100 gSASRec KuaiRand run). Then upload
everything still pending.

1. **Delete exactly that path,** in one commit (`delete_folder` / `HfApi.delete_folder`) with a clear commit message.
   Delete nothing else. Record the commit id. Re-check the account's private storage, and state whether HF counts the
   freed space right away.
2. **Upload in this order,** verifying each (remote file list and sizes) before the next:
   1. trainer inputs (your dry-run-checked `trainer_upload.py`; aborts if a repo is not private);
   2. `eval-data publish kuairand --private`;
   3. the KuaiRand checkpoint `sasrec-ssm-logq-d64-trainval`.

   If a 403 returns at any step, stop, and report exactly what landed and what did not.
3. **Keep the script:** copy `trainer_upload.py` into `docs/artifacts/seqrec-encoder/hub-upload/`. It lives in your
   scratchpad, which dies with the VM.
4. **Docs, on `dev/hstu-hub`:**
   - `checkpoints.md`: the KuaiRand row.
   - `datasets.md`: the trainer inputs on the Hub under `trainer/` per repo and how to fetch them; the KuaiRand eval
     inputs now on the Hub (private); fix the stale "no checkpoint" in § kuairand; note that the old A100
     `gsasrec-d128-shared` checkpoint was deleted to free space (user decision).
   - The link checker must be at zero. Commit; do not push.
5. **Write a final w13 report note:** per upload, the repo, path, visibility, bytes and verified list; the deletion
   commit; and the storage before and after.
