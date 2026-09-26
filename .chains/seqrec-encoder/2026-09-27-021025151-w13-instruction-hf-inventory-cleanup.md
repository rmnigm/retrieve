---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-020810338-w13-instruction-free-space-upload-rest.md"
nextStep: "w13: after the current uploads, inventory every pinkmeme HF repo, classify it, delete only verified exact duplicates, and write an inventory note with a proposed delete list for the user."
created: "2026-09-26T23:10:25Z"
---

# W13 instruction: HF inventory and stale cleanup (user request)

User, 2026-09-27: "also clean up some other stale things from hf if there is something there, save at first the
latest trained models (or the gbce ones), than re-derivable data."

**Keep priority,** highest first:
1. the new `sasrec-ssm-logq-*` checkpoints;
2. the published gBCE `gsasrec-*` checkpoints;
3. re-derivable data (eval inputs, trainer inputs).

**After** the uploads in progress are done (do not interrupt them):

1. **Inventory** every repo under `pinkmeme`, models and datasets: its top-level dirs and sizes (current revision),
   visibility and last-modified date.
2. **Classify each dir:**
   - (A) new models;
   - (B) the gBCE published checkpoints;
   - (C) **exact duplicates** of an A/B item elsewhere: identical LFS sha256/oid for every file (for example a model repo
     mirroring `checkpoints/gsasrec-*` in an eval repo);
   - (D) re-derivable data (say how it is derived: the ETL command);
   - (E) stale or unknown (not referenced in the docs; superseded runs, scratch or test uploads);
   - (F) **unrelated to this project** (for example `deberta-v3-hh-rlhf-reward`). Never touch F.
3. **Delete only (C),** keeping the copy that the docs reference (`docs/system/checkpoints.md`, `hub.py`, the fetch
   paths). Grep the repo for each path before deleting. One commit per repo, with a clear message. Record the commit
   ids. If anything is doubtful, it is not (C).
4. **Propose, do not delete, (E) and (D):** a table with the path, size, why it looks stale (evidence), and what
   would break if it were removed.
5. **Write an inventory note** `$(/scratch/briefs/chain-ts)-w13-hf-inventory.md` (branch w13-hub):
   - the full table;
   - what was deleted (C);
   - the proposed deletions with the space freed;
   - the storage total before and after.

   Its `nextStep` is the delete list for the user to approve. Then stay idle.

Never make anything public. Never print the token or env files.
