---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-192919660-user-add-d256-kuairand-d64.md"
nextStep: "W10 adds train_on_val to the trainer (review, merge); then e-runs reruns KuaiRand d64 and d128 on train+val for a fixed 4 epochs, test only."
created: "2026-09-26T21:58:42Z"
---

# User decision: KuaiRand final models refit on train + val

- **Diagnosis** (`2026-09-26-234034101-e-k64-diagnosis.md`): temporal drift. The model is 1-2 days stale; the val
  day's most-popular list beats k64 on test about 7×. Not capacity, not cold items, not the metric.
- **The user picked the refit option.**
  - Add `train_on_val` to the trainer: val-day sequences become training data, the run uses a fixed
    `num_epochs` with no val early stop, and test is scored on the final model.
  - Rerun k64 and k128 with 4 epochs (k64 peaked on val at epoch 3).
  - The first k64 (train only) stays in the record as the drift evidence.
- **Deferred:** R10 F1 (atomic `_resume.pt` write).
