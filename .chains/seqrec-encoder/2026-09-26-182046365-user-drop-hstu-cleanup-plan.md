---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-181840268-user-final-scope-e1c-recipe.md"
nextStep: "When W5 (KuaiRand ETL) reports, close it and dispatch W6 (trainer cleanup: delete HSTU and gBCE, E1c defaults) from /scratch/briefs/W6-cleanup.md as a w6-cleanup chain note; review; merge; then W7 (KuaiRand memory fit + launch overhead); review; merge; then the KuaiRand go for e-runs."
created: "2026-09-26T15:20:46Z"
---

# User decision: HSTU is not needed; the trainer keeps the new loss, the perf work and the better loop

User, 2026-09-26 ~15:25Z: "yeah drop everything except new loss and perf optimizations and better
trainer. hstu is not needed."

## Interpretation (sent to the user; not yet contradicted)
- **Delete from `evaluation/training/`:**
  - `HSTUBlock`, `RelativeBias`, `use_time`, `time_buckets` and the time-gap embedding;
  - `hidden_dim` with `in_proj`/`out_proj`;
  - the `encoder` switch and `BLOCKS`;
  - `gbce` as a training loss, with its per-position negatives.
- **Keep:**
  - sampled softmax with logQ and `normalize`;
  - the GPU-resident batching, compile, warmup, the R1 fixes, W&B, and `train run FIELD=VALUE`;
  - loading and scoring of the published gSASRec checkpoints (`encode.py`, legacy key rename,
    `D128_DROP05_DEFAULTS`, `normalize` false for them).
- **The E1c recipe becomes the `TrainConfig` defaults.**
- **The ETL `timestamps` column stays.** It is data layout, it is already prepped, and KuaiRand is
  mid-prep. The user may overrule this.

## Sequencing (2-agent cap)
- **W6 cleanup:** CPU. Proof: same seed, same batches, identical per-step losses for the E1c recipe
  before and after, plus Gate A (published checkpoints score identically). Review follows.
- **W7:** the KuaiRand memory fit plus launch overhead (`2026-09-26-171131468-pending-efficiency-step.md`).
  It uses the GPU only in the gaps between runs. Review follows.
- **KuaiRand d64 and d128** run on the merged trainer. The yambda and goodreads d128 runs use the current
  code; the recipe is the same.
