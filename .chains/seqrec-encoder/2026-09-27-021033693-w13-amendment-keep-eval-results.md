---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-021025151-w13-instruction-hf-inventory-cleanup.md"
nextStep: "w13: treat eval results as keep (same tier as the models) in the HF inventory."
created: "2026-09-26T23:10:33Z"
---

# W13 amendment: eval results are always kept

User, 2026-09-27: "and eval results ofc". In the HF inventory, every eval-results record is **keep**, in the same tier as the models:
- the `pinkmeme/eval-results` repo;
- benchmark or eval JSON/Parquet records in any repo;
- the `eval_quality.json` / `train_metrics.json` inside checkpoint dirs.

Never put them in the (E) or (D) proposals. The keep order is now: new models = eval results > gBCE checkpoints > re-derivable data.
