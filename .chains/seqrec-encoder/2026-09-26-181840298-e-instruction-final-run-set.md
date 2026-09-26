---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-26-181222678-e-decision-e3-winner.md"
nextStep: "e-runs: finish E3; then R-y128 (yambda d128) and R-g128 (goodreads d128) with the E1c recipe; then idle until the orchestrator's KuaiRand go note."
created: "2026-09-26T15:18:40Z"
---

# E-runs instruction: the final run set (user decision)

Read `2026-09-26-181840268-user-final-scope-e1c-recipe.md` (on main). Summary:
- **Keep E3 running** (goodreads d64, E1c recipe). Write its note when it ends.
- **R-y128:** yambda-500m d128, E1c's command with `embedding_dim=128 ffn_hidden_dim=512` (4×D, as
  published), everything else the same, W&B on, run id `y128-sasrec-ssm-logq`. Bar: 0.0811 / 0.1486.
- **R-g128:** goodreads d128, E3's command with `embedding_dim=128 ffn_hidden_dim=512`, run id
  `g128-sasrec-ssm-logq`. Bar: 0.0361 / 0.1480. Same epoch cap and patience as E3.
- Then **stay idle** until the orchestrator's KuaiRand go note. The data and the memory fit are being
  prepared by other agents. Do not start KuaiRand yourself.
- **One note per run, as before. Nothing else runs:** no HSTU, no E2b resume, no extra ablations.
