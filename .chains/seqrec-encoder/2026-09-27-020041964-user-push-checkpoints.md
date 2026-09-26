---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-27-015208887-final-dev-hstu-seqrec-encoder.md"
nextStep: "W13 uploads the 7 final checkpoints to the Hub (dry-run first); dev/hstu is NOT merged into staging (user)."
created: "2026-09-26T23:00:41Z"
---

# User: push checkpoints to the Hub; do not merge into staging yet

User, 2026-09-27: "push checkpoints, do not merge into staging yet". This authorizes the Hub upload of the 7 final
models: yambda d64/d128/d256, goodreads d64/d128/d256, and KuaiRand d64 refit. The published gSASRec checkpoints are
untouched. The upload uses `train upload-checkpoint`, which is private by default for new repos.
