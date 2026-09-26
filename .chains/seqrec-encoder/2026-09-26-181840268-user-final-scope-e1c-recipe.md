---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-181309915-yambda-d64-success-e1c.md"
nextStep: "e-runs: after E3, yambda d128 then goodreads d128 (E1c recipe). w5-kuairand-etl: stage KuaiRand with timestamps (CPU). Then a perf/memory worker makes KuaiRand d64/d128 fit and run fast; then e-runs trains KuaiRand d64 and d128."
created: "2026-09-26T15:18:40Z"
---

# User decision: the E1c recipe is final; the six runs that remain

User, 2026-09-26 ~15:20Z: "stick with e1c recipe, drop everything else, except needed perf optimizations
and so. train d=128 for kuairand and d64, and d64/128 for goodreads, yambda500m - and that's it for your
runs, probably."

- **Recipe:** `encoder=sasrec`, the published gSASRec body (2 blocks, 2 heads, ffn 256 at d64, dropout 0.5),
  and `loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096
  logq=true`, with warmup 1000, compile, lr 1e-3, B 256, L 200. Only D, epochs, patience and eval
  cadence vary.
  - Open for d128: does ffn scale with D (the published gSASRec d128 used ffn 4×D = 512)? Use 4×D,
    as the published gSASRec runs did.
- **Runs:**

  | dataset | d64 | d128 |
  |---|---|---|
  | yambda-500m | E1c (done) | new |
  | goodreads | E3 (running) | new |
  | KuaiRand | new | new |

- **Dropped:** HSTU runs (E2b resume, HSTU on goodreads), E4 as it was, the Pinterest ideas, and the
  LLaMA block.
  - Whether to delete the HSTU code is not decided. Ask the user; do not delete on my own.
- **Kept:** the "needed perf optimizations".
  1. The KuaiRand memory fit (32M-row table). Dense AdamW on one table at d128 is ~65.6 GB by
     arithmetic; this OOMed on the A100 with gBCE.
  2. The launch-overhead step (`2026-09-26-171131468-pending-efficiency-step.md`), only as far as it
     makes these runs faster.
- **Bars:**
  - yambda d128: 0.0811 / 0.1486.
  - goodreads d128: 0.0361 / 0.1480.
  - KuaiRand: no published baseline. The A100 gSASRec d128 run's result, if one exists, is the
    reference; otherwise the numbers are recorded as they are.
