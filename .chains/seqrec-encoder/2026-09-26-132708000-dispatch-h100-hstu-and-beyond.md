---
chain: "seqrec-encoder"
branch: "main"
nextStep: "Worker on the H100 pod: branch dev/hstu off staging, fetch goodreads-work-id + yambda-500m (with their published gSASRec checkpoints for the baseline numbers), reproduce the baseline eval, then implement and run the candidate encoder ladder below, goodreads first."
created: "2026-09-26T13:27:08Z"
---

# Dispatch: a better, bigger history encoder than gSASRec (HSTU and beyond), on one H100

Scheduled by the user on 2026-09-26. This supersedes the "not yet scheduled"
status of `.chains/hstu-and-distributed-training/2026-09-26-103000000-dispatch-brief-not-yet-scheduled.md`
(read it: Part B's efficiency list still applies; its "research only, no code"
restriction on HSTU is lifted by the user: run the experiments). Multi-GPU/DDP is
out of scope: one H100.

## Primary request (user's words, condensed)
Substitute gSASRec with HSTU and more efficient training. Do not limit yourself to
HSTU (Argus and other follow-ups are fair game). Goal: a good, embedding-table-based
encoder over the user's interaction history that works across datasets with
minimal hand tuning of features, a generic model better and bigger than gSASRec.
Order: goodreads and yambda first; only on success, KuaiRand.

## Environment
- Pod: 1x H100 80GB HBM3, RunPod, checkout `/workspace/retrieve` on `staging`.
  Data goes on the container disk (`RETRIEVE_DATA_ROOT=/data`), never `/workspace`
  (`docs/system/storage.md`). Use `rp-sync`, not bare `uv sync`. Each job its own
  `TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>`. Record the GPU model with every number.
- `CLAUDE.md` rules apply, except "the A100 is the machine": this step runs on the
  H100. Serialize GPU work (one training at a time) unless two runs provably fit;
  prefer sequential, well-chosen runs to sprawling sweeps.
- Data: `uv run --directory evaluation eval-data fetch goodreads-work-id --dims d64,d128 --include-checkpoints`
  and `eval-data fetch yambda-500m` (+ its checkpoints; see `docs/system/checkpoints.md`).
  Yambda trainer inputs come from `eval-data yambda prep --variant 500m` if the Hub
  repo lacks them. KuaiRand is not on the Hub: `eval-data kuairand download|convert|prep`
  (Zenodo, 13.6 GB raw, md5-verified; see `docs/system/datasets.md`).

## Baseline to beat (same protocol, same eval code)
`evaluation/training/evaluate.py`, full-catalog ranking, test split:
- yambda-500m gSASRec d64 (bf16 + fused AdamW): NDCG@10 0.0813, NDCG@100 0.1029,
  R@10 0.0384, R@100 0.1489 (`docs/system/checkpoints.md`).
- goodreads-work-id gSASRec d{64,128,256}: read `eval_quality.json` of the
  fetched checkpoints; re-run the eval once on the H100 to confirm the number reproduces.
- KuaiRand: the A100 pod's gSASRec d128 run (negs 128) is still training; no published
  baseline exists. Compare at matched dim and wall-clock budget.
Success = beats the best gSASRec checkpoint of the **same embedding dim** on test
NDCG@10 **and** R@100 on both goodreads and yambda-500m, using one shared recipe
(the same hyperparameters across datasets except what scales with data, e.g. epochs).
"Bigger" is allowed in the body (layers, width, heads); the retrieval-facing embedding
dim D stays one the benchmark uses (64/128/256).

## Hard constraints on the model
- Output contract unchanged: `[N, D]` item embeddings (a plain embedding table,
  optionally projected) and `[B, D]` query embeddings from `encode.py`, dot-product
  scored. No generative/beam-search retrieval, no side-feature towers that need
  per-dataset engineering. Allowed inputs: item id, position, timestamp (if the
  dataset has one) and an action/event type (if it has one); both optional.
- Generic: one code path for all datasets; no per-dataset feature code.
- Thin code (`docs/contracts/coding-guidelines.md`): add the new encoder next to
  `GSASRec` in `evaluation/training/`, selected by a config field; if it wins
  everywhere, it replaces gSASRec as the default (no self-compat shims), the old
  checkpoints stay loadable only as far as `encode.py` already needs them.

## Candidate ladder (cheap to expensive; stop climbing when gains stop)
The research findings below (appended by the orchestrator) rank these; roughly:
1. Training recipe on the existing model: sampled softmax / CE with many negatives
   (in-batch + uniform mix, logQ correction) vs gBCE; bf16 + fused AdamW + TF32 as
   defaults; `torch.compile` on the train step.
2. Modern transformer block (pre-norm RMSNorm, SwiGLU, RoPE or relative bias,
   SDPA/flash attention), wider/deeper.
3. HSTU block (pointwise SiLU attention, relative position + time-bucket bias, gating),
   pure PyTorch first; Meta's Triton kernels only if the PyTorch version is the bottleneck.
4. Argus-style ideas (next-item + feedback-prediction decomposition) only if the
   dataset has an action signal and steps 1-3 plateau.
Efficiency items worth measuring (each with a before/after number): the dense
`[P, negs, D]` negative gather (chunk or share negatives across the batch),
sparse/row-wise optimizer state for the item table (KuaiRand's 32M x 128 table
with AdamW moments was the A100 OOM), bf16 table, DataLoader workers/pinned memory.

## Deliverables (worker, rule 7)
- Branch `dev/hstu` off `staging`, commits as you go, **push the branch
  to origin after every meaningful milestone** (the pod is rented). Do not merge
  into staging; the orchestrator does.
- `docs/validation.md`: results marked **not yet validated / not citable**;
  `docs/system/checkpoints.md` (and `architecture`/`evaluation` pages if touched)
  kept in sync with the code. Raw scripts + logs + result JSON under
  `docs/artifacts/seqrec-encoder/`. Leave `docs/roadmap.md` alone.
- Chain notes in `.chains/seqrec-encoder/` on the pod at each milestone (baseline
  reproduced, each ladder step's result, decision to go to KuaiRand or not). Chains
  are local-only: never commit `.chains/`.
- Publishing a new checkpoint to the Hub: only after the orchestrator says so.
- Nesting: you may dispatch one `sonnet` subagent for web research; no other workers.
- Do not stop or delete the pod. When done (or blocked), write the final chain note
  with what passed, what was skipped, what is still unverified, and stay idle.
