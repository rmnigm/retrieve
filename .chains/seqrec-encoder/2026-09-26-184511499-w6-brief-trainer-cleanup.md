---
chain: "seqrec-encoder"
branch: "w6-cleanup"
parent: "2026-09-26-182613777-user-keep-gbce.md"
nextStep: "W6 (opus, agent w6-cleanup, worktree /scratch/wt/cleanup, branch dev/hstu-cleanup): delete HSTU and use_time, keep both losses, E1c defaults; bit-identical equivalence and Gate A; report as a w6 chain note."
created: "2026-09-26T15:45:11Z"
---

# W6 brief: trainer cleanup (delete HSTU, keep gBCE and logQ sampled softmax, E1c recipe as defaults) (dispatched as sent)

Fork from the user's keep-gBCE note on `main`. Scope: `evaluation/training/`, its tests and the named docs. What merges back: `dev/hstu-cleanup` after review.

You are a constrained worker. Read `CLAUDE.md` (rule 8 thin code, rule 9 no secret dumps) and
`docs/contracts/coding-guidelines.md` (D2 no self-compat, D4, D5, D6).
- **No subagents.**
- **The GPU only for the equivalence check below**, and only when `nvidia-smi` shows no other training
  process. The run worker's jobs run from `/scratch/wt/runs`. If the GPU is busy, do the CPU part, write a
  note that names the GPU step as pending, and wait for the orchestrator's go.
- Venv: `export UV_PROJECT_ENVIRONMENT=/venvs/wt-cleanup; uv sync --all-packages --all-groups --extra official`
  in `/scratch/wt/cleanup`. **Not rp-sync.**

## Step
The user: "drop everything except new loss and perf optimizations and better trainer. hstu is not
needed." Read the decision notes on `main`:
- `2026-09-26-182046365-user-drop-hstu-cleanup-plan.md`
- `2026-09-26-181840268-user-final-scope-e1c-recipe.md`

In `evaluation/training/`:
1. **Delete:**
   - `HSTUBlock`, `RelativeBias`;
   - `use_time`, `time_buckets`, the time-gap embedding and the timestamp plumbing in model, dataset,
     evaluate and encode (the parquet column may exist; the trainer simply does not read it);
   - `hidden_dim`, `in_proj`, `out_proj`;
   - the `encoder` config field and `BLOCKS`: one block type, the existing `SASRecBlock` math, unchanged.
   **Keep both losses** (user, 2026-09-26: "don't drop the old loss though, two options: gbce or ssm logq"):
   `loss: gbce | sampled_softmax`, with gBCE's per-position negatives and `gbce_t`, and sampled softmax with
   logQ and `normalize`. The R1/W3 config rejections stay (normalize and logq only with sampled_softmax).
   No alias, no shim.
2. **`TrainConfig` defaults = the E1c recipe:**
   - body: embedding_dim 64, num_blocks 2, num_heads 2, ffn_hidden_dim 256, dropout 0.5;
   - normalize True, temperature 0.05, num_negatives 8192, inbatch_negatives 4096, logq True;
   - warmup_steps 1000, compile True, early stop on ndcg@10;
   - lr 1e-3, B 256, L 200.
   Take the exact values from `docs/artifacts/seqrec-encoder/e1c-yambda-d64-sasrec-ssm-logq/` (its
   config.json / command.sh); the artifact wins over this list.
3. **Published checkpoints keep loading:** `load_model_for_eval` with the legacy key rename,
   `D128_DROP05_DEFAULTS`, `TrainConfig.load` dropping unknown keys from old config.json files
   (`negs_per_pos`, `encoder`, `hidden_dim`, `use_time`, `time_buckets`, and so on; `loss` and `gbce_t` stay valid), and `normalize` False for them. Hub
   `EPOCH_SNAPSHOT_PATTERN` stays valid.
4. **Tests:** delete the HSTU cases; keep the gBCE ones. Keep the mask, sampled-softmax/logQ and logq_correction
   gates, and the config rejections that still apply. No new scaffolding.
5. **Docs:** in `docs/system/datasets.md` § Training and `docs/system/checkpoints.md`, remove HSTU/use_time
   and describe the two losses, with sampled softmax + logQ as the default. In `docs/decisions.md` § Sequential encoder: the encoder is the gSASRec
   body; two losses, gBCE (the published baseline) and logQ sampled softmax (the default, user decision after
   E1c); HSTU was tried (E2a/E2c) and dropped. Link to the validation rows; do not restate the numbers. In `docs/validation.md`, gate rows for
   deleted features become one line ("removed with the code") where they no longer describe code that exists.
   Grep `AGENTS.md`, `README.md`, `docs/` and `retrieve/docs/` for `hstu`, `use_time` and
   `encoder=` and fix every copy (rule 4). Also in `datasets.md` § kuairand, the sentence "The staged copy under
   `data/kuairand` predates the column: re-run `prep` to get it" is stale: it was re-prepped with timestamps on this pod
   (`2026-09-26-184409449-w5-report.md`), so drop it. Leave the artifacts dirs and the recorded results as they are.

## Gates (proof for a refactor, contract §5)
- **Equivalence (GPU, short):**
  - Before the change, on `dev/hstu` code, run the E1c recipe on yambda-500m d64 for
    `max_batches_per_epoch=50 num_epochs=1` with the seed fixed and `compile=false`. Record the
    per-step losses.
  - After the change, run the same with the new defaults.
  - The per-step losses must be **bit-identical** (`==` on the logged floats, or dump the tensors). If
    they differ, state the mechanism; do not loosen the check.
- **gBCE unchanged:** the same 50-step check with `loss=gbce` (the Gate B recipe) must also be
  bit-identical before and after.
- **Gate A:** the published goodreads d64 and yambda d64 checkpoints score the same 4-decimal test
  numbers through the new loader as before (reuse `docs/artifacts/seqrec-encoder/gate-a/gate_a.py`).
- `ruff check retrieve evaluation`, `ruff format --check evaluation/training`, the evaluation suite
  (`CUDA_VISIBLE_DEVICES=""`), and the link checker: all clean.

## Branch/worktree
`/scratch/wt/cleanup`, branch `dev/hstu-cleanup` off `dev/hstu`. Commit; do not push or merge.

## Out of scope
Performance and memory work (W7 comes next), the ETL timestamps column, KuaiRand, and anything outside
`evaluation/training`, its tests and the named docs.

## Return
- A chain note `/workspace/retrieve/.chains/seqrec-encoder/$(/scratch/briefs/chain-ts)-w6-report.md`:
  - frontmatter: branch `w6-cleanup`, parent = this brief, `nextStep` for the orchestrator, `created` in UTC ISO;
  - body: lines deleted/added, the equivalence and Gate A results, the verify outputs, and anything unverified.
- Then stay idle.
