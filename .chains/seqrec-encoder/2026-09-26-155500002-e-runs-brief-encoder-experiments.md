---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-26-155500001-w1-merged-dispatch-e-runs.md"
nextStep: "e-runs worker (opus): execute this brief (E0, E1, E2, decision, E3, then E4 if budget), writing one e-runs note per run."
created: "2026-09-26T12:39:12Z"
---

# E-runs brief: encoder experiments E0-E4 (dispatched as sent)

Fork from `main`: the run worker branch. Scope: GPU runs, docs/validation.md rows, artifacts. What merges back: `dev/hstu-runs` (docs and artifacts only) after the orchestrator reads it.

---


You are a constrained worker dispatched by the dev/hstu orchestrator. Read `CLAUDE.md`,
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Hard rules.**
- No subagents. Do not use Agent/Task or Workflow, and do not spawn another Claude process.
- Never print, cat, grep or dump env or secrets files (`/etc/retrieve-pod.env`, `*.env`,
  `secrets.env`), and never run `env`/`printenv`/`set`/`declare -x`. Read single variables by
  name only.
- Do not use `rp-sync`: it resets `UV_PROJECT_ENVIRONMENT`. Set up with
  `export UV_PROJECT_ENVIRONMENT=/venvs/wt-runs; uv sync --all-packages --all-groups --extra official`.
- **One GPU job at a time.** You are the only GPU user. Never run two trainings at once.
- **No code changes to `evaluation/training/` or anything else under `evaluation/`.** If a run
  exposes a bug, stop, state the mechanism with evidence, write it into your report, and wait. The
  orchestrator routes code fixes through review.
- Nothing is citable: every number is "not yet validated". No Hub uploads.
- The code wins over docs. State the mechanism of a bug before touching anything.

## Step
Run the focused experiment set that decides whether an HSTU-style encoder with softmax attention,
trained with sampled softmax, beats gSASRec at the same D on yambda-500m and goodreads-work-id.
The user's budget is about 10 GPU-hours from now. Choose the order below and do not add runs
beyond it.

## Read only
`evaluation/training/config.py` (`TrainConfig`, the field names), `docs/system/datasets.md` §
Training (the `train run FIELD=VALUE` CLI), `/scratch/briefs/W1-report.md` (Gate A/B numbers,
compile, timings), `docs/artifacts/seqrec-encoder/gate-a/gate_a.py` (reuse it for re-scoring),
and `docs/validation.md` § Seqrec encoder rewrite. Background on every earlier step is in the chain notes under
`/workspace/retrieve/.chains/seqrec-encoder/`; read only the W1 report note (`*-w1-report.md`) if you need more than this brief.

## Data
- yambda-500m: `/data/yambda-500m/trainer` (has `timestamps` in dataset-epoch seconds; the id map
  is byte-identical to the checkpoints'). Published checkpoints are in `/data/yambda-500m/checkpoints/`.
- goodreads-work-id: `/data/goodreads-work-id/trainer` (has `timestamps`, unix s; the id map is
  byte-identical to the Hub copy; test rows differ from the Hub `test.parquet` only in same-second
  tie order). Published checkpoints are in `/data/goodreads-work-id/checkpoints/`.

## Bars (test, full catalog, same test file, same eval code)
- yambda d64: the published checkpoint re-scored on `trainer/test.parquet` gives **0.0846 / 0.1563**
  (ndcg@10 / recall@100). Gate B's retrain gave 0.0837 / 0.1558. Beat the higher (published).
- yambda d128: published re-scored gives 0.0811 / 0.1486.
- goodreads d64 / d128: to be re-scored in E0 on `/data/goodreads-work-id/trainer/test.parquet`.

## Shared recipe (identical across datasets; only epochs, patience and eval cadence scale)
`loss=sampled_softmax normalize=true temperature=0.05 num_negatives=8192 inbatch_negatives=4096
batch_size=256 max_seq_length=200 learning_rate=1e-3 weight_decay=0 warmup_steps=1000 compile=true
early_stop_metric=ndcg@10 wandb_enabled=false seed=42`.
- **HSTU body:** `encoder=hstu hidden_dim=256 num_blocks=4 num_heads=4 ffn_hidden_dim=1024
  dropout=0.2 use_time=true`. If `hstu` ignores ffn_hidden_dim, say so and move on.
- **Model selection is on val, never test.** Record test only for the chosen checkpoint.

## Runs, in order (each: `TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<id>`,
`checkpoint_dir=/scratch/ckpt/<id>`, log `/scratch/logs/seqrec/<id>.log`, nohup in the
background; wake on PID exit, not on a log grep)

- **E0 (minutes):** re-score the published goodreads d64 and d128 checkpoints on
  `trainer/test.parquet` with the new eval path. These are the goodreads bars.
- **E1: yambda d64, `encoder=sasrec`** with the published body (2 blocks, 2 heads, ffn 256,
  dropout 0.5) and the shared sampled-softmax recipe. Measures the loss change alone.
  `num_epochs=100 patience=10 eval_every=2`.
  - **Collapse check:** after the first val eval, if coverage@10 < 0.01 while the Gate B run had
    0.04, stop the run and report it (the shared-negatives collapse that gBCE showed).
- **E2: yambda d64, HSTU body + shared recipe.** Same epochs and patience. Before launching,
  write the predicted epoch time in the run's `prediction.md`.
- **Decision after E1/E2** (write it down in the report): pick the higher **val** ndcg@10. If
  neither beats the yambda d64 bar on test, still run E3 with the higher of the two, and flag it.
- **E3: goodreads d64, the winner's config.** Goodreads has ~8x yambda's train users. Run
  `max_batches_per_epoch` unset and `eval_every=1`, and time the first epoch. Then set
  `num_epochs` and `patience` so the run fits within 2.5 h (restart it with the new cap if
  needed, noting that in the report).
- **E4 (only if at least 3 h of budget remain and the winner beat both d64 bars):** yambda d128 with
  the winner (`embedding_dim=128`, all else the same) against 0.0811 / 0.1486.

## Record for every run
test ndcg@10, ndcg@100, recall@10, recall@100, coverage@10; best epoch; best val ndcg@10;
epoch time (median); samples/s; peak GPU memory; GPU name; sampled sm_mhz; total wall time;
config.json; delta vs its bar. Collect them into one table.

## Branch/worktree
Worktree `/scratch/wt/runs` on branch `dev/hstu-runs` (created for you off `dev/hstu`, which
already contains the reviewed code). Commit only docs and artifacts:
- `docs/validation.md`: one table under § Seqrec encoder rewrite, all "not yet validated / not
  citable", H100.
- `docs/artifacts/seqrec-encoder/<run-id>/`: command.sh, config.json, train_metrics.json,
  eval_quality.json, and prediction.md where required. Keep these small: no checkpoints and no
  logs over ~1 MB.

## Out of scope
Code changes, KuaiRand, Hub uploads, extra ablations (no logQ, no LlamaBlock, no pointwise
attention, no hyperparameter sweeps), `docs/roadmap.md`.

## Verify commands
`python3 scripts/check_doc_links.py` (zero broken) before each commit.

## Return: chain notes, not scratch files
Your trail lives in `/workspace/retrieve/.chains/seqrec-encoder/` on chain branch **`e-runs`**
(the directory is git-excluded; write by absolute path and never commit it). Never edit an existing
note; always write a new one. Format:

```
---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "<filename of the previous e-runs note; the first one's parent is this brief's note>"
nextStep: "<one concrete action, <=1000 chars>"
created: "<UTC ISO>"
---

# <short title>
```

The filename is `$(/scratch/briefs/chain-ts)-e-<slug>.md`: a UTC+3-frame timestamp that is always
later than the newest note (this pod has no tzdata, so `TZ=` tricks silently give UTC).
- **One note per run, once it finishes:** its result row, the delta against its bar, the 2-line
  verdict, the paths to its artifacts, and anything surprising, with the mechanism if it is a
  problem.
- **One note for the E1/E2 decision.**
- **A final note:** the complete table, success yes/no against both d64 bars, what was skipped,
  and what is unverified.
- If you are blocked (a bug, a collapse, or budget), write a note whose `nextStep` names what the
  orchestrator must decide, then stay idle.

Also commit the docs/artifacts after each run on `dev/hstu-runs` (do not push). Then stay idle
at the end.
