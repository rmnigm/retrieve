---
chain: "seqrec-encoder"
branch: "w2-etl"
parent: "2026-09-26-134900000-user-amendment-opus-only-two-agents.md"
nextStep: "W2 (opus, herdr agent w2-etl, pane w1:p4, worktree /scratch/wt/etl, branch dev/hstu-etl, CPU only): implement this brief and hand back with a report."
created: "2026-09-26T10:49:30Z"
---

# W2 brief: timestamps in trainer inputs + goodreads/yambda prep (dispatched as sent)

Fork from `main`: the worker branch for W2. Scope: the ETL files, their tests, the dataset sections of datasets.md, and the data under /data. What merges back: `dev/hstu-etl` after review.

**Note:** the brief's `rp-sync` instruction was wrong. It resets `UV_PROJECT_ENVIRONMENT` from the pod env file; see the report.

---

You are a constrained worker dispatched by the dev/hstu orchestrator (another Claude
session in a herdr pane on this pod). Read `CLAUDE.md` (hard rules),
`docs/contracts/coding-guidelines.md` and `docs/contracts/agent-orchestration.md` §5 first.

**Standing lines.** The code wins over any doc, note or memory: if the wiki and the code
disagree, trust the code and fix the doc. State the mechanism of a bug before fixing it.

**No subagents.** Do not use the Agent/Task tool, Workflow, or spawn any other agent or
Claude process, not even for research. Do all of the work yourself in this session.

**No GPU.** Everything here is CPU. Run with `CUDA_VISIBLE_DEVICES=""`. Another worker
holds the GPU.

## Step
1. Make the three trainer-input ETLs (`evaluation/eval_datasets/etl/goodreads.py`,
   `yambda.py`, `kuairand.py`) keep a `timestamps` column (list[int64], **unix seconds**,
   same length and order as `item_ids`) in `train/val/test.parquet`. Today each ETL
   aggregates `timestamp` next to `item_id` and lists are gathered in lockstep by `timesplit`,
   then dropped at the final `select` (goodreads.py ~582-624, yambda.py ~139-178, kuairand
   `cmd_prep`). Convert units in the ETL if the raw unit is not seconds, and say so in
   datasets.md. For val/test the `timestamps` column covers the input history, not `targets`.
   `item_ids`/`targets` and `item_id_map.json` must not change.
2. Goodreads trainer inputs: `eval-data goodreads` download + convert + `prep` (see
   `--help`; `~/datasets -> /data/_raw` symlink exists; a 48 MB partial
   `/data/_raw/goodreads-ucsd/raw/goodreads_books.json.gz` may resume) to
   **`/data/goodreads-work-id/trainer/`**. Memory: this container's cgroup limit is
   **~251 GB** (not the 2 TB `free` shows); the ETL notes a ~129 GB peak — watch it.
3. Yambda-500m re-prep with timestamps: `eval-data yambda prep --variant 500m` to
   **`/data/yambda-500m/trainer.new/`** (do NOT touch `/data/yambda-500m/trainer/`, the
   other worker is training from it).

## Gates (bit-exact)
- **G-yambda:** `trainer.new/item_id_map.json` byte-identical (sha256) to
  `trainer/item_id_map.json` **and** to the Hub copy `/data/yambda-500m/item_id_map.json`;
  `item_ids` (and `targets` for val/test) columns of each of train/val/test in `trainer.new`
  equal to `trainer` row for row (compare with polars/pyarrow `equals`, record hashes).
  If `trainer/item_id_map.json` differs from the Hub copy, stop and report: that is a finding.
- **G-goodreads:** `/data/goodreads-work-id/trainer/item_id_map.json` sha256-equal to the Hub
  copy `/data/goodreads-work-id/item_id_map.json`, and the trainer `test.parquet` user
  histories/targets consistent with the Hub's `/data/goodreads-work-id/test.parquet` (read its
  schema first; state exactly what you compared). Mismatch = report with the mechanism.
- `timestamps` present, int64, same list lengths as `item_ids`, non-decreasing within a row
  (report the count of violations rather than "fixing" them).
- KuaiRand: code change only, no data run (raw not fetched). Existing
  `tests/eval_datasets/test_kuairand.py` stays green; extend a test only if one already pins
  the output schema.

## Read only
`evaluation/eval_datasets/etl/{goodreads,yambda,kuairand}.py`, `evaluation/eval_datasets/timesplit.py`
(or wherever `timesplit` lives), `evaluation/tests/eval_datasets/`, `docs/system/datasets.md`
§ yambda, § goodreads, § kuairand, `docs/validation.md`.

## Branch/worktree
Worktree `/scratch/wt/etl` on branch `dev/hstu-etl` (off `dev/hstu`). Own venv:
`export UV_PROJECT_ENVIRONMENT=/venvs/wt-etl; rp-sync /scratch/wt/etl`, keep it exported.
Commit on your branch; do not push or merge. Long jobs: `nohup ... > /scratch/logs/seqrec/<job>.log 2>&1 &`,
then poll.

## Model + concurrency
opus, CPU only. Concurrent worker W1 (GPU) owns `evaluation/training/`,
`evaluation/eval_datasets/hub.py`, `evaluation/tests/training/`, `docs/system/checkpoints.md`
and `docs/system/datasets.md` § Training. Do not edit those. You own the ETL files, their tests,
and the dataset sections of datasets.md.

## Out of scope
Training code, Hub publishing (never upload anything), KuaiRand download, `docs/roadmap.md`,
renaming/moving `/data/yambda-500m/trainer*` (the orchestrator swaps `trainer.new` in later),
the evaluation-cleanup backlog (e.g. goodreads `except Exception` in `cmd_download`) unless it
actually breaks your run — then report it with the mechanism before changing it.

## Verify commands
```
cd /scratch/wt/etl
ruff check retrieve evaluation
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
python3 scripts/check_doc_links.py   # from repo root, zero broken
```

## Return
- Branch `dev/hstu-etl` with commits (message ends with the Co-Authored-By line from your
  harness attribution instructions).
- `docs/validation.md`: G-yambda and G-goodreads rows (hashes, row counts, pass/fail),
  marked not citable (data gates, not results). `docs/system/datasets.md` dataset sections
  describe the `timestamps` column and unit.
- `docs/artifacts/seqrec-encoder/etl-timestamps/`: the comparison script and its small JSON
  output (hashes, counts, prep wall time, peak RSS). Keep it small.
- A final message in this pane: what passed, what was skipped, what is unverified, the exact
  data paths produced. Then stop and stay idle.
