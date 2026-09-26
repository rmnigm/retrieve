---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-133927000-architecture-plan-encoder-ladder.md"
nextStep: "Pod orchestrator (Opus 5.5, herdr pane on rp-h100-hstu): read this note, its two parents and the research note; create dev/hstu off origin/staging; dispatch the implementation worker (fable) in a herdr pane on a worktree; after every code change dispatch a review agent (deslop + python-review skills); then run the gates and the ladder through worker agents, one GPU job at a time."
created: "2026-09-26T10:45:00Z"
---

# Dispatch: pod planner-orchestrator for dev/hstu

You are the **planner-orchestrator** for this line of work, a Claude Code session
(Opus 5.5) in a herdr pane on the H100 pod `rp-h100-hstu`. The laptop session that
set up this pod is the user's top-level session; it does not run jobs on this pod.
**You control everything that happens on this pod** — data prep, training,
evaluation, docs — but through agents you dispatch, not by writing the code yourself.

## Read first (in order)
1. `CLAUDE.md` (hard rules), `docs/contracts/agent-orchestration.md`,
   `docs/contracts/coding-guidelines.md`.
2. `.chains/seqrec-encoder/2026-09-26-132708000-dispatch-h100-hstu-and-beyond.md` —
   the goal, the success bar, the baseline numbers, constraints on the model.
   Where it says "worker", read "the agents you dispatch".
3. `.chains/seqrec-encoder/2026-09-26-133927000-architecture-plan-encoder-ladder.md` —
   the architecture plan (from a read-only fable planner). You own it now: verify
   against the code on `origin/staging`, amend where wrong.
4. `.chains/seqrec-encoder/*research*` — web research on HSTU, Argus and
   follow-ups (appended by the laptop session). Reconcile it with the plan before
   dispatching; if it argues for a different ladder order or loss, decide and write
   down why in your first chain note.
5. `.chains/hstu-and-distributed-training/` and `.chains/e4-kuairand/` for context.

## What the user asked for (their words, condensed)
Substitute gSASRec with HSTU and more efficient training; do not limit yourself to
HSTU (Argus or other follow-ups). A good, embedding-table-based encoder over
interaction history that works across datasets with minimal hand tuning of
features: generic, better and bigger than gSASRec. Goodreads and yambda first;
only on success, KuaiRand. Build on a new branch `dev/hstu` off `staging`, **not
on staging**: all work merges into `dev/hstu` and is pushed there; merging to
`staging` is the user's call later.

## Pod state (set up by the laptop session)
- 1x H100 80GB HBM3, 64 vCPU, ~2 TB RAM, 500 GB container disk. Checkout
  `/workspace/retrieve` on `staging` (= origin/staging ae0290d). gh, HF, Claude logged in.
- `.chains/` synced from the laptop (86 notes) and in `.git/info/exclude`: never commit it.
- Skills installed from the user's agent-skills repo into `~/.claude/skills`
  (deslop, python-review, validation-review, chains, diagnose, challenge-plan, ...).
- Data on `/data`: `goodreads-work-id` and `yambda-500m` eval inputs + published
  gSASRec checkpoints d64/d128/d256 (`eval-data fetch ... --include-checkpoints`).
  `/data/yambda-500m/trainer/` has `train/val/test.parquet` + `item_id_map.json`
  from `eval-data yambda prep --variant 500m` (no timestamps column; hash-check vs
  the Hub `item_id_map.json` before trusting it). Goodreads trainer inputs do not
  exist yet: `goodreads all` then `prep` (`~/datasets -> /data/_raw` symlink is
  in place; a partial download may sit in `/data/_raw/goodreads-ucsd/raw`, resumable).
  KuaiRand not fetched.
- Logs of the laptop's setup commands: `/scratch/logs/`.

## How you run the work
- **Dispatch through herdr** (`herdr --skill` for the CLI; you run inside herdr so
  `HERDR_ENV=1`): `herdr pane split --current --direction right|down --no-focus`,
  then `herdr agent start <name> --kind claude --pane <id> -- --model <model>`,
  then `herdr agent prompt <name> "<brief>"`. Give each agent a self-contained brief
  (paste the chain sections it needs under `Chain context:`); agents do not see chains.
- **Models (rule 7):** the core training-module rewrite (plan steps 1-7) = one
  `fable` worker as one big chunk. Runs, monitoring, ETL, debugging, docs = `opus`.
  At most three agents at once; one GPU job at a time.
- **Isolation:** each code-writing agent works in its own git worktree under
  `/scratch/wt/<name>` on a branch off `dev/hstu`; you merge into `dev/hstu` and
  push to origin after each accepted change (the box is rented).
- **Review gate after every code change:** before merging any agent's code into
  `dev/hstu`, dispatch a separate review agent (`opus`) that runs the `deslop`
  skill and then the `python-review` skill on that diff, applies the fixes (or
  hands findings back to the author), and runs `ruff check` / `ruff format --check`
  and the harness suite (`cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q`).
  No merge without a clean review.
- **Docs:** agents update `docs/validation.md` (all results "not yet validated /
  not citable"), `docs/system/checkpoints.md` / `datasets.md` / `evaluation.md` as
  the code changes, and put scripts/logs/JSON under `docs/artifacts/seqrec-encoder/`.
  Leave `docs/roadmap.md` alone. Link checker at zero.
- **Chains:** write a note in `.chains/seqrec-encoder/` at each milestone (plan
  decided, implementation merged, Gate A, Gate B, each ladder result, go/no-go on
  KuaiRand, final). The laptop session pulls `.chains/` back from this pod; keep
  notes handoff-quality, `nextStep` concrete.
- **Hub publishing:** do not publish checkpoints/datasets to the Hub without the user.
- **Pod lifecycle:** never stop or delete the pod.

## Done means
Goodreads and yambda-500m results for the ladder with the winner vs gSASRec at the
same D (NDCG@10 and R@100 on test, full catalog, shared recipe), KuaiRand run if
and only if the winner beats gSASRec on both, all merged and pushed on `dev/hstu`,
docs current, and a final chain note stating what passed, what was skipped and
what is still unverified. Then stay idle in your pane.
