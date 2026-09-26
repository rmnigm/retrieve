---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-155200000-worker-trail-moved-to-chains.md"
nextStep: "Run worker e-runs (opus, pane in /scratch/wt/runs, branch dev/hstu-runs) executes E0-E4 from its brief note; the orchestrator reviews each e-runs note, and routes any code fix through a fix agent plus review."
created: "2026-09-26T12:39:12Z"
---

# dev/hstu-encoder merged; wiki pass pushed; E runs dispatched

- **Merged:** dev/hstu-encoder into dev/hstu at 9c11265 (R1 verdict CLEAN-AFTER-FIXES, 3e8102b).
  The validation.md conflict with the ETL section was resolved by keeping both.
- **Wiki pass:** 02dccfd.
  - AGENTS rule 9: no secret dumps.
  - Contract: the chain trail, the pod override (opus only, two agents, no subagents), the review
    gate, and dev/hstu as the landing branch.
  - storage.md: the rp-sync venv trap and the cgroup memory limit.
  - decisions.md: the sequential-encoder decisions.
  - Pushed.
- **User correction:** "dispatch agents in the future for the work, once again - you are the
  orchestrator". I did that wiki pass and the conflict fix myself. From now on, all docs, code and
  conflict work goes to agents (saved to memory).
- **R1 open follow-ups (not scheduled):**
  1. `--resume` without `_resume.pt` silently starts fresh.
  2. A resumed run's train_metrics.json covers only the resumed epochs.
- **Data:** `/data/yambda-500m/trainer` is now the re-prep with timestamps (`trainer.old` kept).
- **Timestamp helper:** `/scratch/briefs/chain-ts`. The pod has no tzdata, so `TZ=Etc/GMT-3`
  gives UTC.
