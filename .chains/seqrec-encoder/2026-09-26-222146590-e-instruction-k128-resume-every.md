---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-26-221922038-e-e0c-and-k64-restart.md"
nextStep: "e-runs: after k64 ends and its note is written, merge dev/hstu (7de3518) into dev/hstu-runs, re-sync the venv, then the k128 probe and R-k128 with resume_every=5."
created: "2026-09-26T19:21:46Z"
---

# E-runs instruction: pick up resume_every before k128

`resume_every` is merged into dev/hstu (7de3518; R10 CLEAN-AFTER-FIXES). It also fixes two crash-safety bugs in snapshot
retention.
- **Do not touch the running k64.**
- After k64 ends: `git merge dev/hstu` into dev/hstu-runs in /scratch/wt/runs (code only for you), then
  `UV_PROJECT_ENVIRONMENT=/venvs/wt-runs uv sync --all-packages --all-groups --extra official`.
- Run the k128 probe (one table), then R-k128 with `resume_every=5`. Size num_epochs from the probe, at about
  5 h, and remember that the checkpoint time is now ~1/5 per epoch on average.
- **Disk:** before k128, delete k64's `_resume.pt` and any superseded snapshots under its checkpoint dir, keeping
  `best_model.pt` and the eval files. Check `df /data` first.
