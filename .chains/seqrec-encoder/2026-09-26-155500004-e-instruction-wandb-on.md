---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-26-155500003-e-e0-goodreads-bars.md"
nextStep: "e-runs: launch E2 and every later run with wandb on (project seqrec-encoder, run name = run id); leave E1 as it is."
created: "2026-09-26T12:50:13Z"
---

# E-runs instruction: W&B logging on from E2

The user approved W&B logging (2026-09-26). This overrides the brief's `wandb_enabled=false`
for every run launched from now on:
- Pass `wandb_enabled=true wandb_project=seqrec-encoder wandb_run_name=<run id>` (for example
  `e2-yambda-d64-hstu-ssm`).
- E1 is already running: do not restart it. It stays local-only.
- Record the W&B run URL in each run's e-runs note and in its artifacts README.
- **Never print the API key.** It is already in the environment as `WANDB_API_KEY`; do not
  echo it, and do not run `wandb login` with it on the command line.
- If `wandb.init` fails (network, auth), do not retry in a loop and do not print env files. Relaunch
  with wandb off, note it, and carry on.
