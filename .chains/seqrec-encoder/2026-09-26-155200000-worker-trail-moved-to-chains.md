---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-155000000-gates-a-b-etl-merged.md"
nextStep: "On 2026-09-26-155500000-r1-review-report.md (branch r1-review): if CLEAN or CLEAN-AFTER-FIXES, merge dev/hstu-encoder into dev/hstu and push. Then write the e-runs brief note from /scratch/briefs/E-runs.md (parent: the merge note) and start the opus run worker on it."
created: "2026-09-26T12:38:00Z"
---

# Worker trail now lives in chains (branch per agent)

## Decision (user, 2026-09-26)
"Yes, do it, and update the previous steps to chains." Every dispatched agent gets a **chain branch**
in this chain. It holds its brief as sent (the first note, which forks from the `main` note it came
from), each later orchestrator instruction, and its report (the worker writes it itself). The
outcome (merge, reject) goes on `main`.
- **Why:** `/scratch/briefs/` is invisible to the laptop session and dies with the pod. `.chains/`
  is what the laptop pulls back.
- **Rejected:** keeping scratch briefs plus summaries on `main`. It loses the raw reports (W1's
  collapse diagnosis, W2's hashes, R2's findings).
- `/scratch/briefs/*` stays only as a working copy.

## Branches so far
| branch | notes (in order) | outcome on main |
|---|---|---|
| `w1-encoder` | 134930000 brief → 135100000 amendment 1 → 140500000 env rule → 150000000 collapse instruction → 154400000 hub.py instruction → 154800000 report | pending the R1 review |
| `w2-etl` | 134930500 brief → 140500500 env rule → 151400000 report | merged 0f1061e (155000000) |
| `r2-review` | 151900000 brief → 152100000 report (CLEAN) | merged with w2 |
| `r1-review` | 155100000 brief → 155500000 report (being written by R1) | pending |
| `e-runs` | not yet dispatched | — |

## Clock note
Filename timestamps are in the laptop's frame (UTC+3) so that the chain sorts. The `created` field
is UTC. Notes 155000000 and 155100000 are stamped ~15 min ahead of real time. They are kept as
they are (notes are never edited), and later notes sort after them.
