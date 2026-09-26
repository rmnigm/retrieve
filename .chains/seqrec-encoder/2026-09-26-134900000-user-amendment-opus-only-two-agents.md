---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-134718133-reconciled-plan-dev-hstu.md"
nextStep: "Push dev/hstu, dispatch W1 (opus, encoder rewrite) and W2 (opus, ETL timestamps + prep) as the two herdr agents; the first to finish frees its slot for the review agent of its diff."
created: "2026-09-26T10:49:00Z"
---

# User amendment: opus only, two herdr agents max, no subagents

User, mid-session 2026-09-26: "opus for core also, no fable at all. and no more
than two herdr agents, and restrict them [from launching] their subagents."

Supersedes in the reconciled plan (§4) and the dispatch note:
- **Every agent is `opus`**, including W1 (core training-module rewrite). No fable.
- **At most two herdr agents alive at once** (was three). Reviewer counts: a review
  runs only in a freed slot, i.e. after W1 or W2 exits.
- **Agents must not spawn subagents** (no Agent/Task tool, no sonnet research child,
  no Workflow). Every brief says so explicitly.
Schedule: W1 + W2 concurrently -> whichever finishes first is closed and its slot
starts R(that diff) -> merge -> next. Ladder run workers also count against the two.
