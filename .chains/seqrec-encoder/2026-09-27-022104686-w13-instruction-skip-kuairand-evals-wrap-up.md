---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-022012083-w13-instruction-drop-kuairand-ckpt.md"
nextStep: "w13: no KuaiRand uploads at all; finish the approved deletes and squashes without polling; minimal docs; commit; short report; stop."
created: "2026-09-26T23:21:04Z"
---

# W13 instruction: skip the KuaiRand eval inputs too, and wrap up fast (user)

User, 2026-09-27: "same with eval inputs tbh, we can skip them. just finish faster please".
- **No KuaiRand uploads at all.** No polling, no retries. The KuaiRand eval inputs are re-derivable with
  `eval-data kuairand all` (about 20 min).
- Finish the approved deletes and squashes you are in the middle of. Each squash is one API call; **skip the
  storage polling**, and just record one `used_storage` reading at the end.
- **Docs, minimal, on dev/hstu-hub:**
  - datasets.md § kuairand: the eval inputs and the checkpoint are not on the Hub (user choice); rebuild with
    `eval-data kuairand all`, retrain with the artifact `command.sh`;
  - checkpoints.md: no KuaiRand row.
  - Link checker at zero, then commit.
- **Short report note**, `$(/scratch/briefs/chain-ts)-w13-cleanup-report.md`: done or blocked per action, commit ids,
  and blocked calls verbatim. Then stop.
