---
chain: "seqrec-encoder"
branch: "e-runs"
parent: "2026-09-27-013809978-e-blocked-k128-oom.md"
nextStep: "e-runs: no more runs. Write the k128 OOM note (mechanism, peak, traceback excerpt) with a small artifact dir, commit, push dev/hstu-runs, then a final e-runs note listing every run and its artifact dir. Then stay idle."
created: "2026-09-26T22:39:19Z"
---

# E-runs: stop; k128 is not run (user decision)

`2026-09-27-013919889-user-stop-kuairand-at-d64.md` on main. Make sure nothing is left on the GPU. Record the k128 probe OOM in
`docs/artifacts/seqrec-encoder/k128-probe-oom/`: the command, the log tail with the traceback excerpt, and the
arithmetic. Artifacts only, not validation.md. Commit and push dev/hstu-runs. Then write a final e-runs note (its
`nextStep`: none, done) listing each run id, its artifact dir and W&B id, then stay idle. Clean the KuaiRand
`_resume.pt` files and superseded snapshots from /data or /scratch/ckpt, keeping every `best_model.pt` and eval file.
