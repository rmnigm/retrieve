---
chain: "seqrec-encoder"
branch: "w9-ckpt"
parent: "2026-09-26-220149667-e-k64-probe.md"
nextStep: "W9 (opus, agent w9-ckpt, worktree /scratch/wt/ckpt, branch dev/hstu-ckpt, CPU only): add a resume-checkpoint cadence option; report as a w9 chain note."
created: "2026-09-26T19:02:32Z"
---

# W9 brief: resume-checkpoint cadence (KuaiRand per-epoch I/O) (dispatched as sent)

Fork from the KuaiRand d64 probe note (`2026-09-26-220149667-e-k64-probe.md`).
- **The finding:** the trainer writes `_resume.pt` (model + AdamW, 49.2 GB at KuaiRand d64) every epoch,
  which takes ~45 s, plus 15 s per new best. That is ~25% of each ~210 s epoch.
- **Why now:** KuaiRand d128 starts in ~4.5 h and would pay the same cost.
- **Scope:** `evaluation/training/{config,train}.py`, `tests/training/`, `docs/system/datasets.md` § Training.

You are a constrained worker. Read `CLAUDE.md` (rule 9: never print env or secrets files) and
`docs/contracts/coding-guidelines.md`.
- **No subagents. No GPU** (`CUDA_VISIBLE_DEVICES=""`; KuaiRand d64 is training).
- Venv: `export UV_PROJECT_ENVIRONMENT=/venvs/wt-ckpt; uv sync --all-packages --all-groups --extra official`.
  **Not rp-sync.**

## Step
1. State the mechanism first: where `_resume.pt` and `best_model.pt` are written (`train.py` ~166 / ~303), and what
   each save costs in bytes.
2. Add `TrainConfig.resume_every: int` (epochs between `_resume.pt` writes).
   - The default keeps today's behaviour (1), so existing recipes are unchanged.
   - The **final** epoch, including the early-stop exit, always writes it, so a finished run stays resumable.
   - `best_model.pt` is untouched: it is the product.
3. If one cheap, obvious win remains in the same save path, include it and measure it by arithmetic. One
   example is avoiding a redundant `.cpu()` copy or a double serialization. No async or threaded saving:
   that is a new mechanism, out of scope.
4. **Tests:** one parametrized test that pins which epochs write `_resume.pt` for a given `resume_every` and an
   early stop. Use a tiny CPU run or the function in isolation, whichever is thinner. It must fail on an
   off-by-one mutation.
5. **Docs:** datasets.md § Training documents the option and its trade-off (a crash loses up to N-1 epochs).

## Verify
`ruff check retrieve evaluation`, `ruff format --check evaluation/training`,
`cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q`, and `python3 scripts/check_doc_links.py`
(repo docs at zero; notes under `.chains/` are not repo docs) must all be clean. Commit on dev/hstu-ckpt;
do not push or merge.

## Out of scope
Async saves, changing what `_resume.pt` contains, and any other performance work.

## Return
Write the chain note `/workspace/retrieve/.chains/seqrec-encoder/$(/scratch/briefs/chain-ts)-w9-report.md`:
- frontmatter: branch `w9-ckpt`, parent = this brief, `nextStep` for the orchestrator, `created` in UTC ISO;
- body: the mechanism, the change, the mutation check, the verify outputs, and the recommended
  `resume_every` for KuaiRand d128.

Then stay idle.
