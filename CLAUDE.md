# CLAUDE.md — how to work in this repository

Read this, then `docs/plans/00-roadmap.md` (the master plan: the one ordered
work queue, with checkboxes, gates and dependencies), then only the plan
section your step names. Do not start work that the roadmap does not list
or that sits behind an unchecked dependency.

## What this is

`retrieve/` is a GPU retrieval library (PyTorch + Triton; on PyPI as
`torchretrieve`, imports as `retrieve`) that reimplements two industry
papers — SilverTorch (Meta, `articles/silvertorch.md`) and LiNR (LinkedIn,
`articles/linr.md`) — from their text. `evaluation/` is the benchmark and
training harness that produces the numbers for a master's thesis
(its sources are kept out of this repository) and, now, a reproducibility paper (roadmap §0). It is a uv
workspace with one `.venv` at the root; `README.md` has the layout and the
install commands.

Naming that must stay straight in code, docs and prose: our `SilverTorch`
layer is *SilverTorch Algorithm 1 reimplemented* (in the thesis it was
called QuantizedIVF — retired 2026-09-15 in the thesis and `docs/paper/`;
the already-delivered defense talk still says it and stays that way); `meta-recsys/silvertorch`
is Meta's *official* code and, once integrated, `backend="official"`; the
LiNR variants are V1–V4 as in the paper.

## Where things are documented

- `docs/system/*.md` — **how the code works today**, kept in sync with the
  code in the same commit. `architecture.md` (module map, backend dispatch),
  `kernels.md`, `filtering.md`, `testing.md`, `evaluation.md` (harness),
  `datasets.md`, `checkpoints.md`. If code and a system doc disagree, the
  code is right and the doc is a bug — fix the doc.
- `docs/plans/*.md` — **why and in what order**. `00-roadmap.md` is the
  master plan; the others are one-phase detail. Two contracts order
  nothing and apply everywhere: `agent-orchestration.md` (who runs a step,
  on which model, how many at once, where the output lands) and
  `coding-guidelines.md` (what the code should look like, and what is
  deliberately not the goal). Plans carry a status
  blockquote (date, branch, what ran), numbered sections, decisions
  `D1…`, work packages with gates, and — once executed — a validation
  record appended to the same file. The model record is
  `docs/plans/archive/cuda-silvertorch-handoff.md` §13. Finished plans move to
  `docs/plans/archive/` (not maintained, links may rot).
- `docs/plans/*-artifacts/` — raw scripts and outputs behind a plan's
  numbers, kept so they can be re-derived.
- `retrieve/docs/` — the library user guide that ships in the sdist.
- `docs/paper/` — the reproducibility paper's own sections, each row or claim
  citing the plan section that measured it:
  [`reproduction-deviations.md`](docs/paper/reproduction-deviations.md) (paper
  vs us, Meta's code vs Meta's paper, defects reproducing found in our own
  code, and the unexplained residuals) and
  [`provenance-and-disclosure.md`](docs/paper/provenance-and-disclosure.md)
  (hardware, software, what may be cited, what was and was not compared) and
  [`official-vs-reimplementation.md`](docs/paper/official-vs-reimplementation.md)
  (Meta's kernels against ours, from B3: end to end, kernel-only, phase 2,
  parity, memory, and what the comparison cannot say).
- `articles/` — pandoc renderings of the papers. Frozen; cite, never edit.
- `evaluation/results/` — staged campaign outputs (`<name>.json` +
  `.yaml` + `.perkernel/`).

## Hard rules

1. **One GPU, and it is the machine you are on.** Since 2026-09-06 the
   working environment is the A100 box itself (A100-SXM4-80GB, torch
   2.10.0+cu128, triton 3.6.0, nvcc 12.4, Python 3.11). Mac support was
   dropped that day: GPU environments are the only target. Do not build
   CPU emulators, shim headers, or simulations of device code, and do not
   add test scaffolding beyond the existing pytest suites — the user
   rejected that explicitly. Every GPU claim is validated on this box:
   SM clocks **cannot be locked** in this container (`nvidia-smi -lgc`
   is denied, no sudo) — record the sampled `sm_mhz` and the `unstable`
   flag instead, as `docs/plans/evaluation-harness-v2.md` §7 prescribes;
   `ncu` is blocked, use `torch.profiler`. The GPU is shared: serialize
   GPU work, one job at a time. Disk: `/workspace` is a 100 GB quota
   volume; put venvs under `/venvs/` on the local disk.
2. **Nothing is citable until its gate passed.** Harness numbers from a
   branch are not paper material until the roadmap's golden gate for that
   harness is green. Say "not yet validated" rather than quoting them.
3. **Correctness gates are bit-exact where the plan says so** (`torch.equal`
   on scores; ids equal up to ties). Do not loosen a tolerance to make a
   test pass; report the mismatch.
4. **Docs move with code.** A change to behaviour updates the matching
   `docs/system` file in the same commit. Relative markdown links are
   checked by `python3 scripts/check_doc_links.py` (also a pre-commit
   hook); keep it at zero broken links.
5. **Deletion is gated.** Never delete a kernel that still has no
   validated replacement. The model: the two hand-written SilverTorch
   backends went at roadmap B4 only after the official backend's parity
   gate (B2) was green; tag `cuda-cute-backends-final` holds them.
6. **Commits and branches.** Commit only when the user asks. One branch per
   roadmap phase off `main` after Phase A4. Session artifacts (scripts,
   raw JSON) go under `docs/plans/<plan>-artifacts/`, not in the packages.
   **All work ends up on `main` at `origin`** — whatever branch or
   worktree produced it, a step is not finished until it is merged into
   `main` and pushed (`docs/plans/agent-orchestration.md` §6). The
   box is rented; the repository is the only durable artifact.
7. **One orchestrator, constrained workers.** One user-controlled
   orchestrator session dispatches workers that do code, tests,
   evaluations or docs; workers do not widen their own scope, start a step
   the roadmap does not list, or dispatch peers. **At most three workers at
   once**, lowered by GPU serialization (rule 1), plan-level exclusivity and
   the requirement that concurrent workers edit disjoint trees.
   **Model by the shape of the work:** core library rewrites with heavy
   kernel or coding work, and core harness rewrites carrying architecture
   design, go to **`fable`** as one big chunk; routine cleanups, monitoring,
   debugging, docs and one-time experiments go to **`opus`**. **Nesting is
   two levels**, with exactly one exception: a **`sonnet`** nested subagent
   for web deep research. Every worker's result is a validation record
   appended to the plan it executed — a run recorded only in a transcript
   did not happen. Full contract:
   [`docs/plans/agent-orchestration.md`](docs/plans/agent-orchestration.md).
8. **Thin code, documented outside it.** Priorities in order: correct code,
   clean architecture, evals that run. Little defensive programming —
   validate at the boundary, then let it fail loudly. No backward
   compatibility with shapes we invented: redo the part and delete the old
   one (this does not weaken rule 5 — a *kernel* still goes only after its
   replacement's parity gate). No abstraction level without a written
   reason. **No multi-line comment slop — prefer no comment at all**; a
   short one only where the code is genuinely surprising, and it says why.
   The documentation is `docs/system` (what the code does today) and
   `docs/plans` (why, and what was measured), kept current. Tests are the
   gates a plan names, not a deliverable. Not the goal: coverage, cosmetic
   polish, unscheduled baselines, paper speculation. Full contract:
   [`docs/plans/coding-guidelines.md`](docs/plans/coding-guidelines.md).

## Commands

```bash
uv sync --extra official --all-packages          # whole workspace + Meta's ops; --all-packages keeps pytest (it is in the members' dev groups)
uv run --directory retrieve pytest tests/ -x -q  # library suite — GPU only, skips itself without CUDA
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q  # the harness suite (bench / training / eval_datasets): CPU-only by construction, three tests assert it, so hide the GPU
uv run --directory evaluation bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch
uv run --directory evaluation bench campaign --suite filter --resume
uv run --directory evaluation bench check --dataset goodreads     # validate a staged dataset's layout
uv run --directory evaluation eval-data arxiv --help              # dataset ETL + Hub transfer (eval-data fetch|publish)
uv run --directory evaluation train sasrec --help                 # gSASRec training; train upload-checkpoint
uv run --directory retrieve tune-kernels --help  # kernel autotune sweeps (GPU)
ruff check retrieve evaluation && ruff format --check retrieve
python3 scripts/check_doc_links.py
```

`docs/system/testing.md` explains the suite layout and fixtures;
`docs/system/evaluation.md` the harness CLI and config format.

## When you finish a step

Append the validation record to the plan you executed, flip the roadmap
checkbox with date and commit, update the affected `docs/system` file,
run the link checker, and state plainly in your report what passed, what
was skipped, and what is still unverified.

If you are a **worker** (rule 7): write the record, leave the checkbox
alone — the orchestrator flips it after the merge — and hand back the
branch plus the plain statement above.
