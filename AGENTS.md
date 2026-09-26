# AGENTS.md — how to work in this repository

Read this, then [`docs/roadmap.md`](docs/roadmap.md) (the one ordered work
queue: open steps, gates and dependencies), then only the wiki pages your
step names. Do not start work that the roadmap does not list or that sits
behind an open dependency.

## What this is

`retrieve/` is a GPU retrieval library (PyTorch + Triton; on PyPI as
`torchretrieve`, imports as `retrieve`) that reimplements two industry
papers — SilverTorch (Meta, `articles/silvertorch.md`) and LiNR (LinkedIn,
`articles/linr.md`) — from their text. `evaluation/` is the benchmark and
training harness that produces the numbers for a master's thesis
(its sources are kept out of this repository) and, now, a reproducibility paper
([`docs/decisions.md`](docs/decisions.md)). It is a uv
workspace with one `.venv` at the root; `README.md` has the layout and the
install commands.

Naming that must stay straight in code, docs and prose: our `SilverTorch`
layer is *SilverTorch Algorithm 1 reimplemented* (in the thesis it was
called QuantizedIVF — retired 2026-09-15 in the thesis and `docs/paper/`;
the already-delivered defense talk still says it and stays that way); `meta-recsys/silvertorch`
is Meta's *official* code and, once integrated, `backend="official"`; the
LiNR variants are V1–V4 as in the paper.

## Where things are documented

`docs/` is a wiki of what is true **now**: current state, standing
decisions, the open queue. No history. [`docs/index.md`](docs/index.md)
lists every page; [`docs/SCHEMA.md`](docs/SCHEMA.md) holds the conventions
(page frontmatter, tags, links, the contradiction policy).

- [`docs/roadmap.md`](docs/roadmap.md): **what to do next**, the one ordered
  queue of open steps with their gates and dependencies, and the decisions
  waiting on the user. A finished step leaves it.
- [`docs/decisions.md`](docs/decisions.md): **why**, the standing decisions
  and constraints, not reopened without the user.
- [`docs/validation.md`](docs/validation.md): **what holds**, every gate's
  current state and the measured results that stand, each marked citable
  or not.
- `docs/system/*.md`: **how the code works today**, kept in sync with the
  code in the same commit. `architecture.md` (module map, backend
  dispatch), `kernels.md`, `filtering.md`, `testing.md`, `evaluation.md`
  (harness), `datasets.md`, `checkpoints.md`, `storage.md`. Code comments
  cite these pages by section, so keep their headings stable. If code and
  a system page disagree, the code is right and the page is a bug: fix
  the page.
- `docs/contracts/`: two contracts that order nothing and apply
  everywhere: [`agent-orchestration.md`](docs/contracts/agent-orchestration.md)
  (who runs a step, on which model, how many at once, where the output
  lands) and [`coding-guidelines.md`](docs/contracts/coding-guidelines.md)
  (what the code should look like, and what is deliberately not the goal).
- `docs/artifacts/<plan>/`: raw scripts and outputs behind measured
  numbers, kept so they can be re-derived.
- `retrieve/docs/`: the library user guide that ships in the sdist.
- `docs/paper/`: the reproducibility paper's own sections, each number
  traced to [`validation.md`](docs/validation.md) or an artifact:
  [`reproduction-deviations.md`](docs/paper/reproduction-deviations.md) (paper
  vs us, Meta's code vs Meta's paper, defects reproducing found in our own
  code, and the unexplained residuals) and
  [`provenance-and-disclosure.md`](docs/paper/provenance-and-disclosure.md)
  (hardware, software, what may be cited, what was and was not compared) and
  [`official-vs-reimplementation.md`](docs/paper/official-vs-reimplementation.md)
  (Meta's kernels against ours: end to end, kernel-only, phase 2, parity,
  memory, and what the comparison cannot say).
- `articles/`: pandoc renderings of the papers. Frozen; cite, never edit.
- `evaluation/results/`: campaign records (`<suite>/<dataset>-d<dim>.jsonl`
  plus sidecars); `evaluation/golden/`: the golden baseline cells.

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
   flag instead ([`docs/decisions.md`](docs/decisions.md#harness));
   `ncu` is blocked, use `torch.profiler`. The GPU is shared: serialize
   GPU work, one job at a time. Disk: `/workspace` is a small quota
   volume (measured in `docs/system/storage.md`); put venvs under
   `/venvs/` on the local disk.
2. **Nothing is citable until its gate passed.** Harness numbers from a
   branch are not paper material until their gate is green in
   [`docs/validation.md`](docs/validation.md). Say "not yet validated"
   rather than quoting them.
3. **Correctness gates are bit-exact where the gate says so** (`torch.equal`
   on scores; ids equal up to ties). Do not loosen a tolerance to make a
   test pass; report the mismatch.
4. **Docs move with code.** A change to behaviour updates the matching
   `docs/system` file in the same commit. Relative markdown links are
   checked by `python3 scripts/check_doc_links.py` (also a pre-commit
   hook); keep it at zero broken links. One home per fact: when you rename
   a branch, path, flag or count, grep `AGENTS.md`, `docs/`,
   `retrieve/docs/` and `README.md` for the old wording and fix every
   copy, not just the page you were already editing.
5. **Deletion is gated.** Never delete a kernel that still has no
   validated replacement. The model: the two hand-written SilverTorch
   backends went at roadmap B4 only after the official backend's parity
   gate (B2) was green; tag `cuda-cute-backends-final` holds them.
6. **Commits and branches.** Commit only when the user asks. One branch per
   roadmap step (`dev/<step>`) off `staging`. Session artifacts (scripts, raw JSON) go
   under `docs/artifacts/<plan>/`, not in the packages. **All work ends up
   on `staging` at `origin`**: whatever branch or worktree produced it, a
   step is not finished until it is merged into `staging` and pushed
   ([`docs/contracts/agent-orchestration.md`](docs/contracts/agent-orchestration.md)
   §6). The box is rented; the repository is the only durable artifact.
7. **One orchestrator, constrained workers.** One user-controlled
   orchestrator session dispatches workers that do code, tests,
   evaluations or docs; workers do not widen their own scope, start a step
   the roadmap does not list, or dispatch peers. **At most three workers at
   once**, lowered by GPU serialization (rule 1), subtree exclusivity and
   the requirement that concurrent workers edit disjoint trees.
   **Model by the shape of the work:** core library rewrites with heavy
   kernel or coding work, and core harness rewrites carrying architecture
   design, go to **`fable`** as one big chunk; routine cleanups, monitoring,
   debugging, docs and one-time experiments go to **`opus`**. **Nesting is
   two levels**, with exactly one exception: a **`sonnet`** nested subagent
   for web deep research. **On the pods the user's override is in force:**
   opus for every worker, at most two agents, no subagents
   ([contract §3](docs/contracts/agent-orchestration.md#3-model-routing)).
   Briefs and reports live in `.chains/` as a branch per agent (§5).
   Every worker's result lands in the repository
   as current state in [`docs/validation.md`](docs/validation.md) and the
   affected `docs/system` page; a run recorded only in a transcript did
   not happen. Full contract:
   [`docs/contracts/agent-orchestration.md`](docs/contracts/agent-orchestration.md).
8. **Thin code, documented outside it.** Priorities in order: correct code,
   clean architecture, evals that run. Little defensive programming —
   validate at the boundary, then let it fail loudly. No backward
   compatibility with shapes we invented: redo the part and delete the old
   one (this does not weaken rule 5 — a *kernel* still goes only after its
   replacement's parity gate). No abstraction level without a written
   reason. **No multi-line comment slop — prefer no comment at all**; a
   short one only where the code is genuinely surprising, and it says why.
   The documentation is the `docs/` wiki: `docs/system` for what the code
   does today, `docs/decisions.md` and `docs/validation.md` for why and
   what was measured, kept current. Tests are the gates a step names, not a
   deliverable. Not the goal: coverage, cosmetic polish, unscheduled
   baselines, paper speculation. Full contract:
   [`docs/contracts/coding-guidelines.md`](docs/contracts/coding-guidelines.md).

9. **Never print secrets.** Do not `cat`, `grep`, `head` or otherwise dump an env or
   secrets file (`/etc/retrieve-pod.env`, `/workspace/.pod-home/secrets.env`, any `.env`),
   and do not run `env`, `printenv`, `set` or `declare -x`. They hold live tokens, and a
   transcript is not private. Read one non-secret variable by name
   (`echo "$UV_PROJECT_ENVIRONMENT"`) or test that it is set. Every dispatch brief repeats
   this rule.

## Commands

```bash
uv sync --extra official --all-packages          # whole workspace + Meta's ops; --all-packages keeps pytest (it is in the members' dev groups)
uv run --directory retrieve pytest tests/ -x -q  # library suite — GPU only, skips itself without CUDA
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q  # the harness suite (bench / training / eval_datasets): CPU-only by construction, three tests assert it, so hide the GPU
uv run --directory evaluation bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch
uv run --directory evaluation bench campaign --suite filter --resume
uv run --directory evaluation bench check --dataset goodreads     # validate a staged dataset's layout
uv run --directory evaluation eval-data arxiv --help              # dataset ETL + Hub transfer (eval-data fetch|publish)
uv run --directory evaluation train run --help                    # Encoder training (gbce | sampled_softmax + logQ); train upload-checkpoint
uv run --directory retrieve tune-kernels --help  # kernel autotune sweeps (GPU)
ruff check retrieve evaluation && ruff format --check retrieve
python3 scripts/check_doc_links.py
```

`docs/system/testing.md` explains the suite layout and fixtures;
`docs/system/evaluation.md` the harness CLI and config format.

## When you finish a step

Update [`docs/validation.md`](docs/validation.md) to the new state of every
gate the step touched (what passes now, on which environment, what was
skipped, what is still unverified), put the raw scripts and outputs under
`docs/artifacts/<plan>/`, update the affected `docs/system` page, remove
the step from [`docs/roadmap.md`](docs/roadmap.md) and add any new open
work there, run the link checker, and state plainly in your report what
passed, what was skipped, and what is still unverified. Write current
state, not a narrative of the run: replace a superseded result rather
than appending to it.

If you are a **worker** (rule 7): update `docs/validation.md` and the
system page, leave the roadmap alone (the orchestrator edits it after the
merge), and hand back the branch plus the plain statement above.
