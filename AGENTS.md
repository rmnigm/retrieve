# AGENTS.md — how to work in this repository

Read this, then [`research/roadmap.md`](research/roadmap.md) (the one ordered work
queue: open steps, gates and dependencies), then only the wiki pages your
step names. Do not start work that the roadmap does not list or that sits
behind an open dependency.

## What this is

`retrieve/` is a GPU retrieval library (PyTorch + Triton; package
`torchretrieve`, not yet on PyPI (roadmap F4); imports as `retrieve`). `evaluation/` is the benchmark and
training harness that produces the numbers for a master's thesis
(its sources are kept out of this repository) and, now, a reproducibility paper
([`research/decisions.md`](research/decisions.md#goal-and-scheduling)). It is a uv
workspace with one `.venv` at the root; `README.md` has the layout and the
install commands.

**The study** (user, 2026-10-11): SilverTorch (Meta,
`articles/silvertorch.md`) as **Meta's official SilverTorch code, adapted**
— `meta-recsys/silvertorch` with our listed improvements, each gated —
against **LiNR** (LinkedIn, `articles/linr.md`) V1–V3, which we implement
from the paper's text. That comparison is the paper and the released code.

Naming that must stay straight in code, docs and prose:
- **The adapted official code** lives in the `silvertorch/` workspace
  member under Meta's own names (package `silvertorch`, ops `torch.ops.st.*`,
  `SilverTorch(backend="cuda")`, its only backend; changes in its
  `CHANGES.md`; RENAME, user ruling 2026-10-16). In the paper and user docs
  it is "Meta's official SilverTorch code, adapted (improvements listed in
  CHANGES)", never "a fork". `backend="torch"` is reserved for a pure-torch
  SilverTorch path; none exists.
- **Meta's code as published** (pinned `21aa35e`, `backend="official"` while
  it was in the workspace) is the "as published" column of
  [Meta as published vs adapted](research/paper/meta-published-vs-adapted.md),
  not an arm of the study. It left the library and the workspace at RENAME;
  its last bit-exact parity run against the adapted code stands behind the
  tag `meta-published-parity-final`. Records keep the names they were
  written with (`official-fork`, `official`); `bench merge` maps them.
- **Our Triton SilverTorch reimplementation** (`SilverTorch(backend="triton")`
  and its `backend="torch"` reference: SilverTorch Algorithm 1
  reimplemented, called QuantizedIVF in the thesis; the already-delivered
  defense talk still says it and stays that way) is **out of the paper and
  deleted from the code** (tag `triton-silvertorch-final` holds it); its
  records on the Hub are never cited. **Our LiNR bloom filter** is deleted
  with it (user, 2026-10-15): LiNR runs the exact filter only, and the
  bloom in the study is Meta's, inside SilverTorch.
- **LiNR** variants are V1–V3 as in the paper (our int8 V4 left the
  library in campaign v2; [decisions](research/decisions.md#campaign-v2-user-2026-10-08)).

## Where things are documented

Two trees, one set of conventions ([`research/SCHEMA.md`](research/SCHEMA.md):
page frontmatter, tags, links, the contradiction policy). Both say what is
true **now**; no history.

- **`research/`** is the agentic research wiki: the queue, the decisions,
  what was measured, the paper drafts, the process. Not for users of the
  release. [`research/index.md`](research/index.md) lists every page.
- **`docs/`** is the released project's documentation (how the library and
  the harness work), and [`retrieve/docs/`](retrieve/docs/) the library user
  guide that ships in the sdist. [`docs/index.md`](docs/index.md) lists them.
  Project pages may link into `research/` for a rationale or a measured
  number; they never carry run state (a branch, a pod, a step's progress).

In `research/`:

- [`research/roadmap.md`](research/roadmap.md): **what to do next**, the one ordered
  queue of open steps with their gates and dependencies, and the decisions
  waiting on the user. A finished step leaves it.
- [`research/decisions.md`](research/decisions.md): **why**, the standing decisions
  and constraints, not reopened without the user.
- [`research/validation.md`](research/validation.md): **what holds**, every gate's
  current state and the measured results that stand, each marked citable
  or not.
- `research/contracts/`: two contracts that order nothing and apply
  everywhere: [`agent-orchestration.md`](research/contracts/agent-orchestration.md)
  (who runs a step, on which model, how many at once, where the output
  lands) and [`coding-guidelines.md`](research/contracts/coding-guidelines.md)
  (what the code should look like, and what is deliberately not the goal).
- `research/artifacts/<plan>/`: raw scripts behind measured numbers, kept so
  they can be re-derived; the raw outputs (records, JSON dumps, logs) are
  on the Hub under `artifacts/<plan>/`, listed in
  [`research/artifacts/hub-index.md`](research/artifacts/hub-index.md).
- `research/paper/`: the reproducibility paper's own sections, each number
  traced to [`validation.md`](research/validation.md) or an artifact:
  [`claims.md`](research/paper/claims.md) (the papers' claims and C1–C7 for
  this study),
  [`reproduction-deviations.md`](research/paper/reproduction-deviations.md) (paper
  vs us, Meta's code vs Meta's paper, defects reproducing found in our own
  code, and the unexplained residuals),
  [`meta-published-vs-adapted.md`](research/paper/meta-published-vs-adapted.md)
  (what our adaptation of Meta's code changed, and the parity gate that
  holds it to the published code),
  [`provenance-and-disclosure.md`](research/paper/provenance-and-disclosure.md)
  (hardware, software, what may be cited, what was and was not compared),
  [`results.md`](research/paper/results.md) (the results draft),
  [`abstract.md`](research/paper/abstract.md) (the submitted abstract),
  [`ecir-criteria.md`](research/paper/ecir-criteria.md) and
  [`release-and-licenses.md`](research/paper/release-and-licenses.md).
- [`research/environment/storage.md`](research/environment/storage.md): the
  GPU pods' disks, what lives where, the budget, the Hub quota.
- `articles/`: pandoc renderings of the papers. Frozen; cite, never edit.
  Gitignored, not shipped in git (likely copyrighted third-party text) —
  present on a working box, but a fresh clone won't have it; citations to
  it are for the box you're working on, not the shipped repo.
- Campaign records: written as JSONL to a gitignored local results tree
  during a run, aggregated into `results.parquet`, and kept on the
  private Hub repo `pinkmeme/eval-results` (`bench upload` / `bench
  fetch`; [`research/artifacts/hub-index.md`](research/artifacts/hub-index.md));
  `evaluation/golden/`: the golden baseline cells.

In `docs/`:

- `docs/system/*.md`: **how the code works today**, kept in sync with the
  code in the same commit. `architecture.md` (module map, backend
  dispatch), `kernels.md`, `filtering.md`, `testing.md`, `evaluation.md`
  (harness), `datasets.md`, `checkpoints.md`, `storage.md` (where the
  harness keeps data). Code comments cite these pages by section, so keep
  their headings stable (`check_doc_links.py` checks every `§` citation).
  If code and a system page disagree, the code is right and the page is a
  bug: fix the page.
- `retrieve/docs/`: the library user guide that ships in the sdist,
  written for someone who installed the package and has no clone.

## Hard rules

1. **GPU work runs on pods, one job per GPU.** The orchestrator runs on
   the user's laptop or on a RunPod pod; GPU work runs on RunPod pods
   launched with `infra/runpod/pod.sh` (image: torch 2.10.0+cu128,
   triton 3.6.0, CUDA 12.8, Python 3.11). A100-SXM4-80GB is the reference
   GPU for every citable number. GPU environments are the only target: do
   not build CPU emulators, shim headers, or simulations of device code,
   and do not add test scaffolding beyond the existing pytest suites —
   the user rejected that explicitly. Every GPU claim is validated on a
   pod GPU and records it: SM clocks **cannot be locked** in the container
   (`nvidia-smi -lgc` is denied, no sudo) — record the sampled `sm_mhz`
   and the `unstable` flag instead
   ([`research/decisions.md`](research/decisions.md#harness)); `ncu` is blocked,
   use `torch.profiler`. Serialize per GPU, not per pod: one job per GPU
   at a time, each with its own `CUDA_VISIBLE_DEVICES`, inductor cache
   and results path ([`research/roadmap.md`](research/roadmap.md#multi-gpu-execution)).
   Disk: `/workspace` is small (measured in `research/environment/storage.md`);
   data, venvs and scratch go on the container disk (`/data`, `/venvs/`,
   `/scratch`).
2. **Nothing is citable until its gate passed.** Harness numbers from a
   branch are not paper material until their gate is green in
   [`research/validation.md`](research/validation.md). Say "not yet validated"
   rather than quoting them.
3. **Correctness gates are bit-exact where the gate says so** (`torch.equal`
   on scores; ids equal up to ties). Do not loosen a tolerance to make a
   test pass; report the mismatch.
4. **Docs move with code.** A change to behaviour updates the matching
   `docs/system` file in the same commit. Relative markdown links are
   checked by `python3 scripts/check_doc_links.py` (also a pre-commit
   hook); keep it at zero broken links. One home per fact: when you rename
   a branch, path, flag or count, grep `AGENTS.md`, `docs/`,
   `research/`, `retrieve/docs/` and `README.md` for the old wording and fix every
   copy, not just the page you were already editing.
5. **Deletion is gated.** Never delete a kernel that still has no
   validated replacement. The model: the two hand-written SilverTorch
   backends went only after the official backend's parity gate was green
   (tag `cuda-cute-backends-final` holds them), the Triton SilverTorch
   backend with our LiNR bloom filter went after the adapted backend's
   review ACCEPT and the final run (tag `triton-silvertorch-final`), and
   Meta's code as published went at RENAME, after the adapted code's last
   parity run against it (tag `meta-published-parity-final`).
6. **Commits and branches.** Commit only when the user asks. One branch per
   roadmap step (`dev/<step>`) off `staging`. Session scripts go under
   `research/artifacts/<plan>/`, not in the packages; raw outputs (records,
   JSON dumps, logs) never go in git — publish them with `bench upload`
   to `pinkmeme/eval-results` (campaign legs as `<leg>/`, plan outputs as
   `artifacts/<plan>/`) and add the manifest's sha256 to
   [`research/artifacts/hub-index.md`](research/artifacts/hub-index.md).
   **All work ends up
   on `staging` at `origin`**: whatever branch or worktree produced it, a
   step is not finished until it is merged into `staging` and pushed
   ([`research/contracts/agent-orchestration.md`](research/contracts/agent-orchestration.md)
   §6). The pods are rented; the repository is the only durable artifact.
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
   ([contract §3](research/contracts/agent-orchestration.md#3-model-routing)).
   Briefs and reports live in `.chains/` as a branch per agent (§5).
   Every worker's result lands in the repository
   as current state in [`research/validation.md`](research/validation.md) and the
   affected `docs/system` page; a run recorded only in a transcript did
   not happen. Full contract:
   [`research/contracts/agent-orchestration.md`](research/contracts/agent-orchestration.md).
8. **Thin code, documented outside it.** Priorities in order: correct code,
   clean architecture, evals that run. Little defensive programming —
   validate at the boundary, then let it fail loudly. No backward
   compatibility with shapes we invented: redo the part and delete the old
   one (this does not weaken rule 5 — a *kernel* still goes only after its
   replacement's parity gate). No abstraction level without a written
   reason. **No multi-line comment slop — prefer no comment at all**; a
   short one only where the code is genuinely surprising, and it says why.
   The documentation lives outside the code: `docs/system` for what the code
   does today, `research/decisions.md` and `research/validation.md` for why and
   what was measured, kept current. Tests are the gates a step names, not a
   deliverable. Not the goal: coverage, cosmetic polish, unscheduled
   baselines, paper speculation. Full contract:
   [`research/contracts/coding-guidelines.md`](research/contracts/coding-guidelines.md).

9. **Never print secrets.** Do not `cat`, `grep`, `head` or otherwise dump an env or
   secrets file (`/etc/retrieve-pod.env`, `/workspace/.pod-home/secrets.env`, any `.env`),
   and do not run `env`, `printenv`, `set` or `declare -x`. They hold live tokens, and a
   transcript is not private. Read one non-secret variable by name
   (`echo "$UV_PROJECT_ENVIRONMENT"`) or test that it is set. Every dispatch brief repeats
   this rule.

## Commands

```bash
uv sync --all-packages                           # whole workspace incl. the silvertorch extension; --all-packages keeps pytest (it is in the members' dev groups)
uv run --directory retrieve pytest tests/ -x -q  # library suite — GPU only, skips itself without CUDA
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q  # the harness suite (bench / training / eval_datasets): CPU-only by construction, three tests assert it, so hide the GPU
uv run --directory evaluation bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch
uv run --directory evaluation bench campaign --suite filter --resume
uv run --directory evaluation bench check --dataset goodreads     # validate a staged dataset's layout
uv run --directory evaluation eval-data arxiv --help              # dataset ETL + Hub transfer (eval-data fetch|publish)
uv run --directory evaluation train run --help                    # Encoder training (gbce | sampled_softmax + logQ); train upload-checkpoint
uv run --directory retrieve tune-kernels --help  # kernel autotune sweeps (GPU)
ruff check retrieve evaluation && ruff format --check retrieve evaluation
python3 scripts/check_doc_links.py
```

`docs/system/testing.md` explains the suite layout and fixtures;
`docs/system/evaluation.md` the harness CLI and config format.

## When you finish a step

Update [`research/validation.md`](research/validation.md) to the new state of every
gate the step touched (what passes now, on which environment, what was
skipped, what is still unverified), put the raw scripts under
`research/artifacts/<plan>/` and the raw outputs on the Hub (`bench upload`, a
row in `research/artifacts/hub-index.md`), update the affected `docs/system` page, remove
the step from [`research/roadmap.md`](research/roadmap.md) and add any new open
work there, run the link checker, and state plainly in your report what
passed, what was skipped, and what is still unverified. Write current
state, not a narrative of the run: replace a superseded result rather
than appending to it.

If you are a **worker** (rule 7): update `research/validation.md` and the
system page, leave the roadmap alone (the orchestrator edits it after the
merge), and hand back the branch plus the plain statement above.
