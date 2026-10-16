# AGENTS.md — how to work in this repository

## What this is

`retrieve/` is a GPU retrieval library (PyTorch + Triton; package `torchretrieve`, imports as
`retrieve`): LiNR V1–V3 (LinkedIn's paper, implemented from its text) with the exact attribute
filter and the Triton kernels behind them, and `SilverTorch`, Meta's official SilverTorch code,
adapted. `silvertorch/` is that adapted code: `meta-recsys/silvertorch` vendored under Meta's own
names (package `silvertorch`, ops `torch.ops.st.*`), Apache-2.0 with Meta's licence and headers,
every change listed in [`silvertorch/CHANGES.md`](silvertorch/CHANGES.md). `evaluation/` is the
benchmark harness (`bench`), the dataset ETL (`eval_datasets`) and the encoder trainer
(`training`). It is a uv workspace with one `.venv` at the root; `README.md` has the layout and
the install commands.

Naming that must stay straight in code, docs and prose:

- `SilverTorch(backend="cuda")` is Meta's official SilverTorch code, adapted (improvements listed
  in `CHANGES.md`), never "a fork". It is the only `SilverTorch` backend; `backend="torch"` is
  reserved for a pure-torch SilverTorch path, and none exists.
- Meta's code as published is not in this repository. The adapted code was gated bit-exact
  against it before it left; the git tag `meta-published-parity-final` holds that last parity
  run. Harness records written before the rename carry the backends' old names (`official-fork`,
  `official`); `records.read_records` maps `official-fork` to `cuda`.
- LiNR variants are V1–V3 as in the paper. LiNR runs the exact filter only; the bloom filter in
  this repository is Meta's, inside `SilverTorch`.

## Where things are documented

- `docs/system/*.md`: **how the code works today**, kept in sync with the code in the same
  commit. `architecture.md` (module map, backend dispatch), `kernels.md`, `filtering.md`,
  `testing.md`, `evaluation.md` (harness), `datasets.md`, `checkpoints.md`, `storage.md` (where
  the harness keeps data). Code comments cite these pages by section, so keep their headings
  stable (`check_doc_links.py` checks every `§` citation). If code and a system page disagree,
  the code is right and the page is a bug: fix the page. [`docs/index.md`](docs/index.md) lists
  the pages.
- [`retrieve/docs/`](retrieve/docs/): the library user guide that ships in the sdist, written
  for someone who installed the package and has no clone.
- [`silvertorch/CHANGES.md`](silvertorch/CHANGES.md): every change to Meta's code, with what
  gated it.

Both say what is true **now**; no history.

## Hard rules

1. **GPU environments are the only target.** The reference GPU is the A100-SXM4-80GB (torch
   2.10.0+cu128, triton 3.6.0, CUDA 12.8, Python 3.11; `infra/runpod/` builds that image). Do not
   build CPU emulators, shim headers or simulations of device code, and do not add test
   scaffolding beyond the existing pytest suites. SM clocks cannot be locked in a container
   (`nvidia-smi -lgc` is denied): record the sampled `sm_mhz` and the `unstable` flag instead;
   `ncu` is usually blocked, use `torch.profiler`. One job per GPU at a time, each with its own
   `CUDA_VISIBLE_DEVICES`, inductor cache and results path.
2. **Correctness gates are bit-exact where the test says so** (`torch.equal` on scores; ids equal
   up to ties). Do not loosen a tolerance to make a test pass; report the mismatch.
3. **Docs move with code.** A change to behaviour updates the matching `docs/system` page in
   the same commit. Relative markdown links, backticked repo paths, CLI subcommands and `§`
   citations are checked by `python3 scripts/check_doc_links.py` (also a pre-commit hook); keep
   it at zero problems. One home per fact: when you rename a path, flag or count, grep
   `AGENTS.md`, `docs/`, `retrieve/docs/` and `README.md` for the old wording and fix every copy.
4. **Deletion is gated.** Never delete a kernel that still has no validated replacement: a kernel
   goes only after its replacement's parity gate is green.
5. **Thin code, documented outside it.** Priorities in order: correct code, clean architecture,
   evals that run. Little defensive programming: validate at the boundary, then let it fail
   loudly. No backward compatibility with shapes we invented: redo the part and delete the old
   one (rule 4 still holds for kernels). No abstraction level without a written reason. No
   multi-line comment slop, prefer no comment at all; a short one only where the code is
   genuinely surprising, and it says why. The documentation lives in `docs/system`. Tests are the
   gates a change names, not a deliverable. Not the goal: coverage, cosmetic polish.
6. **Nothing a run produces is a result until its gate passed.** `bench report` marks a results
   tree NOT CITABLE unless `--gate` names a passed gate and the records agree (no failed or
   partial record, a clean tree). Say "not yet validated" rather than quoting a number.
7. **Never print secrets.** Do not `cat`, `grep`, `head` or otherwise dump an env or secrets
   file (any `.env`, a pod's env file), and do not run `env`, `printenv`, `set` or `declare -x`.
   They hold live tokens, and a transcript is not private. Read one non-secret variable by name
   (`echo "$UV_PROJECT_ENVIRONMENT"`) or test that it is set.
8. **Commits.** Commit only when asked. Raw outputs (records, JSON dumps, logs) never go in git;
   `bench upload` publishes a results tree to a HF dataset repo of yours (`RETRIEVE_HUB_OWNER`).

## Commands

```bash
uv sync --all-packages                           # whole workspace incl. the silvertorch extension (built from source; needs nvcc); --all-packages keeps pytest (it is in the members' dev groups)
uv run --directory retrieve pytest tests/ -x -q  # library suite — GPU only, skips itself without CUDA
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q  # the harness suite (bench / training / eval_datasets): CPU-only by construction, three tests assert it, so hide the GPU
uv run --directory evaluation bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch
uv run --directory evaluation bench campaign --suite filter --resume
uv run --directory evaluation bench report results           # results.parquet + report.md of a results tree
uv run --directory evaluation bench merge <tree>... --out tables   # the merged tables of several trees
uv run --directory evaluation bench check --dataset goodreads     # validate a staged dataset's layout
uv run --directory evaluation eval-data arxiv --help              # dataset ETL + Hub transfer (eval-data fetch|publish)
uv run --directory evaluation train run --help                    # Encoder training (gbce | sampled_softmax + logQ); train upload-checkpoint
uv run --directory retrieve tune-kernels --help  # kernel autotune sweeps (GPU)
ruff check retrieve evaluation && ruff format --check retrieve evaluation
python3 scripts/check_doc_links.py
```

`docs/system/testing.md` explains the suite layout and fixtures;
`docs/system/evaluation.md` the harness CLI and config format.

## When you finish a change

Update the affected `docs/system` page, run the suites the change touches (library on a GPU,
harness on the CPU), `ruff` and the link checker, and state plainly what passed, what was skipped
and what is still unverified. Write current state, not a narrative of the run.
