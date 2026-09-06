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
(`docs/thesis/`) and, now, a reproducibility paper (roadmap §0). It is a uv
workspace with one `.venv` at the root; `README.md` has the layout and the
install commands.

Naming that must stay straight in code, docs and prose: our `SilverTorch`
layer is *SilverTorch Algorithm 1 reimplemented* (in the thesis it was
called QuantizedIVF — that name is being retired); `meta-recsys/silvertorch`
is Meta's *official* code and, once integrated, `backend="official"`; the
LiNR variants are V1–V4 as in the paper.

## Where things are documented

- `docs/system/*.md` — **how the code works today**, kept in sync with the
  code in the same commit. `architecture.md` (module map, backend dispatch),
  `kernels.md`, `filtering.md`, `testing.md`, `evaluation.md` (harness),
  `datasets.md`, `checkpoints.md`. If code and a system doc disagree, the
  code is right and the doc is a bug — fix the doc.
- `docs/plans/*.md` — **why and in what order**. `00-roadmap.md` is the
  master plan; the others are one-phase detail. Plans carry a status
  blockquote (date, branch, what ran), numbered sections, decisions
  `D1…`, work packages with gates, and — once executed — a validation
  record appended to the same file. The model record is
  `docs/plans/cuda-silvertorch-handoff.md` §13. Finished plans move to
  `docs/plans/archive/` (not maintained, links may rot).
- `docs/plans/*-artifacts/` — raw scripts and outputs behind a plan's
  numbers, kept so they can be re-derived.
- `retrieve/docs/` — the library user guide that ships in the sdist.
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
   lock clocks with `sudo nvidia-smi -lgc 1410` before timing (`-rgc`
   after); `ncu` is blocked in the container, use `torch.profiler`. The
   GPU is shared: serialize GPU work, one job at a time.
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
5. **Deletion is gated.** The CUDA C++ and CuTe backends are removed only
   after the official backend's parity gate (roadmap Phase B). Never
   delete a kernel that still has no validated replacement.
6. **Commits and branches.** Commit only when the user asks. One branch per
   roadmap phase off `main` after Phase A4. Session artifacts (scripts,
   raw JSON) go under `docs/plans/<plan>-artifacts/`, not in the packages.
7. **Subagents.** Respect any cap the user sets on agent count as a total
   including children; child agents for search only on lightweight models
   (`sonnet` / `haiku`), never the default model.

## Commands

```bash
uv sync                                          # whole workspace (Mac: torch wheel will fail; library code still reads fine)
uv run --directory retrieve pytest tests/ -x -q  # library suite — GPU only, skips itself without CUDA
cd evaluation && uv run pytest retrieval/tests/ --ignore=retrieval/tests/test_silvertorch_algo_reverse.py  # CPU-only harness tests
uv run --directory evaluation evaluate --config config/arxiv/d128-filter.yaml --algo silvertorch --output /tmp/st.json
uv run --directory evaluation run-evaluation --eval-type filter
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
