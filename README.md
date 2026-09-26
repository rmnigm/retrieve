# retrieve

GPU retrieval framework for recommender systems (PyTorch + Triton). My master's thesis.

The library half ships on PyPI as [`torchretrieve`](https://pypi.org/project/torchretrieve/); this repo also holds the training / evaluation harness used in the thesis.

## Layout

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two members sharing a single `.venv` at the workspace root:

- [`retrieve/`](retrieve/) — the library: kernels, modules, correctness tests. Published to PyPI as `torchretrieve`; imports as `retrieve`.
- [`evaluation/`](evaluation/) — training and benchmark harness; depends on `retrieve` editable. Not published.

Datasets, virtual environments and campaign scratch live *outside* the repo, on
the GPU box's local disk — see [`docs/system/storage.md`](docs/system/storage.md)
for the paths, the environment variables and the space budget.

## Install (library only)

If you just want the modules:

```bash
pip install torchretrieve
```

Source-only distribution — Triton kernels JIT-compile on first call. Requires a CUDA-capable GPU. See [`retrieve/README.md`](retrieve/README.md) for the quick example and module list.

## Setup (full workspace)

For working on the library or running the evaluation harness:

```bash
uv sync --extra official --all-packages   # from this directory — populates ./.venv with both packages installed editable, Meta's official extension, and pytest (in the members' dev groups)
```

After `uv sync`, run from anywhere in the workspace:

```bash
# library correctness suite (GPU-only; skips itself without CUDA)
uv run --directory retrieve pytest tests/

# one algorithm on one dataset / suite (harness v2; every option after --suite narrows)
uv run --directory evaluation bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch

# a whole campaign: one child process per (dataset, dim, algo, backend), JSONL under results/
uv run --directory evaluation bench campaign --suite filter --resume
```

`cd`-ing into a member works too — uv finds the workspace root automatically.

The lockfile lives at the root (`uv.lock`); the per-member lockfiles are obsolete.

## Docs

**Documentation** ([`docs/`](docs/index.md)) is a wiki of what is true now:
[`index.md`](docs/index.md) lists every page and
[`SCHEMA.md`](docs/SCHEMA.md) its conventions. Start with
[`roadmap.md`](docs/roadmap.md) (the open work queue, gates and
dependencies), [`decisions.md`](docs/decisions.md) (standing decisions and
constraints) and [`validation.md`](docs/validation.md) (which gates pass and
the measured results that stand). Two contracts apply everywhere:
[`agent-orchestration.md`](docs/contracts/agent-orchestration.md) (one
orchestrator, constrained workers, which model takes which work, and the
rule that everything lands on `staging` at `origin`) and
[`coding-guidelines.md`](docs/contracts/coding-guidelines.md) (thin code, no
self-compat, no comment slop, and what is deliberately not the goal). Raw
scripts and outputs behind the numbers are in
[`docs/artifacts/`](docs/artifacts/). Agents start at
[`AGENTS.md`](AGENTS.md).

**System design** ([`docs/system/`](docs/system/)) — how the thing works, kept in
sync with the code:

- [`architecture.md`](docs/system/architecture.md) — module map, what each retrieval family does.
- [`kernels.md`](docs/system/kernels.md) — Triton kernel internals and the official-backend adapter.
- [`filtering.md`](docs/system/filtering.md) — clause / Bloom filter semantics, paper vs. implementation.
- [`testing.md`](docs/system/testing.md) — the correctness / parity / compile suites.
- [`evaluation.md`](docs/system/evaluation.md) — the benchmark harness.
- [`datasets.md`](docs/system/datasets.md) — dataset ETL and the SASRec training pipeline.
- [`checkpoints.md`](docs/system/checkpoints.md) — trained models + HF Hub workflow.
- [`storage.md`](docs/system/storage.md) — the box's disks, what may live on each, and the space budget.

**Library user guide** ([`retrieve/docs/`](retrieve/docs/)) — ships in the sdist,
written for someone who installed `torchretrieve` and does not have this repo:
[getting-started](retrieve/docs/getting-started.md),
[modules](retrieve/docs/modules.md),
[filtering-and-quantization](retrieve/docs/filtering-and-quantization.md),
[indexing-and-ops](retrieve/docs/indexing-and-ops.md).

**Papers** ([`articles/`](articles/)) — pandoc renderings of the three papers
this repo reproduces or benchmarks against (SilverTorch, LiNR, Yambda). Frozen
source material: cited by the docs, never edited.
