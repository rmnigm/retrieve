# retrieve

GPU retrieval framework for recommender systems (PyTorch + Triton). My master's thesis.

The library half is the package `torchretrieve` (not yet published on PyPI; release is roadmap F4); this repo also holds the training / evaluation harness used in the thesis.

**The study.** A reproducibility study of two industrial GPU retrieval papers: **SilverTorch** (Meta), run as Meta's official SilverTorch code adapted by us (improvements listed in its CHANGES file, each gated bit-exact against the published code), against **LiNR** (LinkedIn) V1–V3, implemented here from the paper's text. Our earlier Triton reimplementation of SilverTorch is not part of the paper or of the released code ([decisions](research/decisions.md#goal-and-scheduling)).

## Layout

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with three members sharing a single `.venv` at the workspace root:

- [`retrieve/`](retrieve/) — the library: kernels, modules, correctness tests. Package `torchretrieve` (not yet on PyPI); imports as `retrieve`.
- [`evaluation/`](evaluation/) — training and benchmark harness; depends on `retrieve` editable. Not published.
- [`official-fork/`](official-fork/) — our fork of Meta's `meta-recsys/silvertorch` (Apache-2.0, Meta's licence and headers kept; changes in [`official-fork/CHANGES.md`](official-fork/CHANGES.md)): package `silvertorch_fork`, ops `torch.ops.stfork.*`, installed by the `official` extra beside the pinned upstream.

A third member, `official-fork/` — Meta's official SilverTorch code, adapted (Apache-2.0, Meta's licence and headers kept) — joins when its review accepts (roadmap META-FORK).

Datasets, virtual environments and campaign scratch live *outside* the repo, on
the GPU pod's container disk (pods are launched with [`infra/runpod/pod.sh`](infra/runpod/pod.sh)) — see [`docs/system/storage.md`](docs/system/storage.md)
for where the harness looks and the environment variables, and
[`research/environment/storage.md`](research/environment/storage.md) for the pods' disks and space budget.

## Install (library only)

If you just want the modules, once the package is on PyPI (roadmap F4):

```bash
pip install torchretrieve
```

Until then, install from a clone: `pip install ./retrieve`.

Source-only distribution — Triton kernels JIT-compile on first call. Requires a CUDA-capable GPU. See [`retrieve/README.md`](retrieve/README.md) for the quick example and module list.

## Setup (full workspace)

For working on the library or running the evaluation harness:

```bash
uv sync --extra official --all-packages   # from this directory — populates ./.venv with both packages installed editable, Meta's official extension and our fork of it, and pytest (in the members' dev groups)
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

Two trees, one set of conventions ([`research/SCHEMA.md`](research/SCHEMA.md)):

- **Project documentation** ([`docs/`](docs/index.md)): how the library and the
  harness work, kept in sync with the code. [`docs/index.md`](docs/index.md)
  lists every page.
- **Research wiki** ([`research/`](research/index.md)): the agentic working
  wiki behind the paper — the work queue ([`roadmap.md`](research/roadmap.md)),
  standing decisions ([`decisions.md`](research/decisions.md)), what is
  validated and measured ([`validation.md`](research/validation.md)), the
  process contracts, the paper drafts ([`research/paper/`](research/paper/))
  and the raw scripts behind every number
  ([`research/artifacts/`](research/artifacts/)). Agents start at
  [`AGENTS.md`](AGENTS.md).

**System design** ([`docs/system/`](docs/system/)) — how the thing works, kept in
sync with the code:

- [`architecture.md`](docs/system/architecture.md) — module map, what each retrieval family does.
- [`kernels.md`](docs/system/kernels.md) — Triton kernel internals (LiNR and the filters) and the adapter over Meta's SilverTorch ops.
- [`filtering.md`](docs/system/filtering.md) — clause / Bloom filter semantics, paper vs. implementation.
- [`testing.md`](docs/system/testing.md) — the correctness / parity / compile suites.
- [`evaluation.md`](docs/system/evaluation.md) — the benchmark harness.
- [`datasets.md`](docs/system/datasets.md) — dataset ETL and the SASRec training pipeline.
- [`checkpoints.md`](docs/system/checkpoints.md) — trained models + HF Hub workflow.
- [`storage.md`](docs/system/storage.md) — where the harness keeps data, results and caches, and its environment variables.

**Library user guide** ([`retrieve/docs/`](retrieve/docs/)) — ships in the sdist,
written for someone who installed `torchretrieve` and does not have this repo:
[getting-started](retrieve/docs/getting-started.md),
[modules](retrieve/docs/modules.md),
[filtering-and-quantization](retrieve/docs/filtering-and-quantization.md),
[indexing-and-ops](retrieve/docs/indexing-and-ops.md),
[adding-a-dataset](retrieve/docs/adding-a-dataset.md).

**Papers** ([`articles/`](articles/)) — pandoc renderings of the two papers
this repo reproduces (SilverTorch, LiNR), plus the Yambda benchmark sources the
`yambda` ETL adapts. Frozen source material: cited by the docs, never edited;
gitignored, so not in a fresh clone.
