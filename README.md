# retrieve

GPU retrieval framework for recommender systems (PyTorch + Triton). My master's thesis.

The library half ships on PyPI as [`torchretrieve`](https://pypi.org/project/torchretrieve/); this repo also holds the training / evaluation harness used in the thesis.

## Layout

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two members sharing a single `.venv` at the workspace root:

- [`retrieve/`](retrieve/) — the library: kernels, modules, correctness tests. Published to PyPI as `torchretrieve`; imports as `retrieve`.
- [`evaluation/`](evaluation/) — training and benchmark harness; depends on `retrieve` editable. Not published.

## Install (library only)

If you just want the modules:

```bash
pip install torchretrieve
```

Source-only distribution — Triton kernels JIT-compile on first call. Requires a CUDA-capable GPU. See [`retrieve/README.md`](retrieve/README.md) for the quick example and module list.

## Setup (full workspace)

For working on the library or running the evaluation harness:

```bash
uv sync   # from this directory — populates ./.venv with both packages installed editable
```

After `uv sync`, run from anywhere in the workspace:

```bash
# from the root, targeting a member:
uv run --directory retrieve pytest tests/
uv run --directory evaluation -- bash run_per_algo.sh conf/500m/d128-quality.yaml

# or cd in (uv finds the workspace root automatically):
cd retrieve   && uv run pytest tests/
cd evaluation && ./run_per_algo.sh conf/500m/d128-quality.yaml
```

The lockfile lives at the root (`uv.lock`); the per-member lockfiles are obsolete.

## Docs

- [`docs/system/architecture.md`](docs/system/architecture.md) — module map, what each retrieval family does.
- [`docs/system/kernels.md`](docs/system/kernels.md) — Triton kernel internals.
- [`docs/system/testing.md`](docs/system/testing.md) — running the correctness suite.
- [`docs/system/evaluation.md`](docs/system/evaluation.md) — running the benchmark harness.
- [`docs/system/checkpoints.md`](docs/system/checkpoints.md) — trained models + HF Hub workflow.
- [`docs/system/filtering.md`](docs/system/filtering.md) — clause / Bloom filter API.
