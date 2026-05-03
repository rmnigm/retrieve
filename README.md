# retrieve

GPU retrieval framework for recommender systems (PyTorch + Triton). My master's thesis.

## Layout

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two members sharing a single `.venv` at the workspace root:

- [`retrieve/`](retrieve/) — the library: kernels, modules, correctness tests.
- [`evaluation/`](evaluation/) — training and benchmark harness; depends on `retrieve` editable.

## Setup

```bash
uv sync   # from this directory — populates ./.venv with both packages installed editable
```

After `uv sync`, run from anywhere in the workspace:

```bash
# from the root, targeting a member:
uv run --directory retrieve pytest tests/
uv run --directory evaluation evaluate --config conf/500m-d128.yaml

# or cd in (uv finds the workspace root automatically):
cd retrieve   && uv run pytest tests/
cd evaluation && uv run evaluate --config conf/500m-d128.yaml
```

The lockfile lives at the root (`uv.lock`); the per-member lockfiles are obsolete.

## Docs

- [`docs/system/architecture.md`](docs/system/architecture.md) — module map, what each retrieval family does.
- [`docs/system/kernels.md`](docs/system/kernels.md) — Triton kernel internals.
- [`docs/system/testing.md`](docs/system/testing.md) — running the correctness suite.
- [`docs/system/evaluation.md`](docs/system/evaluation.md) — running the benchmark harness.
- [`docs/system/checkpoints.md`](docs/system/checkpoints.md) — trained models + HF Hub workflow.
- [`docs/system/filtering.md`](docs/system/filtering.md) — clause / Bloom filter API.
