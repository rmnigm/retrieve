# retrieve

GPU retrieval for recommender systems (PyTorch + Triton): the library `torchretrieve` and the
benchmark harness behind a reproducibility study of two industrial GPU retrieval papers.

**The study.** **SilverTorch** (Meta), run as Meta's official SilverTorch code adapted by us
(improvements listed in [`silvertorch/CHANGES.md`](silvertorch/CHANGES.md), each gated bit-exact
against the published code before it left this repository), against **LiNR** (LinkedIn) V1–V3,
implemented here from the paper's text, on public datasets with real and synthetic attribute
filters.

## Layout

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with three
members sharing a single `.venv` at the workspace root:

- [`retrieve/`](retrieve/) — the library: kernels, modules, correctness tests. Package
  `torchretrieve` (not on PyPI); imports as `retrieve`.
- [`silvertorch/`](silvertorch/) — Meta's official SilverTorch code (`meta-recsys/silvertorch`),
  adapted (Apache-2.0, Meta's licence and headers kept; changes in
  [`silvertorch/CHANGES.md`](silvertorch/CHANGES.md)): package `silvertorch`, ops
  `torch.ops.st.*`, a dependency of `retrieve/`, built from source at install (needs an `nvcc`
  matching your torch wheel).
- [`evaluation/`](evaluation/) — the benchmark harness (`bench`), the dataset ETL (`eval-data`)
  and the encoder trainer (`train`); depends on `retrieve` editable. Not published.

[`results/`](results/README.md) (not a workspace member) holds the paper's table and figure data,
the final run's records they are computed from, the script that computes them, and the final
run's configuration.

Datasets, virtual environments and results live *outside* the repo; see
[`docs/system/storage.md`](docs/system/storage.md) for where the harness looks and the
environment variables. The numbers were taken on an A100-SXM4-80GB with torch 2.10.0+cu128,
triton 3.6.0, CUDA 12.8 and Python 3.11.

## Install (library only)

```bash
pip install ./silvertorch ./retrieve      # from a clone; both build from source against your CUDA toolkit
```

Triton kernels JIT-compile on first call. Requires a CUDA-capable GPU. See
[`retrieve/README.md`](retrieve/README.md) for the quick example and module list.

## Setup (full workspace)

For working on the library or running the evaluation harness:

```bash
uv sync --all-packages   # from this directory — populates ./.venv with the three members installed editable (the silvertorch extension builds from source) and pytest (in the members' dev groups)
```

Then, from anywhere in the workspace (`cd`-ing into a member works too; uv finds the root):

```bash
# library correctness suite (GPU-only; skips itself without CUDA)
uv run --directory retrieve pytest tests/

# harness suite (CPU-only by construction)
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q
```

## Running an evaluation and reading its numbers

Stage a dataset first ([`docs/system/datasets.md`](docs/system/datasets.md): the ETL per
dataset, or `eval-data fetch` from a Hub copy of yours, `RETRIEVE_HUB_OWNER`), then:

```bash
cd evaluation
# one algorithm on one dataset / suite (every option after --suite narrows the grid)
uv run bench run --dataset arxiv --dim 128 --suite filter --algo silvertorch

# a whole suite: one child process per (dataset, dim, algo, backend), JSONL records under results/
uv run bench campaign --suite filter --resume

# the numbers: results.parquet (one row per timed entry) and report.md (per cell: recall, latency, status)
uv run bench report results

# several results trees into one set of tables (records, timings, samples) with a manifest
uv run bench merge results-a results-b --out tables
```

What a cell measures, the record format, resume and the suites are in
[`docs/system/evaluation.md`](docs/system/evaluation.md).

**The paper's tables and figures** are rebuilt from the final run's records by
[`results/build_tables.py`](results/build_tables.py); [`results/README.md`](results/README.md) maps
each CSV to its table or figure and states the selection rules. The final run did not use
`evaluation/config/suites.yaml` as shipped: its suites and the configurations it actually executed
are in [`results/final-run-config/`](results/final-run-config/suites.yaml).

## Docs

**System design** ([`docs/`](docs/index.md)) — how the code works, kept in sync with it:

- [`architecture.md`](docs/system/architecture.md) — module map, what each retrieval family does.
- [`kernels.md`](docs/system/kernels.md) — Triton kernel internals (LiNR and the filters) and the adapter over Meta's SilverTorch ops.
- [`filtering.md`](docs/system/filtering.md) — clause / Bloom filter semantics, paper vs. implementation.
- [`testing.md`](docs/system/testing.md) — the correctness / parity / compile suites.
- [`parity.md`](docs/system/parity.md) — the adapted SilverTorch code against Meta's published code: the gate, its recorded results, how to rerun it.
- [`evaluation.md`](docs/system/evaluation.md) — the benchmark harness.
- [`datasets.md`](docs/system/datasets.md) — dataset ETL and the SASRec training pipeline.
- [`checkpoints.md`](docs/system/checkpoints.md) — trained models + HF Hub workflow.
- [`storage.md`](docs/system/storage.md) — where the harness keeps data, results and caches, and its environment variables.

**Library user guide** ([`retrieve/docs/`](retrieve/docs/)) — ships in the sdist, written for
someone who installed `torchretrieve` and does not have this repo:
[getting-started](retrieve/docs/getting-started.md),
[modules](retrieve/docs/modules.md),
[filtering-and-quantization](retrieve/docs/filtering-and-quantization.md),
[indexing-and-ops](retrieve/docs/indexing-and-ops.md),
[adding-a-dataset](retrieve/docs/adding-a-dataset.md).

## Licence

This repository is under its [`LICENSE`](LICENSE); `silvertorch/` stays Apache-2.0 with Meta's
`LICENSE` and file headers.
