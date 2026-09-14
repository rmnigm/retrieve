# torchretrieve

GPU retrieval for recommender systems — PyTorch `nn.Module`s backed by fused Triton kernels.

A library for the candidate-generation / ANN stage of a recsys stack: full-scan KNN, 1-bit Hamming KNN, INT8 IVF (SilverTorch), prefilter / postfilter compositions, and a Bloom / exact attribute-filter family that fuses into the scoring kernels. Every module follows the same `register_index → forward` shape and runs on a single GPU.

## Install

```bash
pip install torchretrieve
```

Source-only distribution. Triton kernels JIT-compile on first call against your local toolchain — no prebuilt CUDA wheels to match. You need a CUDA-capable GPU and a working `torch` + `triton` install (declared as dependencies). Modules that ship a pure-PyTorch fallback accept `backend="torch"`; `SilverTorch` additionally offers `backend="official"`, Meta's own `meta-recsys/silvertorch` kernels behind the same layer (the `official` extra, built from source against your CUDA toolkit; eager-only).

> **Note.** Install pulls in `torch>=2.4` and `triton>=3.0`; the import name is `retrieve`, not `torchretrieve`.

## Quick example

```python
import torch
from retrieve import SilverTorch

# 1M items, dim 128 — INT8 IVF + INT8 ANN
N, D, k = 1_000_000, 128, 10
item_embs = torch.randn(N, D, device="cuda")

ann = SilverTorch(k=k, n_lists=1024, n_probe=16).cuda()
ann.register_index(item_embs)

queries = torch.randn(4, D, device="cuda")
topk_ids, topk_scores = ann(queries)        # ([4, 10], [4, 10])
```

For attribute-filtered retrieval, swap to `filter_mode="bloom"` (with `m_bits` / `k_hash`) or `filter_mode="exact"` and pass `query_clause_attrs` to `forward`. The filter is fused into the same probe+score kernel — no `[B, P, D]` intermediates touch HBM.

## What's in the box

| Module | What it does |
| --- | --- |
| `FullScanKNN` | Exhaustive matmul + top-K. Reference / small-N. |
| `OneBitKNN` | OPORP 1-bit quantization + Hamming top-K. |
| `SimHashKNN` | SimHash 1-bit quantization + Hamming top-K; `k_bits` may exceed `D`. |
| `PostfilterKNN`, `PostfilterKNNInt8` | KNN then attribute filter. |
| `PrefilterKNN` | Attribute filter then KNN over the candidate set. |
| `SilverTorch` | IVF + INT8 ANN with optional fused bloom / exact filter (paper Algorithm 1). |
| `BloomFilter`, `ExactAttributeFilter` | Standalone `FilterModule`s; compose via `combine_masks` / `combine_indices`. |
| `KMeansTorch` | Index-build helper (clusters for IVF). |
| `quantize_int8`, `quantize_oporp_1bit`, `quantize_simhash_1bit` | Quantization utilities used by the modules above. |

Triton is the default backend; modules that have a pure-PyTorch path accept
`backend="torch"` for `torch.compile` / Inductor users, and `SilverTorch` also
accepts `backend="official"` (Meta's kernels, eager-only).

## Docs

User guide (in this package, under [`docs/`](docs/)):

- [`getting-started.md`](docs/getting-started.md) — install, the shared lifecycle, first examples.
- [`modules.md`](docs/modules.md) — which module to pick + per-module API reference.
- [`filtering-and-quantization.md`](docs/filtering-and-quantization.md) — attribute-filtered retrieval and the quantization utilities.

System / internals documentation lives in the repository (not in the sdist)
under `docs/system/`:

- `architecture.md` — module map, what each retrieval family does.
- `kernels.md` — Triton kernel internals and the official-backend adapter.
- `filtering.md` — clause / Bloom filter API.
- `testing.md` — running the correctness suite.
- `evaluation.md` — running the benchmark harness.
- `datasets.md` — dataset ETL and the SASRec training pipeline.
- `checkpoints.md` — trained models + HF Hub workflow.

## License

Apache-2.0.
