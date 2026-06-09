# Getting started

`torchretrieve` is a GPU retrieval library for the candidate-generation / ANN stage of a
recommender stack: PyTorch `nn.Module`s backed by fused Triton kernels. Every retrieval module
follows the same shape — register an item index once, then call it on a batch of query vectors.

## Install

```bash
pip install torchretrieve
```

Source-only distribution: the Triton kernels JIT-compile on first call against your local
toolchain, so there are no prebuilt CUDA wheels to match. You need:

- a CUDA-capable GPU,
- `torch >= 2.4` and `triton >= 3.0` (pulled in as dependencies).

> The import name is **`retrieve`**, not `torchretrieve`:
>
> ```python
> from retrieve import SilverTorch, OneBitKNN
> ```

Modules that ship a pure-PyTorch path accept `backend="torch"` and run without Triton (still on
GPU). The default is `backend="triton"`.

## The shared lifecycle

Every retrieval module is an `nn.Module` with the same three-step contract:

```python
module = SomeKNN(k=10, ...).cuda()   # 1. construct (k = how many results per query)
module.register_index(item_embs)     # 2. register the corpus: item_embs is [N, D]
topk_ids, topk_scores = module(queries)   # 3. score a batch: queries is [B, D]
```

`forward` returns a pair of `[B, k]` tensors:

- `topk_ids` — `int64` global item ids (row position in the registered `item_embs`),
- `topk_scores` — `float32` similarity scores, descending per row.

When fewer than `k` results survive (a filter rejected candidates, or the candidate set was
smaller than `k`), trailing slots are padded with `id = -1` and `score = -inf`.

## Example: SilverTorch (IVF + INT8 ANN)

`SilverTorch` is the scalable default — an IVF index over INT8-quantized embeddings, scored by a
fused probe+score kernel. Good for large `N`.

```python
import torch
from retrieve import SilverTorch

N, D, k = 1_000_000, 128, 10
item_embs = torch.randn(N, D, device="cuda")

ann = SilverTorch(k=k, n_lists=1024, n_probe=16).cuda()
ann.register_index(item_embs)

queries = torch.randn(4, D, device="cuda")
topk_ids, topk_scores = ann(queries)   # ([4, 10], [4, 10])
```

`n_lists` is the number of IVF clusters; `n_probe` is how many of them each query scans. More
probes = higher recall, more work.

## Example: OneBitKNN (1-bit Hamming)

The LiNR family scores the full corpus directly. `OneBitKNN` quantizes embeddings to 1 bit per
dimension (16× smaller than fp16) and scores by Hamming similarity.

```python
import torch
from retrieve import OneBitKNN

N, D, k = 200_000, 128, 10
item_embs = torch.randn(N, D, device="cuda")

knn = OneBitKNN(k=k).cuda()
knn.register_index(item_embs)

queries = torch.randn(8, D, device="cuda")
topk_ids, topk_scores = knn(queries)   # ([8, 10], [8, 10])
```

## Where to next

- [`modules.md`](modules.md) — which module to pick, and the per-module API reference.
- [`filtering-and-quantization.md`](filtering-and-quantization.md) — attribute-filtered retrieval
  and the standalone quantization utilities.

For internals (Triton kernel design, the benchmark harness, the test suite) see the repo-level
`docs/system/` directory.
