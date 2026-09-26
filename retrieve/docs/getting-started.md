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
- `torch >= 2.10` and `triton >= 3.6` (pulled in as dependencies). The floors are the
  validated versions: validated on an A100-SXM4-80GB (sm_80) with torch 2.10.0+cu128,
  triton 3.6.0 and Python 3.11. Older releases may work but are untested.

> The import name is **`retrieve`**, not `torchretrieve`:
>
> ```python
> from retrieve import SilverTorch, OneBitKNN
> ```

Modules that ship a pure-PyTorch path accept `backend="torch"` and run without Triton (still on
GPU). The default is `backend="triton"`.

The package has two public layers and two helper namespaces:

| namespace | what it holds |
| --- | --- |
| `retrieve` / `retrieve.modules` | the `nn.Module`s: `SilverTorch`, the LiNR paper variants `LiNRV1`–`LiNRV4`, the primitives they compose, the two filters; the two builders and `OfficialConfig`; `retrieve.modules.official` for Meta's own modules |
| `retrieve.ops.triton` / `.reference` / `.official` | the kernels behind them, one namespace per backend with the same op names and signatures (`import retrieve.ops.triton` registers `torch.ops.retrieve.*`) |
| `retrieve.indexing` | index-build math: `KMeans`, the IVF layouts, the quantizers, the bloom hash builders |
| `retrieve.functional` | query-time glue: `masked_topk`, `compact_mask`, `combine_masks` / `combine_indices`, `post_filter_topk` |

`import retrieve` imports no kernel; a module resolves its backend's op namespace when it is
constructed. [`indexing-and-ops.md`](indexing-and-ops.md)
lists the two helper namespaces and the ops.

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
probes = higher recall, more work. `ann.set_query_params(n_probe=32)` changes it later without a
rebuild; `ann.build_timings` reports the seconds each build phase took.

The same, as one fluent chain — and the way to load an index you saved earlier without
re-running k-means:

```python
from retrieve import SilverTorchBuilder

ann = (SilverTorchBuilder(k=k, n_lists=1024, n_probe=16)
       .set_item_embeddings(item_embs)
       .set_device("cuda")
       .build())
torch.save(ann.state_dict(), "ann.pt")

same = SilverTorchBuilder(k=k, n_lists=1024, n_probe=16).set_state_dict(torch.load("ann.pt")).build()
```

## Example: the LiNR variants

The LiNR family scores the full corpus directly. The paper's four variants ship as modules:
`LiNRV1` (dense fp16-input scan + mask), `LiNRV2` (filter → candidates → exact rescoring), `LiNRV3`
(1-bit Hamming top-`candidate_pool` → exact rescoring) and `LiNRV4` (int8 dense + mask), each
optionally holding a filter. V3 alone, no filter:

```python
import torch
from retrieve import LiNRV3

N, D, k = 200_000, 128, 10
item_embs = torch.randn(N, D, device="cuda")

v3 = LiNRV3(k=k, candidate_pool=2000).cuda()
v3.register_index(item_embs)

queries = torch.randn(8, D, device="cuda")
topk_ids, topk_scores = v3(queries)   # ([8, 10], [8, 10])
```

With an attribute filter, through the builder (`LiNRBuilder(variant, **kwargs)` takes the
variant's constructor keywords):

```python
from retrieve import ExactAttributeFilter, LiNRBuilder

v3 = (LiNRBuilder("v3", k=k, candidate_pool=2000)
      .set_item_embeddings(item_embs)
      .set_filter(ExactAttributeFilter(), item_attrs)     # [N, C, A_max] int64, -1 padded
      .build())
topk_ids, topk_scores = v3(queries, query_attrs)          # [B, C] int64, -1 = inactive clause
```

The primitives the variants compose (`OneBitKNN`, `PrefilterKNN`, …) are public too — see
[`modules.md`](modules.md).

## Where to next

- [`modules.md`](modules.md) — which module to pick, and the per-module API reference.
- [`filtering-and-quantization.md`](filtering-and-quantization.md) — attribute-filtered retrieval.
- [`indexing-and-ops.md`](indexing-and-ops.md) — `retrieve.indexing` (k-means, the IVF layouts,
  the quantizers, the bloom hash), `retrieve.functional` and the three op namespaces.

For internals (Triton kernel design, the benchmark harness, the test suite) see the repo-level
`docs/system/` directory.
