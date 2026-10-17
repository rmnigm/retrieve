---
title: parity
created: 2026-10-17
updated: 2026-10-17
type: concept
tags: [library, silvertorch, parity]
sources: [retrieve/tests/parity/test_official.py]
---

# Parity of the adapted SilverTorch code with Meta's published code

[`silvertorch/`](../../silvertorch/) is Meta's SilverTorch code
(`meta-recsys/silvertorch`, commit `21aa35e28b6dd9a91e9ee35efb0857715e86bda7`),
adapted. Every change was gated against the published code at that commit,
built in the same environment. This page gives the gate, its recorded results,
the parity tests in this repository, and how to run the comparison against the
published code.

## The gate

- **Contract.** On the int32 score path the `[B, k]` scores are equal bit for
  bit (`torch.equal`) and the ids equal up to permutation within tied scores.
  Meta's bloom filter keeps Meta's semantics (no false negatives against the
  exact predicate; the partial masks equal the full mask).
- **Reference build.** Meta's code at `21aa35e`, built with host and nvcc
  `-O3` (`NVCC_APPEND_FLAGS="-O3 -Xcompiler -O3"`; Meta's `setup.py` passes no
  `-O`, so its host code otherwise builds at `-O0`). Kernels unmodified.
- **Grid.** arXiv d128 / d256, Goodreads d128, PubMed d768 and LAION 30M d256;
  no filter, the exact filter, and the bloom filter (partial and full path,
  Meta's width); `n_probe` 24 to 2,048; batch 1 / 16 / 64; k 100 / 1000;
  eager and CUDA-graph.
- **Environment.** NVIDIA A100-SXM4-80GB, torch 2.10.0+cu128, CUDA 12.8,
  Python 3.11.

## Recorded results

| check | result |
|---|---|
| Meta's own test suite run on the adapted code | 122 passed |
| ids and scores `torch.equal` to the published code, the ≤ 10M grid | 276 / 276 cells |
| the same at 30M (LAION d256), bloom partial and full path | 64 / 64 cells |
| latency, adapted / published, eager | faster on every cell: median 0.230, max 0.764 over 300 cells; 30M bloom median 0.279, max 0.467 |
| configured bloom width with the exact recheck (an adapted-code option, off in the study) | 0 false positives on every cell; the rechecked bloom `torch.equal` to the exact filter |

The upstream repository has one later commit (`22c2007`, CCCL 3 / CUDA 13
compatibility), applied verbatim; on CUDA 12.8 it selects the old code path,
so the compiled behaviour is unchanged.

## The parity tests in this repository

[`retrieve/tests/parity/`](../../retrieve/tests/parity/) holds the GPU parity
tests (`uv run --directory retrieve pytest tests/parity/ -q`; they skip without
CUDA). Each calls one op or module and a reference on the same inputs;
[testing](testing.md#what-each-parity-file-asserts) gives every assertion.

| file | compares | gate |
|---|---|---|
| [`test_official.py`](../../retrieve/tests/parity/test_official.py) | the adapted ops (`torch.ops.st.*`: the upstream-shaped `fused_kmean_ann`, `_with_partial_masks`, `bloom_index_build`, the bloom searches, and `search`, `probe_prep`, the prepared searches) and the `SilverTorch` layer, against a dense torch oracle over the probed positions, against the upstream-shaped ops, and against `quantize_int8` | int32 path: scores `torch.equal`, ids up to ties (plain, exact filter with reverse clauses, the layer); fp16 path: relative error ≤ 2⁻¹⁰ and top-k Jaccard ≥ 0.99 against int32; bloom: no false negatives against the exact predicate, false-positive rate recorded, partial masks equal the full mask; the rechecked bloom `torch.equal` to the exact filter; the op schemas pinned to `21aa35e` |
| [`test_clause_mask.py`](../../retrieve/tests/parity/test_clause_mask.py), [`test_clause_compact.py`](../../retrieve/tests/parity/test_clause_compact.py), [`test_compact_order.py`](../../retrieve/tests/parity/test_compact_order.py) | the exact clause filter's Triton kernels against pure torch | masks, ids and counts `torch.equal`, in ascending item order |
| [`test_fused_masked_knn_topk.py`](../../retrieve/tests/parity/test_fused_masked_knn_topk.py), [`test_gemv_scores.py`](../../retrieve/tests/parity/test_gemv_scores.py), [`test_oporp_1bit_match_topk.py`](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py) | the LiNR scoring kernels against gather + bmm + topk, cuBLAS, and a torch popcount | top-k ids up to ties with scores within a stated bound (`fused_masked_knn_topk`), `torch.equal` (the GEMV, the popcount) |
| [`test_accumulation.py`](../../retrieve/tests/parity/test_accumulation.py) | every scoring kernel against an fp64 computation of the same operation | an accumulation bound fp32 meets and fp16 is asserted to miss |

The comparison with the published code itself (the recorded results above)
needs both builds, in two environments (below); the tests here run on the
adapted build alone.

## Comparing against the published code

Both builds register the package `silvertorch` and the op namespace
`torch.ops.st`, so they cannot load in one process. The comparison runs in two
environments on the same GPU:

1. **Published.** A venv with torch 2.10.0+cu128, `ninja`, `setuptools`,
   `wheel` and an `nvcc` matching the torch wheel (CUDA 12.8):
   ```bash
   NVCC_APPEND_FLAGS="-O3 -Xcompiler -O3" pip install --no-build-isolation --no-deps \
     "silvertorch @ git+https://github.com/meta-recsys/silvertorch@21aa35e28b6dd9a91e9ee35efb0857715e86bda7"
   ```
2. **Adapted.** A second venv with the same torch, and
   `pip install --no-build-isolation --no-deps ./silvertorch` from this
   repository (its `setup.py` already builds at `-O3`).
3. **Same inputs, same ops.** The upstream-shaped ops keep their published
   signatures in the adapted code: `st::fused_kmean_ann` (no filter / exact
   filter scoring), `st::bloom_index_build`, `st::bloom_index_search_batch`
   (full path) and `st::bloom_index_search_batch_return_partial_response` with
   `st::fused_kmean_ann_with_partial_masks` (partial path). Build one seeded
   input set (a cluster-sorted int8 item table with its CSR offsets, centroids,
   int8 queries, item attributes and a query expression), save it with
   `torch.save`, call the ops on it in each venv, and save the outputs. The
   parity tests' input builders (`make_probe_family` in
   `retrieve/tests/parity/conftest.py`; `make_index`, `make_query` and the
   attribute helpers in `retrieve/tests/conftest.py`) produce inputs in the
   layout these ops take.
4. **Compare** the two outputs with `torch.equal` on the scores and on the ids
   up to ties at the k-th score.

The co-design records of the study (`codesign`, `codesign-laion30m`,
`nsweep-codesign` in
[`results/final-run-config/suites.yaml`](../../results/final-run-config/suites.yaml))
come from Meta's code as published at `21aa35e`, run eagerly through the
harness with its bloom filter over the whole index or over the searched
clusters only (`backend` `published`, `bloom_path` full / partial); the records
are in [`results/records/`](../../results/README.md#the-records).
