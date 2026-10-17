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
adapted. Every change was gated against
the published code at that commit, built in the same environment, before the
published code was removed from the workspace. This page gives the gate, its
recorded results, what this repository still checks, and how to rerun the
comparison against the published code.

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
| ids and scores `torch.equal` to the published code, the ≤ 10M grid | 276 / 276 cells at the final revision; 1,488 / 1,488 on a second machine at an earlier revision |
| the same at 30M (LAION d256) | bloom partial + full 64 / 64 at the final revision; no filter and exact filter 72 / 72 at an earlier revision |
| latency, adapted / published, eager | faster on every cell: median 0.230, max 0.764 over 300 cells; 30M bloom median 0.279, max 0.467 |
| configured bloom width with the exact recheck (an adapted-code option, off in the study) | 0 false positives on every cell; the rechecked bloom `torch.equal` to the exact filter |

The upstream repository has one later commit (`22c2007`, CCCL 3 / CUDA 13
compatibility), applied verbatim; on CUDA 12.8 it selects the old code path,
so the compiled behaviour is unchanged.

## What this repository checks

[`retrieve/tests/parity/test_official.py`](../../retrieve/tests/parity/test_official.py)
gates the adapted ops against an independent dense torch oracle, through the
ops and through the `SilverTorch` module (GPU only; see
[testing](testing.md)). It does not load the published code.

## Rerunning the comparison against the published code

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
   `torch.save`, call the ops on it in each venv, and save the outputs.
4. **Compare** the two outputs with `torch.equal` on the scores and on the ids
   up to ties at the k-th score.

The co-design suites of the final run (`codesign`, `codesign-laion30m`, `nsweep-codesign` in
[`results/final-run-config/suites.yaml`](../../results/final-run-config/suites.yaml)) are
historical in the same way: they ran the published code through a harness arm (`backend`
`published`, `bloom_path` full / partial) that left the harness with the published code, and the
shipped harness rejects them.

The cross-build harness that produced the recorded results was not carried
into this repository; step 3 is the part a rerun has to write. The input
builders `test_official.py` uses (`make_probe_family` in
`retrieve/tests/parity/conftest.py`; `make_index`, `make_query` and the
attribute helpers in `retrieve/tests/conftest.py`) produce inputs in the layout
these ops take.
