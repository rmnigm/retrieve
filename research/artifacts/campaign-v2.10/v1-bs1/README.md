# V1-BS1: LiNR V1's single-query GEMV and batched-row filter masks (campaign-v2.11 candidate)

Roadmap AFTER-QUEUE 3 (kernel order (2)): `torch.compile(max-autotune)` V1 beat our Triton V1.

| C3 measurement | compiled / ours |
|---|---|
| bs 1, 0.8-3 M (d-run, synth) | 0.53-0.55 |
| bs 1, 10-30 M (d-run, synth) | 0.88-0.89 |
| bs 16, selective multi-clause sweeps (v-pod1-run, real filters, `all4`) | 0.85-0.86 |

Raw outputs are on the Hub at `artifacts/v1-bs1` ([hub-index](../../hub-index.md)). Pod b, A100-SXM4-80GB, 2026-10-10.
**NOT CITABLE.**

## 1. Profile first (`prof.py`, `prof/`)
- **bs 1:** the dense matmul is the forward, for example 956 of about 1,280 µs at arXiv 3 M d128. cuBLAS runs `gemv2N`
  there, a CUDA-core kernel at about 0.8 TB/s. Top-k is about 180 µs, the mask 38-333 µs.
- **bs 16, `all4`:** the fused clause mask is the forward. It is 1336 of 2,012 µs on goodreads and 5226 of 7,386 µs on
  arXiv. The per-row kernel reloads the item tile's int64 attrs for every row.

## 2. The changes (all bit-exact: ids and scores `torch.equal`)
- **`gemv_scores`** (B = 1). cuBLAS picks its kernel by (B, N, D) (`gemv_probe.py`, `gemv1_probe.py`):

  | shape | cuBLAS kernel |
  |---|---|
  | B 1 at d128 / d256 (0.8-3 M) | `gemv2N` |
  | B 2 | `gemmSN` |
  | B 4+, and d768 / 10 M at B 1 | cutlass tensor-core |

  `gemv2N` accumulates one fp32 FMA per k, in order (`order_probe.py`; at B 2 `gemmSN` sums two k halves). The Triton
  GEMV runs that order, one item per lane: `torch.equal` and 2.0-2.2× faster. `PostfilterKNN.register_index` checks it
  against cuBLAS on 8 item rows, and `score()` takes it only at B = 1 where it matched. So scores are cuBLAS's on every
  shape: PubMed d768 keeps cuBLAS. `register_index` first creates the cuBLAS handle. The check would otherwise make the
  process's first cuBLAS call at the build's memory peak, which failed on PubMed 10 M in a two-layer process.
- **Batched-row masks** (B > 1): `clause_mask_scores` and `bloom_match_scores`. One program per item tile loads its
  attrs / signature words once for 16 rows. At B = 1 the per-row kernels are unchanged.

## 3. Gate (`gate_v1.py` against campaign-v2.11, library 1d390792; one process, swapped build order, 8 windows)
Rerun at v2.11 (Hub `artifacts/v1-bs1-v211`) after a first run against staging 1cba173 (= v2.10) with the same
numbers; ST-TOPK, the only library change between the two, does not touch V1's paths.
68 cells. Ids and scores are `torch.equal` in every cell, none slower:

| cells | after / before |
|---|---|
| goodreads d128 clause | 0.36-0.79 (`all4` bs 16 0.45, bs 64 0.36) |
| goodreads d128 bloom | 0.72-0.86 |
| arXiv d128 clause | 0.33-0.77 (`all4` bs 16 0.41) |
| arXiv d128 bloom | 0.68-0.83 |
| arXiv d256 clause | 0.35-0.79 |
| arXiv d256 bloom | 0.62-0.85 |
| PubMed d768 clause | 0.71-1.000 (`all5` bs 16 0.707; bs 1 1.000, cuBLAS kept) |
| PubMed d768 bloom | 0.91-1.000 |

- Each set was run eager and graph, at bs 1 / 16 / 64. PubMed is eager bs 1 / 16 plus graph bs 1: 32 bs-16 graphs would
  not fit beside two 10 M arms.
- The per-row mask kernels are unchanged source, so they keep their SASS.
- `earlier/` on the Hub holds the first gate runs, made before the GEMV's int64 fix and the early cuBLAS handle.
- **Suites:** library 886 passed on the staging-merged tree (ST-TOPK came in; it touches only SilverTorch's probe
  top-k); harness 572.
- **New tests:** `test_gemv_scores.py` (cuBLAS-equal scores whatever the calibration; the GEMV engages at d128 / 800 k;
  the torch backend never takes it); `gemv_scores` in the op registry.

## 4. Open
- V1's full-width top-k is now the largest piece at bs 16 (405 µs goodreads, 1.19 ms arXiv). ST-TOPK's block top-k
  (now on staging) applies to `masked_topk` unchanged in principle: a follow-up.
- At d768 / 10 M, B = 1, cuBLAS's tensor-core GEMM stays.

## Files
- `prof.py`: V1 per-kernel-family breakdown.
- `gemv_probe.py`, `gemv1_probe.py`, `order_probe.py`: cuBLAS kernel / order / speed probes.
- `gate_v1.py`: the keep-rule gate.
