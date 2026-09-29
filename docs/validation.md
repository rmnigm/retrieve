---
title: validation
created: 2026-09-26
updated: 2026-09-29
type: summary
tags: [validation, testing, harness]
sources: [retrieve/tests/, evaluation/tests/, evaluation/golden/, evaluation/results/, docs/artifacts/, docs/artifacts/hub-index.md]
contested: true
---

# What is validated now

The current state of every gate, and the measured results that stand.
A number here is paper material only where its row says **citable**
(CLAUDE.md rule 2). When a step changes a gate's state, update its row;
when a result is superseded, replace it. Raw scripts and outputs live
under [artifacts/](artifacts/).

**Environment all rows were measured on**: A100-SXM4-80GB, torch
2.10.0+cu128, triton 3.6.0, nvcc 12.4 (CUDA 12.8 runtime), Python 3.11,
Meta's `silvertorch` at the pinned commit. SM clocks cannot be locked; the
sampled clock under load was 1410 MHz. A new box is a new environment:
rerun the library suite before trusting any row on it.

## Library gates

| gate | state | notes |
|---|---|---|
| Library suite (`retrieve/tests`, GPU, `official` extra installed) | **green**, 799 passed at the last full run (branch `dev/pubmed-fixes`, cold inductor and Triton caches; nothing skipped) | no tolerance loosened; the cross-backend LiNR tolerances tightened from `1e-3` to `2e-6`; every tolerance stated at its call site; see [testing](system/testing.md) |
| Exact LiNR scoring precision (L1) | **green**: `PostfilterKNN` (± mask) and `PrefilterKNN` (dense; torch and Triton candidates) return fp32 scores within `2e-6` of an fp64 dot of their own fp16 inputs and select the fp64 top-100 up to ties inside that bound, on a YFCC-shaped case whose top-100 spans ~20 fp16 quanta; an fp16 score of the same dots misses (recall 0.88 / 0.92), asserted on the same inputs. Four of the five cases go red with the fp16 output restored (the Triton kernel already wrote fp32) ([`test_accumulation.py`](../retrieve/tests/parity/test_accumulation.py)) | the cuBLAS path's tensor-core accumulator measures 1.1e-6 from fp64 at D=128 (a plain fp32 sum ~1e-7); storage stays fp16 ([decisions](decisions.md#library)) |
| Triton vs reference ops, every parity file | **bit-exact** (`torch.equal` scores, ids up to ties) for `codesigned_probe_score` (+ bloom) and `codesigned_probe_score_exact` on the compact CSR layout (also against a loop-built oracle and, for bloom, the row-wise subset test), and for `oporp_1bit_match_topk_*`, `clause_mask`, the compaction ops; **not bit-exact** for `fused_masked_knn_topk` | the fused kernel's fp32 `tl.sum` and the reference's `bmm` reduce in different orders: measured drift ≤ 6e-8 at D ≤ 128 on unit-norm data, gated at `atol=1e-6` |
| Kernel identities (Q3) | **bit-exact**: indirect OPORP over `arange(N)` ≡ full scan; bloom op with an all-pass query signature ≡ no-bloom op; `clause_compact` ≡ `compact_mask(clause_mask)` on counts and the `[:counts]` prefix; row alone ≡ row in batch (fused, both probe scorers, OPORP); item-table permutation permutes ids only (fused, `codesigned_probe_score`, OPORP) | Triton only, on this box |
| Kernel cutoffs and degenerate rows (Q3) | **green**: both sides of `_P_BUCKETS[0]` / `_N_BUCKETS[0]` and `P % block` ∈ {0, 1} (read from the kernels' constants, regime asserted); `count = 0` / `1` rows give exact `(-1, -inf)` tails | |
| Addressing past 2³¹ elements (kernel-opt) | **green**: [`test_large_offsets.py`](../retrieve/tests/correctness/test_large_offsets.py), output axis (`B = 144, N = 16M`: `clause_mask`, `clause_compact`, `fused_masked_knn_topk`, `oporp_1bit_match_topk_indirect`) and item axis (a `[140M, 16]` table: `bloom_match`, `bloom_compact`, `clause_mask`, `clause_compact`, `oporp_1bit_match_topk_full`), planted answers. All five cases fail on the pre-change code (illegal address; `bloom_match` also overflowed `grid_y`), and each kernel's widening was mutation-checked | skipped below 48 / 24 GiB free. The probe scorers' `B·P ≥ 2³¹` output axis has the same `row_base` but no large case. Timing of the narrow path: all 13 cases within noise of the pre-change kernels, interleaved ([artifact](artifacts/kernel-opt/phase1.md); [kernels](system/kernels.md#addressing)) |
| Unwritten-slot (poisoned `torch.empty`) | **green** for every kernel writing into `torch.empty`: `fused_masked_knn_topk`, `codesigned_probe_score` (+ bloom), `codesigned_probe_score_exact`, OPORP full and indirect (no poison in the output); the two compaction ops write `[:counts]` only, so there the prefix equals the reference and the poison **survives** past `counts` (red if a tail store comes back) | allocation hit counted, so a moved allocation fails the test |
| Compaction tail (L2) | **green**: every consumer of `clause_compact` / `bloom_compact` bounds its reads by `counts`. Audited site by site ([kernels](system/kernels.md#clause_compact--fused-clause-eval--stream-compaction)), then with the tail poisoned to an out-of-range id across the whole suite: only the tests pinning the old `-1` tail and the reference `fused_masked_knn_topk` fed a Triton compaction (it gathered the whole row) failed; that op and the reference OPORP indirect now gather through `where(n < counts, id, 0)`. Gate: `LiNRV2` (Triton / torch, clause / bloom), `LiNRV3` (Triton / torch) and `combine_indices` return `torch.equal` results with the tail poisoned to `-1` and to `2**40`, in a subprocess; red with the reference op's bound removed or with `LiNRV2` dropping `counts` ([`test_compact_order.py`](../retrieve/tests/parity/test_compact_order.py)) | Triton compaction's full output is no longer a function of its inputs outside `torch.use_deterministic_algorithms(True)` |
| Op boundary (kernel-opt) | **green**: [`test_op_boundary.py`](../retrieve/tests/correctness/test_op_boundary.py). Every Triton op raises `ValueError` on a non-contiguous item-side table (11 cases); before, this was a silent per-call copy. All 11 go red with the check disabled. The 7 non-power-of-two rejection cases went with L4: those widths now pad (row *Non-power-of-two widths*) | [kernels](system/kernels.md) conventions |
| Tuner rule (kernel-opt) | **green**: `tune._choose` keeps the shipped default inside a 3 % noise band and rejects a candidate more than 5 % slower in any regime (`test_tune_smoke.py`, 4 synthetic cases). `--json-out` records SM clocks, commit and versions. No sweep has been run under the rule yet | |
| Build-time int8 quantization (kernel-opt) | **green**: `quantize_int8_global` quantizes chunk by chunk (optionally in `sort_perm` order, so SilverTorch writes its cluster-sorted table directly). The codes are `torch.equal` to the one-shot formula, and the transient stays under half the fp32 table. The one-shot form measured 672 MiB of transient on a 384 MiB table (`test_quantize.py`). This is the SilverTorch OOM on pubmed 10M × 768 that the E2 worker reported: two fp32 table copies on top of a 28.6 GiB table. **Re-run on pubmed 10M × 768**: the build completes on triton and official, peak allocated ≈ 38 GiB (quantize +0.5 GiB beyond its int8 output), official cells run end to end; Triton then stopped at the first forward on the power-of-two limit, which L4 lifted (row *Non-power-of-two widths*); D1's pubmed filter leg has run the Triton cells since (row *pubmed*) ([pubmed](artifacts/kernel-opt/pubmed/README.md)) | [artifact](artifacts/kernel-opt/phase4.md) |
| Build-time 1-bit quantization (pubmed-fixes) | **green**: `quantize_oporp_1bit` / `quantize_simhash_1bit` project and pack in 65,536-row chunks into a preallocated output. Bits `torch.equal` to the one-shot chain on arXiv (3.0M × 128) and yfcc10m (10M × 192), OPORP and SimHash; transient 10.0 → 0.26 GiB (arXiv OPORP), 50.3 → 0.55 GiB (yfcc10m OPORP); build ≤ 0.3 s either way. On pubmed (10M × 768) the one-shot chain OOMs ("Tried to allocate 28.61 GiB", as in D1) and the chunked build takes +2.2 GiB. `test_quantize.py` pins bit-exactness and a fixed per-chunk transient (red with chunking disabled) | [kernels](system/kernels.md#quantize_oporp_1bit-retrieveindexing), [artifact](artifacts/pubmed-fixes/README.md#fix-2-chunked-1-bit-build) |
| Short candidate lists and missing filters (kernel-opt) | **green** (`test_linr.py`): `fused_masked_knn_topk` rejects `P < k` with `ValueError` on both backends, where before Triton raised a torch `RuntimeError` and the twin padded; `PrefilterKNN` returns `[B, k]` with exact `(-1, -inf)` tails for `P` in {0, 3} and the backends agree; OPORP indirect returns `min(k, P)` columns on both backends (Triton returned `k`; at `P = 0` it crashed), compiled `dynamic=True` included; `LiNRV3` rejects `k > candidate_pool`; clause attrs to a filterless variant raise `ValueError` (an `assert` / `AttributeError` before) | |
| Non-power-of-two widths (L4) | **green**: every Triton kernel with a `tl.arange` over the embedding or word width (`codesigned_probe_score*`, `fused_masked_knn_topk` over `D`; OPORP, `bloom_match`, `bloom_compact` over `W`) aranges over `next_power_of_2` and masks the pad lanes to 0 on both sides; tables are not padded. Parity at the true width: D = 192 and 768 (probe scorers bit-exact, `fused_masked_knn_topk` at its file's atol 1e-6), OPORP `W = 3` / `12` bit-exact (full and indirect), bloom ops at `W = 3` / `12` `torch.equal` (op level only: `BloomFilter` keeps `m_bits` a power of two). Layers: SilverTorch (3 filter modes), LiNR V2 / V3 (clause, bloom) triton vs torch at D = 192 / 768, eager and compiled, all pass. At power-of-two widths the kernels are SASS-identical to `staging` `bcee67c` (30 of 30), and D = 128 timing is within noise (interleaved, 5 rounds, new/base 0.92-1.01 against noise bands 0.012-0.21). Run since on D1's yfcc10m (D = 192) and pubmed (D = 768) filter legs. Padding was not enough at `D_PAD = 1024`: row *Probe-scorer tile per width* | [kernels](system/kernels.md#padding), [artifact](artifacts/l4-pow2-pad/README.md) |
| Probe-scorer tile per width (pubmed-fixes) | **green**: `codesigned_probe_score*` ship a tile per `D_PAD` bound (`CONFIGS`, `_host.tile_for_width`): 256 × 4 up to 256 (the old default), 128 × 4 at 512, and at 1024 64 × 4 (none, bloom) / 128 × 4 (exact). The old fixed 256 × 4 tile at `D_PAD = 1024` (pubmed) compiled to 32 registers and a ~10 KB/thread spill stack (2,300-3,000 `LDL`; zero at `D_PAD ≤ 256`), 10-15× slower at B = 16 than the new tile (6.6 / 7.6 / 7.1 ms vs 0.63 / 0.53 / 0.51 ms none / bloom / exact, 2.3 M items, n_probe 24). Picks by `tune-kernels`' own rule, one width at a time (it keeps both D = 768 entries; at D = 384 it moved 256 × 4 → 128 × 4, geomean 0.95 / 0.92), and a realistic-layout sweep. Parity: both scorers `torch.equal` against the reference at D = 64, 128, 192, 768 through the shipped tile, the tile-cutoff cases at the D = 768 tile, and every sweep config `torch.equal` to 256 × 4 at every width. D = 128 / 192 compile to SASS identical to `staging` (7 of 7 cubins) and time within noise, interleaved 5 rounds (new/base 0.93-1.05 against bands 0.10-0.17); at D = 768 the public ops are 10-13× faster at B = 16 and ~2× at B = 1 | [kernels](system/kernels.md#silvertorch-kernels), [artifact](artifacts/pubmed-fixes/README.md#fix-1-d-aware-probe-scorer-tiles) |
| Op registry | **green**: all 10 `retrieve::` ops have a same-name, same-signature reference twin; none declares a mutable argument; every input `torch.equal` after the op and after its twin; `torch.library.opcheck` passes on `clause_compact` / `bloom_compact` (every utility, under deterministic mode, which fills the unwritten tail); the six official schemas equal the strings pinned for 21aa35e | |
| CUDA-graph replay | **bit-exact**: `reduce-overhead` SilverTorch (3 filter modes), LiNR V1-V3 × clause/bloom (and B=1), OneBit/SimHash replay on two queries they were not captured on, each `torch.equal` to eager, zero `cudagraph_skips` | |
| No wide intermediate in a forward | **green** on Triton for SilverTorch and LiNR V1-V4: no ≥ 3-d module-level tensor above B·N elements; the torch backend (which materializes `[B, ·, C, A_max]`) is the control | custom ops are opaque to the recorder: it checks module-level ops only, not a kernel's own buffers |
| Roofline anchors (Q3) | 4 GiB device-to-device copy **1754 GB/s** read+write (median of 20 windows, 1752-1766); empty Triton kernel launch **11.7 µs** median back to back (10.3-14.1) — `unstable`: SM clock 1275-1410 MHz during the copy, memory 1593 MHz ([artifact](artifacts/q3/roofline.json)) | the yardstick for any "% of HBM" claim is achieved GB/s over 1754, not a datasheet figure (1555 GB/s is the 40 GB part; this 80 GB part's spec is ~2039, of which the copy reaches 86 %). The "~1.4 TB/s ≈ 90 % of HBM" scorer claim appears nowhere in this repository and is **not yet validated** until restated against 1754 |
| L1 + L2 kernel timings | interleaved base / new, 3 rounds, 3M × 128, B=16 unless noted, SM clock 1410 MHz on every window but one (predictions written first, [artifact](artifacts/l1-l2/README.md#timing)). L1's cost, as predicted: `PostfilterKNN` masked **+38.7 %**, unmasked +35.5 %, `PrefilterKNN` dense +37.9 %, B=1 +7.2 %; the GEMM is unchanged, the extra is `torch.topk`'s radix select and `masked_fill` over an fp32 `[B, N]` buffer. L2's saving: `clause_compact` **−11.7 %** (B=1: −1.4 %, inside its 2.3 % noise band), `bloom_compact` −5.3 % (B=1 −5.2 %). Reference ops +0.9 / +1.7 % (the `counts` bound). Every other case, the Triton KNN kernels included, within 0.2 % | not yet validated as a harness number; end-to-end cell latency not remeasured |
| Official vs Triton, SilverTorch phase 3, int32 score path | **bit-exact** (`torch.equal`) on every regime; `tests/parity/test_official.py` 43/43 | the fp16 path differs only at ties (below) |
| Official bloom masks | no false negatives on AND (partial mask equals full mask on probed docs); NOT is a complement, so it has no false positives and may have false negatives | FPR at matched memory 0 on synthetic attributes; **not 0 on real attributes**, row below |
| Official vs Triton on bloom cells (pubmed-fixes) | **not a defect: the two arms' blooms admit different false positives.** D1's bloom cells whose `score_max_abs_diff` broke the fp16 floor (arXiv `c0_maincat` 0.095-0.100, pubmed `c0_mesh` 0.109, `c2_year` 0.011) reproduce to four digits under `bench run` (seed 0, both `n_probe`), and every returned id was checked against the exact clause: pubmed `c0_mesh` official holds 8,812-11,321 false-positive ids in its top-1000s (2,331-2,695 of 8,428 rows), Triton 244-303 (75-95 rows), and **every** row over 0.01 (1,570-1,836) holds one; `c2_year` 430-442 against 3-5, its one row over 0.01 likewise. An extra false positive shifts every later rank, and in pubmed's short rows (median 303-361 finite of 1,000) the neighbouring scores are far apart. No finite score on an `id == -1` slot and no `-inf` on a real id on either side, so no masking leak in the adapter or `masked_topk`. arXiv `c0_maincat` is one query (c0 = 29) whose official top-1000 held 315 false positives, and only on the attrs table before the license fix (its 452,732 INT64_MIN license values went into both arms' bloom indexes): on today's table the same cell has 0 false positives on both sides and sits at the 4.83e-4 floor | D1's `bloom_fp_rate` is the Triton filter's only (`oracle.pass_counts` over the harness's own filter module), so the record carries no official false-positive rate; arXiv's other pre-fix bloom cells were also built over the pre-fix attrs ([artifact](artifacts/pubmed-fixes/README.md#fix-3-official-vs-triton-bloom-divergence)) |
| Stream compaction determinism | **byte-identical** on counts and the `[:counts]` prefix across launches and processes (the tail is unwritten); every consumer's output byte-identical whatever the tail holds | ascending item order on both backends |
| LiNR V2 torch vs triton | **not exact, now inside 2e-6**: both return fp32 sums of the same fp16 products (L1), so they differ only in reduction order; id sets equal up to ties within `2e-6` in `test_linr.py`. The harness-level jaccard (0.99900-0.99975 with the torch side rounding to fp16) has not been remeasured | `linr_v1_filter_mask` and `linr_v4` are 1.0 / 0.0 |
| SilverTorch torch vs triton | **exact**: jaccard 1.0, diff 0.0 on every filter cell | |

## Harness gates

| gate | state | notes |
|---|---|---|
| Harness suite (`evaluation/tests`, CPU with `CUDA_VISIBLE_DEVICES=""`) | **green**: 257 passed / 1 skipped on `chore/code-cleanup` on the A100 box. L1's fp32-output `mm` / `bmm` (`out_dtype=`) have no CPU kernel; the three call sites now branch on `is_cuda` and use fp32 operands on CPU ([kernels](system/kernels.md#score-conventions)), CUDA unchanged | the skip reads the raw `_raw/yfcc10m/query.metadata.public.100K.spmat`, which is not on the box. pytest's `pythonpath` makes the suite import the checkout it sits in; before that, a worktree on the shared venv tested the main checkout |
| Convention gates in the harness suite | **green** | one reader per `RETRIEVE_*` variable (`test_env_readers.py`); the dependency direction with a file-count floor and stale-entry failures (`test_dependency_direction.py`, which dropped four stale edges and nine unused library names); every `config/*.yaml` × every suite through `load_matrix` (`test_config.py`). Each was checked to go red on a planted violation |
| `ruff check evaluation` (B, C4, SIM, RUF100, BLE001, PLC0415 on top of E, W, F, I, UP) | **clean**, ruff 0.15.6 | per-file ignores with reasons: ETL inline imports, goodreads' broad `except` (its own cleanup) ([evaluation](system/evaluation.md#lint)) |
| `ruff format --check evaluation` | **clean**, ruff 0.15.6 | the pre-commit format hook covers `retrieve/` and `evaluation/` ([evaluation](system/evaluation.md#lint)) |
| `scripts/check_doc_links.py` | **0 problems**: links, 120 backticked repo paths, 116 `bench` / `eval-data` / `train` subcommands | subcommands read from the click sources with `ast`; floors of 60 paths / 50 calls catch a broken pattern; `docs/log.md` (history) is exempt from the path check and `evaluation/data` (gitignored) is allowed. Needs the gitignored `articles/` present: a fresh clone or worktree reports its four `articles/` links broken. It checks link targets, not `#section` anchors |
| Golden baseline (`evaluation/golden/`) against the v2 harness | Rerun on `dev/l1-l2` against the pre-change tree on this box, same command both sides (quality only, eager; [artifact](artifacts/l1-l2/README.md#golden-cells)). 8 of the 9 comparable cells (goodreads LiNR V2-V3, SilverTorch triton / official at `n_probe` 24 and 32; arXiv SilverTorch triton at 24 and 32) are **identical** to the pre-change tree (max \|diff\| 0 over 24 oracle and held-out metrics each), after L1 and again after L2. `linr_v1_filter_mask` **moves by design** (L1's fp32 scores): every oracle metric up, `recall_oracle@100` +2.9e-5, `@1000` +1.5e-4, now equal to `linr_v2`/triton's (already fp32) at @100 and @500; held-out within ±3e-5. V1's golden cell was **re-derived** accordingly (old harness, L1 + L2's library, twice, identical on every quality column; [golden README](../evaluation/golden/README.md#provenance)), so V1 now meets it at 0.0: the old harness's new numbers equal the v2 harness's. The standing residuals are unchanged: arXiv `silvertorch` recall@100 2.0e-6; `linr_v2` 4.5e-4 and `linr_v3` 1.7e-5 against their golden JSONs, cause unidentified and predating `dev/kernel-opt` | the golden `torch` and `linr_v4` cells have no counterpart in today's suites; the old `linr_v4` residual (recall@100 7.3e-5) is not rerunnable |
| Graph latency against the golden | passes at batch 8 and 16 (ratio 0.96-1.03 at matched clock) | batch 1 is inside the golden's own repeat noise (up to 21 %) |
| CUDA-graph capture | every capturable arm captured, `cudagraph_skips == 0` | `official` is not capturable and records a null entry with a reason |
| Resume after SIGTERM | passes, no duplicate records | |
| Rerun byte-identical in quality | **passes** on all 12 rerun records of the goodreads leg | |
| Ids identical across `eager` and `graph` | **not run** | |
| Batch scaling `median_ms(bs=16) < 16 × median_ms(bs=1)` | passes, worst ratio 14.4 | |

## Official against our Triton reimplementation (citable, contested)

The head-to-head on goodreads (797,084 items) and arXiv (2,988,996 items),
d128, `n_lists 1024`, `n_probe` 24 and 32, k = 100. Full tables:
[artifacts/official-silvertorch/b3/tables.md](artifacts/official-silvertorch/b3/tables.md).
The paper's account is [paper/official-vs-reimplementation.md](paper/official-vs-reimplementation.md).

Contested: the head-to-head's own run record called these numbers "not
citable until the gate is re-run", while the paper section treats them as
citable because the step is closed. The user decides; until then, quote
them with that caveat. [kernels](system/kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend)
explains the mechanism behind the kernel-only split.

- **End to end, batch 16, our Triton is the fastest arm in every cell**:
  1.5× (goodreads none), 1.9× (goodreads bloom), 3.3× (goodreads clause),
  1.3× (arXiv none), 1.2× (arXiv bloom), 10.1× (arXiv clause) against
  official at `n_probe` 24, k = 100; across `n_probe` 24 and 32, 1.3-1.6×
  unfiltered, 1.2-1.9× bloom, 2.9-10.1× clause. At batch 1, 1.1-2.0×, inside batch-1 noise for bloom.
- **Meta's scorer kernel is faster than ours in every cell** (b3, padded layout): 1.15-3.2× on
  arXiv, 10.9-17.8× on goodreads. The cause was our padded IVF layout: on
  goodreads `n_probe × max_cluster_size` is 611,520 slots for about 18.7k
  real items (97 % padding); Meta reads a CSR. Official gives the time back
  in payload preparation (268-834 µs over 73-93 launches, against our
  34-58 launches).
- **After TF-9 / TF-1** (compact CSR probe layout + transposed bloom,
  branch `dev/kernel-opt`; b3 methodology re-run by
  [h2h.py](artifacts/kernel-opt/h2h.py), one run per tree, padded tree and
  compact tree in alternating processes with official as the control arm;
  [table](artifacts/kernel-opt/h2h.md)). Bloom kernel-only at batch 16,
  ours (scorer + mask) against official fp16 (scorer + mask): **0.90×
  goodreads (34.3 vs 38.1 µs), 1.24× arXiv (105.0 vs 84.5 µs)**, the
  1.3× gate green. Unfiltered 0.82× / 0.88×; exact 2.15× / 2.47×, where ours
  fuses a `C·A_max` attrs read that official receives as a precomputed mask
  outside the scorer class. Our scorer went 338 → 26 µs (none), 525 → 34
  (bloom), 400 → 53 (exact) on goodreads and 131 → 99, 238 → 105,
  268 → 197 on arXiv; eager phases-2+3 wall at batch 16 from 0.76 / 1.02 / 0.82
  to 0.42 / 0.62 / 0.44 ms on goodreads, and 0.45 / 0.78 / 0.48 to
  0.46 / 0.62 / 0.45 ms on arXiv, with 28-44 launches. Int32 parity against
  official: jaccard 1.000000, `score_max_abs_diff` 0 in all six cells.
  SM clock 1275-1410 MHz, sampled per row in the artifact. **Not yet
  validated** as paper material: one run per tree, not re-run after the
  merge to `staging`, and no end-to-end (`bench run`) rerun.
- **Phase 2 alone**: Meta's transposed bloom search beats our row-wise
  `bloom_match` 2.0× at 0.8M items and 6.1× at 3.0M; ours grows 3.3× with
  N, theirs 1.07×. This reproduces the paper's transposed-index claim
  (S13) with Meta's code. Our *fused* bloom forward still beats their
  partial-mask pipeline 1.8-2.1×.
- **Parity**: the int32 path is bit-exact on both datasets and all three
  filter modes. The fp16 path costs recall@100 3.1e-4 on arXiv and 4e-6 on
  goodreads, because arXiv's ranks 100 and 101 sit a fifth of an fp16 ulp
  apart in 95 % of rows.
- **Memory**: the official index is 1.6-2.75× smaller where our padding
  bites and equal on arXiv none/clause. The official exact path's 826 MiB
  peak and 7.2 ms at arXiv clause are our adapter's full-N mask packing,
  not Meta's code; every official exact number is an upper bound.
- **Not measured**: the controlled `bloom_path="full"` ablation (S9; encoded
  as the `codesign` suite, see [Campaign](#campaign-roadmap-d1-in-progress-not-yet-validated)), bloom
  FPR against width (both blooms showed zero false positives, so S8 needs
  roadmap D3), seeds 1-2, k in {500, 1000}.

## Campaign (roadmap D1): in progress, not yet validated

`filter` legs done, in scale order: goodreads (105/105 ok), arxiv (126/126
ok, after the license-clause fix and its all4 rerun), yfcc10m (7/7 ok, L4
confirmed working at D=192), pubmed (44/56 ok — see [datasets](#datasets)
for the `linr_v3` OOM and the 6 h timeout gaps). All four are on the Hub
(`d1/<dataset>`), NOT CITABLE (D1's gate is not green). `deep` and
`codesign` (S9) are running on arxiv, then goodreads.

**Code_version policy for this campaign (user decision, 2026-09-29):**
`code_version` is a whole-tree hash, but a `retrieve/src` fix does not
retroactively invalidate every existing D1 record — only the specific
arms the fix actually changes get rerun; everything else stays at its
original `code_version`, tracked here rather than through the hash. The
pubmed-fixes pass (`code_version` `72e5a90` → `c0e42d1`) is the first
case: its own gates prove goodreads, arxiv, yfcc10m and
pubmed's non-`silvertorch`-triton / non-`linr_v3` cells are numerically
identical between the two versions (SASS-identical kernels at D≤192,
`torch.equal` 1-bit codes at every width tested) — so only pubmed's
`silvertorch/triton` filter cells (retimed, wrong tile) and `linr_v3`
filter cells (previously OOM, now buildable) are rerun at `c0e42d1`.
Every other D1 record stays at `72e5a90`. Do not repeat this reasoning
from scratch for the next fix — re-derive it from that fix's own gates
each time, and update this paragraph.

A cross-scale filter comparison across all four datasets (same operating
point, same completed cells per arm) is in the `campaign-d1` chain,
2026-09-28 — not reproduced here since it is not yet citable and this
page tracks current state, not campaign narrative. Headline shape: exact
LiNR (V1, V2) holds ~0.998-0.9997 recall at every scale measured;
SilverTorch triton stays sub-millisecond at bs=1 through 10M items at
D=192 then rises to 3-4 ms at D=768; SilverTorch's recall at fixed
`n_probe` falls with scale (not scale-invariant — the `deep` sweep's
`n_probe` curves are the right place to read this); official vs triton
SilverTorch are within 1e-4 recall except on yfcc10m (official ~0.01
lower), and official is slower everywhere except pubmed, where it
overtakes triton at bs=16 QPS (598 vs 359). That crossover was the triton
probe scorer's register spill at `D_PAD = 1024`, since fixed (row
*Probe-scorer tile per width*); pubmed's triton cells predate the fix.

`codesign` (S9) wiring: `bloom_path` reaches `OfficialConfig` (unit tests
in `test_algos.py` / `test_config.py`); one early smoke cell (pre-dating
the real `codesign` suite run) found `full` faster than `partial` on one
goodreads cell, one seed, unlocked clocks — not a finding, the real
`codesign` suite run (in progress) supersedes it.

## Datasets

| dataset | state |
|---|---|
| goodreads | staged, layout checked, oracles built |
| arxiv | staged, layout checked. **Clause 1 (license) was corrupt** (`dev/arxiv-etl-fix`, merged 2026-09-27): 452,732 null-license items (15.1 %) and 1,528 of the 10,000 queries carried INT64_MIN instead of bucket `none` ([datasets](system/datasets.md#arxiv)), corrupting every recorded `all4` cell's attribute space (the other sweeps don't read clause 1). **Fixed and installed**: the patched `item_attrs_narrow.pt`/`eval_split.parquet` are live in `/data/arxiv-papers` (verified independently: exactly 452,732 cells changed, all in clause 1, every other cell byte-identical, `bench check` ok at d64/128/256) and republished to `pinkmeme/eval-arxiv-papers`. Pre-fix originals kept as `*.pre-license-fix.bak` in `/data/arxiv-papers`. **Every arxiv `all4` cell recorded before this fix (in `d1/arxiv` on the Hub) is stale and not citable** until re-run under the corrected attrs — `--resume` won't catch this on its own since those cells are `ok`, not `failed`; needs an explicit forced rerun of just the `all4` sweep (42 cells) |
| yfcc10m | our exact oracle reproduces the shipped filtered ground truth. **The exact-algorithm gate passes since L1**: `linr_v1_filter_mask`/triton clause `tags_and`, eager, `--skip-perf`, 10,000 queries: `recall_oracle@1000` **0.9944** (0.9652 and `QualityGateError` on the pre-L1 tree, same command); `LiNRV2(backend="torch")` against the same oracle blob 0.9944 (script). `linr_v2`/triton could not run at d192 before L4's padding (row *Non-power-of-two widths*); not yet run here since. The residual 0.006 is the fp16 item storage: an fp32 table gives 1.0 ([artifact](artifacts/l1-l2/README.md)); not yet validated beyond this one cell |
| pubmed | 10 M slice staged locally (A100 box, 2026-09-26), on the Hub as `d1/pubmed` since D1's filter leg (2026-09-28). D=768 (L4-padded): `linr_v1_filter_mask`, `linr_v2`, `silvertorch` (triton + official) all run; `linr_v2` recall_oracle@100 0.998, matching V1. `linr_v3` could not build in D1 (`torch.OutOfMemoryError` on all 8 cells: the one-shot 1-bit build's full-corpus temporaries); the chunked build fixes it (row *Build-time 1-bit quantization*). One cell since, clause `c0_mesh` seed 0, `--skip-perf`: `recall_oracle@100` 0.9985, `@1000` 0.9995 (V1 on the same cell 0.9985 / 0.9997), index 17.1 GiB, 50.8 GB reserved; the 8 D1 cells are still to run. The `silvertorch/triton` cells were timed on the spilling `D_PAD = 1024` tile (row *Probe-scorer tile per width*) and need a rerun. 52/56 filter cells recorded (44 ok, 8 `linr_v3` OOM, 4 never ran — `linr_v2` ×1 and `silvertorch/triton` ×3 hit the campaign's 6 h per-group timeout, itself too short for D=768 × 10M cells at 20-46 min each); the 4 missing cells are a pending resume pass, not yet run. NOT CITABLE (D1's gate is not green) |
| openalex | 10 M slice staged locally (A100 box, 2026-09-26; OpenAlex fallback for Semantic Scholar SPECTER2, no API key), not on the Hub: `bench check` passes. One filter cell, `linr_v1_filter_mask`/triton clause `field_era`, eager, `--skip-perf` (`partial`): `recall_oracle@1000` 0.9959, held-out `recall@100` 0.505, `recall@1000` 0.741, n = 10,000. Scoped down from an initial 15 M encode after `bench/oracle.py`'s `item_embs.t().contiguous()` OOMed at that size (a second full fp32 copy on top of the item table; the real 768-d limit is ~11-12 M, not 15 M) — 10 M items resharded from the already-encoded 15 M vectors (`torch.equal`-verified), matching the pubmed precedent. SilverTorch-`triton` and LiNR V2/V3 were blocked by the power-of-two `D` limit as on pubmed until L4, not yet run here since; not yet validated ([artifacts](artifacts/e3-openalex/)) |
| kuairand | staged locally (A100 box, 2026-09-26); the dataset files are not on the Hub, the gSASRec checkpoint `gsasrec-d128-shared` is (private, `pinkmeme/eval-kuairand`). `bench check` passes. Trained with one shared item table, `--negs-per-pos 128` (256 OOMs), measured peak 82.9 GB allocated; stopped by patience 5 after epoch 18, best val NDCG@10 0.0361 at epoch 13, test NDCG@10 0.0088 / Recall@100 0.0024 (the 4× val→test drop is partly item cold start, 55 % of test targets never clicked in train; the rest is unexplained). One filter cell, `linr_v1_filter_mask`/triton clause `t_cat1`, eager, `--skip-perf` (`partial`): pass rate 0.0362, `recall_oracle@1000` 0.9996, held-out `recall@100` 0.020, `recall@1000` 0.064, n = 9,910; not yet validated ([artifacts](artifacts/e4-kuairand/)) |

## Still unverified

- Any compiled (`graph`-mode) measurement taken on a warm inductor cache after a library
  change: the FX-graph / AOT-autograd caches replay a stale `triton_op` body
  ([testing](system/testing.md#running)). This is measured in the library suite; whether any
  recorded harness cell was affected has not been checked.

- L1's fp32 scores under `graph` mode (`torch.mm(..., out_dtype=)` inside a CUDA-graph capture):
  the library's compile and `reduce-overhead` replay tests pass, and D1's `graph` cells record
  `ok`, but their ids have not been compared with eager (next line).
- Clause 5 of the campaign gate (ids identical across modes).
- The official exact path without our adapter's mask packing.
- Every row above on a new box, until the library suite has run there.
