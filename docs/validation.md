---
title: validation
created: 2026-09-26
updated: 2026-10-07
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
sampled clock under load was 1410 MHz. A new pod is a new environment:
rerun the library suite before trusting any row on it.

## Library gates

| gate | state | notes |
|---|---|---|
| Library suite (`retrieve/tests`, GPU, `official` extra installed) | **green**, 799 passed at the last full run (branch `dev/pubmed-fixes`, cold inductor and Triton caches; nothing skipped) | no tolerance loosened; the cross-backend LiNR tolerances tightened from `1e-3` to `2e-6`; every tolerance stated at its call site; see [testing](system/testing.md) |
| Exact LiNR scoring precision (L1) | **green**: `PostfilterKNN` (± mask) and `PrefilterKNN` (dense; torch and Triton candidates) return fp32 scores within `2e-6` of an fp64 dot of their own fp16 inputs and select the fp64 top-100 up to ties inside that bound, on a YFCC-shaped case whose top-100 spans ~20 fp16 quanta; an fp16 score of the same dots misses (recall 0.88 / 0.92), asserted on the same inputs. Four of the five cases go red with the fp16 output restored (the Triton kernel already wrote fp32) ([`test_accumulation.py`](../retrieve/tests/parity/test_accumulation.py)) | the cuBLAS path's tensor-core accumulator measures 1.1e-6 from fp64 at D=128 (a plain fp32 sum ~1e-7); storage stays fp16 ([decisions](decisions.md#library)) |
| Triton vs reference ops, every parity file | **bit-exact** (`torch.equal` scores, ids up to ties) for `codesigned_probe_score` (+ bloom) and `codesigned_probe_score_exact` on the compact CSR layout (also against a loop-built oracle and, for bloom, the row-wise subset test), and for `oporp_1bit_match_topk_*`, `clause_mask`, the compaction ops; **not bit-exact** for `fused_masked_knn_topk` | the fused kernel's fp32 `tl.sum` and the reference's `bmm` reduce in different orders: measured drift ≤ 6e-8 at D ≤ 128 on unit-norm data, gated at `atol=1e-6` |
| Kernel identities (Q3) | **bit-exact**: indirect OPORP over `arange(N)` ≡ full scan; bloom op with an all-pass query signature ≡ no-bloom op; `clause_compact` ≡ `compact_mask(clause_mask)` on counts and the `[:counts]` prefix; row alone ≡ row in batch (fused, both probe scorers, OPORP); item-table permutation permutes ids only (fused, `codesigned_probe_score`, OPORP) | Triton only, on the A100 |
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
| Harness suite (`evaluation/tests`, CPU with `CUDA_VISIBLE_DEVICES=""`) | **green**: 282 passed / 4 skipped on `dev/h5` (an A100 pod, CPU). L1's fp32-output `mm` / `bmm` (`out_dtype=`) have no CPU kernel; the three call sites now branch on `is_cuda` and use fp32 operands on CPU ([kernels](system/kernels.md#score-conventions)), CUDA unchanged | the skip reads the raw `_raw/yfcc10m/query.metadata.public.100K.spmat`, which is not staged by default. pytest's `pythonpath` makes the suite import the checkout it sits in; before that, a worktree on the shared venv tested the main checkout |
| Record key carries the input identity (H2) | **green** on `dev/h2` (CPU): two jobs differing only in `--checkpoint` key apart; `--checkpoint` lands as the record's `inputs`; a real D1 arxiv key block (schema 2) keeps the resume key today's arxiv job computes; a schema-2 goodreads record derives `gsasrec-d128-drop0.5-id`; `bench report` refuses two `inputs` under one `(dataset, dim)`. On the real Hub records ([resume_done_set.py](artifacts/h2/resume_done_set.py), pre-H2 tree vs H2 tree over today's matrix at the records' `code_version`) the `--resume` done set is identical: `d1/arxiv` 126 / 126, `d1/arxiv-deep-partial` 803 / 803, `d1/pubmed` 44 / 44 cells | the derivation covers goodreads and the text datasets only; a schema-2 record of another checkpoint dataset (kuairand's E4 cell, yambda's retired suite) raises ([evaluation](system/evaluation.md#resume)). `SCHEMA_VERSION` 3 |
| Held-out metric with no target is null (H7) | **green** on `dev/h7` (CPU): a filter cell whose held-out targets the exact mask all excludes records `heldout: {…: null, n: 0}` end to end (`test_run.py`); a pre-H7 record with `n == 0` and 0.0 means aggregates to null (`test_records.py`). On the real `d1/pubmed` records ([pubmed_null_heldout.py](artifacts/h7/pubmed_null_heldout.py)) exactly the 6 `c3_journal_reverse` cells with an empty held-out side (V1, V2, SilverTorch triton / official at `n_probe` 24 and 32) read null, oracle metrics unchanged, no rerun | `bench report`'s tables skip null values (`_reduce`); the exact-algo oracle gate skips a null oracle recall |
| `bench report` states true provenance (H5) | **green** on `dev/h5` (CPU): a non-citable report names its actual reasons (gate not green, `failed`, `partial` with its `partial_reasons` counted, `env.dirty`, an off-main branch) in the `.tex` banner, the caption mark, the figure watermark and `report.md`; `test_report.py` pins the text of each, and no artifact says "pre-campaign" or "predate" | the reasons are the existing rule-2 checks; none was added or dropped |
| Convention gates in the harness suite | **green** | one reader per `RETRIEVE_*` variable (`test_env_readers.py`); the dependency direction with a file-count floor and stale-entry failures (`test_dependency_direction.py`, which dropped four stale edges and nine unused library names); every `config/*.yaml` × every suite through `load_matrix` (`test_config.py`). Each was checked to go red on a planted violation |
| `ruff check evaluation` (B, C4, SIM, RUF100, BLE001, PLC0415 on top of E, W, F, I, UP) | **clean**, ruff 0.15.6 | per-file ignores with reasons: ETL inline imports, goodreads' broad `except` (its own cleanup) ([evaluation](system/evaluation.md#lint)) |
| `ruff format --check evaluation` | **clean**, ruff 0.15.6 | the pre-commit format hook covers `retrieve/` and `evaluation/` ([evaluation](system/evaluation.md#lint)) |
| `scripts/check_doc_links.py` | **0 problems**: links, 126 backticked repo paths, 131 `bench` / `eval-data` / `train` subcommands | subcommands read from the click sources with `ast`; floors of 60 paths / 50 calls catch a broken pattern; `docs/log.md` (history) is exempt from the path check and `evaluation/data` (gitignored) is allowed. Needs the gitignored `articles/` present: a fresh clone or worktree reports its four `articles/` links broken. It checks link targets, not `#section` anchors |
| Golden baseline (`evaluation/golden/`) against the v2 harness | Rerun on `dev/l1-l2` against the pre-change tree on the A100, same command both sides (quality only, eager; [artifact](artifacts/l1-l2/README.md#golden-cells)). 8 of the 9 comparable cells (goodreads LiNR V2-V3, SilverTorch triton / official at `n_probe` 24 and 32; arXiv SilverTorch triton at 24 and 32) are **identical** to the pre-change tree (max \|diff\| 0 over 24 oracle and held-out metrics each), after L1 and again after L2. `linr_v1_filter_mask` **moves by design** (L1's fp32 scores): every oracle metric up, `recall_oracle@100` +2.9e-5, `@1000` +1.5e-4, now equal to `linr_v2`/triton's (already fp32) at @100 and @500; held-out within ±3e-5. V1's golden cell was **re-derived** accordingly (old harness, L1 + L2's library, twice, identical on every quality column; [golden README](../evaluation/golden/README.md#provenance)), so V1 now meets it at 0.0: the old harness's new numbers equal the v2 harness's. The standing residuals are unchanged: arXiv `silvertorch` recall@100 2.0e-6; `linr_v2` 4.5e-4 and `linr_v3` 1.7e-5 against their golden JSONs, cause unidentified and predating `dev/kernel-opt` | the golden `torch` and `linr_v4` cells have no counterpart in today's suites; the old `linr_v4` residual (recall@100 7.3e-5) is not rerunnable. The goodreads golden cells are on the gSASRec checkpoint `gsasrec-d128-drop0.5-id`, which `config/goodreads.yaml` no longer points at ([encoder switch](#encoder-switch-evals-to-redo)); the golden run pins it with `bench run --checkpoint` ([golden commands](../evaluation/golden/README.md#exact-commands)). **Not yet rerun with the pin** (H2's golden run is waiting for a GPU slot) |
| Graph latency against the golden | passes at batch 8 and 16 (ratio 0.96-1.03 at matched clock) | batch 1 is inside the golden's own repeat noise (up to 21 %) |
| CUDA-graph capture | every capturable arm captured, `cudagraph_skips == 0` | `official` is not capturable and records a null entry with a reason |
| Resume after SIGTERM | passes, no duplicate records | |
| Rerun byte-identical in quality | **passes** on all 12 rerun records of the goodreads leg | |
| Ids identical across `eager` and `graph` | **not run** | |
| Batch scaling `median_ms(bs=16) < 16 × median_ms(bs=1)` | passes, worst ratio 14.4 | |

## Official against our Triton reimplementation (citable, contested)

The head-to-head on goodreads (797,084 items) and arXiv (2,988,996 items),
d128 (goodreads on the gSASRec `gsasrec-d128-drop0.5-id` embeddings, since replaced as the
harness encoder: [encoder switch](#encoder-switch-evals-to-redo)), `n_lists 1024`, `n_probe` 24 and 32, k = 100. Full tables:
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

`filter` legs done, in scale order: goodreads (105/105 ok, on the gSASRec
`gsasrec-d128-drop0.5-id` queries and items, **stale since the encoder switch**:
[encoder switch](#encoder-switch-evals-to-redo)), arxiv (126/126
ok, after the license-clause fix and its all4 rerun), yfcc10m (7/7 ok, L4
confirmed working at D=192), pubmed (44/56 ok — see [datasets](#datasets)
for the `linr_v3` OOM and the 6 h timeout gaps). All four are on the Hub
(`d1/<dataset>`), NOT CITABLE (D1's gate is not green). `deep` and
`codesign` (S9): arxiv `deep` holds 803 of 870 cells (`d1/arxiv-deep-partial`), arxiv
`codesign` and goodreads's two legs have no records; goodreads runs on the E1c encoder. What
remains is roadmap D1-A..G.

**Code_version policy for this campaign (user decision):**
`code_version` is a hash of the `retrieve/src/retrieve` subtree (`bench/measure.py`), but a fix there does not
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

`codesign` (S9) wiring: `bloom_path` reaches `OfficialConfig` (unit tests
in `test_algos.py` / `test_config.py`); one early smoke cell (pre-dating
the real `codesign` suite run) found `full` faster than `partial` on one
goodreads cell, one seed, unlocked clocks — not a finding; the `codesign`
legs (roadmap D1-B, D1-D) decide.

### Cross-scale filter comparison (not yet validated)

The four `filter` legs at the same operating point, over the cells every
arm completed; table and script on the Hub in `artifacts/d1-campaign`
([hub-index](artifacts/hub-index.md)). Redone at roadmap D1-G. Goodreads
is on the gSASRec embeddings (**stale**, [encoder switch](#encoder-switch-evals-to-redo));
yfcc10m is one cell; pubmed is 6 cells, its Triton SilverTorch timed on
the spilling tile (row *Probe-scorer tile per width*).

| dataset | arm | `recall_oracle@100` mean (min) | bs 1 ms | bs 16 QPS | index MiB |
|---|---|---|---|---|---|
| goodreads (stale) | V1 | 0.9997 | 0.558 | 7,456 | 292 |
| | V2 | 0.9997 | 0.611 | 5,714 | |
| | V3 | 0.8772 (0.6946) | 1.309 | | |
| | official, `n_probe` 24 / 32 | 0.8364 / 0.8708 | ~1.02 | | |
| | triton, `n_probe` 24 / 32 | 0.8364 / 0.8708 | 0.561 / 0.567 | ~27.2k | |
| arxiv | V1 | 0.9988 | 1.520 | | |
| | V2 | 0.9988 | 0.905 | | |
| | V3 | 0.7598 | 1.393 | | |
| | official | 0.8373 / 0.8651 | ~1.11 | | |
| | triton | 0.8375 / 0.8653 | 0.551 / 0.615 | ~25.9k | |
| yfcc10m | V1 | 0.9886 | 9.327 | | |
| | V2 | | 24.076 | | |
| | V3 | 0.4967 | 23.43 | | |
| | official | 0.5349 / 0.5595 | 7.44 | | |
| | triton | 0.5454 / 0.5700 | 0.438 / 0.532 | | |
| pubmed | V1 | 0.9978 | 11.468 | | |
| | V2 | | 6.031 | | |
| | official | 0.6744 / 0.7115 | 3.003 | 598 | |
| | triton (pre-fix tile) | 0.6745 / 0.7116 | 3.110 / 4.015 | 359 / 272 | |

Exact LiNR (V1, V2) holds ~0.998-0.9997 recall at every scale; SilverTorch's
recall at fixed `n_probe` falls with scale (the `deep` sweep's `n_probe`
curves are where to read it); Triton SilverTorch stays sub-millisecond at
bs 1 through 10 M items at D = 192.

**Official against Triton by filter kind** (ms at bs 1 / bs 16, official
first). Official clause cells time our `pack_mask` adapter (51.67 MiB × bs,
flat in `n_probe`), not Meta's kernels.

| dataset | clause | bloom |
|---|---|---|
| goodreads (stale) | 1.01 / 2.44 vs 0.53 / 0.53 | 1.16 / 1.33 vs 0.73 / 0.72 |
| arxiv | 1.11 / 7.07 vs 0.53 / 0.54 | 1.10 / 1.32 vs 0.70 / 0.72 |
| yfcc10m (clause only; Meta's kernels never timed) | 7.44 / 72.6 vs 0.49 / 2.92 | — |
| pubmed (Triton pre-fix tile) | 3.00 / 26.96 vs 3.61 / 52.3 | 1.11 / 2.22 vs 3.31 / 47.1 |

**yfcc10m: official recall ~0.0105 below Triton.** The official arm runs
its default `score_path="fp16"`; the bit-exact gate covers the int32 path
only. Held-out recall@100 is identical (0.95493 / 0.95915), but mrr@100
0.7783 → 0.7626, ndcg@100 0.8198 → 0.8074, `jaccard@100` 0.8896.

**Results audit** (the D1 records as of the pause):

- 39 arxiv `deep` official cells hold only the parity `reference` entry,
  so their parity cannot be checked; pubmed bloom `c0c2` official has no
  Triton partner.
- Triton `n_probe` 4 is slower than 8 in `graph` mode at `n_lists` 1664,
  bs 8 / 16 (0.200 vs 0.174 ms, 180 of 180 pairs): `n_probe` 4 is
  dominated.
- Eager-vs-graph bit-exactness cannot be checked from the records (the
  ids are not stored per mode).
- `unstable` fired on 349 of 644 records, 268 from clock drift alone
  (1275 ↔ 1410 MHz).
- Seed spread ≤ 0.0076 recall; V1 and V2 recall agree to 1.7e-5.
- Pubmed `c3_journal_reverse` records carry held-out recall 0.0 where it
  should be null (roadmap H7).

## Encoder switch: evals to redo

The harness encodes the sequential datasets with the current trainer's E1c checkpoints
([decisions](decisions.md#sequential-encoder)): goodreads and yambda-500m
`sasrec-ssm-logq-d{dim}`. yambda-5b keeps `gsasrec-d{dim}` (no new model). Both yambda datasets are out of the study. The new checkpoints are L2-normalized, so both the item table and
the queries change: every oracle, recall and latency number on these datasets is a different
experiment. Text datasets (arxiv, yfcc10m, pubmed, openalex) are unaffected.

The record key carries the input identity (`inputs`: the checkpoint directory name, or the
text `content_dir`; [evaluation](system/evaluation.md#resume)), so a rerun on the new encoder
keys apart from its gSASRec record, and `bench report` refuses a tree that mixes the two
under one `(dataset, dim)`.

| record set | encoder it used | state now | redo |
|---|---|---|---|
| D1 `filter` leg, goodreads d128 (`d1/goodreads`, 105 records) | `gsasrec-d128-drop0.5-id` | **stale**, kept on the Hub as the gSASRec run | full leg on `sasrec-ssm-logq-d128` (roadmap D1) |
| D1 `deep` / `codesign`, goodreads | not started | — | run on the new encoder only (roadmap D1) |
| Official vs Triton head-to-head, goodreads (`b3`, kernel-opt `h2h`) | `gsasrec-d128-drop0.5-id` | stands as a gSASRec-embedding measurement; not comparable to D1's new goodreads cells | rerun on the new encoder before F2 quotes goodreads |
| Golden baseline, goodreads cells (`evaluation/golden/`) | `gsasrec-d128-drop0.5-id` | stands; the gate compares harnesses on fixed inputs | keep on the gSASRec checkpoint (`bench run --checkpoint`); no rerun |
| kuairand filter cell (`artifacts/e4-kuairand`) | `gsasrec-d128-shared` (deleted) | stale, not reproducible; KuaiRand is out of the study | none |
| `quality` suite, yambda-500m | `gsasrec-d{dim}-drop0.5` | retired suite; no current records; yambda is out of the study | none |
| arxiv, yfcc10m, pubmed, openalex (every suite) | text embeddings | unaffected | none |

## Datasets

| dataset | state |
|---|---|
| goodreads | staged, layout checked. Oracles built for the gSASRec d128 inputs only; the harness now encodes with `sasrec-ssm-logq-d{dim}` ([encoder switch](#encoder-switch-evals-to-redo)), whose oracles the next run builds (the blob fingerprint covers the item and query tensors, so a gSASRec blob is never read for them) |
| arxiv | staged, layout checked, on the Hub as `pinkmeme/eval-arxiv-papers`. Clause 1 (license) maps the 452,732 null-license items (15.1 %) and 1,528 of the 10,000 queries to bucket `none` ([datasets](system/datasets.md#arxiv)); a table built before that fix carried INT64_MIN there (check: `item_attrs_narrow.pt` min = -1, `eval_split.parquet` `query_attrs_narrow` min = 0). Only the `all4` sweep reads clause 1; `d1/arxiv`'s 42 `all4` cells are on the fixed table (numbers unchanged to 4 decimals against the corrupt one) |
| yfcc10m | our exact oracle reproduces the shipped filtered ground truth. **The exact-algorithm gate passes since L1**: `linr_v1_filter_mask`/triton clause `tags_and`, eager, `--skip-perf`, 10,000 queries: `recall_oracle@1000` **0.9944** (0.9652 and `QualityGateError` on the pre-L1 tree, same command); `LiNRV2(backend="torch")` against the same oracle blob 0.9944 (script). `linr_v2`/triton could not run at d192 before L4's padding (row *Non-power-of-two widths*); D1's yfcc10m filter leg has since run it (7/7 ok, [Campaign](#campaign-roadmap-d1-in-progress-not-yet-validated)). The residual 0.006 is the fp16 item storage: an fp32 table gives 1.0 ([artifact](artifacts/l1-l2/README.md)); not yet validated beyond this one cell |
| pubmed | 10 M slice under `pubmed-medcpt`, not on the Hub (`pinkmeme/eval-pubmed` is registered, not published: [datasets](system/datasets.md#huggingface-io)). D1's filter leg is `d1/pubmed`. D=768 (L4-padded): `linr_v1_filter_mask`, `linr_v2`, `silvertorch` (triton + official) run; `linr_v2` recall_oracle@100 0.998, matching V1. `linr_v3` builds since the chunked 1-bit build (row *Build-time 1-bit quantization*): one cell, clause `c0_mesh` seed 0, `--skip-perf`: `recall_oracle@100` 0.9985, `@1000` 0.9995 (V1 on the same cell 0.9985 / 0.9997), index 17.1 GiB, 50.8 GB reserved. `d1/pubmed`: 52/56 filter cells recorded (44 ok, 8 `linr_v3` OOM before that fix, 4 never ran — `linr_v2` ×1 and `silvertorch/triton` ×3 hit the campaign's 6 h per-group timeout); the `silvertorch/triton` cells were timed on the spilling `D_PAD = 1024` tile (row *Probe-scorer tile per width*). Roadmap D1-E reruns those arms. NOT CITABLE (D1's gate is not green) |
| openalex | 10 M slice (OpenAlex fallback for Semantic Scholar SPECTER2, no API key), not on the Hub: roadmap E5 restages it. `bench check` passed on the staged copy. One filter cell, `linr_v1_filter_mask`/triton clause `field_era`, eager, `--skip-perf` (`partial`): `recall_oracle@1000` 0.9959, held-out `recall@100` 0.505, `recall@1000` 0.741, n = 10,000. Scoped down from an initial 15 M encode after `bench/oracle.py`'s `item_embs.t().contiguous()` OOMed at that size (a second full fp32 copy on top of the item table; the real 768-d limit is ~11-12 M, not 15 M) — 10 M items resharded from the already-encoded 15 M vectors (`torch.equal`-verified), matching the pubmed precedent. SilverTorch-`triton` and LiNR V2/V3 were blocked by the power-of-two `D` limit as on pubmed until L4, not yet run here since; not yet validated ([artifacts](artifacts/e3-openalex/)) |
| kuairand | **out of the study** ([decisions](decisions.md#datasets)). The one filter cell recorded (`linr_v1_filter_mask`/triton `t_cat1`, recall_oracle@1000 0.9996) used the deleted gSASRec `gsasrec-d128-shared` checkpoint and is not reproducible; its logs are `artifacts/e4-kuairand` on the Hub. The E1c trainer results below stand as trainer results |

### Trainer inputs with `timestamps` (data gates, not citable)

The three trainer-input ETLs (yambda, goodreads, kuairand) write a
`timestamps` column next to `item_ids` ([datasets](system/datasets.md#yambda)).
Environment: the pod, CPU only, branch `dev/hstu-etl`
([artifacts](artifacts/seqrec-encoder/etl-timestamps/): `gates.py` and its JSON).
These are data-integrity gates, not results; nothing here is citable.

| gate | state | notes |
|---|---|---|
| G-yambda: `item_id_map.json` | **passes**: the re-prep (now `/data/yambda-500m/trainer`), the previous prep (`trainer.old`) and the Hub copy are byte-identical, sha256 `9cd9535f…6c3b02` | `/data/yambda-500m/trainer/` (prep 97 s, peak RSS 15.7 GB) |
| G-yambda: `item_ids` / `targets` row for row | **train passes** (91,806 rows, `equals`); **val/test fail row for row, pass as row multisets** (45,796 / 45,932 rows; 11 and 0 rows in the same position) | Mechanism: `cmd_prep` builds val/test with `uid` joins (no `maintain_order`, no sort) and then drops `uid`, so their row order is a fresh draw each run. The old `trainer/test` and the Hub `test.parquet` differ from each other the same way (same rows, 1 in the same position). Contents are unchanged; the order is not reproducible and nothing on disk may be aligned to it by position. Not fixed (out of this step) |
| G-yambda: `timestamps` | **passes**: `List(Int64)`, lengths equal to `item_ids` on every row, 0 rows with a decrease in train/val/test | seconds since the anonymized Yambda epoch, not unix (the raw data has no unix anchor) |
| G-goodreads: `item_id_map.json` | **passes**: `/data/goodreads-work-id/trainer/item_id_map.json` sha256-equal to the Hub copy, `dd6b8005…107265` (797,084 items) | raw fetched fresh (books + interactions_dedup only, via parallel ranged `curl`; `eval-data goodreads download` then verified sizes, gzip and wrote the sha256 manifest), `convert` 19.5 min, `prep` 58 s at peak RSS 24.2 GB |
| G-goodreads: `test.parquet` vs the Hub `test.parquet` (`item_ids`, `targets`, both `list[int64]`) | **fails bit-exact; every difference is a timestamp tie**. Same 313,178 rows in the same user order; 274,254 rows identical; of the 38,924 that differ, 36,576 differ only in `item_ids` order inside runs of equal timestamps, 782 differ in which items fill the oldest end of the 200-item window, always inside the window's leading run of equal timestamps (equal length), and 2,398 differ only in `targets` order (same multiset). Nothing else differs | Mechanism: `cmd_prep` orders each user's events with `sort(["user_id", "ts"])` and nothing breaks ties, and goodreads timestamps tie often (bulk shelving, e.g. many rows at `2007-01-01 00:00`). The order inside a tie comes from the multithreaded scan/join and changes run to run; `list.tail(max_seq + 1)` on the full sequence then keeps a different subset when the cut lands in a tie. Not fixed (out of this step): a tiebreak would also move the output off the Hub copy |
| G-goodreads: `timestamps` | **passes**: `List(Int64)` unix seconds, lengths equal to `item_ids` on every row, 0 rows with a decrease in train (744,332) / val (190,278) / test (313,178) | |
| KuaiRand `timestamps` | **passes**: `List(Int64)` unix seconds, lengths equal to `item_ids` on every row, 0 rows with a decrease in train (561,486) / val (24,503) / test (26,221); `tests/eval_datasets/test_kuairand.py` pins the column | `/data/kuairand` staged by `eval-data kuairand all`; the check script is not committed. The test histories end at 1651850999, 30 min before `test_timestamp_unix` 1651852800 |
| Harness suite on `dev/hstu-etl` | **green**, 235 passed / 4 skipped, CPU (`CUDA_VISIBLE_DEVICES=""`) | |

## Seqrec encoder

Trainer rewrite (`evaluation/training/`, [datasets](system/datasets.md#training--evaluationtraining)).
H100 80GB HBM3 (the pod, not the A100 above), torch 2.10.0+cu128, sm_mhz 1980 in every sample.
**Not yet validated, not citable.** Test is full-catalog on the trainer's `test.parquet`,
recorded only for the checkpoint selected on val ndcg@10.

### Bars

The published gSASRec checkpoint of the same D, re-scored on the same `trainer/test.parquet`
with the same eval code (ndcg@10 / recall@100):

| dataset | d64 | d128 | d256 |
|---|---|---|---|
| yambda-500m | 0.0846 / 0.1563 ([gate A](artifacts/seqrec-encoder/gate-a/)) | 0.0811 / 0.1486 ([gate A](artifacts/seqrec-encoder/gate-a/)) | 0.0814 / 0.1398 ([E0b](artifacts/seqrec-encoder/e0b-yambda-d256-bar/)) |
| goodreads-work-id | 0.0350 / 0.1486 ([E0](artifacts/seqrec-encoder/e0-goodreads-bars/)) | 0.0361 / 0.1480 ([E0](artifacts/seqrec-encoder/e0-goodreads-bars/)) | 0.0354 / 0.1472 ([E0c](artifacts/seqrec-encoder/e0c-goodreads-d256-bar/)) |
| KuaiRand | none: no published KuaiRand-27K checkpoint; calibrated against most-popular lists instead ([temporal drift](#kuairand-temporal-drift)) | not run | not run |

### Final models (the E1c recipe)

gSASRec body, ffn 4×D, sampled softmax (in-batch 4096 + uniform 8192) with logQ: the trainer's
defaults. Only D, epochs, patience and eval cadence vary (yambda: 100 epochs, eval every 2;
goodreads: 75-epoch cap, eval every epoch; patience 10 on both; KuaiRand: 48-epoch cap, eval
every epoch, patience 10, then the train + val refit).

| model | test ndcg@10 | R@100 | ndcg@100 | cov@10 | Δ vs bar (ndcg@10 / R@100) | best epoch (val ndcg@10) | s/epoch (median) | peak GB | W&B |
|---|---|---|---|---|---|---|---|---|---|
| [yambda d64](artifacts/seqrec-encoder/e1c-yambda-d64-sasrec-ssm-logq/) | 0.0945 | 0.1619 | 0.1157 | 0.0382 | **+0.0099 / +0.0056** | 77 (0.1010); early stop at 97 | 11.6 | 10.7 | [zxkmti7a](https://wandb.ai/pinkmeme/seqrec-encoder/runs/zxkmti7a) |
| [yambda d128](artifacts/seqrec-encoder/y128-sasrec-ssm-logq/) | 0.1006 | 0.1662 | 0.1207 | 0.0389 | **+0.0195 / +0.0176** | 99 of 100 (0.1078), still rising | 13.8 | 13.6 | [03idzv11](https://wandb.ai/pinkmeme/seqrec-encoder/runs/03idzv11) |
| [yambda d256](artifacts/seqrec-encoder/y256-sasrec-ssm-logq/) | 0.0966 | 0.1481 | 0.1108 | 0.0401 | **+0.0152 / +0.0083** | 97 of 100 (0.1040), still rising | 18.7 | 19.7 | [rqmh9ai9](https://wandb.ai/pinkmeme/seqrec-encoder/runs/rqmh9ai9) |
| [goodreads d64](artifacts/seqrec-encoder/e3-goodreads-d64-sasrec-ssm-logq/) | 0.0381 | 0.1447 | 0.0695 | 0.1232 | +0.0031 / **−0.0039** | 16 (0.0402); early stop at 26 | 49.4 | 6.7 | [338ohzq4](https://wandb.ai/pinkmeme/seqrec-encoder/runs/338ohzq4) |
| [goodreads d128](artifacts/seqrec-encoder/g128-sasrec-ssm-logq/) | 0.0410 | 0.1518 | 0.0736 | 0.1577 | **+0.0049 / +0.0038** | 34 (0.0440); early stop at 44 | 59.0 | 8.1 | [fcian7qz](https://wandb.ai/pinkmeme/seqrec-encoder/runs/fcian7qz) |
| [goodreads d256](artifacts/seqrec-encoder/g256-sasrec-ssm-logq/) | 0.0418 | 0.1530 | 0.0743 | 0.1616 | **+0.0064 / +0.0058** | 13 (0.0463); early stop at 23 | 80.3 | 11.0 | [b42n4obv](https://wandb.ai/pinkmeme/seqrec-encoder/runs/b42n4obv) |
| [KuaiRand d64](artifacts/seqrec-encoder/k64-refit-sasrec-ssm-logq/): train + val day (`train_on_val=true`), 4 epochs | 0.0276 | 0.0063 | 0.0197 | 0.0009 | no bar; val-day most-popular 0.0314 / 0.0073, all-time most-popular 0.0027 / 0.0007 | none: 4 epochs fixed, the first k64's best epoch + 1 | 260.8 | 62.7 | [f11y636d](https://wandb.ai/pinkmeme/seqrec-encoder/runs/f11y636d) |
| [KuaiRand d64, train only](artifacts/seqrec-encoder/k64-sasrec-ssm-logq/) (the first k64; drift evidence, not the model) | 0.0046 | 0.0016 | 0.0040 | 0.0006 | as above | 3 (0.0232); early stop at 13 | 249.4 | 62.6 | [xkk96p6k](https://wandb.ai/pinkmeme/seqrec-encoder/runs/xkk96p6k) |
| KuaiRand d128 | not run: OOM in backward even with one table ([probe](artifacts/seqrec-encoder/k128-probe-oom/README.md)). The shared 32 M × 128 fp32 table (15.28 GiB) needs weight + two AdamW moments (45.8 GiB) plus two dense gradients held at once, input lookup and output scoring (30.6 GiB): ~78 of 79.18 GiB | | | | | | | | |

Against the success rule ([decisions](decisions.md#sequential-encoder)): five of six
beat the bar on both metrics; goodreads d64 misses on R@100. KuaiRand is outside the rule (no
bar). yambda d128 and d256 were still improving at the 100-epoch cap. yambda d128's s/epoch includes 14 epochs slowed by a shared GPU.

**KuaiRand, measured before it left the study** (A100, gSASRec d128, 128 negatives): peak
77.2 GiB, ~11 min/epoch, test ndcg@10 0.0088, R@100 0.0024; its `t_cat1` filter cell: pass rate
0.0362, `recall_oracle@1000` 0.9996; disk: 13.6 GB raw, a 49 GB `_resume.pt`.

### KuaiRand: temporal drift

From the [diagnosis](artifacts/seqrec-encoder/k64-diagnosis/README.md) of the train-only k64
(`diag.json`, `recency.json`; evals only, H100).

- **Protocol.** Val targets are the clicks of 2022-05-06, the day after train ends; test targets are
  2022-05-07's. The val day is never trained on without `train_on_val`. Every history is capped at
  200. Targets per row, median (mean): val 118 (158), test 246 (320). recall@10 / recall@100
  ceilings: val 0.183 / 0.732, test 0.096 / 0.505. Targets seen in train: val 65.9 %, test 44.7 %.
  The 1.45× recall ceiling gap does not explain a 5× ndcg@10 gap (val 0.0232, test 0.0046).
- **Calibration** (the same `evaluate`, full catalog, on test, ndcg@10 / R@100): most-popular over
  train 0.0027 / 0.0007; most-popular over the val day's targets **0.0314 / 0.0073**; train-only
  k64 0.0046 / 0.0016, and 0.0046 / 0.0032 on train-seen targets only, so cold items are not the
  cause. The refit reaches 0.0276 / 0.0063: 6.0× / 3.9× the train-only model, still 12 % / 14 %
  below yesterday's most-popular list.
- **Curve.** Val ndcg@10 peaks at epoch 3 (0.0232) and drifts to ~0.020 while train loss keeps
  falling (12.7 → 11.3) and coverage@10 rises (0.0007 → 0.0018): overfitting to the train
  distribution, not a popularity collapse.
- **Mechanism.** Next-day clicks follow trending items: 47.6 % of test targets were clicked on the
  val day, 13.0 % only on the val day, 42.3 % in neither train nor the val day. logQ subtracts
  all-time train popularity, the wrong prior for a later day. The train-only model is one day
  stale on val and two on test; more capacity does not address that, training on the val day does.

### Recipe search (yambda-500m d64)

| run | recipe | test ndcg@10 / R@100 | Δ vs bar | verdict |
|---|---|---|---|---|
| [E1](artifacts/seqrec-encoder/e1-yambda-d64-sasrec-ssm/) | gSASRec body, sampled softmax, no logQ | 0.0661 / 0.1093 | −0.0185 / −0.0470 | loss: without logQ the in-batch negatives push popular items down (cov@10 0.0674 against E1c's 0.0382) |
| [E1c](artifacts/seqrec-encoder/e1c-yambda-d64-sasrec-ssm-logq/) | E1 + logQ | 0.0945 / 0.1619 | +0.0099 / +0.0056 | loss: logQ alone takes the same body from −0.0185 to +0.0099; selected |
| [E2a](artifacts/seqrec-encoder/e2a-yambda-d64-hstu-ssm-uniform/) (partial: stopped after epoch 58 of 100) | HSTU body (hidden 256, 4 blocks, 4 heads, time bias), uniform 8192 only | 0.0743 / 0.1379 | −0.0103 / −0.0184 | loss: the same body gains +0.0140 / +0.0121 with E1c's loss (E2c) |
| [E2b](artifacts/seqrec-encoder/e2b-yambda-d64-hstu-gbce/) (partial: stopped in epoch 15 of 100) | HSTU body, per-position gBCE | 0.0290 / 0.0658 | −0.0556 / −0.0905 | loss: gBCE on the HSTU body learns slowly (val 0.0301 and rising at the stop); inconclusive |
| [E2c](artifacts/seqrec-encoder/e2c-yambda-d64-hstu-ssm-logq/) | HSTU body, E1c's loss | 0.0883 / 0.1500 | +0.0037 / −0.0063 | body: with the same loss, HSTU is below gSASRec (−0.0062 / −0.0119 against E1c) at 3.6× the s/epoch |

The loss carries the gain; the HSTU body adds cost, not quality.

### Gates

Artifacts: [gate A](artifacts/seqrec-encoder/gate-a/),
[gate B](artifacts/seqrec-encoder/gate-b-yambda-d64/),
[compile A/B](artifacts/seqrec-encoder/compile/),
[W6 equivalence](artifacts/seqrec-encoder/w6-cleanup-equivalence/).

| gate | state |
|---|---|
| A: published checkpoints through `load_model_for_eval` + `evaluate` reproduce `eval_quality.json` test ndcg@10 / recall@100 to 4 decimals | old and new code give **identical** numbers on all four. goodreads d64 and d128 match to 4 decimals. yambda-500m d64 and d128 **do not** (0.0846/0.1563 vs 0.0813/0.1489; 0.0811/0.1486 vs 0.0751/0.1362), before and after the change alike. Mechanism: the test split on disk is not the one those files were scored on. The d64 checkpoint re-scored on `trainer/val.parquet` gives 0.09198, which matches its own training-time best val (0.09200). `test.parquet` at the data root and under `trainer/` hold the same rows. After the W6 cleanup (`loss` missing from the published `config.json` read as `gbce`): all four numbers `==` the pre-cleanup loader's ([artifact](artifacts/seqrec-encoder/w6-cleanup-equivalence/)). |
| B: retrain yambda-500m d64 on the published recipe, test within ±0.002 | per-position gBCE negatives, `compile=true`, 100 epochs: test ndcg@10 **0.0837**, recall@100 **0.1558**. Against the brief's 0.0813 / 0.1489 (the stale split): +0.0024 / +0.0069, **outside**. Against the published checkpoint re-scored on the same file (0.0846 / 0.1563): −0.0009 / −0.0006, **inside**. Best val 0.0913 (published 0.0920). 10.37 s/epoch median, 8,772 seq/s, 1478 s total (A100 published run: 2809 s), peak 11.2 GB, sm_mhz 1980 |
| B, shared negatives (plan §3) | one `[256]` negative vector per step collapses gBCE (test 0.0160 / 0.0377, 21 distinct items across 2048 users' top-10s); gBCE therefore samples per position ([mechanism](artifacts/seqrec-encoder/gate-b-yambda-d64/README.md)) |
| W6 cleanup equivalence (per-step training losses bit-identical before/after, both losses) | **passes** ([artifact](artifacts/seqrec-encoder/w6-cleanup-equivalence/)). H100, yambda-500m d64, 50 steps, seed 42, `compile=false`: the E1c `config.json` on the pre-cleanup trainer against the bare new `TrainConfig` defaults, and the Gate B gBCE `config.json` on both, give `==` per-step losses (50/50 each); the pre-cleanup trainer run twice is also identical to itself. CPU proxy (synthetic split, 18 steps per loss) bit-identical single-threaded; on 64 threads gBCE differs run to run in the last bit for the same code |
| `torch.compile` of the body | 3.93-4.03 s/epoch compiled against 4.8-5.9 eager (shared-negative gBCE, 3 epochs each, interleaved); kept |
| unit gates | `tests/training/test_encoder.py`: a left-padded row stays finite and padding content does not move the last position; `sampled_softmax_loss` equals a direct `F.cross_entropy` over explicit candidate lists, without and with a hand-built logQ (uneven explicit q; fails on a flipped sign or a corrected positive); `logq_correction` equals a hand-written `log(M·p + K/N)` vector for M = 2, K = 3, N = 4 (fails with the per-slot `/(M+K)`); `TrainConfig` rejects an unknown `loss`, and `normalize` or `logq` with `gbce`; defaults to `sampled_softmax` with `normalize` and `logq` on, `gbce` resolves both off; `TrainConfig.load` reads a `config.json` without `loss` as gbce, re-resolves `normalize`/`logq` when an override changes `loss` (both directions), keeps an explicit override and keeps the saved values when `loss` is unchanged (each case fails on its mutation, hand-checked) (CPU) |
| `resume_every` (`_resume.pt` cadence) | `test_resume_is_written_every_n_epochs_and_on_the_last` pins the write epochs at `num_epochs` 8: N = 1 every epoch; N = 3 → 2, 5, 7; N = 3 with an early stop at 3 → 2, 3; N = 4 with a stop at 5 → 3, 5. Each of three mutations (`epoch %` for `(epoch + 1) %`, the last-epoch off-by-one, dropping `stopping`) fails it (CPU). Snapshot retention across a crash (N = 3, crash in epoch 4, `--resume` retrains from epoch 3 and loads the right best; a changed `checkpoint_dir` spelling; a crash before the first `_resume.pt`) is checked only by uncommitted CPU smoke runs. On the GPU only the default N = 1 has run (k64-refit). `_resume.pt` is still written in place, not atomically ([backlog](backlog.md#sequential-encoder-follow-ups)) |
| `train_on_val` (final fit on train + val) | `test_train_on_val_rows_are_the_tail_of_history_then_targets` pins, on a hand-built 4-row `val.parquet` at L = 4, the rows (tail of `item_ids ++ targets`, left padding, more targets than L), `first`, the positions `target_mask` trains and the `target_frequencies` counts. `test_train_on_val_runs_no_val_eval`: a CPU `train()` never calls `evaluate`, writes `best_model.pt`, feeds the val rows after the train rows with `first` aligned, and stores `best_val_metric` `{}`. Each of six mutations (no `do_eval` guard, no `first` in the mask, val rows before train rows, no concat, an off-by-one in `first`, `if True` for the `{}` guard) fails a test (CPU). `train_on_val=false` unchanged: the new mask and `target_frequencies` `torch.equal` the old ones on 50 random left-padded tensors (uncommitted CPU check). GPU: k64-refit ran it (H100, 4 epochs). Rows with more than 200 val-day clicks keep their last 200, so 979,261 of 3,871,194 KuaiRand val-day transitions (25 %) are not trained |

Unverified: `target_frequencies` (logQ) is checked only by hand on CPU; E2c ran `logq=true` on the GPU;
resume (`--resume` has never run on the GPU).

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
- Every row above on a new pod, until the library suite has run there.
