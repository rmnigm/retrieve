---
title: validation
created: 2026-09-26
updated: 2026-09-26
type: summary
tags: [validation, testing, harness]
sources: [retrieve/tests/, evaluation/tests/, evaluation/golden/, evaluation/results/, docs/artifacts/]
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
| Library suite (`retrieve/tests`, GPU, `official` extra installed) | **green**, 708 passed at the last full run (roadmap Q3's hardened gates included; none turned red) | no tolerance loosened; every tolerance stated at its call site; see [testing](system/testing.md) |
| Triton vs reference ops, every parity file | **bit-exact** (`torch.equal` scores, ids up to ties) for `codesigned_probe_score` (+ bloom), `codesigned_probe_score_exact`, `oporp_1bit_match_topk_*`, `clause_mask`, the compaction ops; **not bit-exact** for `fused_masked_knn_topk` | the fused kernel's fp32 `tl.sum` and the reference's `bmm` reduce in different orders: measured drift ≤ 6e-8 at D ≤ 128 on unit-norm data, gated at `atol=1e-6` |
| Kernel identities (Q3) | **bit-exact**: indirect OPORP over `arange(N)` ≡ full scan; bloom op with an all-pass query signature ≡ no-bloom op; `clause_compact` ≡ `compact_mask(clause_mask)` incl. the `-1` tail; row alone ≡ row in batch (fused, both probe scorers, OPORP); item-table permutation permutes ids only (fused, `codesigned_probe_score`, OPORP) | Triton only, on this box |
| Kernel cutoffs and degenerate rows (Q3) | **green**: both sides of `_P_BUCKETS[0]` / `_N_BUCKETS[0]` and `P % block` ∈ {0, 1} (read from the kernels' constants, regime asserted); `count = 0` / `1` rows give exact `(-1, -inf)` tails | |
| Addressing past 2³¹ elements (kernel-opt) | **green**: [`test_large_offsets.py`](../retrieve/tests/correctness/test_large_offsets.py), output axis (`B = 144, N = 16M`: `clause_mask`, `clause_compact`, `fused_masked_knn_topk`, `oporp_1bit_match_topk_indirect`) and item axis (a `[140M, 16]` table: `bloom_match`, `bloom_compact`, `clause_mask`, `clause_compact`, `oporp_1bit_match_topk_full`), planted answers. All five cases fail on the pre-change code (illegal address; `bloom_match` also overflowed `grid_y`), and each kernel's widening was mutation-checked | skipped below 48 / 24 GiB free. The probe scorers' `B·P ≥ 2³¹` output axis has the same `row_base` but no large case. Timing of the narrow path: all 13 cases within noise of the pre-change kernels, interleaved ([artifact](artifacts/kernel-opt/phase1.md); [kernels](system/kernels.md#addressing)) |
| Unwritten-slot (poisoned `torch.empty`) | **green** for every kernel scoring into `torch.empty`: `fused_masked_knn_topk`, `codesigned_probe_score` (+ bloom), `codesigned_probe_score_exact`, OPORP full and indirect | allocation hit counted, so a moved allocation fails the test |
| Op registry | **green**: all 10 `retrieve::` ops have a same-name, same-signature reference twin; none declares a mutable argument; every input `torch.equal` after the op and after its twin; `torch.library.opcheck` passes on `clause_compact` / `bloom_compact`; the six official schemas equal the strings pinned for 21aa35e | |
| CUDA-graph replay | **bit-exact**: `reduce-overhead` SilverTorch (3 filter modes), LiNR V1-V3 × clause/bloom (and B=1), OneBit/SimHash replay on two queries they were not captured on, each `torch.equal` to eager, zero `cudagraph_skips` | |
| No wide intermediate in a forward | **green** on Triton for SilverTorch and LiNR V1-V4: no ≥ 3-d module-level tensor above B·N elements; the torch backend (which materializes `[B, ·, C, A_max]`) is the control | custom ops are opaque to the recorder: it checks module-level ops only, not a kernel's own buffers |
| Roofline anchors (Q3) | 4 GiB device-to-device copy **1754 GB/s** read+write (median of 20 windows, 1752-1766); empty Triton kernel launch **11.7 µs** median back to back (10.3-14.1) — `unstable`: SM clock 1275-1410 MHz during the copy, memory 1593 MHz ([artifact](artifacts/q3/roofline.json)) | the yardstick for any "% of HBM" claim is achieved GB/s over 1754, not a datasheet figure (1555 GB/s is the 40 GB part; this 80 GB part's spec is ~2039, of which the copy reaches 86 %). The "~1.4 TB/s ≈ 90 % of HBM" scorer claim appears nowhere in this repository and is **not yet validated** until restated against 1754 |
| Official vs Triton, SilverTorch phase 3, int32 score path | **bit-exact** (`torch.equal`) on every regime; `tests/parity/test_official.py` 43/43 | the fp16 path differs only at ties (below) |
| Official bloom masks | no false negatives on AND (partial mask equals full mask on probed docs); NOT is a complement, so it has no false positives and may have false negatives | FPR at matched memory 0 on synthetic attributes |
| Stream compaction determinism | **byte-identical** across launches, processes and campaign reruns | ascending item order on both backends |
| LiNR V2 torch vs triton | **not exact**: jaccard@100 0.99900-0.99975, `score_max_abs_diff` one fp16 ulp of the stored score | the kernel accumulates fp32 and stores fp16; `linr_v1_filter_mask` and `linr_v4` are 1.0 / 0.0 |
| SilverTorch torch vs triton | **exact**: jaccard 1.0, diff 0.0 on every filter cell | |

## Harness gates

| gate | state | notes |
|---|---|---|
| Harness suite (`evaluation/tests`, CPU with `CUDA_VISIBLE_DEVICES=""`) | **green**, 214 passed / 1 skipped on the A100 box, `yfcc10m` staged under `/data` | the skip reads the raw `_raw/yfcc10m/query.metadata.public.100K.spmat`, which is not on the box. pytest's `pythonpath` makes the suite import the checkout it sits in; before that, a worktree on the shared venv tested the main checkout |
| Convention gates in the harness suite | **green** | one reader per `RETRIEVE_*` variable (`test_env_readers.py`); the dependency direction with a file-count floor and stale-entry failures (`test_dependency_direction.py`, which dropped four stale edges and nine unused library names); every `config/*.yaml` × every suite through `load_matrix` (`test_config.py`). Each was checked to go red on a planted violation |
| `ruff check evaluation` (B, C4, SIM, RUF100, BLE001, PLC0415 on top of E, W, F, I, UP) | **clean**, ruff 0.15.6 | per-file ignores with reasons: ETL inline imports, goodreads' broad `except` (its own cleanup) ([evaluation](system/evaluation.md#lint)) |
| `ruff format --check evaluation` | **not clean**: 15 files | waits on the `evaluation/` formatting-only commit; the pre-commit format hook covers `retrieve/` only until then |
| `scripts/check_doc_links.py` | **0 problems**: links, 116 backticked repo paths, 88 `bench` / `eval-data` / `train` subcommands | subcommands read from the click sources with `ast`; floors of 60 paths / 50 calls catch a broken pattern; `docs/log.md` (history) is exempt from the path check and `evaluation/data` (gitignored) is allowed |
| Golden baseline (`evaluation/golden/`) against the v2 harness | 9 of 11 cells match within 7.5e-9 in quality | two residuals unexplained: `linr_v4` recall@100 7.3e-5; arXiv `silvertorch` triton recall@100 2.0e-6. No equivalence with the pre-v2 harness is claimed |
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
- **Meta's scorer kernel is faster than ours in every cell**: 1.15-3.2× on
  arXiv, 10.9-17.8× on goodreads. The cause is our padded IVF layout: on
  goodreads `n_probe × max_cluster_size` is 611,520 slots for about 18.7k
  real items (97 % padding); Meta reads a CSR. Official gives the time back
  in payload preparation (268-834 µs over 73-93 launches, against our
  34-58 launches).
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
- **Not measured**: the controlled `bloom_path="full"` ablation (S9), bloom
  FPR against width (both blooms showed zero false positives, so S8 needs
  roadmap D3), seeds 1-2, k in {500, 1000}.

## Campaign (roadmap D1): not yet validated

- Goodreads `filter` leg: 126 of 126 cells `ok` on the grid of
  [decisions](decisions.md#harness). Quality numbers are not yet
  validated, and the report marks the leg not citable (narrowed mode set).
- arXiv leg, seeds 1-2, the `deep` suite and the S9 ablation cells: not run.
- Measured cost: about 530 s per cell, of which about 330 s is the fixed
  cost of nine `torch.compile` + CUDA-graph captures per cell; the timing
  windows themselves take about 125 s (eager) and 68 s (graph). Graph is
  1.84× faster than eager at the median.
- Instability: 3-5 % of perf entries are `unstable` (window spread over
  5 %), 98.6 % of them at batch 1 or 8, almost all in eager mode.

## Datasets

| dataset | state |
|---|---|
| goodreads, arxiv | staged, layout checked, oracles built |
| yfcc10m | our exact oracle reproduces the shipped filtered ground truth. The first filter cell fails the exact-algorithm gate (`recall_oracle@1000` 0.964 < 0.99) because `PostfilterKNN` scores in fp16 and YFCC's top-1000 spans only about fifteen fp16 quanta; decision open (roadmap) |
| pubmed | ETL written and dry-run on one shard; nothing staged |
| Semantic Scholar, KuaiRand | not started |

## Still unverified

- Clause 5 of the campaign gate (ids identical across modes).
- The official exact path without our adapter's mask packing.
- `bench upload`'s LFS path above 37 MB.
- Every row above on a new box, until the library suite has run there.
