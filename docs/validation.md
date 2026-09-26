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
| Library suite (`retrieve/tests`, GPU, `official` extra installed) | **green**, 755 passed at the last full run (branch `dev/kernel-opt`, on a cold inductor cache; Q3's hardened gates included, none turned red) | no tolerance loosened; every tolerance stated at its call site; see [testing](system/testing.md) |
| Triton vs reference ops, every parity file | **bit-exact** (`torch.equal` scores, ids up to ties) for `codesigned_probe_score` (+ bloom) and `codesigned_probe_score_exact` on the compact CSR layout (also against a loop-built oracle and, for bloom, the row-wise subset test), and for `oporp_1bit_match_topk_*`, `clause_mask`, the compaction ops; **not bit-exact** for `fused_masked_knn_topk` | the fused kernel's fp32 `tl.sum` and the reference's `bmm` reduce in different orders: measured drift ≤ 6e-8 at D ≤ 128 on unit-norm data, gated at `atol=1e-6` |
| Kernel identities (Q3) | **bit-exact**: indirect OPORP over `arange(N)` ≡ full scan; bloom op with an all-pass query signature ≡ no-bloom op; `clause_compact` ≡ `compact_mask(clause_mask)` incl. the `-1` tail; row alone ≡ row in batch (fused, both probe scorers, OPORP); item-table permutation permutes ids only (fused, `codesigned_probe_score`, OPORP) | Triton only, on this box |
| Kernel cutoffs and degenerate rows (Q3) | **green**: both sides of `_P_BUCKETS[0]` / `_N_BUCKETS[0]` and `P % block` ∈ {0, 1} (read from the kernels' constants, regime asserted); `count = 0` / `1` rows give exact `(-1, -inf)` tails | |
| Addressing past 2³¹ elements (kernel-opt) | **green**: [`test_large_offsets.py`](../retrieve/tests/correctness/test_large_offsets.py), output axis (`B = 144, N = 16M`: `clause_mask`, `clause_compact`, `fused_masked_knn_topk`, `oporp_1bit_match_topk_indirect`) and item axis (a `[140M, 16]` table: `bloom_match`, `bloom_compact`, `clause_mask`, `clause_compact`, `oporp_1bit_match_topk_full`), planted answers. All five cases fail on the pre-change code (illegal address; `bloom_match` also overflowed `grid_y`), and each kernel's widening was mutation-checked | skipped below 48 / 24 GiB free. The probe scorers' `B·P ≥ 2³¹` output axis has the same `row_base` but no large case. Timing of the narrow path: all 13 cases within noise of the pre-change kernels, interleaved ([artifact](artifacts/kernel-opt/phase1.md); [kernels](system/kernels.md#addressing)) |
| Unwritten-slot (poisoned `torch.empty`) | **green** for every kernel writing into `torch.empty`: `fused_masked_knn_topk`, `codesigned_probe_score` (+ bloom), `codesigned_probe_score_exact`, OPORP full and indirect, and the two compaction ops, whose scatter launch now writes the `-1` tail itself. The compaction case goes red with the tail store removed | allocation hit counted, so a moved allocation fails the test |
| Op boundary (kernel-opt) | **green**: [`test_op_boundary.py`](../retrieve/tests/correctness/test_op_boundary.py). Every Triton op raises `ValueError` on a non-contiguous item-side table (11 cases) and on a non-power-of-two `tl.arange` extent (7 cases). Before, these were a silent per-call copy and a Triton `CompilationError`. All 18 go red with the checks disabled | [kernels](system/kernels.md) conventions |
| Tuner rule (kernel-opt) | **green**: `tune._choose` keeps the shipped default inside a 3 % noise band and rejects a candidate more than 5 % slower in any regime (`test_tune_smoke.py`, 4 synthetic cases). `--json-out` records SM clocks, commit and versions. No sweep has been run under the rule yet | |
| Build-time int8 quantization (kernel-opt) | **green**: `quantize_int8_global` quantizes chunk by chunk (optionally in `sort_perm` order, so SilverTorch writes its cluster-sorted table directly). The codes are `torch.equal` to the one-shot formula, and the transient stays under half the fp32 table. The one-shot form measured 672 MiB of transient on a 384 MiB table (`test_quantize.py`). This is the SilverTorch OOM on pubmed 10M × 768 that the E2 worker reported: two fp32 table copies on top of a 28.6 GiB table. **Re-run on pubmed 10M × 768**: the build completes on triton and official, peak allocated ≈ 38 GiB (quantize +0.5 GiB beyond its int8 output), official cells run end to end; Triton stops at the first forward on `D=768 must be a power of two`, a separate blocker ([pubmed](artifacts/kernel-opt/pubmed/README.md)) | [artifact](artifacts/kernel-opt/phase4.md) |
| Short candidate lists and missing filters (kernel-opt) | **green** (`test_linr.py`): `fused_masked_knn_topk` rejects `P < k` with `ValueError` on both backends, where before Triton raised a torch `RuntimeError` and the twin padded; `PrefilterKNN` returns `[B, k]` with exact `(-1, -inf)` tails for `P` in {0, 3} and the backends agree; OPORP indirect returns `min(k, P)` columns on both backends (Triton returned `k`; at `P = 0` it crashed), compiled `dynamic=True` included; `LiNRV3` rejects `k > candidate_pool`; clause attrs to a filterless variant raise `ValueError` (an `assert` / `AttributeError` before) | |
| Non-power-of-two widths (kernel-opt finding, unscheduled) | **not supported, by design today**: every Triton kernel with a `tl.arange` over the embedding or word width needs a power of two. Those are `codesigned_probe_score*` (`D`), `fused_masked_knn_topk` (`D`), OPORP and the bloom kernels (`W`); since Phase 2 each raises `ValueError` at the op boundary. Measured on pubmed 10M × 768: SilverTorch triton stops at its first forward (`D=768 must be a power of two`), and LiNR V2 / V3 would stop likewise (OPORP's default `k_bits = D` gives `W = 12`). Official and the cuBLAS paths (V1, V4) run. A limitation for a roadmap decision (masked padding to the next power of two), not fixed here | [pubmed](artifacts/kernel-opt/pubmed/README.md) |
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
| Golden baseline (`evaluation/golden/`) against the v2 harness | Rerun on `dev/kernel-opt` (quality only, eager) ([artifact](artifacts/kernel-opt/golden_compare.md)). All 7 comparable goodreads cells (LiNR V1-V3 and SilverTorch triton / official at `n_probe` 24 and 32) are **identical** to the committed D1 records of the pre-change code: max \|diff\| 0 over 24 oracle and held-out metrics each. Against the golden JSONs: `linr_v1` and goodreads `silvertorch` within 1e-6; arXiv `silvertorch` recall@100 2.0e-6 (the known residual); `linr_v2` 4.5e-4 and `linr_v3` 1.7e-5 **also differ on the pre-change code** (the D1 records carry the same values), so the former "9 of 11 within 7.5e-9" no longer describes this harness and library. The cause is unidentified and predates `dev/kernel-opt`. The golden `torch` and `linr_v4` cells have no counterpart in today's suites | the two old residuals: `linr_v4` recall@100 7.3e-5 (not rerunnable: no suite runs `linr_v4`); arXiv `silvertorch` triton recall@100 2.0e-6 |
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
- **After G-a TF-9 / TF-1** (compact CSR probe layout + transposed bloom,
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
| pubmed | 10 M slice staged locally (A100 box, 2026-09-26), not on the Hub: `bench check` passes, the exact `c0_mesh` oracle is built (pass rate 0.0002). One filter cell, `linr_v1_filter_mask`/triton clause `c0_mesh`, eager, `--skip-perf` (so `partial`): `recall_oracle@1000` 0.9991, held-out `recall@100` 0.9989, n = 8,428. SilverTorch's global int8 quantize OOM is fixed (chunked build, kernel-opt pass); `official` builds and queries (recall@100 0.664/0.709 at n_probe 24/32, held-out recall@100 0.842/0.880, quality-only); SilverTorch-`triton` and LiNR V2/V3 hit the Triton power-of-two `D` limitation (roadmap, Known defects) at native 768; not yet validated ([artifacts](artifacts/e2-pubmed/), [artifacts](artifacts/kernel-opt/pubmed/)) |
| openalex | 10 M slice staged locally (A100 box, 2026-09-26; OpenAlex fallback for Semantic Scholar SPECTER2, no API key), not on the Hub: `bench check` passes. One filter cell, `linr_v1_filter_mask`/triton clause `field_era`, eager, `--skip-perf` (`partial`): `recall_oracle@1000` 0.9959, held-out `recall@100` 0.505, `recall@1000` 0.741, n = 10,000. Scoped down from an initial 15 M encode after `bench/oracle.py`'s `item_embs.t().contiguous()` OOMed at that size (a second full fp32 copy on top of the item table; the real 768-d limit is ~11-12 M, not 15 M) — 10 M items resharded from the already-encoded 15 M vectors (`torch.equal`-verified), matching the pubmed precedent. Same power-of-two `D` limitation as pubmed for SilverTorch-`triton` and LiNR V2/V3; not yet validated ([artifacts](artifacts/e3-openalex/)) |
| kuairand | staged locally (A100 box, 2026-09-26), not on the Hub: `bench check` passes, ETL and two filter protocols (target-derived, business-rule) built. No gSASRec checkpoint yet — GPU queued behind the other GPU work; not yet validated ([artifacts](artifacts/e4-kuairand/)) |

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
| KuaiRand `timestamps` | **code only**: `tests/eval_datasets/test_kuairand.py` pins the column (unix seconds, same windows as `item_ids`); no data run, raw not fetched | the staged `data/kuairand` predates the column |
| Harness suite on `dev/hstu-etl` | **green**, 235 passed / 4 skipped, CPU (`CUDA_VISIBLE_DEVICES=""`) | |

## Seqrec encoder rewrite

Trainer rewrite (`evaluation/training/`, [datasets](system/datasets.md#training--evaluationtraining)).
H100 80GB HBM3 (the pod, not the A100 above), torch 2.10.0+cu128. **Not yet validated, not
citable.** Artifacts: [gate A](artifacts/seqrec-encoder/gate-a/),
[gate B](artifacts/seqrec-encoder/gate-b-yambda-d64/),
[compile A/B](artifacts/seqrec-encoder/compile/).

| gate | state |
|---|---|
| A: published checkpoints through `load_model_for_eval` + `evaluate` reproduce `eval_quality.json` test ndcg@10 / recall@100 to 4 decimals | old and new code give **identical** numbers on all four. goodreads d64 and d128 match to 4 decimals. yambda-500m d64 and d128 **do not** (0.0846/0.1563 vs 0.0813/0.1489; 0.0811/0.1486 vs 0.0751/0.1362), before and after the change alike. Mechanism: the test split on disk is not the one those files were scored on. The d64 checkpoint re-scored on `trainer/val.parquet` gives 0.09198, which matches its own training-time best val (0.09200). `test.parquet` at the data root and under `trainer/` hold the same rows |
| B: retrain yambda-500m d64 on the published recipe, test within ±0.002 | per-position gBCE negatives, `compile=true`, 100 epochs: test ndcg@10 **0.0837**, recall@100 **0.1558**. Against the brief's 0.0813 / 0.1489 (the stale split): +0.0024 / +0.0069, **outside**. Against the published checkpoint re-scored on the same file (0.0846 / 0.1563): −0.0009 / −0.0006, **inside**. Best val 0.0913 (published 0.0920). 10.37 s/epoch median, 8,772 seq/s, 1478 s total (A100 published run: 2809 s), peak 11.2 GB, sm_mhz 1980 |
| B, shared negatives (plan §3) | one `[256]` negative vector per step collapses gBCE (test 0.0160 / 0.0377, 21 distinct items across 2048 users' top-10s); gBCE therefore samples per position ([mechanism](artifacts/seqrec-encoder/gate-b-yambda-d64/README.md)) |
| `torch.compile` of the body | 3.93-4.03 s/epoch compiled against 4.8-5.9 eager (shared-negative gBCE, 3 epochs each, interleaved); kept |
| unit gates | `tests/training/test_encoder.py`: a left-padded row stays finite and padding content does not move the last position (`sasrec`, `hstu`); `sampled_softmax_loss` equals a direct `F.cross_entropy` over explicit candidate lists, without and with a hand-built logQ (uneven explicit q; fails on a flipped sign or a corrected positive); `logq_correction` equals a hand-written `log(M·p + K/N)` vector for M = 2, K = 3, N = 4 (fails with the per-slot `/(M+K)`); `TrainConfig` rejects an unknown `loss`, and `normalize` or `logq` with `gbce` (CPU) |

### Encoder experiments E0-E4 (H100, not yet validated, not citable)

Bars are the published checkpoints re-scored on the trainer's `test.parquet`: yambda-500m d64
0.0846 / 0.1563, d128 0.0811 / 0.1486; goodreads-work-id d64 0.0350 / 0.1486, d128 0.0361 / 0.1480
(ndcg@10 / recall@100; [E0](artifacts/seqrec-encoder/e0-goodreads-bars/), which matches the
stored goodreads numbers to 4 decimals). Selection is on val ndcg@10; test is recorded only for
the selected checkpoint. sm_mhz 1980 throughout.

| run | test ndcg@10 / ndcg@100 / R@10 / R@100 / cov@10 | Δ vs bar (ndcg@10 / R@100) | best val ndcg@10 (epoch) | s/epoch (median) | seq/s | peak GB | wall |
|---|---|---|---|---|---|---|---|
| [E1](artifacts/seqrec-encoder/e1-yambda-d64-sasrec-ssm/) yambda d64, gSASRec body, sampled softmax (in-batch 4096 + uniform 8192, no logQ) | 0.0661 / 0.0790 / 0.0326 / 0.1093 / 0.0674 | −0.0185 / −0.0470 | 0.0740 (95 of 100) | 10.9 | 8,337 | 10.7 | 26 min |
| [E2a](artifacts/seqrec-encoder/e2a-yambda-d64-hstu-ssm-uniform/) yambda d64, HSTU body (hidden 256, 4 blocks, 4 heads, `use_time`), sampled softmax (uniform 8192 only). **Stopped after epoch 58 of 100 by user decision** | 0.0743 / 0.0950 / 0.0364 / 0.1379 / 0.0450 | −0.0103 / −0.0184 | 0.0781 (55) | 39.2 | 2,249 | 9.4 | 47 min (to the stop) |
| [E2c](artifacts/seqrec-encoder/e2c-yambda-d64-hstu-ssm-logq/) yambda d64, HSTU body as E2a, sampled softmax (in-batch 4096 + uniform 8192) **with logQ** | 0.0883 / 0.1075 / 0.0425 / 0.1500 / 0.0423 | **+0.0037** / −0.0063 | 0.0921 (21; early stop at 41) | 42.3 | 2,155 | 11.9 | 34 min |
| [E2b](artifacts/seqrec-encoder/e2b-yambda-d64-hstu-gbce/) yambda d64, HSTU body as E2a, per-position gBCE (K=256, t=0.75). **Stopped during epoch 15 of 100 by user decision** | 0.0290 / 0.0422 / 0.0123 / 0.0658 / 0.0003 | −0.0556 / −0.0905 | 0.0301 (13, still rising) | 41.1 | 2,193 | 11.6 | 12 min (to the stop) |

Unverified: `target_frequencies` (logQ) is checked only by hand on CPU; E2c ran `logq=true` on the GPU;
`use_time` on goodreads timestamps (E2a used it on yambda); resume (`--resume` has never run on the GPU).

## Still unverified

- Any compiled (`graph`-mode) measurement taken on a warm inductor cache after a library
  change: the FX-graph / AOT-autograd caches replay a stale `triton_op` body
  ([testing](system/testing.md#running)). This is measured in the library suite; whether any
  recorded harness cell was affected has not been checked.

- Clause 5 of the campaign gate (ids identical across modes).
- The official exact path without our adapter's mask packing.
- `bench upload`'s LFS path above 37 MB.
- Every row above on a new box, until the library suite has run there.
