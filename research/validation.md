---
title: validation
created: 2026-09-26
updated: 2026-10-17
type: summary
tags: [validation, testing, harness]
sources: [retrieve/tests/, evaluation/tests/, evaluation/golden/, research/artifacts/, research/artifacts/hub-index.md]
---

# What is validated now

The current state of every gate, and the measured results that stand. A
number is paper material only where its row says **citable** (AGENTS.md
rule 2). When a gate's state changes, update its row; when a result is
superseded, replace it. Raw scripts are under [artifacts/](artifacts/), raw
outputs on the Hub ([hub-index](artifacts/hub-index.md)).

**The study** ([decisions](decisions.md#goal-and-scheduling)): Meta's
official SilverTorch code, adapted, against LiNR V1-V3 (exact filter only),
on goodreads, arXiv, PubMed, LAION and Yambda. **The paper's numbers come
from one run, the [FULL-RERUN](#full-rerun) at tag `final-freeze`.** What it
did not remeasure (the adapted code's gates and their timings, the data
facts) is final and citable with its source named. Everything measured
before it is [superseded](#superseded-results). The paper itself is kept
outside this repository.

**Environment of every row** (unless the row names another): A100-SXM4-80GB,
torch 2.10.0+cu128, triton 3.6.0, CUDA 12.8 (nvcc 12.8.93, the pod image),
Python 3.11. SM clocks cannot be locked; the sampled clock under load is
1410 MHz. A new pod is a new environment: rerun the library suite there
before trusting any row on it.

## Library gates

| gate | state | notes |
|---|---|---|
| Library suite (`retrieve/tests`, GPU, the `silvertorch` extension installed) | **green**: 642 passed, 0 skipped, 0 failed (2026-10-16, after RENAME); Meta's own suite inside `silvertorch/` 122 passed; the compile gate (`test_silvertorch_compile.py`: 16 `st` ops `opcheck`, fullgraph, export) is in the suite | no tolerance loosened; every tolerance stated at its call site ([testing](../docs/system/testing.md)) |
| RENAME: the adapted code under Meta's names (`silvertorch`, `torch.ops.st.*`, `SilverTorch(backend="cuda")`, the only backend) | **green**: before / after `torch.equal` on the gate grid's int32 cells, 45 grid lines, 300 cells × 16 pool batches, **0 differ** ([artifact](artifacts/rename/README.md)); harness suite (CPU) 1110 passed, 3 skipped | records keep the names they were written with; `records.read_records` maps `official-fork` → `cuda` (`RENAMED_BACKENDS`) |
| Meta's code as published (pinned `21aa35e`) | **left the workspace at RENAME**; its last parity run against the adapted code (T1 / T6 / C13 bit-exact at int32) stands behind the tag `meta-published-parity-final` | the library's parity tests are now the adapted code against the dense torch oracle (`test_official.py`) |
| Exact LiNR scoring precision (L1) | **green**: `PostfilterKNN` (± mask) and `PrefilterKNN` return fp32 scores within `2e-6` of an fp64 dot of their own fp16 inputs and select the fp64 top-100 up to ties; an fp16 score of the same dots misses (recall 0.88 / 0.92) ([`test_accumulation.py`](../retrieve/tests/parity/test_accumulation.py)) | storage stays fp16 ([decisions](decisions.md#library)) |
| Triton vs reference ops | **bit-exact** (`torch.equal` scores, ids up to ties) for `oporp_1bit_match_topk_*`, `clause_mask`, `clause_mask_scores`, `clause_compact`, `gemv_scores`; **not bit-exact** for `fused_masked_knn_topk` | the fused kernel's fp32 `tl.sum` and the reference's `bmm` reduce in different orders: drift ≤ 6e-8 at D ≤ 128 on unit-norm data, gated at `atol=1e-6` |
| Kernel identities (Q3) | **bit-exact**: indirect OPORP over `arange(N)` ≡ full scan; `clause_compact` ≡ `compact_mask(clause_mask)` on counts and the `[:counts]` prefix; row alone ≡ row in batch; an item-table permutation permutes ids only | |
| Kernel cutoffs and degenerate rows (Q3) | **green**: both sides of `_P_BUCKETS[0]` / `_N_BUCKETS[0]` and `P % block` ∈ {0, 1}; `count = 0` / `1` rows give exact `(-1, -inf)` tails | |
| Addressing past 2³¹ elements | **green** ([`test_large_offsets.py`](../retrieve/tests/correctness/test_large_offsets.py)): output axis (`B = 144, N = 16M`) and item axis (a `[140M, 16]` table), planted answers; every case failed on the pre-change code | skipped below 48 / 24 GiB free ([kernels](../docs/system/kernels.md#addressing)) |
| Unwritten slot (poisoned `torch.empty`) | **green** for every kernel writing into `torch.empty`; the compaction op writes `[:counts]` only, so the poison survives past `counts` (red if a tail store comes back) | |
| Compaction tail (L2) | **green**: every consumer of `clause_compact` bounds its reads by `counts`; `LiNRV2`, `LiNRV3` and `combine_indices` return `torch.equal` results with the tail poisoned to `-1` and to `2**40` ([`test_compact_order.py`](../retrieve/tests/parity/test_compact_order.py)) | Triton compaction's full output is not a function of its inputs outside `torch.use_deterministic_algorithms(True)` |
| Stream compaction determinism | **byte-identical** on counts and the `[:counts]` prefix across launches and processes; ascending item order on both backends | |
| Op boundary | **green** ([`test_op_boundary.py`](../retrieve/tests/correctness/test_op_boundary.py)): every Triton op raises `ValueError` on a non-contiguous item-side table | [kernels](../docs/system/kernels.md) conventions |
| Tuner rule | **green**: `tune._choose` keeps the shipped default inside a 3 % noise band and rejects a candidate more than 5 % slower in any regime (`test_tune_smoke.py`) | no sweep has run under the rule |
| Build-time int8 quantization | **green**: `quantize_int8_global` quantizes chunk by chunk; codes `torch.equal` to the one-shot formula, transient under half the fp32 table (`test_quantize.py`); PubMed 10M × 768 builds at ≈ 38 GiB peak | [artifact](artifacts/kernel-opt/phase4.md) |
| Build-time 1-bit quantization | **green**: `quantize_oporp_1bit` / `quantize_simhash_1bit` pack in 65,536-row chunks; bits `torch.equal` to the one-shot chain; transient 10.0 → 0.26 GiB on arXiv OPORP; PubMed 10M × 768 builds where the one-shot chain OOMs | [kernels](../docs/system/kernels.md#quantize_oporp_1bit-retrieveindexing), [artifact](artifacts/pubmed-fixes/README.md#fix-2-chunked-1-bit-build) |
| Short candidate lists and missing filters | **green** (`test_linr.py`): `fused_masked_knn_topk` rejects `P < k`; `PrefilterKNN` pads `(-1, -inf)` for `P` in {0, 3}; OPORP indirect returns `min(k, P)` columns; `LiNRV3` rejects `k > candidate_pool` | |
| Non-power-of-two widths (L4) | **green**: every Triton kernel aranges over `next_power_of_2` and masks the pad lanes; parity at D = 192 and 768 and OPORP `W = 3` / `12`; at power-of-two widths the kernels are SASS-identical to before | [kernels](../docs/system/kernels.md#padding), [artifact](artifacts/l4-pow2-pad/README.md) |
| V1-FUSE: LiNR V1's clause filter folded into the score write (`clause_mask_scores`) | **bit-exact, ships**: ids `torch.equal` to the mask kernel + `where` composition, eager and graph, on every cell; 0 cudagraph skips | [kernels](../docs/system/kernels.md#clause_mask--fused-clause-eval-emitting-b-n-bool) |
| V1-BS1: single-query GEMV (`gemv_scores`) and batched-row clause masks | **bit-exact**: 68 cells, ids + scores `torch.equal`, none slower; `register_index` checks the GEMV against cuBLAS on 8 rows (`calibration_rows`, int64, tested past 2²⁴ items) and takes it only where equal | [kernels](../docs/system/kernels.md#postfilterknn-dense-path--cublas-and-gemv_scores-for-one-query), [artifact](artifacts/campaign-v2.10/v1-bs1/README.md) |
| V1-TOPK: LiNR V1's top-k on the two-level block top-k (`functional.two_level_topk`) | **bit-exact**: 144 cells, ids + scores `torch.equal`, none slower | below B·N 2²³ V1 runs `torch.topk` ([artifact](artifacts/campaign-v2.11/v1-topk/README.md)) |
| V2-HIGHP / V2-FILL: `fused_masked_knn_topk`'s split scoring loop and wide `-inf` fill | **bit-exact**: 64 / 64 and 120 / 120 cells, ids + scores `torch.equal` to the code they replaced | the int32 / int64 count choice rests on register allocation; re-check on a toolchain change ([V2-HIGHP](artifacts/campaign-v2/v2-highp/README.md), [V2-FILL](artifacts/campaign-v2/v2-fill/README.md)) |
| CLAUSE-SKIP: clause kernels skip inactive clauses | **bit-exact** on goodreads and the 10-clause synth tables, every LiNR and SilverTorch exact path | one accepted cost: goodreads `all4` V2 bs 1 graph 1.095 ([artifact](artifacts/campaign-v2.6/clause-skip/README.md)) |
| Op registry | **green**: 7 `retrieve::` ops, each with a same-name, same-signature reference twin; none mutates an input; `opcheck` passes on `clause_compact`; the seven `st::` schemas equal the strings pinned in `test_official.py` | |
| CUDA-graph replay | **bit-exact**: `reduce-overhead` SilverTorch (3 filter modes), LiNR V1-V3 with the exact filter (and B=1), OneBit / SimHash replay on queries they were not captured on, each `torch.equal` to eager, zero `cudagraph_skips` | |
| No wide intermediate in a forward | **green** for SilverTorch and LiNR V1-V3 on their kernel backends: no ≥ 3-d module-level tensor above B·N elements | custom ops are opaque to the recorder |
| Item-range filter masks | **bit-exact**: `evaluate_mask(q, start, end)` ≡ `evaluate_mask(q)[:, start:end]` for `ExactAttributeFilter` on both backends; transient of a 1,000-item range below `B·N/16` bytes (`test_filters.py`) | |
| SilverTorch bloom masks | no false negatives on AND (partial mask equals full mask on probed docs); NOT has no bloom form | false positives allowed; graded against the exact oracle |
| LiNR V2 torch vs triton | **inside 2e-6**: same fp16 products summed in fp32 in different orders; id sets equal up to ties (`test_linr.py`) | |
| Roofline anchors (Q3) | 4 GiB device-to-device copy **1754 GB/s** read + write; empty Triton kernel launch **11.7 µs** median ([artifact](artifacts/q3/roofline.json)) | the yardstick for any "% of HBM" claim is achieved GB/s over 1754 |

## Adapted SilverTorch gates

The adapted code ([`silvertorch/`](../silvertorch/CHANGES.md), one row per
change in its `CHANGES.md`), gated against Meta's code as published at the
time both were installed (`backend="official-fork"` against
`backend="official"`; ops `stfork`). Review: **ACCEPT**. The FULL-RERUN did
not rerun this grid, so these rows are **final and citable**, with their Hub
subtrees as the source (`artifacts/meta-fork/*`,
[hub-index](artifacts/hub-index.md)); scripts in
[`artifacts/meta-fork/`](artifacts/meta-fork/README.md). Timings on one A100
pod unless named.

| gate | state | notes |
|---|---|---|
| upstream sync 21aa35e → 22c2007 | **applied, built** | |
| Meta's own suite on the adapted code | **pass**, 122 passed | |
| X: `torch.equal` to Meta's published code at `-O3` (ids and scores) | **pass** on every cell run: **276 / 276** at the head (arXiv d128 / d256, goodreads d128, PubMed d768; none / exact / bloom partial + full with Meta's semantics; n_probe 24 … 2048; bs 1 / 16 / 64; k 100 / 1000; eager + graph); 1,488 / 1,488 on a second pod at an earlier head; at 30 M (LAION d256) bloom partial + full 64 / 64 at the head | 30 M none / exact last ran at `8ff11e0` (72 / 72); covered at the head by the ≤ 10 M cells and the library tests, not at 30 M |
| speed against Meta's published code at `-O3` | **pass**, faster on every cell: eager adapted / published median 0.230, max 0.764 over the 300 X cells; 30 M bloom median 0.279, max 0.467 | |
| S: no host sync, no copy | **pass**: `set_sync_debug_mode("error")` ok, 0 D2H, 0 pageable H2D, 9-10 kernels a forward (published: 3-6 syncs, 50 kernels) | `syncs.py` |
| capture | **pass**: 1 `cudaGraphLaunch` a replay, `torch.equal` to eager over 16 batches | `graph_launches.py` |
| `torch.compile` | **pass**: `opcheck` on all 16 ops; none / exact / bloom traces `fullgraph`, one graph, `torch.equal` to eager; `torch.export` equals eager | |
| R: the configured-width bloom (C12) with the recheck (C13) | **pass**: 0 false positives on every cell; recall against an independent exact filter 1.0 or ties at the k-th score; the rechecked bloom `torch.equal` to the exact path | the recheck is off in the study ([decisions](decisions.md#goal-and-scheduling)) |
| C10: cluster / tile skip | **no change**, measured: Meta's scorer already skips at lane granularity | |
| C11: scorer per D | **changed**: the cooperative-load scorer for D ≥ 128 | |
| C14: exact filter layout | **bit-exact**: 1,056 cells, scores and ids `torch.equal` to `final-freeze` (goodreads, arXiv, PubMed d768, LAION 1M-30M, up to 65,536 lists) | timings NOT CITABLE (the paper's come from the FR clause row) ([artifact](artifacts/clause-opt/), [kernels](../docs/system/kernels.md#cuda--the-adapted-forward)) |

## Harness gates

| gate | state | notes |
|---|---|---|
| Harness suite (`evaluation/tests`, `CUDA_VISIBLE_DEVICES=""`) | **green**: 1110 passed, 3 skipped (CPU, after RENAME); the 3 GPU-marked tests pass on a GPU | the skips are the GPU-marked tests |
| `ruff check` / `ruff format --check` on `retrieve` and `evaluation` | **clean**, ruff 0.15.6 | per-file ignores with reasons ([evaluation](../docs/system/evaluation.md#lint)) |
| `scripts/check_doc_links.py` | **0 problems** | it does not check `#anchor` fragments |
| Convention gates | **green**: one reader per `RETRIEVE_*` variable (`test_env_readers.py`); the dependency direction (`test_dependency_direction.py`); every `config/*.yaml` × every suite through `load_matrix` (`test_config.py`) | each goes red on a planted violation |
| Record key carries the input identity (H2) | **green**: two jobs differing only in `--checkpoint` key apart; `bench report` refuses two `inputs` under one `(dataset, dim)` | `SCHEMA_VERSION` 3 ([evaluation](../docs/system/evaluation.md#resume)) |
| Held-out metric with no target is null (H7) | **green**: a cell whose held-out targets all fail the filter records `heldout: {…: null, n: 0}` | |
| `bench report` states true provenance | **green**: a non-citable report names its actual reasons (gate not green, `failed`, `partial` with its reasons, `env.dirty`, an off-main branch) (`test_report.py`) | |
| `code_version` names the imported library | **green** (`test_measure.py`): the tree hash of the imported checkout, `dirty` and a content hash for uncommitted edits | [evaluation](../docs/system/evaluation.md#resume) |
| Inductor cache keyed by `code_version` | **green**: `measure.inductor_cache_dir` returns `<base>/<code_version>`; `bench campaign` hands children the caller's base | [evaluation](../docs/system/evaluation.md#inductor-cache) |
| Synthetic selectivity attrs | **green** (`test_synth_filter.py`): pass rate within 1 % of target where `N·p ≥ 10^4`, nested pass sets, byte-identical rebuilds; draws `--draw d` beside draw 0; the cluster-correlated variant puts a query's 10 nearest items over 80 % passing | seed 20261008 ([datasets](../docs/system/datasets.md#synthetic-selectivity-attrs)) |
| `filters.query_attrs` | **green** (`test_inputs.py`) | |
| Grid and keys in `suites.yaml` | **green** (`test_config.py`): every (suite, dataset) loads; job / cell counts pinned in `GRID` | |
| Interleaved groups (`--interleave`) | **green**: `measure.latency_group` runs A, B, A, B windows; resume after dropping one arm re-runs that group whole | [evaluation](../docs/system/evaluation.md#interleaved-groups) |
| Per-variant `ids_sha256` and `ids_sha256_canon` | **green**: deterministic per `(bs, k, mode)`; the canonical hash orders rows by (score desc, id asc) | |
| Per-query sidecars | **green**: the sidecar mean equals the record's `recall@k` (largest diff 1.1e-16 on arXiv) | [evaluation](../docs/system/evaluation.md#per-query-sidecar) |
| Quality cache for seed-free arms | **green, bit-exact**: V1, V2 and the postfilter at seed 1 equal seed 0 metric by metric | [evaluation](../docs/system/evaluation.md#quality-cache) |
| Quality-only suites (`perf: false`) | **green**: records `ok` with `perf` null | |
| The item-chunked exact oracle and `bench oracle` | **green, bit-exact, 0 tie flips** against one-shot blobs on arXiv and PubMed; peak 1.6 GB above the resident inputs ([artifact](artifacts/campaign-v2/harness-core/oracle_gate.py)) | a tie flip stays possible on another sweep ([evaluation](../docs/system/evaluation.md#oracle-blob-v4)) |
| Unfiltered cells get the exact oracle | **green** (CPU): `none` cells build the exact unfiltered top-k (`run.NO_ATTRS`) | [evaluation](../docs/system/evaluation.md#self-match) |
| Self-match exclusion | **green** (CPU and a GPU smoke): on `self_match` datasets every filter cell drops the query's own item from the oracle row and the module's ids; held-out recall there is self-retrieval | [evaluation](../docs/system/evaluation.md#self-match), [script](artifacts/selfmatch/smoke.sh) |
| V3 `candidate_pool_frac` and `pool_passing_mean` | **green**: the pool is `max(2000, round(frac × mean pass count))`; both recorded | the 2000 floor binds where `frac × N·p < 2000` |
| V3 `k_bits` | **green**: `k_bits` reaches V3's stage 1; `k_bits` = D equals the default build bit for bit | |
| Bloom width as build params (`m_bits` / `k_hash`) | **green**: a gridded cell gets its own `bloom` and key | |
| Kernel sums and scopes in `profile_once` | **green**: `kernels_us` / `kernels_calls` over every kernel, profiler ranges excluded; `kernel_scopes` split them into scorer / topk / epilogue / other | |
| `--profile` kernel lists complete | **green on a GPU**: counts constant over 30 sessions per arm | mechanism of the original loss open; each profiled entry can pay up to ≈ 12.5 s |
| Quality pass on the device | **green**: quality and per-query sidecars byte-identical, 31 / 31 cells, against the host loop | [artifact](artifacts/campaign-v2/h-qloop/README.md) |
| One `bench run` frees each arm before the next | **green**: four 30 M-class arms in one process without OOM; quality bit-identical to per-arm processes | [artifact](artifacts/h-armfree/gate.sh) |
| `torch.compile` arms | **green**: `compile` in the key, `compile_s` apart from `build_s` | no bit-exact gate between eager and compiled ([script](artifacts/cv2-harness-data/compile_parity.py)) |
| The clock sampler reads the visible GPU | **green** (CPU): `measure.smi_device()` feeds `nvidia-smi -i` | records timed off GPU 0 before this fix carry GPU 0's clock fields; none is in the final run's accepted set |
| Per-job clock log | **green**: a `bench campaign` child's log holds `nvidia-smi -q -d CLOCK` before and after and its under-load histogram | |
| `bench campaign`'s per-group timeout | **green**: 48 h default (`cli.TIMEOUT_H`) | |
| `partial` for `modes` stamped per job | **green** | |
| The `postfilter` baseline | **green**: ids and scores `torch.equal` to a hand-computed matmul → topk(`alpha*k`) → filter reference at every alpha | [evaluation](../docs/system/evaluation.md#the-postfilter-baseline) |
| `bench merge` tables from the FR legs | **green**: all `fr/` subtrees merge; the accepted rows equal `build_results.py`'s frame row for row | [evaluation](../docs/system/evaluation.md#merged-tables-bench-merge), [artifact](artifacts/export-prep/README.md) |
| FR records reproduce on the export code | **green**: 127 quality-only cells, every `quality` block and sidecar equal to the FR record with the same key | [artifact](artifacts/export-prep/README.md) |
| Golden baseline ([`evaluation/golden/`](../evaluation/golden/README.md)) | today's harness meets the golden LiNR cells at 0 (recall, precision; ndcg ≤ 1.6e-9) after re-derivation for L1's fp32 scores and L5's fp32 accumulation ([R-RES](artifacts/campaign-v2/r-res/README.md)) | the golden goodreads cells run on `gsasrec-d128-drop0.5-id` (pinned with `--checkpoint`); arXiv `silvertorch` recall@100 2.0e-6 unexplained; the golden SilverTorch cells were taken on the deleted Triton backend |

## FULL-RERUN

The final run ([full-rerun](artifacts/full-rerun/README.md), with the
extension legs of [full-rerun-ext](artifacts/full-rerun-ext/README.md) and
SYNTH-MID): every number the paper cites, at tag `final-freeze`. Its legs are
on the Hub under `fr/` ([hub-index](artifacts/hub-index.md)).

| gate / leg | state | notes |
|---|---|---|
| **FULL-RERUN gate** ([fr-gate](artifacts/fr-gate/README.md); Hub `artifacts/fr-gate`) | **GREEN** (2026-10-16). (0) all 106 `fr/` trees verified against the Hub manifests. (1) The plan at the final scope: 79 legs, **3,577 cells**. (2) `bench merge`: **3,577 accepted, all `ok`**, one per planned cell, **0 gaps**; 250 keep the `unstable` flag. (3) D1-G: (a) eager vs graph `ids_sha256` equal on **4,189 / 4,189** pairs; (b) **162 quality-only cells rerun** at the tags, equal to every FR record with the key (170 / 170); (c) batch scaling `median_ms(bs 16) < 16 × median_ms(bs 1)` on 3,282 pairs, 0 failures, 1 listed known exception (goodreads-synth LiNR V2 torch `compile=max-autotune` `p1`, k 1000 eager, 24.9×; [decisions](decisions.md#harness)); (d) `manifest.yaml` 0 problems | D1-G (b) did not cover 18 legs whose oracle caches were on another pod (about 20-25 GPU-min to cover; not done) |
| Release cleanup changed nothing the final run executes | **green** ([release-cleanup](artifacts/release-cleanup/README.md)): the files a final-run leg imports are byte-identical to `final-freeze`; every suite resolves to the same jobs | |
| **Results tables** ([`build_results.py`](artifacts/results-draft/build_results.py), `product_recall.py`, `gls_adapted.py`; page from `page/pagedata.py` + `results_md.py`; Hub `artifacts/results-draft/20261016a-fr`) | **built, citable** (the FR gate is green): 11,095 records, one `results.parquet`, `pooled_variance.csv`, `probesweep.csv` | the standing findings are below |
| SYNTH-MID: 20 / 30 / 50 % synthetic pass rate on LAION 10 M / 30 M and a timed `n_probe` sweep ([artifact](artifacts/full-rerun-synth-mid/README.md)) | **measured at `final-freeze`, read into the results tables** | trimmed: bloom only, k 100, no 65,536 lists at 20-50 %, the sweep at bs 16 only; 20 sweep cells at n_probe ≤ 256 kept flagged `unstable` |

**Findings of the results tables** (seed 0 unless named):

- 312 tuned cells. SilverTorch reaches recall 0.90 / 0.95 within a quarter
  of the lists in 219 and is faster than the best LiNR at that recall in
  216 (median 0.28 of its time); 51 more reach it at 50-75 % of the lists
  (SilverTorch faster in 13); 42 not reached.
- Synthetic filters (k 100): SilverTorch beats the best LiNR at 0.90 on
  every synthetic filter at every batch (1.06-34×); at 10-50 % on the
  10-30 M sets 3.7-14× (LAION's real filters at 10-30 M: 0.8-7.4×). Above
  10 % SilverTorch's time at a fixed setting grows with the pass rate; at
  0.95 / bs 64 the two are even at 50 % (0.99-1.02×). LiNR V3 is the fastest
  LiNR at one query on 20-50 % at 10-30 M (0.43-0.74 of the faster exact
  variant); V1 wins at bs 16 / 64.
- Probe sweep (LAION-synth 30M, bs 16): the smallest `n_probe` reaching 0.90
  moves with the filter (256-1,024 at 4,096 lists, 1,024-4,096 at 16,384);
  a fixed 1,024 at 4,096 reaches it everywhere at up to 4× the per-filter
  optimum.
- Bloom against the exact clause filter (separate runs, unstable pairs
  dropped): 0.71-0.87 on PubMed MeSH, 0.77-0.88 on LAION's domain filter at
  deep probes, 1.1-1.6 on LAION tags4 at 120-370 items per list, about 1.0
  at 450-600, 0.88 at 1,800; recall difference ≤ 0.000002.
- Seed spread, pooled per dataset: SilverTorch time range 0.3-4.6 % of the
  median, recall range ≤ 0.008; datasets without seeds use the mean over
  datasets (2.7 %); the error bar on every SilverTorch number is ± half
  that. 0 winner changes at the seed-0 pick over 183 tuned cells.
- **Not covered:** SYNTH-MID's trim (above); arXiv's local-pass-rate check.

### Inputs to the final run (method, not results)

Checks whose outcome set a choice the final run carries. Their numbers are
not for the paper; the choice stands.

| check | state | notes |
|---|---|---|
| IVF tune: SilverTorch `n_lists` per dataset (Hub `artifacts/ivf-tune`) | **done** | goodreads 4096, arXiv 2048, PubMed 4096 (0.95 not reached under the `n_lists` / 4 cap); the final run's stage-1 curves pick its timed points |
| Synthetic-filter rates and grid | **done** | ten uniform rates 0.001-1 (`layout.SYNTH_RATES`), nested pass sets ([datasets](../docs/system/datasets.md#synthetic-selectivity-attrs)) |
| One timed job per GPU on a two-GPU pod (Hub `artifacts/m1-v25`, `artifacts/m1`) | **no interference resolved above about 0.7 %** | the basis for one job per GPU ([roadmap](roadmap.md#multi-gpu-execution)) |
| k-means seeding on LAION 30 M (Hub `artifacts/kmeans-test/20261013`) | **no verdict flips** | k-means++ moves recall by at most +0.0033 at the cap; the final run keeps random init, 10 iterations |
| The exact arms' recall shortfall on PubMed (Hub `artifacts/pubmed-diag`) | **not a defect** | 86-99 % of the missed items are fp16 input rounding orders, the rest ties |

## Datasets

| dataset | state |
|---|---|
| goodreads | staged, layout checked; the harness reads the E1c checkpoints `sasrec-ssm-logq-d{64,128,256}` (`item_id_map.json` sha256-equal to the data root's); every query and item vector has L2 norm 1 |
| arxiv | staged, on the Hub as `pinkmeme/eval-arxiv-papers`. Clause 1 (license) maps the 452,732 null-license items and 1,528 of the 10,000 queries to bucket `none` ([datasets](../docs/system/datasets.md#arxiv)); check: `item_attrs_narrow.pt` min = -1, `eval_split.parquet` `query_attrs_narrow` min = 0 |
| pubmed (`pubmed-medcpt-dot`, the final run's) | on the Hub as `pinkmeme/eval-pubmed@10d48d05` (private). MedCPT scores by dot product, so vectors are stored as shipped, not normalised (`config/pubmed.yaml` `normalize: false`); item norms mean 8.86, CV 3.4 %. Exact top-100 overlap of dot product vs cosine on queries with ≥ 100 passing items: unfiltered 0.53, `c0_mesh` 0.92 |
| laion1m-30m | staged through `eval-data ingest`; `bench check` ok. 30,000,000 distinct-caption items + 10,000 held-out caption queries from Re-LAION parts 0-1, 1,430,979 domains; encode (bf16 nomic-embed-text-v1.5, Matryoshka d256) 10.1 k captions/s; query↔target cosine median 0.80. Not on the Hub (waits for the user's release decision) ([datasets](../docs/system/datasets.md#laion30m)) |
| yambda-500m | **in, unfiltered**: from `pinkmeme/eval-yambda-500m` with its checkpoints; `test.parquet` is the Hub's `trainer/test.parquet` (45,932 users); `bench check` d64 / d128 / d256 ok |
| out of the study | YFCC-10M, OpenAlex, KuaiRand, yambda-5b ([decisions](decisions.md#datasets)); no result of theirs is cited |

### Trainer inputs with `timestamps` (data gates, not citable)

The trainer-input ETLs (yambda, goodreads) write a `timestamps`
column next to `item_ids` ([datasets](../docs/system/datasets.md#yambda);
[artifacts](artifacts/seqrec-encoder/etl-timestamps/)).

| gate | state | notes |
|---|---|---|
| yambda `item_id_map.json` | **passes**: the re-prep, the previous prep and the Hub copy are byte-identical | |
| yambda `item_ids` / `targets` row for row | **train passes; val / test pass only as row multisets** | `cmd_prep` builds val / test with `uid` joins and no sort, so their row order is a fresh draw each run; nothing may be aligned to it by position. Not fixed |
| yambda `timestamps` | **passes**: lengths equal to `item_ids`, non-decreasing | seconds since the anonymized Yambda epoch |
| goodreads `item_id_map.json` | **passes**: sha256-equal to the Hub copy (797,084 items) | |
| goodreads `test.parquet` vs the Hub copy | **fails bit-exact; every difference is a timestamp tie** (38,924 of 313,178 rows) | `sort(["user_id", "ts"])` breaks no ties. Not fixed: a tiebreak would move the output off the Hub copy |
| goodreads `timestamps` | **pass**: unix seconds, lengths equal, non-decreasing | |

## Seqrec encoder

Trainer (`evaluation/training/`, [datasets](../docs/system/datasets.md#training--evaluationtraining)).
H100 80GB HBM3, torch 2.10.0+cu128, sm_mhz 1980. **Not citable.** Test is
full-catalog on the trainer's `test.parquet`, for the checkpoint selected on
val ndcg@10.

### Bars

The published gSASRec checkpoint of the same D, re-scored on the same
`trainer/test.parquet` with the same eval code (ndcg@10 / recall@100):

| dataset | d64 | d128 | d256 |
|---|---|---|---|
| yambda-500m | 0.0846 / 0.1563 ([gate A](artifacts/seqrec-encoder/gate-a/)) | 0.0811 / 0.1486 ([gate A](artifacts/seqrec-encoder/gate-a/)) | 0.0814 / 0.1398 ([E0b](artifacts/seqrec-encoder/e0b-yambda-d256-bar/)) |
| goodreads-work-id | 0.0350 / 0.1486 ([E0](artifacts/seqrec-encoder/e0-goodreads-bars/)) | 0.0361 / 0.1480 ([E0](artifacts/seqrec-encoder/e0-goodreads-bars/)) | 0.0354 / 0.1472 ([E0c](artifacts/seqrec-encoder/e0c-goodreads-d256-bar/)) |

### Final models (the E1c recipe)

gSASRec body, ffn 4×D, sampled softmax (in-batch 4096 + uniform 8192) with
logQ: the trainer's defaults. Only D, epochs, patience and eval cadence vary.

| model | test ndcg@10 | R@100 | ndcg@100 | cov@10 | Δ vs bar (ndcg@10 / R@100) | best epoch (val ndcg@10) | s/epoch | peak GB |
|---|---|---|---|---|---|---|---|---|
| [yambda d64](artifacts/seqrec-encoder/e1c-yambda-d64-sasrec-ssm-logq/) | 0.0945 | 0.1619 | 0.1157 | 0.0382 | **+0.0099 / +0.0056** | 77 (0.1010) | 11.6 | 10.7 |
| [yambda d128](artifacts/seqrec-encoder/y128-sasrec-ssm-logq/) | 0.1006 | 0.1662 | 0.1207 | 0.0389 | **+0.0195 / +0.0176** | 99 of 100 (0.1078), still rising | 13.8 | 13.6 |
| [yambda d256](artifacts/seqrec-encoder/y256-sasrec-ssm-logq/) | 0.0966 | 0.1481 | 0.1108 | 0.0401 | **+0.0152 / +0.0083** | 97 of 100 (0.1040), still rising | 18.7 | 19.7 |
| [goodreads d64](artifacts/seqrec-encoder/e3-goodreads-d64-sasrec-ssm-logq/) | 0.0381 | 0.1447 | 0.0695 | 0.1232 | +0.0031 / **−0.0039** | 16 (0.0402) | 49.4 | 6.7 |
| [goodreads d128](artifacts/seqrec-encoder/g128-sasrec-ssm-logq/) | 0.0410 | 0.1518 | 0.0736 | 0.1577 | **+0.0049 / +0.0038** | 34 (0.0440) | 59.0 | 8.1 |
| [goodreads d256](artifacts/seqrec-encoder/g256-sasrec-ssm-logq/) | 0.0418 | 0.1530 | 0.0743 | 0.1616 | **+0.0064 / +0.0058** | 13 (0.0463) | 80.3 | 11.0 |

Against the success rule ([decisions](decisions.md#sequential-encoder)): five
of six beat the bar on both metrics; goodreads d64 misses on R@100.

### Recipe search (yambda-500m d64)

| run | recipe | test ndcg@10 / R@100 | verdict |
|---|---|---|---|
| [E1](artifacts/seqrec-encoder/e1-yambda-d64-sasrec-ssm/) | gSASRec body, sampled softmax, no logQ | 0.0661 / 0.1093 | without logQ the in-batch negatives push popular items down |
| [E1c](artifacts/seqrec-encoder/e1c-yambda-d64-sasrec-ssm-logq/) | E1 + logQ | 0.0945 / 0.1619 | selected |
| [E2c](artifacts/seqrec-encoder/e2c-yambda-d64-hstu-ssm-logq/) | HSTU body, E1c's loss | 0.0883 / 0.1500 | below gSASRec at 3.6× the s/epoch |

The loss carries the gain; the HSTU body adds cost, not quality.

### Gates

| gate | state |
|---|---|
| A: published checkpoints reproduce their `eval_quality.json` | goodreads d64 / d128 match to 4 decimals; yambda-500m d64 / d128 do not, because the test split on disk is not the one those files were scored on ([gate A](artifacts/seqrec-encoder/gate-a/)) |
| B: retrain yambda-500m d64 on the published recipe | within ±0.002 of the published checkpoint re-scored on the same file (−0.0009 / −0.0006) ([gate B](artifacts/seqrec-encoder/gate-b-yambda-d64/)); one shared negative vector per step collapses gBCE, so gBCE samples per position |
| W6 cleanup equivalence | per-step losses bit-identical before / after, both losses ([artifact](artifacts/seqrec-encoder/w6-cleanup-equivalence/)) |
| unit gates | `tests/training/test_encoder.py`: padding, `sampled_softmax_loss`, `logq_correction`, `TrainConfig` resolution, `resume_every` cadence, `train_on_val` rows (CPU) |

Unverified: `target_frequencies` (logQ) is checked only by hand on CPU;
`--resume` has never run on the GPU; `_resume.pt` is written in place, not
atomically.

## Still unverified, known defects

- A compiled (`graph`-mode) measurement on a warm inductor cache after a
  library change can replay a stale `triton_op` body
  ([testing](../docs/system/testing.md#running)); whether a recorded cell
  outside the final run was affected is not checked.
- The adapted backend at 30 M without a filter and on the exact path at the
  head (last run at `8ff11e0`): rerun before stating "bit-exact at 30 M"
  for those paths.
- `eval-data pubmed plan` trips NCBI's throttle (HTTP 503 on most of a
  16-way burst; `_remote_size` turns it into size 0) and `_fetch_one`
  accepts a GET that ends early when its HEAD failed
  ([artifact](artifacts/pubmed-restage/README.md)).
- The quality subset is a 10k prefix of the query file, not a seeded
  sample; safe only because the current files are shuffled.
- `scripts/check_doc_links.py` cannot parse `for … in f(...)` in a CLI
  module.
- Every row above on a new pod, until the library suite has run there.

## Superseded results

Everything measured before the final run (earlier campaigns, the NUMBERS
legs, the official-vs-Triton head-to-heads, the gSASRec-encoder evals) is
superseded and not for the paper. Git holds the old rows at tag
`pre-release-cleanup`; the Hub keeps their records. Our Triton SilverTorch,
LiNR with a bloom filter and YFCC-10M are out of the study and have no rows.
