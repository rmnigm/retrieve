# Evaluation harness v2 — correct measurements, one config matrix, a third of the code

> **Status:** planned 2026-09-05 on `feat/cute-dsl-scorer` (nothing implemented). Target box:
> A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0. Authored on the Mac (no GPU): every
> "current behaviour" claim below cites a file:line that was read; every "will measure X"
> claim is gated by a GPU work package (§6). Supersedes the open harness half of
> [refactor-validation-handoff.md](refactor-validation-handoff.md) and roadmap §4a/§4b
> ([00-roadmap.md](00-roadmap.md)); does not touch `retrieve/` except two library nits in §7.
>
> Question this plan answers: *what does the harness have to measure for the reproducibility
> paper to be comparable with SilverTorch ("Evaluation") and LiNR ("Model Inference
> Benchmarking"), and what is the smallest code that measures it correctly, end to end, for
> every dataset × dim × backend (`triton`, `torch`, `cuda`, `cute`) without babysitting?*
>
> **Amendment, same day (2026-09-05, after this plan was written).** The user decided to adopt
> Meta's official kernels as the reference backend and to delete the CUDA C++ and CuTe backends
> ([silvertorch-official-integration.md](silvertorch-official-integration.md), decisions D1/D7/D9).
> Read every `cuda` / `cute` backend mention below as **`official`**, with one difference: the
> official ops cannot be CUDA-graph captured, so the `graph` mode of §2 applies to `triton` and
> `torch` only and `official` cells record `mode: graph` as `null` (reason `not_capturable`). The
> `PATHS` table in §3.1 has three backends, WP-0's golden run uses `triton` + `torch` only, WP-4's
> jaccard gate compares `official` vs `triton` on `silvertorch`, and WP-5's campaign is
> `{triton, torch, official}`. **Ordering authority:** [00-roadmap.md](00-roadmap.md) §1 Phase C.

## 1. Where it starts

`evaluation/retrieval/` is 3,884 lines (3,424 code + 460 tests) across 30 files plus 19 YAMLs
(668 lines). The previous refactor ([evaluation-refactor.md](archive/evaluation-refactor.md), E1–E8)
deleted dead weight and added `SweepContext`/`FilterAssets`, `AlgoBase`, `PerfStats`, the oracle
fingerprint and CPU tests, but deliberately left the **measurement methodology untouched** ("Out
of scope: no behavioral changes to measurement methodology") — which is where the problems are:

| file | lines | verdict |
|---|---|---|
| `sweep.py` | 552 | 4-level loop + 9 helpers threading 2 context bags; `_make_perf_row` (461–503) is the only place the schema lives |
| `measure.py` / `passes.py` | 191 / 182 | the measurement path (audited below) |
| `loaders.py` + `queries_cache.py` + `encode.py` | 396 + 117 + 117 | three files for "load item/query tensors"; the `users_limit` trim lives in two of them |
| `config.py` / `context.py` | 146 / 74 | 6 dataclasses (`EncodeConfig`, `FilterSweepCfg`, `FilterCfg`, `EvalConfig`, `SweepContext`, `FilterAssets`) for a config that is one dict |
| `algos/` (7 files) | 563 | 5 wrappers of ~35 lines each + a Protocol, a base class, a registry, a capability table |
| `cli/run_evaluation.py`, `stage_results.py`, `upload_results.py` | 314 / 74 / 206 | 56 lines of `LogSinks` tee (112–167), `EVAL_TYPES` file lists (35–54); rename `results/x/` → `x.perkernel/` (62–64); HF mirror of that layout |
| `results_io.py`, `metrics.py`, `oracle.py` | 34 / 171 / 170 | fine; metrics syncs per batch (150–154) |

### Verdicts on the ten measurement concerns

1. **Cold-L2 `do_bench`, p20/p80, no p99/QPS/mean/vector/build time — real.**
   `measure_forward_cuda` (measure.py:144–153) times via `triton.testing.do_bench(..., quantiles=[0.5,0.2,0.8])`,
   whose ~256 MiB L2 flush before every call the file itself documents (measure.py:109–110). Both
   papers report warm, replayed-traffic numbers (SilverTorch: 5,000 replayed requests, "50 warm-up
   batches followed by 100 test batches", P99 budget + QPS; LiNR: mean + p95 by batch size). Cold
   L2 penalises exactly the small hot structures serving keeps warm (centroids, hash seeds, the
   query batch) while the 3M×128 index does not fit a 40 MB L2 either way. There is no throughput,
   no p95/p99, no mean, no vector and no build time; `MIN_SAMPLES = 30` (measure.py:24) is far too
   few for a p99. `p20/p80` exist only because that is what `do_bench` returns.
2. **What is timed is cudagraph replay + static-buffer input copy, with `dynamic=True` — real, and
   unverified.** Every algo is compiled in its constructor (`AlgoBase._finalize`, `_helpers.py:58`)
   so the *quality* pass also runs through inductor; ragged chunks (`QUALITY_BATCH_SIZE = 16`,
   skip-mask holes, passes.py:89–95) are why `dynamic=True` and `recompile_limit = 64`
   (evaluate.py:58) exist at all. Neither paper serves through CUDA-graph replay (LiNR: TF/PyTorch
   custom ops; SilverTorch: a PyTorch predictor), and WP-6 measured the gap: eager 0.31–1.29 ms vs
   graph 0.06–0.29 ms for the same forward (`cute-dsl-scorer-artifacts/wp6/graphs.md`). The harness
   never asserts `cudagraph_skips == 0` (graphs.py:208 does), so whether a cell actually replayed
   a graph is unknown; `dynamic=True` also makes the library fight SymInts (main.py:180–186). Fix:
   quality in eager; perf as two labelled variants, `eager` (paper-comparable) and `graph`
   (`mode="reduce-overhead", dynamic=False, fullgraph=True`, one capture per bs, skips asserted 0).
3. **Memory columns do not mean what a reader thinks — real, worse than suspected.**
   `index_mem_mib` is an allocator delta around `build_algorithm` (sweep.py:282–288): it *excludes*
   the standalone filter built earlier (sweep.py:141–154) for mask algos but *includes* the
   codesigned attrs for `silvertorch` — not comparable across algos, and not what the thesis says it
   is ("общий размер всех тензоров модуля поиска и фильтрации", main.tex:673). `fwd_scratch_mib`
   is **0.0 on 6,066 of the 7,104 checked-in rows** (all non-deep-sweep campaigns): under graph
   replay nothing is allocated in the window (measure.py:128–139), so the metric is dead;
   `peak_mem_mib` is therefore just process-resident bytes (item_embs fp32 + pool + graph pools).
   Fix: `index_mib = Σ buffers` of the algo incl. its filter submodule (deterministic, matches the
   thesis definition), `filter_mib` separately, `peak_fwd_mib` from the eager window only.
4. **Wide, repetitive rows — real.** 810 rows = 270 cells × 3 batch sizes in
   `results/goodreads/d128-filter.json`; quality is recomputed per `k` (sweep.py:251–263 rebuilds the
   algo per `k`) and copied onto every bs row. The `cell` string (sweep.py:483) omits `impl` and
   `params`: it **collides on 90 / 162 rows** in the two deep sweeps, which is why the CUDA handoff §7
   join script had to build its own key. `extra.params` values are stringified (sweep.py:500).
5. **Global backend axis — real.** Only `SilverTorch` dispatches four ways (architecture.md:377–393);
   `PostfilterKNN*` ignore the flag entirely (linr_v4.py:8–9), so on `none` cells `linr_v1`/`linr_v4`
   `torch` rows duplicate `triton` rows and `cuda` rows for any LiNR algo are torch-path numbers
   labelled `cuda`. Filter modules are built once per *algo* backend (sweep.py:141), so
   `[triton, cuda, torch]` builds two identical Triton filters. Fix: backends are declared per algo
   in the suite, the registry maps `(algo, filter_kind, backend)` to the code path that actually runs
   (`path` column), duplicates collapse, filters are keyed by filter backend.
6. **Oracle vs held-out — both are needed, and the second is missing on filter cells.** Filter cells
   score only against the filtered brute-force oracle (sweep.py:395–415); LiNR's tables report
   "recall label@2000" on the filtered sets, i.e. against labels. The oracle build already evaluates
   the exact mask per query (oracle.py:96–103); recording whether the held-out item passes it is
   one extra gather, after which held-out recall on filter cells is the same quality pass with a
   second target tensor. `k_gt = max(ks)` is fine.
7. **Robustness — mostly real.** `device` is hard-coded (sweep.py:488); `--filter-kind` is
   `type=str` (evaluate.py:35) and a typo yields an empty JSON; `--resume` is all-or-nothing per
   (config, algo) because rows are only written at the end (evaluate.py:106–110, run_evaluation.py:90–93)
   — a crash in hour 5 of a 6-hour algo loses everything; `perf_pass_cached` returns a zero-latency
   row when every user is skipped (passes.py:141–142); `_autotune_prewarm` silently skips
   `bs > n_users` (sweep.py:431–432). **Blocking bug:** with `users_limit` on a checkpoint dataset
   the cache returns pre-trimmed queries (queries_cache.py:77–87) and `load_query_attrs` then raises
   on the row-count check (loaders.py:308–312, called at evaluate.py:74–76) — every shipped goodreads
   filter config has `users_limit: 10000` (goodreads/d128-filter.yaml:26), so the golden gate cannot
   even run on this branch (handoff §5 recorded it, unfixed). The 314-line orchestrator is a 60-line
   loop once the child logs to a file itself and rows stream to JSONL.
8. **Statistics — real.** One seed everywhere (`seed: 0` in all 19 YAMLs), no repeated timing
   windows, no spread, no CI, no per-query vector; `MIN_SAMPLES = 30`.
9. **Config — real.** `diff goodreads/d128-filter.yaml d256-filter.yaml` is 4 lines (output,
   `gt_subdir`, checkpoint, a comment); yambda/goodreads/arxiv quality YAMLs differ only in paths.
   19 files carry the same `encode:` block, the same algorithm list, the same sweeps. Recommendation
   in §3.3: one YAML per dataset with a `dims:` list plus one `suites.yaml`; ~150 lines total.
10. **Other findings.** TF32 is pinned twice (evaluate.py:63–64 and measure.py:73–76).
    `torch._dynamo.reset()` runs between sweeps only (sweep.py:105), so graph pools accumulate across
    the cells of a sweep. No GPU-clock locking or recording; provenance is `gpu/torch/commit` only
    (sweep.py:441–458) — no driver, CUDA, triton, dirty-tree, clocks, timestamp, config hash.
    `accumulate_metrics` syncs per 16-query chunk (metrics.py:154). Eager `bloom` forwards pay a pageable H2D copy per call
    (`build_query_signatures` builds its salt with `torch.tensor(_SALT, device=cuda)`, graphs.md
    "Manual CUDAGraph" note) — a library artifact that would inflate the new `eager` numbers.
    The thesis methodology (main.tex:662–676) says one process per (dataset, dim, bs, filter, impl);
    the code isolates per (config, algo) (run_evaluation.py:250–257) — bs and filters share a process.

## 2. Measurement protocol

One **cell** = `(dataset, dim, filter_kind, sweep, algo, backend, params, seed)`. Per cell, in
one process, in this order. Everything below is `eager` unless labelled `graph`.

1. **Environment, once per process.** Seed torch/CUDA; TF32 off (one function); record
   provenance: GPU name, driver, CUDA runtime, torch/triton versions, git commit + dirty flag,
   hostname, UTC start, `nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,power.limit`
   sampled *after* warm-up. Clock locking is recommended, not enforced (`nvidia-smi -pm 1 &&
   nvidia-smi -lgc 1410` in the runbook); the harness warns when `clocks.sm` drifts >5 % from the
   first cell. One 3-matmul GPU warm-up (keep `warm_gpu_once`).
2. **Inputs, once per (dataset, dim).** Item embeddings, queries, held-out targets, query attrs;
   `users_limit` applied in exactly one place (prefix, as today, so quality stays golden-comparable).
   Filter modules: one per *filter backend* (`triton` for triton/cuda/cute cells, `torch` for torch
   cells). Per sweep: synthesised query attrs, skip mask, exact oracle top-`max(ks)` (fingerprint
   cache as today) **plus** `target_in_filter[u]` (held-out item passes the exact mask) and
   `pass_rate` (exact; bloom cells also record the bloom rate → `bloom_fp_rate`) — the axis both
   papers organise their tables around (LiNR high/low pass-rate; SilverTorch FP rate vs bits).
3. **Build.** `build_s` = wall time of algo construction + `register_index` with a sync on each
   side. `index_mib = Σ numel·itemsize` over `algo.buffers()` (the filter is a submodule, so mask
   algos include it, codesigned includes its attrs); `filter_mib` for the filter alone. No allocator
   deltas. Per-phase build time (k-means / quantize / assemble) stays a roadmap item; it needs a
   3-line `build_timings` dict in `SilverTorch.register_index` (main.py:132–143), not harness code.
4. **Quality, eager, once per cell at `k_max = max(ks)`.** Stream all kept users in chunks of 16
   (the OOM bound from passes.py:31–36 stands), accumulate `recall/ndcg/precision/mrr` at every `k`
   in `ks` from the one top-`k_max` list (exact for every algo here: same candidate set, same
   scores, `torch.topk` sorted). Targets: **oracle** on filter cells, **held-out** always (on filter
   cells restricted to users with `target_in_filter`, and `n_queries_heldout` recorded). Metrics
   accumulate as running sums on device; one `.item()` per metric at the end. Exact algos assert
   `recall_oracle@k ≥ 0.99` (fp16 tolerance per main.tex:706); a failure kills the run.
   **Amended by §8.2 K:** backends are no longer the innermost loop — each runs in its own
   process — so cross-backend parity is not free any more. Correctness lives where it always
   belonged, in the library's parity suite (**O** §5 T1–T7, roadmap B2, `torch.equal` on scores);
   the harness keeps only the *wiring* check that C4 and **O** WP-6 gate on, via a spill file:
   the first backend of a `(dataset, dim, algo, params, seed)` group writes its top-`k_max` ids to
   `results/_parity/<key>.npy` (≤ 40 MB, one alive at a time, deleted when the group closes) and
   later backends record `jaccard@k` and `score_max_abs_diff` against it. ~15 lines in `run.py`.
5. **Perf, per `(k, bs, mode)`** with `mode ∈ {eager, graph}` and `module.k = k` set before each
   variant (a WP-1 check confirms no layer bakes `k` into buffers at register time). Inputs: the
   fixed-seed pool of 4,096 query batches (and attr batches) rotated round-robin, identical across
   backends and modes. No L2 flush: the pool rotation makes index reads cold naturally while the
   structures serving keeps hot stay hot. `graph` = `torch.compile(mode="reduce-overhead",
   dynamic=False, fullgraph=True)` of a per-bs callable; after warm-up assert
   `counters["inductor"]["cudagraph_skips"] == 0` and one `cudaGraphLaunch` per call (as graphs.py
   130–144 does) — otherwise the cell is an error, not a mislabelled number. Protocol per variant:
   50 warm-up calls → sync → **3 windows** of `N = clamp(2 s / median_est, 1000, 5000)` calls, each
   call bracketed by CUDA events on the current stream, wall clock around the window with one
   sync at the end. From the window with the median median: `median_ms, mean_ms, p95_ms, p99_ms,
   min_ms, n`, `qps = N·bs / wall_s` (closed-loop single client — the in-process analogue of the
   papers' client-side QPS), `host_gap_ms = wall/N − mean_gpu_ms` (wp4 `wall_vs_gpu.txt`'s number,
   diagnostic), and `spread = (max − min) / median` of the three window medians; `spread > 0.05`
   sets `unstable: true`. The per-call vector of the chosen window goes to the samples parquet.
   `peak_fwd_mib` = `max_memory_allocated − allocated_before` over the first eager window only
   (graph mode allocates nothing). The first eager call runs under
   `torch.cuda.set_sync_debug_mode("warn")` to catch hidden host syncs. `--profile` (optional,
   off by default) wraps one eager call in `torch.profiler` and stores per-kernel CUDA µs
   (top 8 kernels) — the wp4 `kernel_only.py` split, 25 lines.
6. **Seeds and repeats.** Timing repeats are the 3 windows above (no rebuild). Seeds change the
   IVF (k-means), the OPORP projection (`v3_seed`) and the pool; v1/v2/v4 are seed-invariant in
   quality, so `seeds: [0, 1, 2]` applies only to the `deep` suite and the headline `filter` cells
   (d128 `c0_*`, `all4`); report median across seeds with min–max whiskers; paired tests offline.
7. **Comparability, stated once in the paper.** Same as SilverTorch: A100, D=128, INT8 IVF,
   n_probe grid incl. **24** (their production setting; add it to the deep grid), bloom bits/FP-rate,
   mean latency at bs=16 vs recall (their Fig. 5), QPS and P99 from the same vector. Same as LiNR:
   V1–V4 by batch size (1, 16), mean + p95, label recall on filtered sets, high/low pass-rate
   bucketing by `pass_rate`. Necessary differences: in-process (no RPC, no user tower, no
   5,000-request replay), 0.8–5.4 M items vs 10–80 M / 15.5 M, 80 GB card, k ≤ 1000 vs 1024/2000,
   closed-loop QPS rather than open-loop load, no live updates. `eager` is the comparable number;
   `graph` is the deployed-best-case number and is reported alongside, never instead.
8. **Cell cost.** Quality once per cell (not per k), perf 3 k × 3 bs × 2 modes × 3 windows ≈ 2 min
   per cell on A100; the campaign in §6 WP-5 is ~700 cells ≈ 24 h (today: 3 builds per cell).
   Two amendments pull in opposite directions and roughly cancel: §8.2 A (build params sweep
   separately from query params) removes ~5 of every 6 k-means builds in the `deep` suite, while
   §8.2 K (a process per backend, ~200–250 boundaries instead of ~80) adds a dataset reload and a
   CUDA context init per boundary — *est.* 30–60 s each, 2–4 h over the campaign. WP-5 records
   both so the next plan does not have to guess.

## 3. Target architecture

### 3.1 Files (`evaluation/retrieval/`, budget 1,450 code + 350 tests ≈ 1,800 lines, from 3,884)

| file | lines | owns |
|---|---|---|
| `config.py` | 80 | `load_matrix(dataset.yaml, suites.yaml, cli overrides) -> list[Job]`; `{dim}` templating; grid → explicit param combos (`itertools.product`, 5 lines) |
| `data.py` | 220 | arxiv / SASRec loaders, encode cache (mtime + max_seq + users_limit key), query attrs, sweep qa + skip mask, filter modules by filter backend. Merges `loaders.py`, `queries_cache.py`, `encode.py` |
| `oracle.py` | 130 | as today + `target_in_filter`, `pass_rate` (bloom rate when a bloom module is given) in the cached blob |
| `metrics.py` | 90 | recall/ndcg/precision/mrr as device running sums for a list of ks; `jaccard_at_k` |
| `algos.py` | 220 | the five `nn.Module` wrappers (filter as submodule), `ALGOS` dict, `FILTER_KINDS` and `PATHS` tables (`PATHS[(algo, filter_kind, backend)] -> "triton"|"torch"|"cublas"|"cuda"|"cute"`), `build(job, data)`. Merges `algos/*` |
| `bench.py` | 180 | `timed_build`, `index_bytes`, `latency(fn, pool, mode)` → dict + samples, `graph_callable(module, bs)` with skip assertion, `profile_once`, `provenance()` |
| `run.py` | 250 | the loop: jobs → per (dataset, dim) inputs → per (filter_kind, sweep) assets + oracle → per (algo, params, seed) build → per backend quality + parity → per (k, bs, mode) perf → `append_record`. Fails loudly; no skip that is not logged with the reason and counted |
| `cli.py` | 120 | `bench run`, `bench campaign`, `bench report` (click group); resume by reading existing cell keys from the JSONL |
| `report.py` | 200 | JSONL + parquet → the thesis/paper tables and figures (§6 WP-6) |
| `tests/` | 350 | config expansion, metrics vs reference, oracle padding/fingerprint/target_in_filter, record schema, PATHS table covers ALGOS × FILTER_KINDS × backends, `k`-slice invariance on a tiny CPU exact algo |

`upload_results.py` becomes a 60-line mirror of `results/` (no staging layout to know about).
`eval_datasets/` and `training/` are untouched. Console scripts: `bench` replaces `evaluate`,
`run-evaluation`, `stage-results`.

**Classes that survive, with justification.** (a) `Job` — one frozen dataclass: the fully resolved
cell spec (dataset, dim, suite, filter_kind, sweep, algo, backend, params, ks, batch_sizes, seed,
paths). It exists because `dataclasses.asdict(job)` *is* the record's key block and the resume
key; a dict would need the same field list written three times. (b) The five algo `nn.Module`s
in `algos.py` — needed for `buffers()` (memory), `.k` (perf per k), `torch.compile`, and their
`forward` is the readable spec of each cascade. No base class: `__init__` builds layers and
registers the filter as a submodule, `forward` is 3–8 lines; 20 lines each. Nothing else.

### 3.2 Output: one JSONL record per cell + one parquet of latency samples

`results/<suite>/<dataset>-d<dim>.jsonl`, appended by the subprocess the moment a cell finishes
(this is the crash-resilience the handoff deferred; `--resume` = skip keys already present).
Nested, not wide — `perf` is a list, `polars.read_ndjson(...).explode("perf")` flattens it.

```json
{"dataset":"goodreads","dim":128,"suite":"filter","filter_kind":"clause","sweep":"c0_genre",
 "algo":"silvertorch","backend":"triton","path":"triton","params":{"n_lists":1024,"n_probe":24},"seed":0,
 "n_items":797043,"n_queries":9859,"n_queries_heldout":8231,"pass_rate":0.21,"k_max":1000,
 "build_s":12.3,"index_mib":297.8,"filter_mib":0.0,
 "quality":{"oracle":{"recall@100":0.5890,"ndcg@100":0.6516,"recall@500":…},
            "heldout":{"recall@100":0.041,…},"jaccard_vs_first@100":1.0,"score_max_abs_diff":0.0},
 "perf":[{"k":100,"bs":1,"mode":"eager","n":2000,"median_ms":0.52,"mean_ms":0.53,"p95_ms":0.58,
          "p99_ms":0.66,"min_ms":0.49,"qps":1880.0,"host_gap_ms":0.05,"spread":0.01,"unstable":false,"peak_fwd_mib":41.2},
         {"k":100,"bs":1,"mode":"graph","n":5000,"median_ms":0.13,…}],
 "env":{"gpu":"NVIDIA A100-SXM4-80GB","driver":"580.65.06","cuda":"12.8","torch":"2.10.0+cu128",
        "triton":"3.6.0","commit":"f0838d0","dirty":false,"host":"gpu-a100","sm_mhz":1410,
        "mem_mhz":1593,"started":"2026-09-10T08:00:00Z","config_sha":"…"}}
```

**Amended by §8.2** (B, C, D, E, F, and the `load` key): the record also carries
`schema_version`, `status`, `code_version`, `git_branch`/`python`/`clocks_locked` in `env`, and
each `perf` entry carries `iqr_ms`, the two outlier counts and `load: "closed_loop"`.

`results/<suite>/samples/<dataset>-d<dim>.parquet`: one row per `perf` entry — key columns, `k`,
`bs`, `mode`, `ms: list[float]` (≈ 35 MB per campaign). JSONL for records (greppable, diffable,
append-only, nests); parquet for the vectors (8 M floats). `polars`/`pyarrow` are dependencies already.

### 3.3 Config: one YAML per dataset + one `suites.yaml` (5 files, ≈ 150 lines, from 19 / 668)

```yaml
# config/goodreads.yaml (12 lines)
data_dir: data/goodreads-work-id
checkpoint: data/goodreads-work-id/checkpoints/gsasrec-d{dim}-drop0.5-id/best_model.pt
dims: [64, 128, 256]
encode: {batch_size: 512, num_workers: 8, max_seq_length: 200}
users_limit: 10000
filters:
  attrs: item_attrs_narrow.pt
  reverse: clause_is_reverse_narrow.pt
  clause: {c0_genre: [0], c1_lang_reverse: [1], c2_format: [2], c3_year: [3], c0c1: [0, 1], all4: [0, 1, 2, 3]}
  bloom:  {c0_genre: [0], c2_format: [2], c3_year: [3]}      # forward-only: no reverse clause
```

`arxiv.yaml` has `content_dir: content{_d{dim}}` (d256 → `content`) and no `checkpoint`;
`yambda-500m.yaml` / `yambda-5b.yaml` have no `filters`. `gt_subdir` is derived (`gt_d{dim}`),
`output` is the CLI's `--out`, `split`/`device`/`query_emb_path` go away (never varied).

```yaml
# config/suites.yaml (≈ 45 lines)
quality:                       # unfiltered, vs held-out
  datasets: [goodreads, yambda-500m, yambda-5b, arxiv]
  filter_kinds: [none]
  ks: [100, 200, 400]
  batch_sizes: [1, 16]
  algos: {linr_v1: [triton], linr_v3: [triton, torch], linr_v4: [triton], silvertorch: [triton, torch, cuda, cute]}
filter:                        # vs oracle + held-out-in-filter
  datasets: [goodreads, arxiv]
  filter_kinds: [clause, bloom]
  ks: [100, 500, 1000]
  batch_sizes: [1, 8, 16]
  algos: {linr_v1: [triton, torch], linr_v2: [triton, torch], linr_v3: [triton, torch],
          linr_v4: [triton, torch], silvertorch: [triton, torch, cuda, cute]}
  seeds: {default: [0], headline: {sweeps: [c0_genre, c0_maincat, all4], dims: [128], seeds: [0, 1, 2]}}
deep:
  datasets: [goodreads, arxiv]
  dims: [128]
  filter_kinds: [clause, bloom]
  ks: [100, 200, 400]
  batch_sizes: [1, 8, 16]
  seeds: [0, 1, 2]
  algos: {silvertorch: [triton, cuda, cute, torch], linr_v3: [triton, torch]}
  params:                      # dict-of-lists = grid; list-of-dicts = explicit combos
    silvertorch: {n_lists: [1664, 8192], n_probe: [4, 8, 24, 32, 128, 256]}
    linr_v3: {candidate_pool: [2000, 4000, 8000, 16000, 32000]}
bloom: {m_bits: 1024, k_hash: 5}   # shared defaults, overridable per suite
```

**Amended by §8.2 A**: `params:` splits into `build:` (`n_lists`) and `query:` (`n_probe`,
`candidate_pool`), because the index is built once per build config and re-queried per query
config — 2 builds in the `deep` sweep above, not 12. Sweep entries also accept `disabled: true`
(§8.2 J).

`config.load_matrix` expands this to `Job`s; `PATHS` collapses backends that run the same code
(`linr_v1`/`linr_v4` on `none` → one job with `path: cublas`) and logs each collapse once.
`is_valid_combo` (`n_probe ≤ n_lists`) stays as 3 lines in `algos.build`.

### 3.4 CLI

```
bench run      --dataset goodreads --dim 128 --suite filter [--algo A] [--backend B] [--filter-kind K]
               [--sweep S] [--k 100] [--bs 1] [--seed 0] [--mode eager|graph] [--skip-quality|--skip-perf]
               [--profile] [--out results] [--force]
bench campaign --suite filter|quality|deep|all [--dataset D] [--dim N] [--resume|--force] [--out results]
bench report   results [--tables docs/thesis/tables] [--figures docs/thesis/figures]
```

`bench campaign` is a loop: per `(dataset, dim, algo, backend)` in the suite —
**amended by §8.2 K**, was `(dataset, dim, algo)` — `subprocess.run(["uv", "run", "bench", "run",
...], stdout=open(log, "a"), stderr=STDOUT)` with one log per child under `results/_logs/`, a
one-line summary, non-zero rc recorded, loop continues (`--resume` fills gaps). Process isolation
is the one thing the old orchestrator got right and the new boundary takes it further: a backend
is the thing under test, so it gets the process, and no dynamo cache, allocator arena or CUDA
graph pool outlives it. The cost is a dataset reload per boundary (§2.8) and parity by spill file
rather than by shared memory (§2.4).

## 4. What NOT to abstract

- **No context bags.** `SweepContext`/`FilterAssets` go; `run.py` passes the four input tensors and
  the per-sweep assets as plain function arguments (≤ 6 per call) or one local dict `assets`.
- **No stats dataclasses.** `PerfStats`/`QualityStats` go; `bench.latency` and `metrics.finalize`
  return dicts that are merged into the record as-is. New statistic = one key.
- **No algo framework.** No `AlgoBase`, no `RetrievalAlgo` Protocol, no `_finalize`, no
  `runtime_checkable`; `ALGOS = {"linr_v1": LinrV1, …}` and two module-level dict tables.
- **No config object model.** No `EvalConfig`/`EncodeConfig`/`FilterCfg`/`FilterSweepCfg`; YAML →
  dict → `Job`. No YAML anchors/merges, no env interpolation, no schema library.
- **No logging framework in the orchestrator.** No `LogSinks`, no tee, no `current.log` symlink;
  the child writes its own log, the parent redirects stdout to a file.
- **No results-IO layer, no staging layout, no composite `cell` string.** `report.py` reads JSONL
  with `polars.read_ndjson`; key columns are columns and joins are `group_by` on them.
- **No timing strategy classes.** `mode` is a string and `graph_callable` is one function; no
  `Measurer`, no statistics registry, no "pass" base class.
- **No exception swallowing, no retries, no CPU/device option, no `--eval-type` presets.**

## 5. What gets deleted

| what | reason |
|---|---|
| `context.py`, `passes.py`, `results_io.py`, `algos/_helpers.py`, `algos/__init__.py` split, `cli/stage_results.py`, `cli/run_evaluation.py`, `cli/evaluate.py` | replaced by `run.py`/`bench.py`/`algos.py`/`cli.py` (§3.1) |
| `triton.testing.do_bench` dependency, L2 flush, auto-extending `rep_ms` loop, `MIN_SAMPLES`, `MEM_REPS` | replaced by the event-per-call windows of §2.5 |
| `torch.compile(dynamic=True)` in constructors, `recompile_limit = 64`, `_autotune_prewarm` | quality is eager; perf compiles per bs statically; warm-up covers capture |
| `index_mem_mib` (allocator delta), `fwd_scratch_mib`, `peak_mem_mib`, `p20_ms`, `p80_ms`, `device`, `suite`, `cell`, stringified `extra.params` | replaced by `index_mib`, `filter_mib`, `peak_fwd_mib`, `p95/p99/mean/min/qps`, key columns, native params. Old columns are not carried: this is a schema break, and the old JSONs stay readable under `results/archive/` |
| 19 YAMLs, `_defaults` anchors, `EVAL_TYPES`, `gt_subdir`, `output`, `split`, `device`, `query_emb_path`, `content_subdir` | the matrix of §3.3 |
| `triton_knn` alias, `SUPPORTED_FILTER_KINDS` vs `PATHS`, per-`k` rebuilds | one table; quality at `k_max` sliced, perf sets `module.k` |
| `stage-results`, `.perkernel/`, `upload_results` layout logic, `run-logs` tee | see §3.4 |
| checked-in `evaluation/results/*.json` from the harness's output path | moved to `results/archive/` (pre-schema, pre-oracle-fix, per roadmap §2 not citable) |

## 6. Work packages

Each WP is one PR; `ruff check` clean on touched code; CPU tests green on the Mac; GPU gates
run on the A100 box. Effort in focused days.

- **WP-0 — golden baseline on the old harness (GPU, 0.5 d).** Commit the 3-line fix for the
  `users_limit` row-count bug (`load_query_attrs` trims `qa[:n]` to the already-trimmed queries);
  run `evaluate --config config/goodreads/d128-filter.yaml --algo <each of 5> --filter-kind clause
  --sweep c0_genre` (`triton`, `torch`) and `arxiv/d128-filter` `c0_maincat` for `silvertorch`
  (`--backend cuda cute`) into `golden/` — the only cross-check the new quality numbers get.
- **WP-1 — `bench.py`, `metrics.py`, `algos.py` + tests (CPU, 2 d).** Latency windows, graph
  callable with skip assertion, `index_bytes`, provenance, profile; device-accumulated metrics
  (test: equal to today's per-row means to 1e-9 on random data); the five wrappers with filter as
  submodule and `.k` settable; `PATHS` table with a test that it covers `ALGOS × FILTER_KINDS ×
  {triton, torch, cuda, cute}` and agrees with architecture.md's dispatch table. Check that no
  layer bakes `k` into `register_index` (grep + one test per layer on CPU where possible).
- **WP-2 — `config.py`, `data.py`, `oracle.py` + tests (CPU, 1.5 d).** Matrix expansion (test:
  job count and keys for each suite), `{dim}` templating, grid expansion, single `users_limit`
  site, filter modules keyed by filter backend, oracle blob v4 with `target_in_filter` and
  `pass_rate` (test on the existing tiny CPU oracle fixtures), bloom pass rate.
- **WP-3 — `run.py`, `cli.py`, deletions, docs (1.5 d).** The loop, JSONL append, resume by key,
  the campaign loop; delete everything in §5; rewrite `docs/system/evaluation.md` to the new
  protocol (§2 verbatim), update `architecture.md`'s pointer, archive
  `evaluation-refactor.md` and the harness half of `refactor-validation-handoff.md`.
- **WP-4 — GPU gate (1 d).** `bench run --dataset goodreads --dim 128 --suite filter --filter-kind
  clause --sweep c0_genre`, all algos and backends. Gates: (1) `quality.oracle.recall@k`/`ndcg@k`
  equal to WP-0 golden within 1e-6 for every `(algo, backend, k)` — tie order under the `k_max`
  slice is the only permitted difference; (2) `graph` `median_ms` within 5 % of golden `median_ms`
  at the same bs (the old number *was* graph replay minus the L2 flush; a larger delta must be
  explained by the flush via a throwaway `--flush-l2`, then deleted); (3) `cudagraph_skips == 0`
  in every graph cell; (4) `jaccard_vs_first@100 == 1.0` for `torch` vs `triton` on exact algos and
  `cuda`/`cute` vs `triton` on `silvertorch`; (5) no `unstable` cell at locked clocks; (6) kill the
  child mid-run and confirm `--resume` continues at the next cell.

> **Superseding decision, 2026-09-15 (user).** *"We don't care about
> reproducing old results now, we're improving all code and rewriting, then
> testing and profiling, then running the full evals step by step."* The golden
> baseline was a bridge: it existed to prove the harness rewrite had not
> changed quality. It is now **informational, not a gate**. WP-4's clauses (1)
> and (2) — equality with A1's numbers and with A1's latency — no longer block
> anything; what still blocks is the harness being *correct on its own terms*:
> clauses (3), (4) and (6), the parity spill, and the official cell running end
> to end. The amendments below stand as the record of what the golden
> comparison could and could not support, and the two residuals remain worth
> knowing — but no step waits on them. **Consequence for the paper: no claim of
> equivalence with the pre-v2 harness may be made from these numbers** (P G9's
> provenance section says what was and was not compared).
>
> **WP-4's gates, amended 2026-09-15 — after the gate ran, by the orchestrator
> with the user.** The run is §12; the amendments below are what it showed the
> clauses could and could not mean. They are recorded here rather than edited
> into the text above, because a gate rewritten to match its own result is
> worth nothing to a reader. Each names the evidence.
>
> - **(1) quality ≤ 1e-6 — kept, with two recorded residuals.** 9 of 11 golden
>   cells pass with a worst *passing* delta of **7.5e-9**. Two do not, and
>   neither is a harness defect. **`linr_v4` (both backends, 7.3e-5)**: the
>   int8 path (`PostfilterKNNInt8`, int32 `>>5` → fp16) produces boundary ties
>   whose order depends on the **query-batch chunk shape**, and the two
>   harnesses chunk differently (v2's `QUALITY_CHUNK = 16` against the golden's
>   64; below `_PAD_M = 17` the module pads, which 16 always trips). §12 killed
>   the alternatives by measurement: not the library (§11.2 has it
>   bit-identical), not the `k_max` slice (rerun at `--k 100` reproduces v2's
>   *own* number to the last digit), not eager-vs-compiled (bit-identical).
>   The clause's "tie order under the `k_max` slice" is therefore too narrow —
>   ties reach the metric by a second route the clause does not name.
>   **Correction, 2026-09-15 (C5 ran L4-b): the attribution is falsified.** v2
>   at quality chunk 64 lands **2.9e-4** from the golden at `recall@100`, four
>   times *further* than chunk 16's 7.3e-5, and moves k=500/1000 by 2.3–2.6e-4
>   where chunk 16 was within 1e-5. Batch shape does move `linr_v4`, but "the
>   golden batched at 64" is not what separates the two harnesses. **The 7.3e-5
>   residual is unexplained**, alongside arxiv's 2.0e-6.
>   **arxiv `silvertorch` `recall@100` (2.0e-6)** is **unattributed**: not the
>   slice and not the batch shape (both tested), ≈ 2 single-hit changes in
>   10,000 rows. Accepted as a bounded residual, not explained.
> - **(2) graph latency within 5 % — the clause was comparing two different
>   estimators of the same clock, and is re-specified as *matched-estimator,
>   bs ≥ 8*.** Every v2 perf entry records `sm_mhz` as a single under-load
>   sample (1410); the golden's 1155 is a whole-run 30 s-cadence median
>   *dominated by idle and between-cell samples* — filtering that trace to
>   `utilization > 50 %` gives median 1410, min 1410. The normalisation
>   therefore injected a flat ×1.221, four times the threshold, and the
>   orchestrator's instruction to pass `--golden-sm-mhz 1155` was wrong. At the
>   matched estimator: **66/66 bs=8/16 rows pass**, ratios 0.957–1.030, no
>   threshold touched (`gate-matched-clock.txt`). **bs=1 is excluded from the
>   criterion**, because the baseline cannot support it: the *golden harness's
>   own repeats* spread up to **21.1 %** at bs=1 against ≤ 0.4 % at bs=8 and
>   ≤ 0.1 % at bs=16, while v2's own windows are ≤ 0.3 %. The noise is in the
>   baseline, and no harness can hit 5 % against it. `--flush-l2` was not run;
>   the bs≥8 rows bound any flush effect at ≤ 3 %.
> - **(3), (6) — unchanged and passed.** 20/20 and one clean kill/resume.
> - **(4) jaccard — the roadmap's clause passed; this plan's wording conflates
>   two different properties and is split.** `official` vs `triton` on
>   goodreads `silvertorch`: **0.999849 ≥ 0.99**, plan cache off. SilverTorch
>   `torch` vs `triton`: **exactly 1.0**, `score_max_abs_diff 0.0`, both
>   datasets. Two rows do not meet it: **`linr_v2` torch-vs-triton** (jaccard
>   0.998743, `score_max_abs_diff` 9.77e-3) and **`official` on arxiv** (0.985,
>   against a threshold the roadmap states for goodreads only). "Exact algo"
>   describes exact *filtering*, not bit-identical arithmetic across two
>   implementations of an fp16 dot product; the clause as written demands the
>   latter. The `linr_v2` divergence is **reproduced independently by the
>   golden** (torch 0.99969470 vs triton 0.99927376, 4.2e-4 recall) and so is
>   not a harness artifact. It is carried as **roadmap L4**, to be settled
>   before D1.
> - **(5) no `unstable` cell at locked clocks — not evaluable in this
>   container** (`nvidia-smi -lgc` denied, no sudo), and 10 of the 18 flags are
>   an artifact: `clocks_drift` compares a start-of-process idle sample (1155)
>   against under-load samples (1410), i.e. it fires on the GPU *boosting*.
>   Eight flags are real bs=1 window spread. Also recorded: `clocks_locked` is
>   `true` on the arxiv cells because the field means "within 2 % of
>   `expected_sm_mhz`" and the box happened to be there — the opposite of what
>   §8.2 F wanted it for. Both are harness defects, carried as open item L4-c.

- **WP-5 — campaign rerun (GPU, ~24 h wall + 0.5 d).** Locked clocks, `bench campaign --suite all`:
  four datasets, all dims, all backends incl. `cuda`/`cute` (clause cells on cuda are legal per
  cuda-silvertorch-handoff §7). Closes roadmap §2 (goodreads oracle rerun) and §4b items 1, 3, 4,
  5, 7 in one pass. Gate: `bench report` runs with no missing cells; `median_ms(bs=16) <
  16·median_ms(bs=1)`; ids identical across `mode`; a rerun is byte-identical in quality.

> **WP-5 staged, 2026-09-15 (orchestrator, on the user's steer "running the
> full evals step by step").** WP-5 is written as one ~24 h campaign. It runs
> as five stages instead, each resumable and each leaving the paper strictly
> better off than the stage before, so that losing the box costs one stage
> rather than the run. `bench campaign --resume` already keys on
> `(cell, code_version)`, so a stage boundary costs nothing but a process
> restart. **No kernel or library change may land between stages** — the tree
> hash is in the resume key, and a change silently invalidates every cell
> recorded before it (§8.2 B).
>
> | stage | what | why this order | rough wall |
> |---|---|---|---|
> | **D1-a** | goodreads + arxiv, `filter` suite, d128, seed 0, `{triton, torch, official}` | the headline filtered cells: every table in the thesis and §C.4's core claims rest on these, and they are the cells C4 already exercised, so a failure here is a harness problem and not a surprise | ~4–6 h |
> | **D1-b** | the same cells at seeds 1, 2 (headline sweeps only, per `suites.yaml`) | closes P gap **G4** (seed variance). Cheap insurance: without it every headline number is a single sample, which a reviewer will ask about | ~8–12 h |
> | **D1-c** | `quality` suite, all four datasets, all dims | the unfiltered recall tables, incl. the two yambda datasets that no longer appear anywhere else. **Both yambda sets (18 GB) fit on `/data` simultaneously** — the stage-run-prune dance in earlier notes was forced by the 26 GB network quota and is unnecessary ([storage.md](../system/storage.md)) | ~3–4 h |
> | **D1-d** | `deep` suite (2 builds × 6 query configs per §8.2 A, seeds 0–2) | the recall-vs-latency Pareto curves — the most cuttable stage if time runs short, and the one whose absence is easiest to explain | ~6–8 h |
> | **D1-e** | the S9 co-design ablation cells (`OfficialConfig(bloom_path="full")`) | O §9's fairness ablation; needs the official arm of D1-a to have landed | ~1 h |
>
> **Wall-time corrected 2026-09-16, from measurement — the stage estimates
> above are wrong, and so is WP-5's "≈ 24 h".** D1-a's first job took **4,390 s**
> and covered **nine sweeps** (six `clause`, three `bloom`) at **365–535 s
> each**, plus ~65 s per new oracle. The "9–13 min per cell" figure that fed the
> estimates came from C4, which only ever ran **one** sweep (`c0_genre`); the
> `filter` suite runs all nine. The unit is therefore ≈ **490 s per (algo,
> backend, sweep, params)**, not per cell.
>
> The matrix at d128 alone is **833 jobs**: goodreads `filter` 165 / `quality` 7
> / `deep` 216, arxiv 198 / 7 / 240. At the measured rate **D1-a is ~25–30 h,
> not 4–6**, and the full five-stage campaign is **days, not one night**.
>
> This does not invalidate anything already recorded — the cells are correct and
> the staging still works — but the ordering now has to be read as *value per
> GPU hour*, because the tail will not all fit. In that light: **D1-a's
> goodreads leg alone (~13 h) gives a complete filtered picture of the primary
> dataset**, which is what the thesis's core tables need; the three headline
> sweeps (`c0_genre`, `c0_maincat`, `all4`) carry those tables, and the other
> six feed the pass-rate analysis. If the rental horizon is short, the trim that
> costs least is **the six non-headline sweeps at seed 0**, then **D1-d** (the
> deep Pareto sweeps, 456 of the 833 jobs).

> **Grid narrowed 2026-09-16 (user), mid-D1-a.** Two changes, made because the
> measured campaign is days rather than one night:
>
> - **`torch` leaves the perf grid** — *"torch backends are less interesting
>   now, priority for triton (fastest?) or official meta version"*. That is 5 of
>   11 jobs per dataset, ≈45 % of the wall time. Cross-backend parity does not
>   depend on the campaign: the library's parity suite is bit-exact, and B3
>   measured `torch` vs `triton` at jaccard **1.0** with `score_max_abs_diff`
>   **0.0** on SilverTorch across both datasets.
> - **Modes narrow to `eager`, plus `graph` on `triton` for the headline
>   sweeps.** The user first asked for graph-only; that was put back because
>   Meta's `official` backend **cannot be captured** (O D7, and O §3 names the
>   two sync sites), so a graph-only campaign would leave the prioritised arm
>   with no latency at all — and `eager` is the mode both papers' published
>   figures are comparable to, and the one B3's head-to-head used.
>
> **Consequence to be honest about:** `bench/run.py` stamps a narrowed mode set
> as `status: partial`, `partial_reasons: ["modes"]`. That is the harness
> describing itself correctly, but it means `report.py`'s citability check will
> mark these records NOT CITABLE. Whether a *deliberate, plan-recorded*
> narrowing should be distinguished from an accidental one is a change to what
> the paper may claim, so it goes to the user rather than being made quietly.

> **Gate, unchanged from WP-5** but evaluated per stage: `bench report` runs
> with no missing cells for that stage, `median_ms(bs=16) < 16·median_ms(bs=1)`,
> ids identical across `mode`, and a rerun byte-identical in quality — the last
> of which is only meetable because L3 and L5 landed first (before them,
> `linr_v2`/`linr_v3` on `triton` could not reproduce themselves at all).
> **Latency caveat carried from C4:** report `sm_mhz_load`, never
> `sm_mhz_idle`, and treat any bs=1 comparison narrower than the baseline's own
> ~21 % spread as noise.

- **WP-6 — `report.py` (1.5 d).** From the JSONL/parquet only: the thesis tables
  (`tab:recall_nofilter`, `tab:pareto_arxiv`/`goodreads`, `tab:batch_scaling`, the memory table at
  main.tex:820–823) as LaTeX, the Pareto figures, deep-sweep curves with seed whiskers, latency
  violins from samples, a backend-parity table (`triton`/`cuda`/`cute` eager vs graph, the
  wp6 table regenerated from the real index), and the paper's comparison table placing our
  `eager` bs=16 mean latency / p99 / QPS / pass-rate rows next to SilverTorch's and LiNR's
  reported numbers with the §2.7 differences as footnotes. Also emits the methodology paragraph
  for main.tex §"Методология замеров" so text and code cannot drift again.

## 7. Risks / open questions

- **`k`-slice equivalence and `module.k` mutation.** If any layer sizes a buffer by `k` at
  register time, WP-1 finds it and that algo rebuilds per `k` (a per-algo flag, not a design
  change). Tie-order differences may break byte-identity with golden; the gate is 1e-6.
- **Eager bloom is inflated by the library's per-call salt tensor** (§1.10). Fix in `retrieve`
  (register the salt as a buffer) before WP-5, else the LiNR-comparable bloom rows carry a
  pageable-copy artifact. Second library nit: `SilverTorch.build_timings` for phase times.
- **Static-shape graph capture per bs** triples captures vs today's one dynamic graph; ~20 s per
  capture × 3 bs × cells is inside the 2-min budget but should be measured in WP-4.
- **`torch` backend at bs=16 on arxiv d256 loose filters** materialises `[B, P, D]` (passes.py:31–36);
  keep bs ≤ 16 and let OOM kill the cell loudly (it is a finding about the torch path, not noise).
- **`users_limit` is a prefix, not a sample.** Kept for golden comparability; switching to a
  seeded subsample after WP-4 changes every quality column and should be its own documented delta.
- **Clock locking needs root** on the GPU box; if unavailable, `unstable` flags plus recorded
  `sm_mhz` are the fallback and the paper says so.
- **Held-out recall on filter cells** depends on `target_in_filter` coverage: on tight sweeps
  (`all4`) few users qualify; report `n_queries_heldout` next to it and do not headline it there.
- **Schema break.** Downstream thesis scripts that read the old columns are gone with the LaTeX
  move (roadmap §4 note); `report.py` is the only consumer. Old JSONs remain under `results/archive/`.
- **No precomputed-oracle input** (added 2026-09-06 by roadmap E1). `oracle.py` always *computes*
  the filtered top-K. YFCC-10M ships its own filtered ground truth, staged by the loader as
  `gt_shipped.pt` (format in [../system/datasets.md](../system/datasets.md#yfcc10m)); nothing in
  the harness can read it, so it is validated out-of-band by
  `evaluation/eval_datasets/yfcc_check_gt.py`. WP-2 should give `oracle.py` a
  "load this blob instead of computing" path, keyed the same way the content fingerprint is, and
  with it a per-dataset metric (YFCC's GT is squared L2; every other dataset is inner product on
  L2-normalised embeddings, which is what the loader hard-codes today).

## 8. External practice review — what the field does, and what we take from it

Two surveys of how benchmark harnesses are actually built were run against
this plan (raw reports and URLs:
[evaluation-harness-v2-artifacts/](evaluation-harness-v2-artifacts/README.md)).
One covers retrieval-specific harnesses — ann-benchmarks,
big-ann-benchmarks incl. the NeurIPS'23 filtered YFCC track, VectorDBBench,
MTEB / BEIR / `ir_measures`, cuVS `raft-ann-bench`, FAISS `benchs/`. The
other covers general benchmarking infrastructure — Criterion.rs, Google
Benchmark, nanobench, pytest-benchmark, airspeed velocity, MLPerf
Inference, `torch.utils.benchmark`, Hydra / W&B Sweeps / MLflow / DVC, and
the results-as-data practice of ClickBench, db-benchmark, Conbench and
Codespeed. Neither ran anything; both are literature reviews and each ends
with an explicit unverified list.

### 8.1 Confirmed, no change

Five of the ten conventions the ANN survey found recurring across ≥3
frameworks are already in this plan, arrived at independently, and should
stop being treated as debatable:

- **one appended record per finest unit, whose existence *is* the resume
  mechanism** (§3.2) — ann-benchmarks/big-ann one HDF5 per (dataset, algo,
  args), MTEB one JSON per (model, revision, task), VectorDBBench one row
  per (db, case);
- **provenance inline on the record, not in a side log** (§3.2 `env`) —
  ann-benchmarks HDF5 `attrs`, MTEB's `mteb_version`/`dataset_revision`,
  pytest-benchmark's `machine_info` + `commit_info`, Conbench's machine
  block;
- **sweeps declared as data and Cartesian-producted** (§3.3) —
  ann-benchmarks `run_groups.args`, cuVS `groups.build`/`.search`;
- **closed-loop by default with a separately *named* second mode** (§2.5's
  `eager`/`graph`) — ann-benchmarks per-query vs `--batch`, VectorDBBench
  `serial_runner` vs `concurrent_runner`;
- **cutoff-qualified metric keys** (`recall@100`, not `recall_at_100`) —
  the `ir_measures` grammar. Keep enforcing it for every key `report.py`
  emits.

Three things this plan does *not* do are also confirmed as correct rather
than as shortcuts. **No Docker-per-algorithm**: ann-benchmarks and big-ann
need containers because they aggregate dozens of third-party libraries
with conflicting native dependencies; we have five algorithms behind one
`uv.lock`, and §3.4's subprocess boundary buys the state isolation without
the dependency problem we don't have. **No separate results repository**:
MTEB split its results out to keep thousands of external contributions off
the benchmark's review path; a single-team repo gains only friction.
**No power/$-per-query axis**: big-ann's T3 track needs IPMI sensor access
we do not have on a rented A100, and scoring on power invites exactly the
measurement-integrity disputes a reproducibility paper should not host —
§2.1's recorded clocks and >5 % drift warning are the right-sized version.

### 8.2 Accepted — deltas to the sections above

**A. Build-time and query-time parameters sweep separately (amends §2.8,
§3.3, §3.1 `run.py`).** Every ANN harness surveyed builds the index once
per *build* config and then re-queries it for every *query* config
(ann-benchmarks `args` vs `query_args` with `set_query_arguments`; cuVS
`groups.build` vs `groups.search`; big-ann caps a submission at "1 build +
up to 10 search configs"). We rebuild per param combo. `n_probe` is not a
build parameter: `SilverTorch.__init__` stores it
([main.py:118–119](../../retrieve/src/retrieve/modules/silvertorch.py))
and `register_index` reads it only for two validation checks
(main.py:168–169, 187–192); the k-means at main.py:176 does not depend on
it. So the `deep` suite's `{n_lists: [1664, 8192], n_probe: [4, 8, 24, 32,
128, 256]}` is **2 builds, not 12** — one k-means over 3 M × 128 per
`n_lists`, then six query configs against it, the same mutate-then-measure
pattern §2.5 already uses for `module.k`. The same holds for `linr_v3`'s
`candidate_pool`. Config change: `params:` splits into `build:` and
`query:`; `run.py` grows one loop level (build → for each query config:
quality + perf); the two `register_index` checks move into a
`set_query_params()` that re-validates on mutation. This is the largest
single cost saving in the campaign and it makes the `deep` sweep's build
column meaningful (one build time per built index, not twelve copies).
`[medium]`

**B. The resume key must include a code version (amends §3.1 `cli.py`,
§3.2).** "Skip keys already present in the JSONL" silently treats a cell
as done when the kernel it measured has since changed. asv versions the
*benchmark definition* by the hash of its own source, separately from the
numeric result, precisely so a redefinition is a discontinuity rather than
a silent comparison across semantics. Record `code_version` =
`git rev-parse HEAD:retrieve/src/retrieve` (the library subtree's tree
hash, not the repo commit — docs and plans churn constantly and must not
invalidate a campaign) and make it part of the resume key. `[cheap]`

**C. `schema_version: 1` on every record (amends §3.2).** asv (`version:
2`), Google Benchmark (`json_schema_version`), pytest-benchmark. Costs
nothing on day one, cannot be retrofitted once records exist, and §5 of
this plan is already an explicit schema break — the next one should be
readable rather than guessed. `[cheap]`

**D. `status: ok | failed | partial` on every record (amends §3.2, §3.1
`run.py`).** §3.1 says "no skip that is not logged with the reason and
counted"; putting the reason on the record instead of only in the log
makes `report.py` able to state coverage without parsing logs. `[cheap]`

**E. Robust statistics, not just the mean (amends §2.5).** Criterion,
nanobench, pytest-benchmark and Google Benchmark all warn against the mean
alone on heavy-tailed runtimes, and all keep raw samples. §2.5 already
stores `median/mean/p95/p99/min` and the per-call vector; add **IQR** and
both outlier counts (stddev-based and Tukey-fence) computed at write time
from the vector we already have, and never drop an outlier silently.
`[cheap]`

**F. Provenance additions (amends §2.1, §3.2 `env`).** Add `git_branch`,
`python`, and — the field naive harnesses forget — `clocks_locked` as a
*recorded value*, not an operational habit. Google Benchmark records
`cpu_scaling_enabled` for exactly this reason: without it, a future reader
cannot tell which historical rows were taken at locked clocks.
`CLAUDE.md`'s rule 1 already mandates `nvidia-smi -lgc 1410`; recording it
lets `bench report` refuse to cite a run instead of trusting memory. Same
argument for `dirty` (already planned) — keep it, and have `report.py`
*enforce* it. `[cheap]`

**G. `report.py` emits one flat table before any plot (amends §3.1,
§6 WP-6).** Every harness surveyed puts exactly one denormalisation step
between raw storage and any chart: `data_export.py` → CSV in
ann-benchmarks, big-ann and cuVS; VectorDBBench's `leaderboard.json` *is*
that step. Make `bench report` write `results/flat.csv` (or parquet)
first, unconditionally, and build every table and figure from it. It is
also the artifact to ship with the paper. `[cheap]`

**H. QPS-vs-recall Pareto and a recall-at-budget table as `report.py`'s
default view (amends §6 WP-6).** The one figure the whole ANN field
converges on, plus big-ann's `show_operating_points.py`-style "best recall
at or above a QPS / under a latency budget" table. Every field it needs
(`qps`, `recall@k`, `p99_ms`) is already in the §3.2 record. `[cheap]`

**I. The oracle is a shippable artifact, not just a cache (amends §3.1
`oracle.py`, and F4 in the roadmap).** ann-benchmarks embeds ground truth
in the dataset HDF5, big-ann ships `yfcc100m_query_gt100.bin`, MTEB pins
`dataset_revision` — ground truth is versioned and distributed, never
recomputed per run. E6's content fingerprint already makes our cache
correct; make the blob *portable* (a stable content hash over item
embeddings, query attrs, filter definition and `k_max`, with the hash in
the filename) and publish the oracles alongside the datasets in the
roadmap's F4. This is what makes an external reader able to check our
recall numbers without an A100. `[medium]`

**J. `disabled: true` on a sweep entry (amends §3.3).** ann-benchmarks and
cuVS both retire a config this way rather than by commenting out or
deleting YAML, so the definition and its git blame survive. `[cheap]`

**K. One process per `(dataset, dim, algo, backend)`, not per `(dataset, dim, algo)`
(amends §2.4, §2.8, §3.4).** Every isolation-providing harness surveyed puts a hard process or
container wall around exactly the state that is cheapest to leak — compiled kernels, allocator
arenas, CUDA graph pools — and §1.10 already documents that leak here (`torch._dynamo.reset()`
runs between sweeps only, so graph pools accumulate across the cells of a sweep). ann-benchmarks'
rule is that one process is one thing under test; our thing under test is a *backend*, so the
backend gets the process. The thesis methodology (main.tex:662–676) asked for this too.
**Decision (user, 2026-09-06): take it — in-process cross-backend parity is not worth keeping the
boundary coarse for.** Consequences: parity moves to a spill file (§2.4), the campaign loop keys
on backend (§3.4), and ~200–250 process boundaries each pay a dataset reload and CUDA context
init (§2.8). Correctness itself does not depend on this: the bit-exactness gate is the library's
parity suite (**O** §5 T1–T7 / roadmap B2), not the harness. `[medium]`

### 8.3 Rejected, with the reason

- **Open-loop load generation** (MLPerf's Server scenario). The
  infrastructure survey recommends it; the ANN survey found that *no*
  retrieval harness does it for queries — ann-benchmarks, big-ann, cuVS
  and both of VectorDBBench's query runners are closed-loop, and only
  VectorDBBench's *insert* path is open-loop. Neither paper we reproduce
  reports open-loop query latency either. Verdict: stay closed-loop, and
  make that explicit rather than implicit — record `load: "closed_loop"`
  in the perf entry and say so in §2.7's comparability paragraph. Building
  an arrival-process generator is `[expensive]` for a number nobody we are
  comparing against reports.
- **Hydra, W&B Sweeps, MLflow, Sacred, DVC pipelines.** Each needs a
  server, an agent, or a config object model this plan explicitly rejects
  in §4; W&B's grid resume has a multi-year history of duplicate and
  missing runs. ClickBench and DuckDB's db-benchmark run at comparable or
  larger scale on files-in-the-repo plus a flat CLI, which is what §3
  already specifies.

### 8.4 Effect on the work packages

- **WP-1** additionally: IQR + outlier counts in `bench.latency`;
  `code_version`, `git_branch`, `python`, `clocks_locked` in
  `provenance()`; `load: "closed_loop"` in the perf dict.
- **WP-2** additionally: `build:` / `query:` param split in
  `config.load_matrix` and its test (job count assertions change);
  `disabled: true` honoured and logged; the oracle blob's content hash in
  its filename.
- **WP-3** additionally: `schema_version`, `status`, resume keyed on
  `code_version` too; `run.py`'s extra query-config loop and
  `set_query_params()`; the campaign loop keyed on
  `(dataset, dim, algo, backend)` and the `results/_parity/` spill file
  with its group-close cleanup (§8.2 K); `torch.cuda.memory_reserved()`
  recorded per cell, which is now a leak *detector* rather than a
  mitigation.
- **WP-4** additionally, three gates: reserved GPU memory flat across a
  sweep's cells (±5 %); quality identical for the same `(n_lists,
  n_probe)` whether reached by rebuild or by `set_query_params`; and
  `jaccard_vs_first@100 == 1.0` still reproduced across the process
  boundary via the spill file (this is C4's gate (4) and **O** WP-6's
  gate — the boundary change must not silently drop them).
- **WP-6** additionally: `results/flat.csv` written first; the QPS-vs-recall
  Pareto and the recall-at-budget table as the default view.

One library change follows from A: `SilverTorch.set_query_params(n_probe=…)`
re-running the two `register_index` validations — 10 lines, alongside the
`build_timings` nit in §7.

## 9. Validation record — C1–C3 (2026-09-06, Mac / CPU only, `dev/c1-harness-v2`)

Roadmap C1, C2 and C3 (WP-1, WP-2, WP-3 above, with the §8.4 additions) are authored and
CPU-tested; the GPU gate is C4 (WP-4). Nothing here is citable (CLAUDE.md rule 2).

**What ran.** `cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest retrieval/tests/ -q`:
82 passed (8 files) on torch CPU; `ruff check retrieval training/evaluate.py` clean;
`python3 scripts/check_doc_links.py` at zero. The end-to-end tests (`test_run.py`,
`test_cli.py`) drive `run.run` and `bench run` / `bench campaign` on a 24-item, 8-query
pre-encoded fixture with `backend="torch"`, `mode="eager"`, shrunken latency windows: record
schema (§3.2 + §8.2 B–F), resume by key with `code_version`, a failed cell recorded and the
loop continuing, the §2.4 exact-algo gate, the parity spill file across two "backends", one
real child process per group with per-child logs and `campaign.log`.

**What is in the tree** (`evaluation/retrieval/`, `wc -l` incl. docstrings): `bench.py` 349,
`metrics.py` 111, `algos.py` 328, `config.py` 365, `data.py` 291, `oracle.py` 264, `run.py`
582, `cli.py` 190, `upload.py` 60, `encode.py` 117 = 2,657 code; tests 1,699 (8 files +
`conftest.py`). Budget in §3.1 was 1,450 + 350; the overrun is docstrings (≈ ⅓), the
record assembly in `run.py`, `data.py`'s two loaders, and tests that lock the old harness's
numbers (metrics to 1e-9, the d128 cell sets, the dispatch table). Deleted per §5: 30 old
files / 4,339 lines (commit `f021179`). The as-built reference is
[../system/evaluation.md](../system/evaluation.md).

**Deviations from this plan** (also listed in the system doc): parity spill is `.npz` with
ids + scores under a hash of the key block minus `backend`, cleaned by `bench campaign` per
`(dataset, dim, algo)`; samples are a JSONL sidecar rather than a parquet (append per cell);
resume re-runs `failed` / `partial` records; the campaign child is `python -m retrieval.cli
run` on the same interpreter; `EXACT_ALGOS = (linr_v1_filter_mask, linr_v2)`; `--flush-l2`
not implemented; `training/evaluate.py` moved to the new `metrics` API (5 lines) so the old
per-row wrappers could go; `bench campaign` forwards `--skip-quality / --skip-perf /
--profile`. `upload-results` is the 60-line mirror of §3.1. `report.py` is D4.

**Unverified until C4 (A100):** every Triton / `graph` / CUDA-event / `nvidia-smi` /
SASRec-encode path; the `official` cells (roadmap B1 must land first — until then they are
`status: failed` with `NotImplementedError`); quality vs A1's golden to 1e-6; graph latency
within 5 % of the old numbers; `cudagraph_skips == 0`; `jaccard_vs_first@100 == 1.0` torch
vs triton across the process boundary; reserved memory flat across a group; kill-and-resume
mid-run; the per-cell wall time (§2.8's 2 min estimate). The C4 command:
`uv run bench run --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre`.

### C4 library prerequisites — validation record, 2026-09-06, A100 (`dev/c4-library-fixes`)

The three library changes the C4 gate needs before it can be re-run, from the coordinator's
three C4 findings. Branch `dev/c4-library-fixes` = `development` + B4's deletion +
`dev/c4-harness-gate`. Environment: A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, nvcc
12.8, Python 3.11, venv `/venvs/c4` built with `--extra official`. Clocks unlocked (H §7
fallback applies to anything timed here). **No evaluation was run** — tests and small probes
only, so nothing here is citable (CLAUDE.md rule 2).

**1. Compaction kernels as opaque custom ops** (`dd8b7b5`, verifying the drafted `5815275`).
`clause_compact` / `bloom_compact` are `@torch.library.custom_op` (`mutates_args=()`,
`device_types="cuda"`, `register_fake`) instead of `@triton_op`: under `triton_op`, inductor's
TTIR mutation analysis walks a store address back through every argument of the `tt.call` that
produced it, and `compact_store`'s address is `base + intra`, data-dependent on the pass mask,
so the index buffers (graph inputs) are reported as mutated and cudagraph trees skip the whole
forward. New gate
[`retrieve/tests/compile/test_linr_compile.py`](../../retrieve/tests/compile/test_linr_compile.py)
mirrors `algos.py`'s LiNR builds and `bench.py:graph_callable`'s compile settings
(`mode="reduce-overhead"`, `dynamic=False`, `fullgraph=True`, 5 warm-ups) and reads the same
`counters["inductor"]["cudagraph_skips"]`.

| cell | skips, on `custom_op` | skips, on `triton_op` (negative control) |
|---|---|---|
| `linr_v1` × clause / bloom | 0 / 0 | 0 / 0 |
| `linr_v2` × clause / bloom | 0 / 0 | **1** / 0 |
| `linr_v3` × clause / bloom | 0 / 0 | **1** / 0 |

Scores `torch.equal` to eager on all six, ids equal up to ties. Only `clause_compact` actually
tripped the analysis at these shapes; `bloom_compact` is converted anyway — same
data-dependent store, and the failure mode is silent. `torch.export` still works on the
custom-op form (the graph holds `torch.ops.retrieve.clause_compact.default` and round-trips
bit-exact); what is lost is the `triton_kernel_wrapper` HOP.
`tests/compile/test_export_kernel_ref.py` gates `codesigned_probe_score_exact`, still a
`triton_op`, and passes. **Tests:** `parity/test_clause_compact.py` +
`test_bloom_compact.py` + `correctness/test_compact.py` + `test_filters.py` + `tests/compile/`
= **67 passed**.

**2. Deterministic k-means** (`d5d824b`). `KMeansTorch._cluster_sums` replaces `index_add_`'s
float atomics with `bincount` (counts) + a float64 one-hot GEMM `onehot[n_lists, W] @
embs[W, D]` accumulated panel by panel with `addmm_`. No floating-point atomics and no
scheduling-dependent order: a GEMM's reduction order is a function of its shapes alone.
float64 costs 8 % over float32 here (DMMA runs at the float32 SIMT rate) and cannot be demoted
to TF32 by a caller's `set_float32_matmul_precision("high")`, so **nothing global is toggled**.
Initialisation, iteration count, chunked assignment and the float32 result dtype are unchanged.

Measured at N=200k, D=128, n_lists=1024, n_iter=10 (`torch.cuda.synchronize`, sampled clock
not recorded — this is a ratio, not a citable latency):

| | wall time | two seed-0 fits |
|---|---|---|
| old (`index_add_`) | 142.3 ms | not equal in general |
| new (order-fixed) | 173.7 ms — **1.22×** | `torch.equal` on centroids *and* assignments |

Centroid delta vs the atomic implementation: **one update from a fixed assignment** (the
chaos-free measure) is max abs **3.3e-6** on sums of order 1e2, and that residual is the
*atomic* side's float32 accumulation error. A **full fit** is not compared elementwise: Lloyd
is chaotic and the two were seen both 6.0e-8 and 2.1e-2 apart on different runs, purely by
which side flipped an assignment first — the atomic reference is not stable against itself.
What is asserted instead is the objective: mean squared distance to the assigned centroid
differs by **5.4e-9 relative**. **Tests:** `correctness/test_kmeans.py` = **5 passed in 10.0 s**;
`parity/test_official.py` = 43 passed.

**3. `-1` id sentinel in the Triton SilverTorch epilogue** (`8df7e9a`, plan O §14.7).
`_cps_finish` / `_cpse_finish` apply `torch.where(torch.isfinite(topk_scores), topk_ids, -1)` —
one capture-safe elementwise op, no host sync — so the Triton backend meets `interfaces.py`'s
`-1 / -inf` contract the way `masked_topk` does for `torch` and `official`. New
`TestFewSurvivorsSentinel` in `correctness/test_silvertorch.py` builds an index with a known
survivor count (5 items carry the queried value; one query row asks for a value no item
carries) and asserts, for exact and bloom mode, that every `-inf` slot has id `-1` and that
triton vs torch ids compare with no finiteness normalisation first (scores `torch.equal`, ids
equal up to ties). All 4 fail without the change. `test_official.py`'s T6 `-inf`-slot
normalisation is left in place and still passes — it is now a no-op. **Tests:**
`parity/test_codesigned_probe_score.py` + `test_codesigned_probe_score_exact.py` +
`correctness/test_silvertorch.py` + `tests/compile/` + `parity/test_official.py` =
**167 passed**.

**Suites.** Full library suite on the A100: `516 passed, 3 failed` in 100 s. The three failures
are **pre-existing on `dev/b4-delete-cuda-cute`**, not from this work (confirmed by stashing
every change on this branch): `test_linr.py::test_unknown_backend_is_rejected[*-simhash]` calls
`SimHashKNN(k=K, backend=backend)` and dies with `TypeError: missing 1 required positional
argument: 'k_bits'` before the expected `ValueError` — the lambda in B4's `010681d` omits
`k_bits`. One-line test fix, left to B4's owner. Harness CPU suite: `103 passed, 1 skipped`.
`ruff check retrieve` clean and `ruff format --check` clean on all Python; the 11 `E501`s
`ruff check evaluation` reports are pre-existing in `evaluation/eval_datasets/`.
`check_doc_links.py` at 0.

**Two notes for the coordinator.**

1. **A1's golden must be re-derived before the 1e-6 gate can be evaluated.** Every SilverTorch
   cell in A1's golden was produced with the atomic k-means, so its centroids — and everything
   downstream of them: cluster membership, the int8 scale, every quality column — are one
   arbitrary draw from a distribution the old code could not reproduce. Comparing a
   deterministic re-run against them at 1e-6 is not a meaningful test. The LiNR cells are also
   affected: A1's `graph` numbers for `linr_v2` / `linr_v3` on triton were compiled-eager
   (finding (ii)), so their latencies are not graph latencies. Both re-derivations want the
   same GPU lane.
2. **The sentinel can change SilverTorch quality numbers**, on the triton backend only.
   `evaluation/retrieval/metrics.py` is unchanged and its `_hits` masks on `ids != -1` only,
   never on score finiteness — so before this fix a filtered-out item's id sitting in a `-inf`
   slot could be counted as a hit, and counted into `jaccard_vs_first@k`. Rows with fewer than
   `k` survivors can therefore move; the movement is **downward only** (spurious hits removed),
   and rows with ≥ `k` survivors have no `-inf` slot in the top-`k` and are unaffected. `torch`
   and `official` numbers do not change. The harness was deliberately not touched.

**Operational finding.** Inductor's on-disk FX-graph cache (`/tmp/torchinductor_root`) does not
invalidate when a `@triton_op` body's *Python* source changes: after editing `_cps_finish` the
compiled cells silently kept running the pre-edit epilogue, and only a cache clear (or
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`) exposed the change. Anyone editing a kernel host
wrapper and re-running compiled cells must clear it, or the numbers are from the old code.

## 10. Validation record — WP-0 / roadmap A1, 2026-09-06, A100-SXM4-80GB

> Model: [cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360).

**Environment.** A100-SXM4-80GB, driver 570.195.03, CUDA 12.8, torch
2.10.0+cu128, triton 3.6.0, Python 3.11. Branch `dev/a1-golden`; every row
carries `extra.commit = 70bafc4`, the commit the cells were produced at.
Runbook: [evaluation-harness-v2-artifacts/a1_golden_run.sh](evaluation-harness-v2-artifacts/a1_golden_run.sh),
stages `golden step4 step7`.

**Clocks were NOT locked — this container cannot.** `nvidia-smi -lgc 1410`
returns *"The current user does not have permission to change clocks"* and
there is no `sudo` binary, so §7's fallback applies: the runbook samples
`clocks.sm` every 30 s into `evaluation/golden/_logs/clocks.csv` instead.
Under load the SM clock sat at **1140 MHz** (the application-clock default,
not the 1410 MHz boost the plan asks for), 210 MHz idle, 28-31 °C, no
thermal excursion. **The quality columns are unaffected; the latency
columns are not clock-controlled and must not be quoted as if they were.**
C4 compares quality within 1e-6 (H WP-4 gate 1) — that gate stands; its
gate 2, `graph` median within 5 % of golden, has to be read against this,
and the cleanest fix is for C4 to run on a box where clocks can be pinned.

### 9.1 Golden cells — 11/11 green

99 rows, 9 per cell (`ks = [100, 500, 1000]` × `batch_sizes = [1, 8, 16]`),
every `(k, bs)` present and every quality column populated. Wall time 35.9
min total, 2.5-4.6 min per cell (each is a fresh process; the first pays the
oracle build, 62 s for goodreads `c0_genre`, then it is cached by
fingerprint). `users_limit: 10000`; goodreads keeps 9,859 of 10,000 users on
`c0_genre` (the rest have no surviving clause), arxiv keeps 10,000/10,000.

| cell | k | recall@k | ndcg@k | median_ms bs=1 | bs=8 | bs=16 |
|---|---|---|---|---|---|---|
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 0.884007 | 0.913704 | 0.1868 | 0.3097 | 0.4701 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 0.847281 | 0.875401 | 0.2012 | 0.3197 | 0.4790 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 0.819665 | 0.848330 | 0.2020 | 0.3198 | 0.4801 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 100 | 0.999695 | 0.999781 | 0.4340 | 0.9449 | 1.6256 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 500 | 0.999657 | 0.999729 | 0.4506 | 0.9574 | 1.6389 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 1000 | 0.999627 | 0.999696 | 0.4500 | 0.9576 | 1.6382 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 100 | 0.999695 | 0.999781 | 0.4600 | 1.0907 | 1.9004 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 500 | 0.999657 | 0.999729 | 0.4756 | 1.1021 | 1.9098 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | 0.999627 | 0.999696 | 0.5475 | 1.1028 | 1.9136 |
| goodreads-d128-c0_genre-linr_v2-torch | 100 | 0.999695 | 0.999781 | 0.9320 | 4.6478 | 9.1234 |
| goodreads-d128-c0_genre-linr_v2-torch | 500 | 0.999657 | 0.999729 | 0.9088 | 4.6556 | 9.1410 |
| goodreads-d128-c0_genre-linr_v2-torch | 1000 | 0.999627 | 0.999696 | 0.8409 | 4.6580 | 9.1373 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 0.999279 | 0.999483 | 0.3679 | 1.6625 | 3.2321 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 0.999372 | 0.999503 | 0.4577 | 1.6842 | 3.2532 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 0.999388 | 0.999501 | 0.4602 | 1.6953 | 3.2561 |
| goodreads-d128-c0_genre-linr_v3-torch | 100 | 0.876957 | 0.909106 | 0.4742 | 1.8755 | 3.4024 |
| goodreads-d128-c0_genre-linr_v3-torch | 500 | 0.724322 | 0.774934 | 0.5834 | 1.8952 | 3.4184 |
| goodreads-d128-c0_genre-linr_v3-torch | 1000 | 0.617541 | 0.676164 | 0.5791 | 1.8919 | 3.4193 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 0.876978 | 0.909125 | 0.4347 | 1.3807 | 2.4714 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 0.724367 | 0.774973 | 0.5496 | 1.4048 | 2.4867 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 0.617544 | 0.676167 | 0.4720 | 1.4036 | 2.4864 |
| goodreads-d128-c0_genre-linr_v4-torch | 100 | 0.982054 | 0.987055 | 0.5993 | 1.1424 | 1.8444 |
| goodreads-d128-c0_genre-linr_v4-torch | 500 | 0.986094 | 0.988973 | 0.6142 | 1.1539 | 1.8551 |
| goodreads-d128-c0_genre-linr_v4-torch | 1000 | 0.987316 | 0.989631 | 0.7399 | 1.1551 | 1.8556 |
| goodreads-d128-c0_genre-linr_v4-triton | 100 | 0.982054 | 0.987055 | 0.6273 | 1.3012 | 2.5555 |
| goodreads-d128-c0_genre-linr_v4-triton | 500 | 0.986094 | 0.988973 | 0.6470 | 1.3123 | 2.1000 |
| goodreads-d128-c0_genre-linr_v4-triton | 1000 | 0.987316 | 0.989631 | 0.6462 | 1.3135 | 2.1002 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 0.912643 | 0.935841 | 0.6128 | 3.4215 | 6.8018 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 0.847044 | 0.876175 | 0.6074 | 3.4368 | 6.8182 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 0.788911 | 0.823259 | 0.6025 | 3.4635 | 6.8771 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 0.912759 | 0.935927 | 0.2107 | 0.4509 | 0.8570 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 0.847199 | 0.876303 | 0.2721 | 0.4686 | 0.8727 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 0.789083 | 0.823403 | 0.2777 | 0.4767 | 0.8859 |

Reading these: `linr_v2` is the exact filtered top-K baseline, so its ~0.9997
is the tie-order ceiling rather than a recall loss; `torch` and `triton`
agree to ~1e-4 on every algo (tie order, not arithmetic); `silvertorch`
triton is 3.2-7.9× faster than its torch fallback at bs=16 while matching it
to 1.2e-4 on recall.

### 9.2 Handoff steps

| step | result |
|---|---|
| 1 — eval CPU tests | **pass**, 33 passed (`retrieval/tests/`, minus the GPU-only reverse-clause suite) |
| 4 — compile gate, `TORCH_LOGS=graph_breaks` on `linr_v3 --skip-quality` | **pass**: exit 0 and **zero** graph breaks on the branch. The step's criterion is "no *new* breaks vs `main`"; with none at all the comparison is vacuous, and the `main`-side leg did not run (that worktree's `.venv` was removed when step 6 was deferred, and it needs both fixes below first) |
| 7 — orchestrator smoke | **pass**: killed as soon as the first algo completed (250 s in), resumed, `exit=0`, 1 × `resume: skipping`, 5/5 algo JSONs, and `SUMMARY.txt` / `full.log` / `goodreads__a1_step7-d128-filter.log` in the run-log dir. Output went to a throwaway dir via a copied config, so `evaluation/results/` was untouched |
| 5 — per-kernel ±5 % gates | **deferred 2026-09-06** (user: heavy evals later). Scripted: stage `step5` + [a1_step5_compare.py](evaluation-harness-v2-artifacts/a1_step5_compare.py). Estimate 332 sweep points per side, ~44 min both sides at 4 s/point |
| 6 — golden diff vs `main` | **deferred 2026-09-06** (same). Scripted: stage `step6` + [a1_step6_diff.py](evaluation-harness-v2-artifacts/a1_step6_diff.py); the throwaway `tmp/main-users-limit-fix` worktree carries main's `users_limit` port and still needs the two fixes below |

### 9.3 Three bugs, none of them in the plan

WP-0 was budgeted as "commit a 3-line fix, run 11 cells". It took three
attempts, and each failure was a real defect that only a golden run could
surface — which is the argument for A1 existing at all.

1. **`users_limit` row count** (`fix(A1): users_limit row-count in
   load_query_attrs`, `0129e25`) — the known blocker of §1 verdict 7.
2. **The fetched datasets are the pre-`3b1b5b3` 1-indexed `[N+1, …]`
   artifacts** (`fix(A1): accept the legacy 1-indexed dataset layout`,
   `df6db40`). The loaders, the ETL and `docs/system/datasets.md` expect
   `[N, …]` 0-indexed dense; the copies published on the Hub never moved.
   goodreads crashed in the oracle (797,085 vs 797,084 rows). **arxiv did
   not crash** — attrs and embeddings were both 1-indexed, agreeing with
   each other, so only the held-out target shift was wrong:
   `cos(query, text_emb[target_id])` 0.9891 versus
   `cos(query, text_emb[target_id - 1])` 0.6248, with no error anywhere. An
   arxiv golden cell would have been wrong and plausible. Not a refactor
   regression: `main` and `development` have byte-identical
   `load_filter_assets`. Diagnostic:
   [a1_check_item_alignment.py](evaluation-harness-v2-artifacts/a1_check_item_alignment.py).
3. **K3's shared `@triton.jit` helpers do not survive inductor**
   (`fix(A1): import the shared triton helpers by name, not via the
   module`, `70bafc4`). `common.clause_pass(...)` resolves fine in eager
   Triton — hence 47/47 parity tests and the full library suite green on
   2026-09-02 — but `torch.compile` rebuilds the kernel's globals and
   captures `@triton.jit` callees *by name*, so every compiled filter algo
   died with `NameError('common is not defined')` and **not one golden cell
   could be produced**. This is precisely what handoff step 4 exists to
   catch, and step 4 had never run. Fixed by importing the helpers by name;
   the compiled mask is `torch.equal` to the eager one and
   `pytest -k "clause or bloom or compact"` is 165 passed / 33 skipped.

One operational note, not a code defect: killing a cell mid-`torch.save`
left a truncated 20 MiB `gt_topk_v3_c0_maincat.pt`, and the next run failed
with `PytorchStreamReader failed reading zip archive`. Deleted and rebuilt.
`load_or_build_oracle` should treat an unreadable cache the way it treats a
stale fingerprint — recompute, not raise; folded into WP-2's oracle work.

## 11. Validation record — A1 golden re-derive, 2026-09-15, A100-SXM4-80GB

> Model: [cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360).
> Roadmap eval-queue item 1. Extends §10 (A1's own record); it does not
> replace it. Worker job, run under
> [agent-orchestration.md](agent-orchestration.md): no roadmap checkbox was
> flipped and nothing was merged into `development`.

**Why.** §10's cells were produced against the library *before* the three C4
library fixes (`7a21095`): the non-deterministic atomic k-means, the
compiled-eager LiNR V2/V3 `graph` cells, and no O §14.7 `-1` id sentinel. C4's
gate compares the v2 harness against those cells at `1e-6`, so the comparison
measured the *library* change and not the harness rewrite. The fix is to run
the **old harness unchanged against the current library** and keep everything
else — data, queries, oracle, box — fixed, so every delta is a library delta.

**Environment.** A100-SXM4-80GB, driver 580.159.04, nvcc **12.4** (the only
toolkit installed; correct per
[silvertorch-official-integration.md](silvertorch-official-integration.md)
§13.1), torch 2.10.0+cu128, triton 3.6.0, Python 3.11, `silvertorch` 1.0.0 at
`21aa35e`. Fresh rental: no worktree and no dataset survived the 2026-09-06
session.

- **Worktree** `/workspace/wt/golden`, branch `tmp/golden-rederive` off
  `origin/dev/a1-golden` (`1ccdb27`) with `git checkout 4f52972 -- retrieve/`.
  Library tree `28fda5ae` (= `4f52972:retrieve`); A1's was `6dc72aac`
  (= `70bafc4:retrieve`). Rows carry `extra.commit = 87a9b38`. **Throwaway,
  never merged** — kept alive for the L1 follow-up cell.
- **Venv** `/venvs/golden`, `uv sync --extra official --all-packages`.
- **Datasets** staged to `/workspace/data` (`RETRIEVE_DATA_ROOT`), symlinked as
  `evaluation/data`: goodreads-work-id 2.6 GB (d128 only) + arxiv-papers 1.3 GB
  (`content_d128` only). Confirmed the **pre-`3b1b5b3` 1-indexed `[N+1, …]`
  layout** the Hub still publishes — goodreads `item_attrs_narrow` `[797085,
  4, 4]` against 797,084 ids, arxiv `[2988997, …]` against 2,988,996 — which
  the old harness detects and drops (`df6db40`).
- **GPU shared** with the L1 worker; every cell took `flock /workspace/gpu.lock`
  for its own duration (CLAUDE.md rule 1).
- **Clocks still cannot be locked.** Sampled every 30 s into
  `evaluation/golden/_logs/clocks.csv`: **1155 MHz** median under load (A1:
  1140), 1410 MHz peak, 210 MHz idle, 26-39 °C. Quality unaffected; **latency
  is not clock-controlled** and the interleaved second worker makes it worse
  than A1's. `c4_gate.py --golden-sm-mhz` must be passed **1155**.

**Runbook.** [a1-rederive/a1_golden_rerun.sh](evaluation-harness-v2-artifacts/a1-rederive/a1_golden_rerun.sh),
`STAGES=golden`; a copy of [a1_golden_run.sh](evaluation-harness-v2-artifacts/a1_golden_run.sh)
differing only in: `STAGES` defaults to `golden`; every GPU `uv run` is wrapped
in `flock`; `uv run --no-sync` against `/venvs/golden`; `NO_CLOCK_LOCK=1`;
a private `TORCHINDUCTOR_CACHE_DIR` (the default `/tmp/torchinductor_root` is
shared with the other worker, and §10's C4 note is that it does not invalidate
on a `@triton_op` wrapper source change); `REPO_ROOT` one directory deeper.
Stages `step4` / `step5` / `step6` / `step7` are **out of scope and still
deferred**.

### 11.1 Inputs are provably A1's

Each cell's log (`evaluation/golden/_logs/branch-*.log`) shows
`loaded encoded_queries from cache` and `loaded oracle from cache`, so the
queries, the item embeddings and the ground truth are byte-identical to A1's.
The `encoded_queries_test.pt` cache is keyed on the checkpoint's mtime, which a
fresh `snapshot_download` changes; the checkpoint's mtime was set back to the
blob's recorded `ckpt_mtime` so no re-encode could perturb the queries. `kept
users: 9859 / 10000` (goodreads) and `10000 / 10000` (arxiv) match §10.

### 11.2 Cells — 11/11 green

99 rows, 9 per cell, every `(k, bs)` present and every quality column
populated. 29.4 min of cell time (2.4-3.1 min each), 33 min end to end.

| cell | quality vs A1 | max abs delta | direction |
|---|---|---|---|
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | **identical** | 0 | — |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | **identical** | 0 | — |
| goodreads-d128-c0_genre-linr_v2-torch | **identical** | 0 | — |
| goodreads-d128-c0_genre-linr_v3-torch | **identical** | 0 | — |
| goodreads-d128-c0_genre-linr_v4-triton | **identical** | 0 | — |
| goodreads-d128-c0_genre-linr_v4-torch | **identical** | 0 | — |
| goodreads-d128-c0_genre-linr_v2-triton | moved | 5.07e-6 | mixed |
| goodreads-d128-c0_genre-linr_v3-triton | moved | 2.43e-5 | up |
| arxiv-d128-c0_maincat-silvertorch-triton | moved | 3.50e-5 | mixed |
| goodreads-d128-c0_genre-silvertorch-triton | moved | 7.79e-5 | mixed |
| goodreads-d128-c0_genre-silvertorch-torch | moved | 1.57e-4 | up |

Row-level table: [a1-rederive/quality-diff.md](evaluation-harness-v2-artifacts/a1-rederive/quality-diff.md).
Every moved cell moves identically at all three batch sizes, as it must —
quality does not depend on `bs`.

### 11.3 The roadmap's prediction is wrong, and here is why

The roadmap's status blockquote predicted: *SilverTorch-**triton** quality on
rows with `< k` survivors moves **down** only (`metrics.py::_hits` masks on
`ids != -1`, never on score finiteness); torch / official unchanged.* Three of
its four claims fail.

1. **The `-1` sentinel cannot move a goodreads number at all.** The cached
   oracle says so directly: of 10,000 goodreads `c0_genre` queries, exactly 141
   have **zero** survivors and the other 9,859 have the full 1,000 — and the
   141 zero-survivor rows are precisely the users the harness drops
   (`kept users: 9859 / 10000`). No kept row has `< k` survivors at any of
   `k ∈ {100, 500, 1000}`, so no algo output has a `-inf` slot inside the
   reported slice. On arxiv the sentinel *can* act, but only on 4 / 6 / 11 of
   10,000 rows at `k = 100 / 500 / 1000`.
2. **`torch` moved** — `silvertorch-torch` by up to 1.57e-4, the largest delta
   in the run. Deterministic k-means (C4 fix ii) is backend-independent: it
   changes the centroids, hence the IVF lists and the probed cells, on *every*
   backend. Nothing about the sentinel argument covers this.
3. **Moves are up as well as down.** Every SilverTorch goodreads delta is
   positive; arxiv is mixed; `linr_v3-triton` is positive.
4. **Two LiNR cells moved.** `linr_v2-triton` and `linr_v3-triton` moved while
   their `torch` counterparts are bit-identical, which at the time looked like
   the signature of C4 fix (i) (`clause_compact` / `bloom_compact` as opaque
   custom ops). It was flagged **not verified**, and §11.8 **withdraws it**:
   both cells turn out to be non-deterministic run to run, with a spread that
   covers the whole observed movement. Only the three `silvertorch` moves in
   the table above are real.

**The result that replaces the prediction.** SilverTorch's two backends now
agree *exactly* on goodreads:

| k | A1: torch vs triton | re-derived |
|---|---|---|
| 100 | recall 1.156e-4, ndcg 8.523e-5 | **0.0 / 0.0** |
| 500 | recall 1.550e-4, ndcg 1.279e-4 | **0.0 / 0.0** |
| 1000 | recall 1.717e-4, ndcg 1.444e-4 | **0.0 / 0.0** |

§10 read A1's gap as "tie order, not arithmetic". It was the non-deterministic
k-means: with a deterministic fit both backends build the same index and score
the same items. That is the cleanest available evidence that C4 fix (ii) does
what it claims, and it is *not* something the golden cells were designed to
show.

### 11.4 Consequences for C4

The `1e-6` gate now has a meaningful baseline. Five cells moved by 5e-6 to
1.6e-4, i.e. between 5× and 160× the tolerance — a C4 gate run against the old
cells would have failed on all five and the failures would have been
unattributable. Two things C4 must carry over: `--golden-sm-mhz 1155`, and the
fact that SilverTorch `torch` / `triton` parity is now exact on goodreads, so
gate 4 (`jaccard_vs_first@100`) should be *stricter* than it was, not looser.

### 11.5 Findings outside the cells

1. **The old harness does not import against the current library.**
   `from retrieve.interfaces import Backend` fails: B4 split `Backend` into
   `LinrBackend` / `SilverTorchBackend` and dropped `"cuda"` / `"cute"`. That
   is the only break — every other symbol the harness imports
   (`FilterModule`, `PostfilterKNN`, `PrefilterKNN`, `OneBitKNN`,
   `SilverTorch`, `BloomFilter`, `ExactAttributeFilter`, `PostfilterKNNInt8`)
   still resolves with a compatible signature. `Backend` is used only in type
   annotations (no `get_args`), so the throwaway worktree declares the old
   literal locally
   ([copy](evaluation-harness-v2-artifacts/a1-rederive/harness_compat_backend.py));
   no behaviour and no number changes. **This is the concrete requirement on
   Phase L's compatibility shim**: re-export `Backend` or the frozen harness
   will not import.
2. **The harness CPU suite is only green with CUDA hidden.** `cd evaluation &&
   pytest retrieval/tests/ --ignore=…` gives **3 failed, 98 passed, 3 skipped**
   on this box and **103 passed, 1 skipped** under `CUDA_VISIBLE_DEVICES=""`.
   The three — `test_bench.py::test_latency_windows_and_keys`,
   `test_run.py::test_end_to_end_records`,
   `test_run.py::test_perf_times_with_the_plan_cache_off_and_records_it` —
   assert CPU-only behaviour in their own text ("no CUDA: no first call", "no
   CUDA allocator on this box", `reason == "cuda_unavailable"`). They were
   written when the box had no GPU. Not fixed here (harness v2 code is out of
   this job's scope); the roadmap's "103 passed / 1 skipped" should say under
   which condition.
3. **The Hub still publishes the pre-`3b1b5b3` 1-indexed artifacts** — §10's
   bug 2 — and the `gt_d128/gt_topk_v3_*` oracle blobs are published with them,
   which is what kept this run to 33 minutes instead of hours.
4. **`/tmp/torchinductor_root` is shared between workers on this box.** Any
   concurrent GPU job must set its own `TORCHINDUCTOR_CACHE_DIR`, or §10's
   stale-FX-cache finding applies across workers as well as across edits.

### 11.6 What was not done

`step4`, `step5`, `step6`, `step7` (A1's other halves) — **still deferred**,
untouched. The C4 gate rerun, B3, D1 — not this job. No library code was
changed. Nothing here is citable: CLAUDE.md rule 2 — these cells are the *input*
to C4's gate, and that gate has not run.

### 11.7 L1's gate line — one golden cell against `dev/l1-library-layout`

Run 2026-09-15 on the orchestrator's instruction, after L1's own gates.
Worktree `/workspace/wt/golden` (the frozen A1 harness, untouched), library
subtree swapped to `dev/l1-library-layout` @ `df04e74` (tree `624459b6`),
committed as `52ee677` on `tmp/golden-rederive`. Same box, same
`/workspace/data`, same GPU lock, own `TORCHINDUCTOR_CACHE_DIR`. Both cached
inputs hit again, `kept users: 9859 / 10000`.

**The `torchretrieve` rename does fight the frozen workspace**, as the
orchestrator suspected. L1 renames the distribution `retrieve` →
`torchretrieve` (import name unchanged), so the frozen root `pyproject.toml`
(`dependencies = ["retrieve"]`, `retrieve[official]`,
`[tool.uv.sources] retrieve`) and the frozen `evaluation/pyproject.toml`
(`dependencies = ["retrieve"]`) no longer resolve. Fixed on the throwaway
branch by taking L1's root `pyproject.toml` wholesale and renaming the one
dependency line in `evaluation/pyproject.toml`. **Workspace metadata only** —
no harness code moved, no import changed, `uv sync --extra official` then
succeeds and the old harness imports cleanly against L1 (the
`retrieve.kernels` shim emits its deprecation warning as designed). Anyone
re-pointing a frozen worktree at L1 or later needs the same two-line rename.

**Gate: `goodreads-d128 c0_genre silvertorch triton` — PASS, bit-identical.**
All 9 rows × `recall@k` / `ndcg@k` / `precision@k` / `mrr@k` equal to §11.2's
re-derived JSON exactly, no tolerance applied, and equal again on a second
independent run. This is the most sensitive of the eleven cells — k-means, the
fused filter path and the `-1` sentinel all bear on it — so L1's "no behaviour
change" claim is confirmed against real data, not just unit fixtures.

| cell | runs on L1 | vs §11.2 golden | verdict |
|---|---|---|---|
| `goodreads-…-silvertorch-triton` | 2 | 0 of 36 quality columns differ, both runs | **PASS** |
| `goodreads-…-linr_v3-triton` | 3 | 27 of 36 differ | **not evaluable** — see §11.8 |
| `goodreads-…-linr_v2-triton` | 2 | 24 of 36 differ | **not evaluable** — see §11.8 |

The second cell was added because it was the other one that moved in §11.2. It
did not reproduce — and the reason is not L1.

### 11.8 `linr_v2-triton` and `linr_v3-triton` are non-deterministic run to run

Before reporting the `linr_v3` mismatch as a defect in L1's move, the cell was
repeated on the **identical** L1 tree. It does not reproduce itself:

| `recall@k`, bs=1 | §11.2 golden | L1 run 1 | L1 run 2 | L1 run 3 | spread over the three L1 runs |
|---|---|---|---|---|---|
| `linr_v3-triton` k=100 | 0.877002740643 | 0.877019983864 | 0.876952025615 | 0.876981440444 | **6.796e-05** |
| `linr_v3-triton` k=500 | 0.724373871613 | 0.724341413849 | 0.724360888599 | 0.724337356573 | 2.353e-05 |
| `linr_v3-triton` k=1000 | 0.617545795329 | 0.617553200089 | 0.617552388708 | 0.617545694154 | 7.506e-06 |
| `linr_v2-triton` k=100 | 0.999273760709 | 0.999275789310 | 0.999273760709 | — | **2.029e-06** |
| `linr_v2-triton` k=1000 | 0.999391020874 | 0.999389803697 | 0.999389296547 | — | 5.072e-07 |

Same commit, same data, same `seed: 0`, same cached queries and oracle. The
golden value sits *inside* the run-to-run range in both cells, and the
`|L1 − golden|` distance (3.246e-05 for `linr_v3`) is **smaller** than the
cell's own spread (6.796e-05). So:

1. **The `linr_v3` / `linr_v2` mismatches are not attributable to L1.** There
   is no evidence of a defect in the move; there is also no way to prove its
   absence on these two cells, because they cannot prove anything about
   themselves. `silvertorch-triton` — which *is* deterministic, bit-identical
   across three runs on two different library trees — carries the gate.
2. **§11.2's `linr_v2-triton` (5.07e-6) and `linr_v3-triton` (2.43e-5) moves
   are withdrawn.** Both are inside the noise floor measured here. The
   attribution to C4 fix (i) in §11.3 item 4 does not stand. The three
   `silvertorch` moves and the exact `torch`/`triton` convergence are
   unaffected — `silvertorch` is deterministic, so those remain real.
3. **C4's gate 1 cannot be met on these two cells by any harness.** H WP-4
   asks for `recall@k` / `ndcg@k` within `1e-6`; `linr_v2-triton`'s own noise
   is 2.0e-6 and `linr_v3-triton`'s is 6.8e-5, i.e. 2× and 68× the tolerance.
   No v2 harness can reproduce a single sample of a distribution that wide.
   C4 needs a decision before it runs: exclude the two cells from gate 1,
   gate them on a repeat-derived interval instead of a point, or fix the
   non-determinism first. **This is a blocker for roadmap queue item 2 and it
   is not L1's to fix.**

**Mechanism — hypothesis, not verified.** The two affected cells are exactly
the compiled *triton* LiNR paths; their `torch` counterparts were bit-identical
in §11.2 and `linr_v1` / `linr_v4` triton were too. The leading candidate is
per-process Triton autotuning: each `evaluate` process is fresh (H §8.2 K), so
an autotuner that picks a config by measured time can pick differently run to
run, and a different block/reduction shape resolves score ties in a different
order. That would explain the magnitudes — `linr_v2` is the *exact* filtered
top-K so only ties can move it (2e-6), while `linr_v3`'s 1-bit stage has
massive integer-Hamming ties in the candidate pool, so a reordering there
propagates into stage 2 (7e-5). **Not tested**; testing it means pinning the
autotuner and re-running, which is library work and out of this job's scope.

### 11.9 `retrieve.interfaces.Backend` — do not "fix" the L shim

The orchestrator ruled on §11.5 item 1 on 2026-09-15 and the ruling is
recorded here so the next reader does not undo it. `Backend` was deleted at
roadmap **B4**, when it split into `LinrBackend` / `SilverTorchBackend`; it is
not a path plan **L** moved, and L's shim is scoped to `retrieve.layers` /
`retrieve.kernels`. Re-exporting `Backend` from that shim would resurrect an
API retired a phase earlier, which coding-guidelines D2 forbids. **The correct
place for the alias is where it now lives** — declared locally in the frozen
A1 harness on the throwaway `tmp/golden-rederive` branch
([copy](evaluation-harness-v2-artifacts/a1-rederive/harness_compat_backend.py)).
No change to L1 is needed. §11.5 item 1's closing sentence ("this is the
concrete requirement on Phase L's compatibility shim") is superseded by this
paragraph.

---

## 12. Validation record — C4 / WP-4 GPU gate, 2026-09-15, A100-SXM4-80GB

> Model: [§11](#11-validation-record--a1-golden-re-derive-2026-09-15-a100-sxm4-80gb) above and
> [cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360).
> Roadmap eval-queue item 2. Worker job under
> [agent-orchestration.md](agent-orchestration.md): **no roadmap checkbox was
> flipped and nothing was merged into `development`.** Branch
> `dev/c4-gate-rerun` off `development` @ `afc2ab9`.
> **Verdict: the gate is not green.** Gates 3 and 6 pass outright, gate 1
> passes on 9 of 11 golden cells, gate 4 passes on the cell the roadmap names
> and on the SilverTorch backend pair, gates 2 and 5 fail as written and the
> failures are measurement artifacts explained below, and three substantive
> findings fall out. Nothing here is citable (CLAUDE.md rule 2).

**Environment.** A100-SXM4-80GB, driver 580.159.04, nvcc 12.4, torch 2.10.0+cu128,
triton 3.6.0, Python 3.11, `silvertorch` 1.0.0. Venv `/venvs/retrieve` (the repo's
`.venv`), used directly with `PYTHONPATH` — never `uv run` (the shared venv's editable
`retrieve` pointer may belong to another worktree). Datasets at `/workspace/data`
(`RETRIEVE_DATA_ROOT`, `evaluation/data` symlinked there). Private
`TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-c4`, created empty and confirmed to be the one in
use — §10's stale-FX-cache trap does not apply to these numbers. Sole GPU worker; every
stage still took `flock /workspace/gpu.lock`.

Every record carries `code_version = 0fe440d4bc9013097737e225af59b71ccec94a6d`
(= `afc2ab9:retrieve/src/retrieve`) and `dirty: false` — the **library subtree was clean
for every cell**. `repo_dirty: true` is the artifact scripts in this directory, not
measured code. `provenance.txt` is rewritten by each invocation of the runbook and shows
only the last stage (`gate`); the per-record `env` block is the authority.

**Runbook.** [c4_gate_run.sh](evaluation-harness-v2-artifacts/c4_gate_run.sh),
`PY=/workspace/retrieve/.venv/bin/python`, stages `goodreads` → `arxiv` → `resume` →
`gate`. One change to the script: `GOLDEN_SM_MHZ` is now an environment variable
defaulting to **1155**, §11's sampled median; the hard-coded `1140` was A1's and is
stale. No threshold in `c4_gate.py` was touched.

| stage | window (UTC) | result |
|---|---|---|
| `goodreads` | 08:50:26 – 11:15:28 | 14/14 cells, all `status: ok` |
| `arxiv` | 11:20:06 – 11:51:47 | 6/6 cells, all `status: ok` |
| `resume` | 11:51:47 – 12:10:53 | gate 6, pass |
| `gate` | 12:18:58 | `gate.txt`, exit 1 |

20 records, 360 perf entries, 0 failed cells. All 11 golden cells matched **exactly one**
v2 record at the algo defaults (`coverage`: 0 failures).

### 12.1 The inputs are the golden's inputs, and that was checked, not assumed

The v2 harness keys its own query cache (`encoded_queries_v2.pt`) differently from the old
one (`encoded_queries_test.pt`), so it re-encoded all 313,178 queries from the checkpoint
rather than reading A1's blob. That could have moved every quality column on its own, so
the two caches were compared directly:

| tensor | result |
|---|---|
| `item_embs` `[797084, 128]` | **bit-identical** (`torch.equal`) |
| `queries[:10000]` | **bit-identical** |
| `targets[:10000]`, `n_targets[:10000]` | **bit-identical** |

The old blob stores only the first 10,000 rows (`users_limit` applied before caching); v2
stores all 313,178 and slices at load (`users_limit=10000: keeping the first 10000 of
313178`). Independently, the goodreads oracle **cache hit** —
`oracle_v4_c0_genre_2634b8ef02f0f10c.pt`, whose fingerprint samples the query tensor — so
the queries reaching the oracle are provably the ones it was built from. The legacy
1-indexed `[N+1, …]` attrs layout was detected and dropped on both datasets, as designed.

**An oracle *was* built, once, and it was not a fingerprint mismatch.** arxiv's
`gt_d128/` as published carries only the old-harness `gt_topk_v3_c0_maincat.pt`; there is
no `oracle_v4_*` blob for arxiv at all (goodreads ships both). The build took **~40 s**
(11:20:35 → 11:21:15) for 10,000 queries × 2,988,996 items, not hours. The result was then
cross-checked against the blob the golden actually used:

> `gt_topk_v3_c0_maincat.pt["topk"]` vs the new `oracle_v4_c0_maincat_c5222af9615d538d.pt["topk"]`,
> both `[10000, 1000]` int64 — **exact equal: True**.

So the rebuild changed no ground truth. Nothing else was rebuilt.

### 12.2 Per-cell gate table

`gr/` = goodreads-d128 clause/`c0_genre`, `ax/` = arxiv-d128 clause/`c0_maincat`, seed 0.
`np24`/`np32` = `n_probe`. INFO = no golden cell for those params/backend (checked on
gates 3–5 only), or an inexact algo's parity, which is reported not gated.

| cell | 1 quality | 2 latency | 3 graph | 4 parity | 5 stable |
|---|---|---|---|---|---|
| gr/linr_v1_filter_mask-triton | PASS | FAIL | PASS | INFO (ref) | PASS |
| gr/linr_v1_filter_mask-torch | PASS | FAIL | PASS | **PASS** | FAIL |
| gr/linr_v2-triton | PASS | FAIL | PASS | INFO (ref) | FAIL |
| gr/linr_v2-torch | PASS | FAIL | PASS | **FAIL** | FAIL |
| gr/linr_v3-triton | PASS | FAIL | PASS | INFO | FAIL |
| gr/linr_v3-torch | PASS | FAIL | PASS | INFO | FAIL |
| gr/linr_v4-triton | **FAIL** | FAIL | PASS | INFO (ref) | FAIL |
| gr/linr_v4-torch | **FAIL** | FAIL | PASS | INFO | PASS |
| gr/silvertorch-triton np24 | PASS | FAIL | PASS | INFO (ref) | FAIL |
| gr/silvertorch-triton np32 | INFO | INFO | PASS | INFO | FAIL |
| gr/silvertorch-torch np24 | PASS | FAIL | PASS | INFO | FAIL |
| gr/silvertorch-torch np32 | INFO | INFO | PASS | INFO | FAIL |
| gr/silvertorch-official np24 | INFO | INFO | PASS | **PASS** | FAIL |
| gr/silvertorch-official np32 | INFO | INFO | PASS | **PASS** | FAIL |
| ax/silvertorch-triton np24 | **FAIL** | FAIL | PASS | INFO (ref) | FAIL |
| ax/silvertorch-triton np32 | INFO | INFO | PASS | INFO | FAIL |
| ax/silvertorch-torch np24 | INFO | INFO | PASS | INFO | FAIL |
| ax/silvertorch-torch np32 | INFO | INFO | PASS | INFO | FAIL |
| ax/silvertorch-official np24 | INFO | INFO | PASS | **FAIL** | FAIL |
| ax/silvertorch-official np32 | INFO | INFO | PASS | **FAIL** | FAIL |

Full table: [c4/gate.txt](evaluation-harness-v2-artifacts/c4/gate.txt) (PASS 88, FAIL 127,
INFO 23).

### 12.3 Gate 1 — quality within 1e-6: **9 of 11 golden cells pass**

52 of 66 compared rows pass, and they do not merely pass — the **worst passing delta in
the whole run is 7.5e-9**, three orders of magnitude inside the tolerance. Six of the nine
passing cells are exact to 0.0 on `recall@k`.

**The two cells §11.8 declared unmeetable now pass.** L3's deterministic stream compaction
is confirmed against real data, not just unit fixtures:

| cell | §11.8's measured own-noise | C4 max \|diff\| vs the re-derived golden |
|---|---|---|
| `linr_v2-triton` | 2.0e-6 (2× the tolerance) | **3.6e-11** |
| `linr_v3-triton` | 6.8e-5 (68× the tolerance) | **1.6e-9** |

§11.8 item 3 ("C4's gate 1 cannot be met on these two cells by any harness") is therefore
**withdrawn**: it was true of the pre-L3 library and is false of this one. No decision to
exclude the two cells is needed.

**Two cells fail, and neither failure is the permitted `k_max`-slice difference.**

| cell | k=100 | k=500 | k=1000 |
|---|---|---|---|
| `linr_v4` (**both** backends, identical) `recall@k` | **7.3e-5** | 9.3e-6 | 2.1e-6 |
| `linr_v4` (both backends) `ndcg@k` | 5.2e-5 | 7.3e-6 | 1.6e-6 |
| `ax/silvertorch-triton np24` `recall@k` | **2.0e-6** | 6.0e-12 | 6.0e-12 |
| `ax/silvertorch-triton np24` `ndcg@k` | 1.4e-6 | 7.5e-9 | 6.5e-9 |

Both misses concentrate at the shallowest `k`, which looks exactly like the `k_max` slice
WP-4 gate 1 permits. **It is not.** Both cells were rerun with `--k 100`, i.e. `k_max =
100`, precisely the way the golden ran them
([c4/kmax-diag/](evaluation-harness-v2-artifacts/c4/kmax-diag)):

| cell | v2 at `k_max=1000` | v2 at `k_max=100` | golden |
|---|---|---|---|
| `gr/linr_v4-triton` `recall@100` | 0.982127004293 | **0.982127004293** | 0.982053974462 |
| `ax/silvertorch-triton np24` `recall@100` | 0.884044068151 | **0.884044068151** | 0.884042068159 |

The v2 number is identical to the last digit at both `k_max`. The slice is exonerated and
the misses are real differences between the two harnesses.

**`linr_v4` is attributed, and it is not a defect in either harness.** Three hypotheses
were tested ([c4_linr_v4_probe.py](evaluation-harness-v2-artifacts/c4_linr_v4_probe.py)):

1. *the library changed* — no: §11.2 found `linr_v4` bit-identical between A1 and the
   re-derive on both backends;
2. *eager vs compiled* (the old harness compiled at build time in `AlgoBase._finalize` and
   ran quality through the compiled forward; v2 runs quality eager) — no: eager vs
   `torch.compile(dynamic=True, mode="reduce-overhead")` is **bit-identical**, 0/2048 rows
   differ, `max |Δscore| = 0`;
3. *the query-batch shape* — **yes.** `LiNRV4`'s output depends on how many queries are in
   the call:

   | algo | scoring | chunk 16 vs chunk 64, 2048 rows | max \|Δscore\| | mean set overlap@100 |
   |---|---|---|---|---|
   | `linr_v1_filter_mask` | fp32 cuBLAS | **0 / 2048 rows differ** | 0.0 | 1.000000000 |
   | `linr_v4` | int8 `_int_mm` → `>>5` → fp16 | **1536 / 2048 rows differ** | 1.96e+2 | 0.992065430 |

   The library documents the mechanism itself
   ([knn.py:46–56](../../retrieve/src/retrieve/modules/knn.py)): `PostfilterKNNInt8`
   compresses int32 dots with `>>5` into fp16 before `topk`, and says "topk ordering is
   exact up to ties introduced by the `>>5` range compression (**boundary ties are
   quality-equivalent**)". It also zero-pads batches below `_PAD_M = 17`, and v2's quality
   pass uses `QUALITY_CHUNK = 16`, so v2 takes the padded path where the old harness's
   batching did not. ~0.8 of 100 ids per row change, which moves `recall@100` by 7.3e-5.

   So this is the library's documented, quality-equivalent tie order — *tie order*, which
   WP-4 gate 1 allows, but reached by a route the clause does not name ("under the `k_max`
   slice"), and `c4_gate.py` compares floats and cannot express "ties only". **This is a
   gate-wording question for the orchestrator, not a bug to fix**, and it is the one row
   in this run whose resolution changes whether C4 can ever go green as written.

**The arxiv 2.0e-6 is not attributed.** It is 2× the tolerance on `recall@100` only, with
k=500 and k=1000 bit-exact at 6e-12 — arithmetically about **two single-hit changes across
10,000 queries**. It is not the `k_max` slice (above) and not the batch shape: the same
chunk-16-vs-64 probe on `ax/silvertorch-triton` gives **0 / 4096 rows differ**, ids and
scores identical. The remaining candidate is top-k boundary ties on the handful of arxiv
rows that have fewer than `k` survivors (§11.3 counts 4 such rows at k=100); **not tested,
and it is recorded here as unexplained.**

### 12.4 Gate 2 — graph latency within 5 %: fails as specified, and the failure is the clock normalisation

**As the gate is specified (`--golden-sm-mhz 1155`), 92 of 99 latency rows FAIL**,
including every bs=8 and bs=16 row. That result is an artifact of `c4_gate.py`'s
clock normalisation, and the evidence is decisive.

The script compares two numbers that are not estimates of the same thing:

* the **golden's** clock is `--golden-sm-mhz`, the **median of a 30 s-cadence trace over
  the whole run** — which mostly samples the build, quality and oracle phases, not the
  timing windows;
* the **run's** clock is `perf[].sm_mhz`, a **single sample taken immediately after the
  last timing window's sync, under load**. All 360 perf entries recorded **1410 MHz**, the
  boost clock, without exception.

The two runs' GPUs were at the same clocks throughout:

| trace | n | median | max | min | histogram |
|---|---|---|---|---|---|
| golden (`evaluation/golden/_logs/clocks.csv`) | 65 | 1155 | 1410 | 210 | 1155×55, 1410×9, 210×1 |
| C4 (`c4/clocks.csv`) | 41 | 1155 | 1410 | 210 | 1155×31, 1410×9, 210×1 |

So dividing a 1410 MHz under-load sample by a 1155 MHz whole-run median injects a flat
**1410/1155 = 1.221** bias into every ratio, which is more than four times the 5 % gate.
That, not the harness, is what fails 92 rows.

**At a matched under-load clock** the picture inverts
([c4/gate-matched-clock.txt](evaluation-harness-v2-artifacts/c4/gate-matched-clock.txt),
the same script at `--golden-sm-mhz 1410`, no threshold changed):

| bs | PASS | FAIL | ratio range (v2 / golden) |
|---|---|---|---|
| 1 | 17 | **16** | 0.799 – 1.210 |
| 8 | **33** | 0 | 0.957 – 1.026 |
| 16 | **33** | 0 | 0.975 – 1.030 |

Every bs=8 and bs=16 row is inside 5 %, most inside 3 %. All 16 failures are at bs=1, and
**14 of the 16 have v2 faster than golden**, which is the direction WP-4 (2) itself
predicts ("the golden number is a cold-L2 `do_bench` median and v2 does not flush, so v2 is
expected at or below golden"). The two exceptions are arxiv rows at 1.210 and 1.082.

**bs=1 cannot be a 5 % target, and that is measurable on the golden harness alone.** The
§11.7 repeat runs are the same code, same box, same data, run 2–3 times
(`c4_latency_evidence.py --golden-noise`):

| bs | cells × (k) | max run-to-run spread | median spread |
|---|---|---|---|
| 1 | 9 | **21.1 %** | 14.1 % |
| 8 | 9 | 0.4 % | 0.1 % |
| 16 | 9 | 0.1 % | 0.0 % |

The golden's own bs=1 column carries up to 21 % of noise — wider than the entire
0.799–1.210 range C4 measured — while its bs=8/16 columns are stable to 0.4 %. bs=1 at
these shapes is 0.2–0.9 ms of launch latency for a handful of kernels; there is nothing
there for a 5 % gate to hold on to. v2's *own* windows are tight (window spread ≤ 0.3 % at
bs=1, ≤ 0.1 % at bs=8/16), so the noise is in the baseline, not in the new measurement.

**The `--flush-l2` experiment WP-4 (2) names was not run.** `bench.py` has no flush and
adding one is harness code, i.e. C5's tree, not this job's. It is also not needed to reach
a verdict: the direction is already the one the flush predicts, and the golden's own bs=1
noise covers the whole observed range without invoking it, while at bs=8/16 — where the
golden is stable to 0.4 % — v2 lands within 3 %, bounding any flush effect there at ≤ 3 %.

**Verdict on gate 2: explained, not met.** The bs=8/16 half of the gate is met once the
clocks are compared like for like; the bs=1 half cannot be met by any harness against this
baseline. Two things are the orchestrator's to decide and were deliberately not done here:
whether `c4_gate.py` should compare under-load samples against an under-load golden clock
(the golden run recorded no per-variant clock, only the trace), and whether gate 2 should
apply at bs=1 at all.

### 12.5 Gate 3 — `cudagraph_skips == 0`: **PASS, 20/20 cells**

Every capturable cell measured all 9 graph variants (`9/9 graph entries measured`); no
record carries a `reason` or a null `median_ms` on `triton` or `torch`. On `official` all
four cells are null with `reason: not_capturable` and nothing else, which is the only
accepted reason there (O D7). The three library fixes behind this (H §9, `dd8b7b5`) hold on
real data at every batch size.

### 12.6 Gate 4 — parity: the roadmap's cell passes, two rows fail

**The result §11.4 asked for holds.** SilverTorch `torch` vs `triton` is now
`jaccard@100 = 1.000000` with `score_max_abs_diff = 0.0` on **both** datasets and **both**
`n_probe` values — exact, not merely within tolerance.

**The roadmap's own clause passes**: `gr/silvertorch-official` runs end to end with
`status: ok`, quality present, 9 timed entries, `cache_plans: false` on every one, and
`jaccard@100 = 0.999849` (np24) / `0.999805` (np32), against O WP-6's ≥ 0.99.

Two rows fail:

| row | jaccard@100 | `score_max_abs_diff` | required |
|---|---|---|---|
| `gr/linr_v2-torch` vs `triton` | 0.998743 | 9.766e-03 | 1.0 (exact algo) |
| `ax/silvertorch-official` np24 / np32 vs `triton` | 0.985033 / 0.984882 | 4.827e-04 | ≥ 0.99 |

1. **`linr_v2` torch ≠ triton is pre-existing and is in the golden.** The two golden cells
   disagree by exactly the same amount — `recall@100` 0.9996946954935145 (torch) vs
   0.9992737607088252 (triton) — and the v2 harness reproduces **each** of them to 1e-11.
   The harness is right; LiNR V2, which is the *exact* filtered top-K and is gated at
   `jaccard == 1.0` for that reason, does not agree between its two backends in this
   library. Reported, **not fixed** (out of scope). This is the one row in this run that
   looks like a genuine library bug rather than a measurement or wording issue.
2. **`official` clears 0.99 on goodreads and misses it on arxiv.** The roadmap's gate names
   goodreads only, and the arxiv `official` cells are extra (`BACKENDS_ARXIV` defaults to
   all three). `score_max_abs_diff` is an order of magnitude *smaller* on arxiv
   (4.8e-4 vs 5.5e-3) while jaccard is lower, which is what a looser filter with more
   near-ties at the top-100 boundary looks like. Whether O WP-6's bar should hold on arxiv
   is not settled anywhere and is not settled here.

### 12.7 Gate 5 — no `unstable` cell: fails, and clause 5's precondition does not exist on this box

18 of 20 cells are flagged. Split by cause:

| cause | cells |
|---|---|
| a real window spread > 5 % in at least one perf entry | **8** |
| `clocks_drift` **only**, every window tight | **10** |
| clean | 2 |

The 10 are a pure artifact. `run.py` sets `clocks_drift` when a cell's clock differs by
more than 5 % from `clk0`, sampled once at process start; `clk0` reads 1155 MHz and every
under-load sample reads 1410 MHz, so the flag fires on a 22 % "drift" that is just the GPU
boosting. The log says so literally: `clocks.sm 1410.0 MHz drifted from 1155.0 MHz`.
Related, `clocks_locked` is recorded **`true`** on the arxiv cells — the field is defined
as "`sm_mhz` within 2 % of `expected_sm_mhz`", the default expectation is 1410, and the
box happened to be at 1410. On a container where `nvidia-smi -lgc` is denied that field
records a coincidence, which is the opposite of what §8.2 F wanted it for.

The other 8 are real, and they are the same bs=1 story as gate 2: **11 of the 14
window-spread outliers are at bs=1**, 2 at bs=8, 1 at bs=16, out of 360 perf entries
(3.9 %). The worst are `linr_v3-triton` k=100 bs=1 eager at 22.8 % and
`silvertorch-triton` k=500/1000 bs=1 **graph** at 14.9 % / 14.3 %.

WP-4 clause 5 reads "no `unstable` cell **at locked clocks**". Clocks cannot be locked in
this container (`provenance.txt`: *"The current user does not have permission to change
clocks"*), so the clause's stated precondition is unavailable and H §7's fallback applies
instead. **Gate 5 as written cannot be evaluated here**; what can be said is that 10 of
the 18 flags are a first-sample artifact and the remaining 8 are bs=1 launch-latency noise.

### 12.8 Gate 6 — kill mid-run, `--resume` continues: **PASS**

`docs/plans/evaluation-harness-v2-artifacts/c4/resume/`. Child started 11:51:47 on
`linr_v1_filter_mask` × {triton, torch}, SIGTERM at 12:01:03 once the first record had
landed and the second cell was well under way, 1 record on disk after the kill. The rerun
with `--resume` logged
`resume: ('goodreads', 128, 'linr_v1_filter_mask', 'triton') all 1 cells done` and finished
with `{'skipped': 1, 'ok': 1}` — 2 records, no duplicate, no recomputation of the completed
cell.

### 12.9 What was not done, and what is still unverified

* **No roadmap checkbox flipped, nothing merged** — the orchestrator's call after review.
* **`--flush-l2`** (WP-4 (2)'s named experiment): not run, see §12.4.
* **The arxiv 2.0e-6 gate-1 miss**: not attributed, see §12.3.
* **`linr_v2` torch/triton disagreement**: reported, not investigated in the library and
  not fixed — the library was closed for the day.
* **`--seed 0` only.** The `filter` suite's `headline` block adds seeds 1 and 2 on
  `c0_genre`/`c0_maincat` at d128; WP-4 does not ask for them and they did not run.
* **`bloom` filter cells, other sweeps, other dims**: out of WP-4's scope, not run.
* The 2026-09-06 partial run's outputs were moved to
  [c4/2026-09-06-partial/](evaluation-harness-v2-artifacts/c4/2026-09-06-partial) rather
  than deleted; they carry no `code_version` and are not comparable to these.
* Parity spill files (`_parity/*.npz`, ~600 MB) are regenerated by every run and are now
  gitignored.

### 12.10 Summary for the roadmap

| WP-4 clause | verdict |
|---|---|
| 1 quality within 1e-6 | **9 / 11 golden cells pass** (worst passing delta 7.5e-9). `linr_v4` (both backends) misses by 7.3e-5 — attributed to the int8 path's documented boundary ties moving with the query-batch shape, i.e. tie order, reached by a route the clause does not name. arxiv `silvertorch` misses `recall@100` by 2.0e-6, unattributed. |
| 2 graph latency within 5 % | **Fails as specified** (92/99 rows) because the gate normalises an under-load clock sample against a whole-run median. At matched clocks: **66/66 pass at bs=8 and bs=16**, 16/33 fail at bs=1, 14 of them with v2 *faster*. The golden's own bs=1 noise is up to 21 %. |
| 3 `cudagraph_skips == 0` | **PASS**, 20/20 cells; `official` null with `not_capturable` only. |
| 4 `jaccard_vs_first@100` | Roadmap's clause **PASS** (`gr/silvertorch-official` 0.999849 ≥ 0.99). SilverTorch torch-vs-triton **exactly 1.0** on both datasets. **FAIL** on `linr_v2` torch-vs-triton (0.998743, also present in the golden — a library property) and on `official` **on arxiv** (0.985, a dataset the roadmap's clause does not name). |
| 5 no `unstable` cell | **Not evaluable** — the clause says "at locked clocks" and clocks cannot be locked here. 18/20 flagged: 10 by a first-sample `clocks_drift` artifact, 8 by real bs=1 window spread. |
| 6 kill + `--resume` | **PASS**. |

Three decisions this gate needs before it can be green, all the orchestrator's:
**(a)** whether gate 1's "tie order under the `k_max` slice" covers `linr_v4`'s
shape-dependent int8 ties; **(b)** what clock gate 2 normalises against, and whether it
applies at bs=1 at all; **(c)** whether gate 5 means anything on a box that cannot lock
clocks, and whether `clocks_drift` should be measured against a warm sample.

## 13. Validation record — WP-6 / roadmap D4, `report.py`, 2026-09-15, CPU only (`dev/d4-report`)

> Model: §12 above, and [V §11](evaluation-package-layout.md). Worker job under
> [agent-orchestration.md](agent-orchestration.md): **no roadmap checkbox was flipped and
> nothing was merged into `development`.** Branch `dev/d4-report` off `development`
> @ `e23309c`, worktree `/workspace/wt/d4`.
>
> **This step was dispatched out of order.** D4 normally runs on D1's campaign records; the
> campaign needs the GPU, which another worker held. `report.py` is therefore written
> against the **record schema** and the records that exist today — C4's gate run (schema 1)
> and C5's one-cell check (schema 2) — and nothing in it hard-codes a cell of either.
> **Nothing it emitted today is citable** (CLAUDE.md rule 2), which is also what every
> artifact says about itself.

**Environment.** The A100 box, CPU only: **`CUDA_VISIBLE_DEVICES=""` throughout, no GPU
work at all** (the concurrent B3 worker's timings were not perturbed). Python 3.11.15,
torch 2.10.0+cu128, `/venvs/d4` via `uv sync --all-packages`, ruff 0.15.6 (`uvx`; not in
the lock), datasets symlinked but never read. `retrieve/` untouched.

**What was built.** [`evaluation/bench/report.py`](../../evaluation/bench/report.py)
(≈ 700 lines), the `bench report` subcommand wired into `cli.py` in place of its exit-2
stub, [`evaluation/tests/bench/test_report.py`](../../evaluation/tests/bench/test_report.py),
the [Report section](../system/evaluation.md#report-reportpy) of the system doc, and the
sample output in [d4/](evaluation-harness-v2-artifacts/d4/README.md). One dependency added:
`matplotlib>=3.8` on `retrieve-evaluation` (`uv add --package`; `uv.lock` +761 lines) — the
figures WP-6 names need a plotting library and none was installed.

Shape: one function per artifact and one `ARTIFACTS` dispatch table, no plotting framework,
no config object model (coding-guidelines D3/D4). **It is 975 lines against §3.1's 200-line
budget** — that budget was written before §6 WP-6 enumerated fourteen artifacts; each
function is 20–45 lines and the only shared machinery is one selector, one seed/parameter
reducer, one LaTeX table writer and one figure wrapper. V §5.3's three requirements are
met: `partial` honoured, the subtree `dirty` flag enforced, a torn samples line tolerated. `records.flatten` writes `flat.csv` first
(§8.2 G) and every table and figure is built from it; `records.read_records` is read a
second time for the provenance block alone, because `schema_version`, `partial_reasons`,
`stage` and `error` are **not columns of `flat.csv`** (see "reported, not changed" below).

**What each artifact contains.**

| `--only` name | file | content |
|---|---|---|
| `recall_nofilter` | `tables/tab-recall_nofilter.tex` | `tab:recall_nofilter`: held-out Recall@k on `filter_kind: none` cells, datasets × algos |
| `pareto` | `tables/tab-pareto_<dataset>.tex` | `tab:pareto_<dataset>`: condition × algo — oracle recall, `median_ms`, speedup vs LiNR V1, `index_mib` |
| `batch_scaling` | `tables/tab-batch_scaling.tex` | `tab:batch_scaling`: amortised ms/query per algo × batch size, **each cell carrying its window spread** |
| `memory` | `tables/tab-memory.tex` | `tab:memory`: `index_mib`, datasets × algos |
| `parity` | `tables/tab-backend_parity.tex` | per `(dataset, sweep, algo, backend)`: `path`, `jaccard_vs_first@k`, `score_max_abs_diff`, eager vs graph median and the ratio |
| `recall_at_budget` | `tables/tab-recall_at_budget.tex` | §8.2 H: best recall under each `--budget-ms` p99 budget, and the algo that reached it |
| `paper_comparison` | `tables/tab-paper_comparison.tex` | our `--compare-bs` eager mean / p99 / QPS / pass rate beside SilverTorch's and LiNR's **reported** numbers, with §2.7's differences as a footnote. The published rows are the `PAPER_REPORTED` constant, cited per row to `articles/` |
| `fig_pareto`, `fig_qps_recall`, `fig_batch_scaling` | `figures/*.png` | recall–latency Pareto per dataset, QPS vs recall, amortised latency vs batch with window-spread error bars |
| `fig_deep_sweep` | `figures/fig-deep-sweep-*.png` | one per swept parameter: recall and latency against its value, **whiskers = seed min–max** |
| `fig_latency_violin` | `figures/fig-latency-violin.png` | per-call distributions from the samples sidecar (one torn trailing line tolerated) |
| `methodology` | `methodology.tex` | the thesis's §"Методология замеров" itemize with its constants read **live** out of `measure.latency`'s signature, `inputs.query_pool`'s `n_pool` and `run.{MODES, QUALITY_CHUNK, CLOCK_DRIFT}`, so text and code cannot drift apart again |
| — | `report.md` | the human-readable view: provenance, citability verdict and reasons, the selection the tables used, a coverage table, the failed cells with stage and error, the partial records, the unstable variants with their spread, the artifact list |

**The three constraints of the day, and how they are met.**

1. **Nothing is citable by default (rule 2).** `--gate STEP` is the only way to an unmarked
   artifact, and the **evidence vetoes the flag**: a `failed` or `partial` record, or one
   with `env.dirty`, keeps the marker on even with `--gate`. Otherwise every `.tex` carries
   a `% PROVENANCE: *** NOT CITABLE ***` banner listing the reasons, its caption opens with
   `\textbf{[PRE-CAMPAIGN RECORDS — NOT CITABLE]}`, and every figure gets a diagonal
   watermark. Every artifact carries the `code_version`, the commit, the branch, the GPU,
   the schema version and the run window whether citable or not.
2. **Which clock estimator.** Every latency table's last footnote names it: the per-variant
   **under-load** `perf[].sm_mhz`, with the range observed over exactly the rows behind
   that table. `env.sm_mhz_idle` and a schema-1 `env.sm_mhz` (a whole-run median dominated
   by idle) are provenance only and are never compared with an under-load sample — §12.4's
   artifact. **No clock normalisation is performed anywhere**; the report states the clock,
   it does not divide by it.
3. **`bs = 1` is noise-dominated.** `tab:batch_scaling` prints each cell's window spread
   beside the number and says in the caption that no speedup claim is made from a `B=1`
   column; `fig_batch_scaling` draws spread error bars; the Pareto and comparison tables
   mark every `unstable` variant `†`.

**Failed, partial, unstable — decided, not filtered quietly.** A `status: failed` record
never reaches a number, anywhere, and is listed in `report.md` with its stage and the last
line of its traceback. A `partial` record is used and marked `*`. A perf entry with
`unstable: true` is used and marked `†`. Both marks are explained in every caption, and
both counts are in the banner and in `report.md`. Where a table's shape allows one value
per cell and the records hold several parameter sets, the smallest by canonical-JSON order
is shown **and a caption footnote names all of them**; several seeds reduce to their median
with min–max whiskers in the figures.

**Gates.**

| gate | result |
|---|---|
| `ruff check evaluation` (0.15.6) | **clean** |
| `cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/` | **170 passed, 4 skipped** (baseline at `e23309c`: 164 / 4). +6 = `tests/bench/test_report.py`; `test_cli.py`'s report case changed from "exits 2" to a real end-to-end `bench report` over the campaign's own records. The 4 skips are unchanged (`test_yfcc.py::TestRealSlice`, yfcc10m not staged) |
| `python3 scripts/check_doc_links.py` | **0 broken links** |
| `bench report --help` | renders, rc 0; all 20 other `--help`s unchanged |
| `bench report` end to end on today's records | **18 artifacts, no traceback**, over 24 records (C4's 20 schema-1 + C5's 6 schema-2, 4 deduping by resume key). Output committed as [d4/](evaluation-harness-v2-artifacts/d4/README.md) |
| LaTeX | **not compiled — no TeX toolchain on this box.** Checked structurally by the test: balanced `table` / `tabular` / `itemize` / `minipage`, balanced braces and `$`, the thesis's labels present, no unescaped `_` outside math, `\texttt` and `\label` |

**Labels match `docs/thesis/main.tex` as it now stands** (read, not assumed): the five
table labels are `tab:recall_nofilter`, `tab:pareto_arxiv`, `tab:pareto_goodreads`,
`tab:batch_scaling`, `tab:memory`, the algorithm label is `algo:silvertorch`, and the
captions follow the file's Russian decimal-comma convention (`$0{,}9128$`). `QuantizedIVF`
appears nowhere; `ALGO_LABEL` maps `silvertorch` → "SilverTorch".

**The test** (`tests/bench/test_report.py`, 6 cases, ≈ 10 s, coding-guidelines D6 — a gate,
not a deliverable): every column the tables read still comes out of `records.flatten`
(a `KEY_FIELDS` / `_RECORD_COLUMNS` change breaks it loudly and names the missing column);
every artifact emitted and the LaTeX structurally sound with the thesis's labels; a
`failed` record excluded and a `partial` / `unstable` one marked; citability off by default
and evidence beating `--gate` (a `failed`/`partial` record, a dirty subtree, a `dev/*`
branch); an empty results tree emitting placeholders instead of failing; a schema-1 record building tables and using the under-load clock rather than the
1155 MHz whole-run median.

**Reported, not changed — one `records.py` gap.** `records.flatten` does **not** carry
`schema_version`, `partial_reasons`, `stage` or `error` into `flat.csv` (`_RECORD_COLUMNS`
omits them). `report.py` works around it by reading the JSONL a second time through
`records.read_records` for the provenance block only, which is public API and changes no
behaviour — but if `flat.csv` is meant to be the shippable denormalisation of a record
(§8.2 G), those four fields belong in it. Not done here: another worker's numbers depend on
that module today. Second, smaller: `records.flatten` writes to a path it does not create,
so a caller must `mkdir` first (`report._load` does).

**What is skipped, and what is unverified.**

* **Every number in every artifact.** These are C4's and C5's probe records, not campaign
  data. The step's own output is the *shape*; the numbers are D1's.
* **`tab:recall_nofilter` is empty today** — no `filter_kind: none` cell exists in the
  records that exist; it emits its `--- no matching cells ---` placeholder, which is the
  behaviour under test. It fills in when D1's `quality` suite runs.
* **No LaTeX was compiled**, and no fragment was placed in `main.tex`. Pasting them is
  F5's job, not this one.
* **The deep-sweep and Pareto figures have never seen a real sweep**: today's only swept
  parameter is `n_probe ∈ {24, 32}`, two points. The `deep` suite's `n_lists` × `n_probe`
  grid and `linr_v3`'s `candidate_pool` ladder are D1's.
* **Seed whiskers are exercised on exactly one cell set** (C5's seeds 0/1/2 on
  goodreads `silvertorch`); the multi-seed headline block is D1's.
* **`bench report` has never run on a `deep` or `quality` suite directory**, on more than
  one dim, or on a tree with a `_parity` or `_logs` directory beside the suites (the glob
  is `*/*.jsonl`, so `_logs/*.log` is ignored, but this was reasoned, not run).
* **`--profile` kernel tables are not reported on.** The `kernels` key is in `_PERF_SKIP`,
  so it never reaches `flat.csv`; a per-kernel view would need its own artifact and WP-6
  does not name one.
* The **`matplotlib` addition is unexercised outside this box's Agg backend**, and the
  `uv.lock` change is the one thing in this branch that can conflict on merge.

---

## 14. Validation record — results storage and `bench upload`, 2026-09-15, CPU only (`dev/results-storage`)

> Model: §13 above. Worker job under [agent-orchestration.md](agent-orchestration.md):
> **no roadmap checkbox was flipped, nothing was merged, nothing was made public.**
> Branch `dev/results-storage` off `development` @ `5fd05a6`, worktree
> `/workspace/wt/results`, `/venvs/results`. `CUDA_VISIBLE_DEVICES=""` throughout —
> no GPU work at any point, so the concurrent D1-a campaign in `/workspace/wt/d1a`
> was not perturbed and its worktree was not touched. `retrieve/` untouched.
>
> **Why now, before the volume arrives.** D1 runs as five stages and E5 adds four
> datasets; the records are the only thing the paper may cite (CLAUDE.md rule 2) and
> they live on a rented box whose `/workspace` is a ~26 GB quota. `bench upload`
> was the machinery meant to make them durable and **V §11 listed it "exercised by
> no test"** — it had never been run. It has now.

### 14.1 The policy

Three destinations, by size and by the cost of recreating the file. Written up as
[Results storage](../system/evaluation.md#results-storage); the short version:

| what | where | why |
|---|---|---|
| the JSONL records, `flat.csv`, `report/` (`*.tex`, `report.md`), the validation record | **git** | 79 records exist today and they are **750 KB in total** — kilobytes, line-diffable, and they are the evidence |
| `*.samples.jsonl`, `.perkernel/`, figures | **HF Hub**, `pinkmeme/eval-results`, private | the sidecars behind those same 750 KB are **45 MB — 60×**. Nobody diffs a latency vector, and it is regenerable only by re-running the cell on the GPU |
| `results/_parity/*.npz` (600–680 MB/run), `results/_logs/` | **deleted** | rewritten by every run, and the parity *verdict* (`jaccard_vs_first@k`, `score_max_abs_diff`) is already inside the record. C4, C5 and B3 each deleted theirs by hand; this makes it the written rule. `.gitignore` already covers both, and `upload.files` skips every `_`-prefixed path part |

The 60× ratio is the whole argument and it was measured, not guessed: `750 KB`
of records against `45 MB` of sidecars across C4 + C5 + B3. At D1's ~700 cells the
sidecars extrapolate to ~450 MB — 1.7 % of the quota for data no reader opens.

The two halves stay in step: the Hub copy carries a sha256 per file, and that
manifest is committed beside this plan in
[results-storage/](evaluation-harness-v2-artifacts/results-storage/), so git can
prove what the Hub holds without downloading it.

### 14.2 What was built

[`evaluation/bench/upload.py`](../../evaluation/bench/upload.py) rewritten (54 → 264
lines) and [`evaluation/tests/bench/test_upload.py`](../../evaluation/tests/bench/test_upload.py)
added (15 tests, no network). The command was a bare `upload_folder` mirror with a
required `--repo-id`, `--private` defaulting to *false*, and no provenance of any kind.
It is now:

- **One `--path-in-repo` subtree per invocation**, in **one** `create_commit` — so a
  failure leaves no half-published subtree, and C4's, C5's and B3's runs (different
  commits, different branches, different schema versions) do not blur into one tree.
- **`<prefix>/MANIFEST.json`**: `report.provenance` over the records actually being
  uploaded, plus `{path, bytes, sha256}` per file.
- **A root `README.md` generated from every manifest in the repo** — the existing ones
  are fetched first, so a second upload does not drop the first subtree from the front
  page. Generated from the records; no hand-written prose anywhere in it.
- **`--verify`**: downloads the subtree back and checks every sha256.
- **`--private/--public`, default private**, and `--repo-id` defaulting to
  `upload.RESULTS_REPO = pinkmeme/eval-results` — the registry entry for *results*,
  beside `eval_datasets.hub.EVAL_REPOS` for *datasets*.

**Provenance survives the trip, and this is the point of the step.** Rule 2 is enforced
inside `report.py`; the same verdict now travels with the payload. The one change outside
`upload.py` is that `report._provenance` is renamed **`report.provenance`** and exported
(3 lines: the `def`, its one call site, `__all__`) — deliberately *not* a copy, so the
Hub manifest and the LaTeX banner cannot drift. `--gate STEP` is the only route to
`"citable": true` and the evidence vetoes it exactly as in the tables: `failed`,
`partial`, `env.dirty`, or an `env.git_branch` outside `development` / `main`.

`eval_datasets/hub.py` was **not touched** (the storage worker was live in that area),
and no second HF abstraction was built: one `HfApi`, `create_commit`, `snapshot_download`,
`hf_hub_download`, the same four calls `hub.py` uses.

### 14.3 What was published

`pinkmeme/eval-results`, created **private**, 20 files / 37 MB, three subtrees —
every record that exists on `development` today:

| subtree | source | records | status | schema | branch | commit | files | bytes |
|---|---|---|---|---|---|---|---|---|
| `c4` | `…/evaluation-harness-v2-artifacts/c4/results` | 20 | ok=20 | 1 | `dev/c4-gate-rerun` | `afc2ab9` | 4 | 12,983,727 |
| `c5` | `…/evaluation-package-layout-artifacts/c5/results` | 6 | ok=6 | 2 | `dev/c5-harness-split` | `71430d7` | 3 | 7,563,738 |
| `b3` | `…/official-silvertorch-artifacts/b3/e2e` | 30 | ok=24, partial=6 | 2 | `dev/b3-head-to-head` | `e23309c` | 8 | 16,918,821 |

**All three are `"citable": false`**, and the reasons are the right ones — the machinery
found them, they were not asserted:

- all three: `no --gate given` (correct: no roadmap gate is green for these), and
  `record(s) produced on dev/…` (rule 2, verbatim).
- `b3` additionally: `6 record(s) with status=partial` — B3's `--skip-quality` /
  narrowed cells, caught without anyone remembering they existed.

`c4` carries `schema_version: 1` and 18 `unstable` cells, `c5` 5 and `b3` 14; all of it
is in the manifests. The three manifests and the generated README are committed at
[results-storage/](evaluation-harness-v2-artifacts/results-storage/).

### 14.4 The round trip

Two independent checks, both on the VM's **local disk** (`/tmp`, 262 GB free) — never
`/workspace`. Transcript:
[roundtrip.txt](evaluation-harness-v2-artifacts/results-storage/roundtrip.txt).

1. `--verify` on each upload, against the manifest it had just written: `c4` 4 files,
   `c5` 3, `b3` 8, **every sha256 equal**.
2. A separate full-repo `snapshot_download` into a fresh temp dir afterwards, then
   `diff -r` against the three source trees: **`c4` / `c5` / `b3` IDENTICAL**, byte for
   byte. 36 MB down.

`api.repo_info(...).private` re-read after the last upload: `True`.

### 14.5 Gate

| gate | result |
|---|---|
| `ruff check evaluation` | clean |
| `cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/` | **185 passed, 4 skipped** (baseline at `5fd05a6`: 170/4; the +15 are `test_upload.py` and nothing else changed) |
| a no-network test for the upload path | `tests/bench/test_upload.py`, 15 tests, `HfApi` replaced by a recorder; the real upload is a validation-record item, not a CI dependency (coding-guidelines D6) |
| `bench upload --help` | renders, rc 0 |
| the real upload ran | three subtrees, §14.3 |
| the round trip verified | §14.4 |
| `python3 scripts/check_doc_links.py` | 0 broken links |

### 14.6 Surprises

- **The sidecar ratio is 60:1, not the 5–10× a reader would guess.** C4's
  `goodreads-d128` is a 156 KB record file beside an 8.4 MB samples file. That single
  number is what makes the split obviously right rather than a matter of taste.
- **B3's 6 `partial` records were news.** Nobody had to remember them; `report.provenance`
  reported them from the files, which is the argument for reusing it rather than writing
  a second verdict for the Hub.
- **The Hub's dataset viewer would have tried to parse the records.** The generated README
  carries `viewer: false` front matter — the JSONL is ragged (perf entries differ per
  cell) and the viewer would only fail on it; without front matter the Hub also warns
  about a bare card on every push.
- **The existing sidecars are already committed to git** (45 MB across the three artifact
  dirs), predating this policy. They are left alone: rewriting history would not reclaim
  the quota, and rule 5's instinct — do not delete what has no validated replacement —
  applies until the Hub copy has been round-tripped from a *different* machine. The
  policy binds D1 onward.

### 14.7 Unverified

- **Every number in every record published.** These are C4/C5/B3 probe runs; the
  manifests say so. Nothing here makes them citable.
- **`bench upload` has never run against a live `results/` tree** — the three payloads
  are the static artifact dirs. Specifically untested against a campaign *in flight*
  (D1-a is appending to `/workspace/wt/d1a` right now): a JSONL that grows between
  `sha256()` and `create_commit` would upload a manifest describing a file that no longer
  matches. **Run `bench upload` on a stage that has finished, not one that is running.**
- **No upload larger than 37 MB, and no LFS path exercised.** `create_commit`
  pre-uploads LFS itself, but the ~450 MB D1 sidecar set has not gone through it.
- **The round trip was verified on this box only.** A download onto a *different*
  machine — the real test of "the box is rented" — has not been done.
- **`--path-in-repo ""` (the whole repo root) is untested against a real repo**; only the
  prefixed form ran.
- **Nothing was deleted.** The `.npz` rule is documented and already the practice; no
  parity spill existed to delete at the time of writing.
