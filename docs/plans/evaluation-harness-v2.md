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
- **WP-5 — campaign rerun (GPU, ~24 h wall + 0.5 d).** Locked clocks, `bench campaign --suite all`:
  four datasets, all dims, all backends incl. `cuda`/`cute` (clause cells on cuda are legal per
  cuda-silvertorch-handoff §7). Closes roadmap §2 (goodreads oracle rerun) and §4b items 1, 3, 4,
  5, 7 in one pass. Gate: `bench report` runs with no missing cells; `median_ms(bs=16) <
  16·median_ms(bs=1)`; ids identical across `mode`; a rerun is byte-identical in quality.
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
([main.py:118–119](../../retrieve/src/retrieve/layers/silvertorch/main.py))
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
