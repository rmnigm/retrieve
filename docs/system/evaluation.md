# `evaluation/` retrieval harness

Live reference for the retrieval benchmark used to compare algorithms
across yambda, goodreads, and arxiv. Covers the driver layout, the YAML
config format, the measurement methodology, the output schema, and the
extension points for adding a new algorithm, config, or dataset.

For the algorithm internals (what each `forward(query)` does) see
[kernels.md](kernels.md) and [architecture.md](architecture.md). For how
the datasets on disk got there and how the SASRec checkpoints are
trained, see [datasets.md](datasets.md); for the checkpoint inventory
itself, [checkpoints.md](checkpoints.md). The correctness-only test
suite that gates kernel changes is documented in
[testing.md](testing.md).

## Harness v2 in progress (roadmap Phase C)

The harness is being rewritten to the protocol in
[evaluation-harness-v2.md](../plans/evaluation-harness-v2.md) (H); the
ordered steps are [00-roadmap.md](../plans/00-roadmap.md) Phase C. Steps C1
and C2 landed the first six modules next to the old ones, which keep running
unchanged until C3 deletes them (H §5):

| module | owns | status |
|---|---|---|
| [`retrieval/bench.py`](../../evaluation/retrieval/bench.py) | `setup`, `warm_gpu_once`, `provenance` (GPU, driver, CUDA, torch, triton, commit, dirty, branch, `code_version` = tree hash of `retrieve/src/retrieve`), `clocks`, `timed_build`, `index_bytes` (Σ buffers incl. the filter submodule), `latency` (H §2.5 event-per-call windows, IQR + outlier counts, `load: closed_loop`), `graph_callable` (`reduce-overhead`, `dynamic=False`, `fullgraph=True`, asserts `cudagraph_skips == 0` and one `cudaGraphLaunch` per call, raises `NotCapturable` with the record's `reason`), `profile_once` | authored C1; CPU tests green; GPU paths validated in C4 |
| [`retrieval/metrics.py`](../../evaluation/retrieval/metrics.py) | `accumulator` / `accumulate` / `finalize` — recall, ndcg, precision, mrr at every `k` from one top-`k_max` list as float64 running sums on device (one sync per pass); `ranked=True` reproduces the old per-`k` oracle `nt_k`; `jaccard_at_k`. The old per-row API (`*_at_k`, `accumulate_metrics`, `finalize_metrics`) is kept as wrappers for `passes.py` | rewritten in place; equal to the old per-row means to 1e-9 (`tests/test_metrics.py`) |
| [`retrieval/algos_v2.py`](../../evaluation/retrieval/algos_v2.py) | the five `nn.Module` wrappers (`LinrV1`…`Silvertorch`) with the filter as a submodule, `k` settable, `set_query_params` (`n_probe`, `candidate_pool`; H §8.2 A); `ALGOS`, `FILTER_KINDS`, `BACKENDS = (triton, torch, official)`, `FILTER_BACKEND` (official cells build Triton filters, O §6.2), `CAPTURABLE` (official is eager-only), `PATHS[(algo, filter_kind, backend)]` → the code path that runs (`triton`, `torch`, `cublas`, `cublas+triton`, `cublas+torch`, `official`) or `None`; `build`, `build_filter`, `is_valid_combo` | named `algos_v2.py` because a module cannot coexist with the old `algos/` package; C3 renames it to `algos.py`. `backend="official"` raises `NotImplementedError` until roadmap B1 adds it to `retrieve` |
| [`retrieval/config.py`](../../evaluation/retrieval/config.py) | the H §3.3 matrix: `load_dataset(yaml, dim) -> Dataset`, `load_matrix(dataset_yaml, suites_yaml, suite, **narrows) -> list[Job]`; `{dim}` templating, the `build:` / `query:` param split (H §8.2 A), `disabled: true` sweeps (§8.2 J), the PATHS collapse, `ConfigError` naming the file and key. Schema below | authored C2 (v2 API on top; the old `EvalConfig` API stays below a divider for `cli/evaluate.py` / `sweep.py` until C3). `tests/test_config_v2.py` checks job counts per suite on a fixture and that the real `config/{goodreads,arxiv}.yaml` × `suites.yaml` reproduce the old `d128-{filter,quality}.yaml` cells |
| [`retrieval/data.py`](../../evaluation/retrieval/data.py) | `load_inputs(ds, device, with_filters=True) -> dict` (item embs, queries, targets, `n_targets`, `qa`, `item_attrs`, `clause_is_reverse`, `n_items`, `n_queries`; SASRec encode cached as `<ckpt-dir>/encoded_queries_v2.pt` on the *full* split; `users_limit` applied once, as a prefix, after the eval_split row check), `sweep_qa(qa, clauses)`, `build_filters(kind, inputs, backends, bloom=…)` keyed by *filter* backend (`FILTER_BACKEND`, so `[triton, official]` builds one module), `exact_filter(...)` for the oracle, `query_pool(...)` (the fixed-seed perf pool, same draw as the old harness) | authored C2; the pre-encoded path is tested on a tmp fixture (`tests/test_data.py`), the checkpoint path runs in C4. Imports `encode.py` (the one `training.*` boundary); copies the arxiv numerics from `loaders.py` |
| [`retrieval/oracle.py`](../../evaluation/retrieval/oracle.py) | the exact filtered oracle as **blob v4** (`load_or_build(...) -> dict`, `compute`, `pass_counts`, `pass_rate`, `bloom_fp_rate`), the content fingerprint in the file name, `resume_key(key, code_version)` (+ `KEY_FIELDS`); `bench.code_version()` = `git rev-parse HEAD:retrieve/src/retrieve` or `files:<sha256>` outside git | rewritten in place; the old `compute_filtered_oracle` / `load_or_build_oracle` are wrappers returning `topk` (the old harness now reads/writes v4 blobs under its `gt_subdir`; its `gt_topk_v3_*` files are ignored and can be deleted) |

`tests/test_algos.py` parses the dispatch table in
[architecture.md](architecture.md#backend-dispatch) and asserts `PATHS`
agrees with it, and checks on CPU (`backend="torch"`) that no `retrieve`
layer bakes `k` into a buffer at `register_index` — `module.k = k'` after
registration returns the top-`k'` prefix of the top-`k` result with every
buffer untouched (H §7 first risk: nothing found; `SilverTorch` and the
1-bit KNNs only *validate* `k` at registration, and the wrapper's `k`
setter / `set_query_params` re-run SilverTorch's two checks).

### Config schema (C2, H §3.3 + §8.2 A/J)

Five files replace the 19 old YAMLs (which stay until C3):
[`config/goodreads.yaml`](../../evaluation/config/goodreads.yaml),
[`arxiv.yaml`](../../evaluation/config/arxiv.yaml),
[`yambda-500m.yaml`](../../evaluation/config/yambda-500m.yaml),
[`yambda-5b.yaml`](../../evaluation/config/yambda-5b.yaml) and
[`suites.yaml`](../../evaluation/config/suites.yaml). Their values are
the old configs' (`users_limit: 10000`, the same sweeps, ks, batch sizes,
`n_probe` grid) so C4 can compare against A1's golden cells.

```yaml
# config/<dataset>.yaml — one per dataset; every string may carry {dim}
data_dir: data/goodreads-work-id
checkpoint: data/goodreads-work-id/checkpoints/gsasrec-d{dim}-drop0.5-id/best_model.pt
#   or, for pre-encoded text datasets, a per-dim mapping instead of `checkpoint`:
#   content_dir: {64: content_d64, 128: content_d128, 256: content}   # relative to data_dir
dims: [64, 128, 256]
encode: {batch_size: 512, num_workers: 8, max_seq_length: 200}       # SASRec datasets only
users_limit: 10000                                                    # or null
filters:                                                              # optional
  attrs: item_attrs_narrow.pt                                         # relative to data_dir
  reverse: clause_is_reverse_narrow.pt
  clause: {c0_genre: [0], c1_lang_reverse: [1], all4: [0, 1, 2, 3]}   # name: active clauses
  bloom: {c0_genre: [0], old_sweep: {clauses: [2], disabled: true}}   # long form: disabled
```

`gt_dir` is derived (`<data_dir>/gt_d{dim}`); `output`, `split`, `device`,
`gt_subdir`, `query_emb_path` and `content_subdir` are gone. Unknown keys
raise `ConfigError` naming the file.

```yaml
# config/suites.yaml — a suite = cells run on every listed dataset × its dims
filter:
  datasets: [goodreads, arxiv]
  dims: [128]                       # optional; default: the dataset's dims
  filter_kinds: [clause, bloom]     # none | clause | bloom
  ks: [100, 500, 1000]
  batch_sizes: [1, 8, 16]
  algos: {linr_v2: [triton, torch], silvertorch: [triton, torch, official]}
  params:                           # per algo; dict-of-lists = grid, list-of-dicts = combos
    silvertorch: {build: {n_lists: [1664, 8192]}, query: {n_probe: [4, 8, 24, 32]}}
  seeds: {default: [0], headline: {sweeps: [c0_genre, all4], dims: [128], seeds: [0, 1, 2]}}
  bloom: {m_bits: 1024, k_hash: 5}  # optional per-suite override of the top-level default
bloom: {m_bits: 1024, k_hash: 5}
```

`build:` params rebuild the index; `query:` params (`n_probe`,
`candidate_pool` — the `QUERY_PARAMS` set) are applied with
`set_query_params` to the built index, so the `deep` suite above is two
k-means per `(dataset, sweep, seed)`, not eight. Putting a query param
under `build:` (or vice versa) is a `ConfigError`. `seeds:` is a list, or
the `default` / `headline` form (headline seeds apply to the named sweeps
at the named dims, whatever the filter kind). A backend whose `PATHS`
entry is `None` is skipped and one that runs the same path as an earlier
backend of the same algo and filter kind is collapsed into it (logged
once each); CLI narrows (`dims`, `algos`, `backends`, `filter_kinds`,
`sweeps`, `seeds`, `ks`, `batch_sizes`) filter the suite's lists *before*
that collapse.

`load_matrix` returns `Job`s grouped by `Job.group == (dataset, dim, algo,
backend)` — the campaign's process boundary (H §8.2 K) — in the order
`dim → algo → backend → filter_kind → sweep → build → seed`. A `Job` is one
build: `dataset, dim, suite, filter_kind, sweep, clauses, algo, backend,
path, build, query, ks, batch_sizes, seed, bloom, data: Dataset`;
`job.cells()` lists the `params = build | query` of each cell and
`job.key(params)` is the record's key block (`dataset, dim, suite,
filter_kind, sweep, algo, backend, params, seed`). `none` cells have
`sweep == "full_scan"` and `clauses is None`.

### Oracle blob v4 and the resume key (C2, H §2.2, §8.2 B/I)

`oracle.load_or_build(gt_dir, sweep, k_gt, item_embs=, queries=, targets=,
qa_sweep=, skip_mask=, clauses=, filter_mod=, device=)` returns one dict,
cached at `<gt_dir>/oracle_v4_<sweep>_<fingerprint[:16]>.pt`:

| key | value |
|---|---|
| `version` | `4` |
| `topk` | `[U, k_gt]` int64 exact filtered top-k (0-indexed, `-1` padded; skipped rows all `-1`) |
| `pass_counts` | `[U]` int64 items passing the exact mask; `-1` on skipped rows |
| `pass_rate` | mean of `pass_counts / n_items` over kept rows |
| `targets_in_filter` | `[U, T]` bool, target `t` of user `u` passes the mask |
| `target_in_filter` | `[U]` bool, any target passes (`n_queries_heldout` = its sum) |
| `n_items`, `n_queries`, `n_kept`, `k_gt`, `sweep`, `clauses` | shape of the build |
| `fingerprint` | sha256 over shapes, dtypes and a 64-row linspace sample of `item_embs`, `queries`, `targets`, `qa_sweep`, plus `clauses` and `k_gt` |
| `code_version`, `harness_commit`, `torch`, `created` | provenance |

The fingerprint is in the file name, so a stale blob is never read (a
different dim off the same `data_dir`, regenerated attrs, a retrained
checkpoint, another `users_limit` or clause set each produce a new file),
and the blob is a portable artifact for roadmap F4. The filter passed in
must be exact (`data.exact_filter`): bloom cells build a fresh
`ExactAttributeFilter` for it. Bloom pass rates are not cached —
`oracle.pass_counts(bloom_mod, qa_sweep, skip_mask, device=)` plus
`oracle.bloom_fp_rate(bloom_counts, blob["pass_counts"], n_items)` give
the record's `bloom_fp_rate` (mean per-query `(bloom − exact) / (N −
exact)`).

The resume key is `oracle.resume_key(job.key(params),
bench.code_version())`: canonical JSON of the key block plus the library
subtree's tree hash (`git rev-parse HEAD:retrieve/src/retrieve`;
`files:<sha256>` over the installed sources outside git). A kernel edit
therefore invalidates every cell; doc churn invalidates none. `run.py`
(C3) reconstructs a record's key as `resume_key({k: rec[k] for k in
oracle.KEY_FIELDS}, rec["env"]["code_version"])`.

Everything below this section describes the **old** harness, which is
what `uv run evaluate` still runs.

## Scope

[`evaluation/retrieval/cli/evaluate.py`](../../evaluation/retrieval/cli/evaluate.py)
is the **single-algo** retrieval benchmark CLI — one invocation runs
exactly one algorithm's cells against one YAML config and writes one
JSON file. The orchestrator
[`cli/run_evaluation.py`](../../evaluation/retrieval/cli/run_evaluation.py)
(`uv run run-evaluation`) reads the YAML's `algorithms` list and spawns
`uv run evaluate` once per algo, writing per-algo JSONs into
`cfg.output` (a **directory**). Downstream analysis reads them back via
[`retrieval.results_io.load_results`](../../evaluation/retrieval/results_io.py).

Per-algo process isolation is the point: every algo gets a fresh Python
interpreter, which wipes torch.compile / Triton JIT / CUDA-graph
private-pool state that `_release_algo` and `torch._dynamo.reset()`
can't reclaim between algos in the same process. The SASRec encode
pass is cached on disk by
[`queries_cache.py`](../../evaluation/retrieval/queries_cache.py)
(`<ckpt-dir>/encoded_queries_<split>.pt`, keyed on
`(ckpt mtime, max_seq_length, users_limit)` — `users_limit` is applied
**before** caching so the cache scales with the limited user set), so
the N subprocesses don't pay N × encode cost.

The cache write is guarded by a free-disk budget: if the estimated blob
would exceed `0.7 × free − 4 GiB`, caching is **skipped silently** and
every subsequent algo subprocess re-encodes from scratch (minutes each).
A campaign that feels inexplicably slow on a full disk is usually this.

The driver dispatches by config shape across three datasets:

| Config shape                     | Mode                                         |
|----------------------------------|----------------------------------------------|
| `checkpoint` set                 | Encode queries via SASRec (yambda/goodreads) |
| `checkpoint` unset (`null`)      | Load pre-encoded text embeddings (arxiv)     |
| `filters: null`                  | No filter loop (yambda)                      |
| `filters: {clause, bloom}` block | Filter sweeps (goodreads, arxiv)             |

The query-source branch keys on **`checkpoint` alone**
([`queries_cache.load_or_cache_queries`](../../evaluation/retrieval/queries_cache.py)).
`query_emb_path` only names *where* the pre-encoded tensor lives on the
arxiv path; it does not select the path, and no shipped config sets it.
Setting both `checkpoint` and `query_emb_path` is not a supported
combination — the cache layer would take the SASRec branch and
`query_emb_path` would be ignored.

For each `(filter_kind, sweep, algo, backend, params, K, batch_size)`
cell:

1. Builds the index on top of the dataset's item embeddings.
2. Streams every query through the index in chunks
   (`QUALITY_BATCH_SIZE = 16`) to compute recall/ndcg/precision/mrr@K
   (the **quality pass**) — vs the held-out test targets on
   `filter_kind=none`, vs the cached filtered-FullScan top-K oracle on
   filtered cells.
3. Times the index forward at the configured batch size against a
   fixed-seed pool of cached queries (the **perf pass**, `n_pool = 4096`),
   capturing median / p20 / p80 latency plus peak / transient GPU memory.
4. Writes one row per cell to the per-algo output JSON.

Cell eligibility is declarative (`SUPPORTED_FILTER_KINDS` +
`supports(algo, filter_kind)` checked before building); anything that
*does* raise during construction is a genuine error and kills the run
loudly — silently-vanishing cells were a real failure mode and are gone.
Correctness of the library itself lives in
[`retrieve/tests/`](../../retrieve/tests/) and gates CI separately;
the harness's own unit tests live in
[`evaluation/retrieval/tests/`](../../evaluation/retrieval/tests/) — run
with `cd evaluation && uv run pytest retrieval/tests/ -v`. Four of the
five files (metrics, sweep helpers, oracle fingerprint, config loader)
are CPU-only; `test_silvertorch_algo_reverse.py` allocates on `cuda` and
fails rather than skips on a CPU box, so deselect it there
(`--ignore=retrieval/tests/test_silvertorch_algo_reverse.py`).

## Files

```
evaluation/retrieval/
├── cli/
│   ├── evaluate.py            # single-algo CLI: --config X --algo Y --output <file>
│   ├── run_evaluation.py      # orchestrator: subprocess per (config, algo); --eval-type presets; tee'd logs
│   ├── stage_results.py       # per-algo JSONs → combined <name>.json + .perkernel/ + .yaml
│   └── upload_results.py      # staged results/ → HF dataset repo (--repo-id required)
├── context.py                 # SweepContext / FilterAssets frozen dataclasses
├── sweep.py                   # loop driver: run_sweep → run_filter_kind → run_one_sweep → evaluate_cell
├── loaders.py                 # embedding/attr loaders, apply_users_limit, build_sweep_qa, resolve_path
├── oracle.py                  # filtered-FullScan ground truth + content-fingerprint disk cache
├── queries_cache.py           # disk cache for the SASRec encode pass
├── measure.py                 # PerfStats + CUDA timing/memory primitives (imports torch + triton.testing ONLY)
├── encode.py                  # SASRec model load + query encode (the ONE file that imports training.*)
├── passes.py                  # QualityStats + quality_pass_cached / perf_pass_cached
├── metrics.py                 # recall / precision / mrr / ndcg accumulators
├── results_io.py              # load_rows(file) / load_results(dir)
├── config.py                  # EvalConfig dataclass + load_raw_config / load_eval_config
├── algos/                     # one class per algo, all AlgoBase subclasses
│   ├── __init__.py            # ALGORITHMS + SUPPORTED_FILTER_KINDS + build_algorithm factory
│   ├── _helpers.py            # RetrievalAlgo protocol + AlgoBase (the one torch.compile call)
│   ├── filter.py              # build_filter (clause/bloom → FilterModule) + make_mask
│   ├── linr_v1.py             # LinrV1Algo — covers triton_knn + linr_v1_filter_mask
│   ├── linr_v2.py             # LinrV2Algo — exact filtered top-K via PrefilterKNN
│   ├── linr_v3.py             # LinrV3Algo — V3 → V2 cascade (1-bit prefilter, fp32 rerank)
│   ├── linr_v4.py             # LinrV4Algo — single-stage int8 dense (PostfilterKNNInt8)
│   └── silvertorch.py         # SilvertorchAlgo — IVF + INT8 + (none / bloom / exact) filter_mode
└── tests/                     # harness unit tests (all CPU-only except test_silvertorch_algo_reverse.py)
    ├── test_config.py
    ├── test_metrics.py
    ├── test_oracle.py
    ├── test_silvertorch_algo_reverse.py
    └── test_sweep_helpers.py

evaluation/eval_datasets/      # dataset ETL package (renamed from `datasets` —
│                              # the old name shadowed HuggingFace `datasets` in the venv)
├── yambda.py / goodreads.py / arxiv.py / synth_arxiv.py
├── common.py / constants.py / timesplit.py
└── hf_io.py                   # HF fetch/publish helpers + console scripts

evaluation/config/
├── yambda-500m/d{64,128,256}-quality.yaml
├── yambda-5b/d{64,128}-quality.yaml
├── arxiv/d{64,128,256}-{quality,filter}.yaml
├── goodreads/d{64,128,256}-{quality,filter}.yaml
└── deep_sweeps/{arxiv-d128-silvertorch,goodreads-d128-linr_v3}.yaml
```

Console scripts ([`evaluation/pyproject.toml`](../../evaluation/pyproject.toml)):
`evaluate`, `run-evaluation`, `stage-results`, `upload-results`,
`yambda`, `arxiv`, `goodreads`, `upload-checkpoints`, `eval-fetch`,
`eval-publish`, `eval-publish-checkpoint`.

### Driver flow

`cli/evaluate.py::main` → seeds + TF32 off +
`torch._dynamo.config.recompile_limit = 64` →
`queries_cache.load_or_cache_queries` (SASRec encode or arxiv
pre-encoded load) → `loaders.load_query_attrs` (filter configs only) →
`loaders.apply_users_limit` (idempotent no-op on the cache path, the
real trim on arxiv) → build a `SweepContext` → `sweep.run_sweep(ctx)` →
JSON dump to `--output`.

Run-wide inputs travel in a frozen
[`SweepContext`](../../evaluation/retrieval/context.py) (config,
post-limit query/target tensors, device, `gt_dir`, `k_gt = max(ks)`,
suite, backends, CLI narrows); per-filter_kind filter modules and
attrs travel in a frozen `FilterAssets`, whose per-sweep fields
(`qa_n_sweep`, `skip_mask`, `oracle_topk`, `n_kept`) are stamped by
`run_one_sweep` via `assets.for_sweep(...)`. Adding a new run-wide
input = one field in `context.py`, visible at every loop level.

**Loop nest** ([`sweep.py`](../../evaluation/retrieval/sweep.py)):

```
run_sweep(ctx)                       # pin TF32/matmul precision, warm GPU once
└─ run_filter_kind(kind, fcfg, ctx)  # build one FilterModule per backend + oracle filter
   └─ run_one_sweep(sweep, ...)      # build_sweep_qa → load_or_build_oracle → algo/backend/params/k loops
      └─ evaluate_cell(...)          # reset CUDA state → build algo → quality → prewarm → per-bs perf rows
```

Yambda (`filters: null`) iterates a single synthetic
`("none", full_scan)` cell so the loop body stays uniform. An unknown
`filter_kind` key in the YAML raises at `_select_filter_iter` (it would
otherwise silently fail every `supports` check). Between sweeps the
driver calls `torch._dynamo.reset()` + `empty_cache()` — compiled
graphs from earlier sweeps otherwise stay pinned and OOM large-N
configs.

## Configuration

Configs are YAML, parsed with `yaml.safe_load` into the
[`EvalConfig`](../../evaluation/retrieval/config.py) dataclass.
Each file declares a `_defaults: &defaults` anchor block, merges it at
the top level with `<<: *defaults`, and adds dataset-specific fields
(`checkpoint:` for SASRec datasets, `query_emb_path:` for arxiv).
`load_raw_config` strips top-level keys starting with `_` (the one
place that rule lives — the orchestrator and `stage-results` reuse it).

Example yambda config (`config/yambda-500m/d128-quality.yaml`):

```yaml
_defaults: &defaults
  data_dir: data/yambda-500m
  output: results/yambda/500m-d128
  split: test
  device: cuda
  ks: [100, 200, 400]
  batch_sizes: [1]
  seed: 0
  encode:
    batch_size: 512
    num_workers: 8
    max_seq_length: 200
  algorithms:
    - linr_v1_filter_mask
    - linr_v4
    - silvertorch
    - linr_v3
  backends: [triton, torch]

<<: *defaults
checkpoint: data/yambda-500m/checkpoints/gsasrec-d128-drop0.5/best_model.pt
```

`algo_params[algo]` is a **list of dicts** — each dict is one explicit
combo, taken as-is. Algos with no entry iterate over `[{}]` so the
loop stays uniform. The cross-product is only over `(algos, backends,
combos, ks, batch_sizes)`; `is_valid_combo` drops combos the library
would assert on (`n_probe > n_lists`).

Filter-bench configs (`goodreads/*-filter.yaml`, `arxiv/*-filter.yaml`)
add a `filters:` block declaring `clause` and `bloom` filter kinds,
each with attribute paths and named sweeps (`active_clauses` subsets);
see [filtering.md](filtering.md) for the filter API.

### Field reference

| Field | Type | Purpose |
|---|---|---|
| `checkpoint` | path | SASRec `best_model.pt`. Required for yambda/goodreads, omit for arxiv. The trainer writes a sibling `config.json` which the loader reads for hyperparams; legacy 500M ckpts without one fall back to `D128_DROP05_DEFAULTS` in [`encode.py`](../../evaluation/retrieval/encode.py). |
| `query_emb_path` | path or `null` | Override for the pre-encoded query tensor location on the arxiv path. Defaults to `<data_dir>/<content_subdir>/query_emb.pt`. No shipped config sets it; it does **not** select the pre-encoded path (`checkpoint` does). |
| `data_dir` | path | Holds `item_id_map.json`, `<split>.parquet`, optional `eval_split.parquet` + filter attrs. |
| `content_subdir` | str | Subdir under `data_dir` for `{text_emb,query_emb}.pt` + meta sidecars (arxiv only). Default `content`; arxiv ships `content_d64`, `content_d128`, `content` (= d=256). Sharded synth catalogs (`shard_index.json`) are reassembled on load. |
| `gt_subdir` | str | Subdir under `data_dir` for the oracle caches (since C2: the v4 blobs `oracle_v4_<sweep>_<hash>.pt` described in the harness-v2 section; older `gt_topk_v3_*` files are ignored). Default `gt`. Varying it per dim keeps caches tidy; **correctness no longer depends on it** — the fingerprint is in the file name. |
| `output` | dir path | Directory for the per-algo JSONs. The orchestrator requires it to be set (a `output: null` config exits with a clear error rather than a bare TypeError). |
| `split` | str | `test` (default) or `val`. Inert on the arxiv path (the held-out set is fixed at `heldout.parquet`); it selects the parquet on SASRec paths and is part of the encode-cache filename. |
| `device` | str | `cuda`. The only supported value — output rows hard-code `"device": "cuda"` regardless of what is set here, so a `cpu` config produces mislabelled rows rather than a CPU run. |
| `ks` | list[int] | K-cutoffs. Each emits `recall@K`, `ndcg@K`, `precision@K`, `mrr@K` on its own row. `max(ks)` is the oracle depth `K_GT`. |
| `batch_sizes` | list[int] | Perf-pass batch sizes. Quality is invariant to bs and computed once per `(algo, params, k)`; emitted on every bs row. |
| `seed` | int | Drives `torch.manual_seed`, `torch.cuda.manual_seed_all`, the perf-query-pool generator, and any algo seeds in `algo_params`. |
| `encode.batch_size` / `num_workers` / `max_seq_length` | | SASRec encode-pass knobs; `max_seq_length` must match the trained checkpoint (it is part of the cache key). |
| `algorithms` | list[str] | Subset of [`ALGORITHMS`](../../evaluation/retrieval/algos/__init__.py). |
| `algo_params` | dict[str, list[dict]] | Per-algo parameter combos; no implicit cross-product. |
| `backends` | list of `triton` / `torch` / `cuda` / `cute` | Backend fan-out: each backend adds a row per cell with a `backend` column (default `[triton]`). `cuda` (and `cute`, its CuTe DSL port) only changes what runs for `silvertorch` — see [The `cuda` backend in sweeps](#the-cuda-backend-in-sweeps). |
| `filters` | dict or `null` | Optional filter-bench block. See [filtering.md](filtering.md). |
| `users_limit` | int or `null` | Optional cap on users for ALL cells. Goodreads has 313k test users; cap to e.g. 10000 to speed runs up. Part of the encode-cache key. |

Two CUDA settings the driver pins at startup, in addition to seeds:
`torch.backends.cuda.matmul.allow_tf32 = False` and
`torch.backends.cudnn.allow_tf32 = False` (re-pinned with
`torch.set_float32_matmul_precision("highest")` in
`pin_precision_globals`) — keeps the oracle (cuBLAS `q @ E_t`) and the
algos in the same precision so exact-mask paths don't show ~1e-3 recall
drift on narrow filters.

### CLI overrides

```
uv run evaluate \
    --config config/<dataset>/<name>.yaml \
    --algo <name> \
    --output <output-dir>/<algo>.json \
    [--filter-kind <none|clause|bloom> ...] \
    [--backend {triton|torch|cuda|cute} ...] \
    [--sweep <sweep_name>] \
    [--skip-quality]
```

`--algo` is required and runs a single algorithm — the orchestrator is
what loops over the YAML's `algorithms` list. `--filter-kind` and
`--backend` are repeatable and narrow the run; empty = all configured.
`--backend` is a `click.Choice` and rejects unknown values;
`--filter-kind` is **not** validated — a typo'd kind silently matches
no configured filter and the run writes an empty JSON.
`--sweep` narrows to a single sweep — useful for iterating on one cell
without re-encoding queries. `--skip-quality` drops the quality stream
(quality columns → NaN; the oracle is neither loaded nor built); perf
rows still emit.

### The orchestrator

```
uv run run-evaluation --eval-type {filter|quality|param-sweeps} [--resume|--force]
uv run run-evaluation CONFIG [CONFIG ...] [--stage] [--resume|--force]
                      [--logdir PATH]
                      [-- extra args forwarded to evaluate]
```

`--eval-type` presets expand to the config lists in `EVAL_TYPES`
(`filter` → the six arxiv/goodreads filter YAMLs, **no** staging;
`quality` → the six quality YAMLs + staging; `param-sweeps` → the two
deep-sweep YAMLs + staging). The `yambda-500m` / `yambda-5b` configs are
not covered by any preset — pass them positionally. Positional configs
accept globs; `--logdir` relocates the run-log tree. `--resume` skips any
`(config, algo)` whose output JSON exists and parses as a non-empty
list; `--force` overwrites; with neither, existing outputs abort the
run up front. Everything after `--` is forwarded verbatim to each
`evaluate` subprocess (e.g. `-- --skip-quality --sweep c0_genre`).

Output is tee'd to `results/_runlogs/`: `full.log` (everything),
`SUMMARY.txt` (per-config exit/duration lines + host/GPU header),
`<cfg_name>.log` per config, and a `current.log` symlink while running.

`stage-results <config>` restructures one config's finished outputs
into the campaign layout: concatenates per-algo JSONs into
`<output>.json`, renames the directory to `<output>.perkernel/`, and
copies the YAML to `<output>.yaml` (idempotent). `upload-results
--repo-id <user/repo> [--notes-file notes.md] [--private|--public]
[--dry-run]` mirrors the staged `results/` layout to a HF dataset repo
— the repo id is a required flag, and campaign-specific prose belongs
in the notes file, not in code.

## Algorithms

Six algorithm names are registered (one duplicate alias: `triton_knn` /
`linr_v1_filter_mask` share `LinrV1Algo`). The classes live in
[`evaluation/retrieval/algos/`](../../evaluation/retrieval/algos/),
one per file, all subclassing
[`AlgoBase`](../../evaluation/retrieval/algos/_helpers.py) and
structurally typed by the `RetrievalAlgo` protocol:

```
algo.algo_modules: list[nn.Module]            # for per-cell memory cleanup
algo(q, qa_narrow=None) -> (ids, scores)      # via Module.__call__
```

Each algo's `__init__` builds its `retrieve` layers and ends with
`self._finalize(*modules, filter_mod=...)` — the base wires the
`algo_modules` cleanup list and applies the one canonical
`self.compile(dynamic=True, mode="reduce-overhead")` call, so the whole
forward (filter + index + cascade) becomes a single cudagraph capture.

Eligibility is a table, not exception control flow:

```python
SUPPORTED_FILTER_KINDS = {
    "triton_knn":          {"none", "clause", "bloom"},
    "linr_v1_filter_mask": {"none", "clause", "bloom"},
    "linr_v2":             {"clause", "bloom"},   # candidate source IS the filter
    "linr_v3":             {"none", "clause", "bloom"},
    "linr_v4":             {"none", "clause", "bloom"},
    "silvertorch":         {"none", "clause", "bloom"},
}
```

`run_one_sweep` checks `supports(algo, filter_kind)` and logs one info
line per skip. `build_algorithm(name, item_embs, k, *, filter_kind,
filter_mod, item_attrs_narrow, clause_is_reverse, params, backend)` is
the single factory; any exception it raises is a genuine construction
error and **propagates** (the orchestrator records rc≠0; `--resume`
continues after a fix).

| Name | Class / file | Notes |
|---|---|---|
| `triton_knn` | [`LinrV1Algo`](../../evaluation/retrieval/algos/linr_v1.py) | Pure-torch full-scan KNN via `PostfilterKNN` (the original Triton kernel was removed; both backends run the same `query @ x.T + topk` path). Yambda-config alias. |
| `linr_v1_filter_mask` | [`LinrV1Algo`](../../evaluation/retrieval/algos/linr_v1.py) | Same class as `triton_knn`; canonical name on filter cells (exact mask baseline). |
| `linr_v2` | [`LinrV2Algo`](../../evaluation/retrieval/algos/linr_v2.py) | Exact filtered top-K — the candidate set IS the filter (`filter_mod.evaluate_indices` → `PrefilterKNN`). Recall=1.0 by construction; headline is speed/memory. Filter cells only. |
| `linr_v3` | [`LinrV3Algo`](../../evaluation/retrieval/algos/linr_v3.py) | V3 → V2 cascade: `OneBitKNN(backend=...)` produces top-`candidate_pool` at 1-bit precision; `PrefilterKNN` rescores at full precision. Approximate. Params: `candidate_pool`, `v3_seed`. |
| `linr_v4` | [`LinrV4Algo`](../../evaluation/retrieval/algos/linr_v4.py) | Single-stage int8 dense + optional mask via `PostfilterKNNInt8` (`torch._int_mm`, int8×int8 → int32, IMMA on Ampere+). |
| `silvertorch` | [`SilvertorchAlgo`](../../evaluation/retrieval/algos/silvertorch.py) | IVF + INT8 ANN with `SilverTorch.filter_mode` mapped from `filter_kind`: `"bloom"` → bloom-fused `codesigned_probe_score`; `"clause"` → exact-AND-of-OR-fused `codesigned_probe_score_exact` (accepts `clause_is_reverse` — reverse-clause sweeps run end-to-end); `"none"` → plain IVF + INT8. Params: `n_lists`, `n_probe`, `n_iter`, `m_bits`, `k_hash`, `seed`. |

`make_mask(filter_mod, qa_narrow)` (in
[`algos/filter.py`](../../evaluation/retrieval/algos/filter.py)) is the
one-line helper the mask-based algos call: it routes per-batch query
attrs through `filter_mod.evaluate_mask` and returns `None` when either
input is `None` (unfiltered cell or pure-IVF batch). No item-0 fixup is
needed — the training-side padding row is dropped at the loaders
boundary, so the library is 0-indexed over real items throughout.

### The `cuda` backend in sweeps

`backends: [..., cuda]` is not a uniform third measurement. Two things
about it are easy to misread (everything in this section applies verbatim
to `cute`, the CuTe DSL port of the cuda backend: the sweep treats the two
identically):

**Only `silvertorch` actually changes.** `backend="cuda"` selects a real
CUDA C++ path inside `SilverTorch` only. Every other layer dispatches
`if backend == "triton": … else: <torch>`, so a `cuda` row for
`linr_v1`/`v2`/`v3`/`v4` is a **torch-path measurement labelled
`backend: "cuda"`**. Do not read those rows as a CUDA-vs-Triton
comparison. See
[architecture.md](architecture.md#backend-dispatch) for the dispatch
table.

**Filter modules are built with Triton on `cuda` cells.** There are no
CUDA C++ filter kernels — only SilverTorch's fused probe scoring exists
in C++. So `_build_filter_modules` maps `cuda → triton` when
constructing the standalone `FilterModule`s, keeping the filter kernel
GPU-fast instead of silently dropping to torch. Consequence worth
knowing when reading memory columns: a config with
`backends: [triton, cuda, torch]` builds **two identical Triton filter
indexes** (one keyed `"triton"`, one keyed `"cuda"`), so filter-index
GPU memory for the run is doubled.

The oracle filter is always exact and always Triton-or-first-backend for
the same reason; bloom's false positives must never leak into ground
truth.

**`silvertorch` + `clause` + `cuda` now builds.** It used to be a
footgun: `SUPPORTED_FILTER_KINDS` is backend-blind and `SilverTorch`
rejected `filter_mode="exact"` under `backend="cuda"` at construction, so
a clause sweep with `--backend cuda` killed the run. The cuda backend has
a clause-mask kernel now, so clause cells build and run on it like any
other. The shipped
[`config/deep_sweeps/arxiv-d128-silvertorch.yaml`](../../evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml)
is still bloom-only; add cuda to a clause config only after the
correctness gates in
[cuda-silvertorch-handoff.md](../plans/cuda-silvertorch-handoff.md) §5
have passed on the target box.

## Measurement methodology

### Quality pass

[`quality_pass_cached`](../../evaluation/retrieval/passes.py)
streams every cached query through the index in chunks of
`QUALITY_BATCH_SIZE = 16` and accumulates per-row metrics via
[`accumulate_metrics`](../../evaluation/retrieval/metrics.py),
returning a `QualityStats(recall, ndcg, precision, mrr)`. Quality is
invariant to perf batch size (same per-row scoring math), so it runs
**once per `(algo, params, k)` cell** and attaches to every `bs` row —
the chunk amortises CUDA-launch + Python overhead vs a bs=1 stream.
The constant is 16 (not 64) because `PrefilterKNN[backend="torch"]` on
arxiv/d256 materialises `[B, P, D]` fp32 ≈ 110 GiB at bs=64 on loose
clause filters and OOMs an 80 GB card; bs=16 matches the perf-pass
allocation shape that already fits. Rows whose synthesised qa is all
`-1` for the active clauses are dropped via `skip_mask` (reported as
`n_users_kept`).

For filtered cells the targets are the cached filtered-FullScan top-K
oracle; for `filter_kind=none` and yambda they are the held-out test
items. Recall's denominator is the per-row count of *valid* oracle
targets (`(oracle_row != -1).sum().clamp(max=k)`), not a flat `k` —
tight filters can pass fewer than K items and the oracle pads with
`-1`; zero-target rows fold into the skip mask.

### Oracle

The oracle is built by
[`compute_filtered_oracle`](../../evaluation/retrieval/oracle.py) on
first run: brute-force `q @ E_t` with the *exact* filter mask
(`ExactAttributeFilter` even on bloom cells — bloom's false positives
must not leak into ground truth), then top-`K_GT` with `-inf` ties
mapped back to `-1` so short-fill rows don't score against the algos'
`-1` sentinel. Indices are 0-indexed positions in the (already
pad-row-dropped) `item_embs`.

Disk cache: since C2 `load_or_build_oracle` is a wrapper over the v4
`oracle.load_or_build` and returns its `topk`; the blob lives at
`<data_dir>/<gt_subdir>/oracle_v4_<sweep>_<fingerprint[:16]>.pt` with
the content **fingerprint** (shapes + dtypes + a fixed 64-row linspace
sample of `item_embs`, `queries` and the sweep's `qa_narrow`, plus
`K_GT`) in the file name — so same-shape content changes (a different dim
off the same `data_dir`, regenerated attrs, a retrained checkpoint, a
changed `users_limit`) produce a new file instead of silently reusing
stale ground truth (the failure mode behind the goodreads stale-cache
incident). The old `gt_topk_v3_*` files are ignored and can be deleted.

### Perf pass

[`perf_pass_cached`](../../evaluation/retrieval/passes.py) measures the
index forward in isolation via
[`measure_forward_cuda`](../../evaluation/retrieval/measure.py): it
runs `WARMUP_ITERS = 20` calls to settle JIT compile + cudagraph
capture, takes a clean `max_memory_allocated()` window over
`MEM_REPS = 16` calls (without `do_bench`'s ~256 MiB L2-buster
polluting peak; outputs are kept alive so the window includes
result + scratch), then times via `triton.testing.do_bench` starting
at `rep = 200 ms` and **auto-extending** the budget until at least
`MIN_SAMPLES = 30` iterations fit (capped at `MAX_REP_MS = 3000`) —
otherwise slow kernels collapse to a single cold sample with
`median == p20 == p80`. Returns a `PerfStats(median_ms, p20_ms, p80_ms,
peak_mib, transient_mib)`. The dataclass is the extension point: new
statistics are added as fields and picked up by `_make_perf_row`
without changing the measurement path.

Before timing, `_autotune_prewarm` calls the forward once per batch
size: the kernels ship offline-tuned `DEFAULT_CONFIG`s (no runtime
autotune), but `reduce-overhead` still compiles + captures a cudagraph
per batch-size shape, and that cold cost has been observed leaking into
the first cell's timing window.

### Multi-query pool — why p20/p80 are over queries

The perf pass times against a **fixed-seed pool of `n_pool = 4096`
query batches**, round-robin'd into the timed function. This matters
for IVF-style algorithms (`silvertorch`) where a single fixed query
collapses p20/p80 to one cluster's traversal cost. The pool is sampled
with `torch.Generator(cpu).manual_seed(cfg.seed)` so the same query
indices recur across reruns; on filter cells, `qa_narrow` batches are
sampled from the same row indices, and `skip_mask` rows are excluded
from the pool.

### Memory snapshot ordering

Reading allocator state around `build_algorithm(...)` is brittle;
per-cell ordering is:

```
sync → empty_cache → reset_peak_memory_stats → mem_before
build_algorithm
sync → index_mem = allocated() - mem_before
quality_pass
autotune_prewarm
for bs in batch_sizes: perf_pass
algo_obj.algo_modules.clear(); empty_cache      # _release_algo
```

`algo_modules.clear()` is required — dropping the loop variable alone
leaves the list's references alive and leaks the index into the next
cell's `mem_before`.

### Determinism

`main()` seeds torch + CUDA from `cfg.seed` before any allocation; the
perf-pool generator uses the same seed; algo seeds thread through
`algo_params` (`silvertorch.seed`, `linr_v3.v3_seed`). TF32 is pinned
off. Quality columns must be **byte-identical** across reruns with the
same seed; latency may drift within ~5%.

## Output schema

One row per `(filter_kind, sweep, algo, backend, params, k, bs)` cell.
Column names are load-bearing for downstream joins — existing fields
are never renamed; new fields are additive.

| Field | Type | Notes |
|---|---|---|
| `suite` | str | `yambda` (no filters) or `filter`. |
| `cell` | str | `<filter_kind>_<sweep>_<backend>_bs<N>_k<K>` for cross-row joins (the backend segment was added with the backend fan-out). |
| `filter_kind` | str | `none`, `clause`, or `bloom`. |
| `sweep` | str | Filter sweep name from the config (e.g. `c0_genre`), or `full_scan` on yambda. |
| `impl` | str | Algorithm name. |
| `backend` | str | `triton`, `torch`, `cuda` or `cute` — the `retrieve`-layer backend *requested* for this row. For anything but `silvertorch`, a `cuda` / `cute` row ran the torch path; see [The `cuda` backend in sweeps](#the-cuda-backend-in-sweeps). |
| `device` | str | Always `"cuda"` (the CPU-timing path was removed; the column stays for schema stability). |
| `seed` | int | The `cfg.seed` that produced this row. |
| `batch_size`, `k` | int | First-class columns. |
| `n_users_kept` | int | Users this sweep evaluated (skip mask drops users with no surviving narrow clauses). |
| `median_ms`, `p20_ms`, `p80_ms` | float | Latency over the multi-query pool. |
| `peak_mem_mib` | float | `max_memory_allocated()` over the perf window. |
| `index_mem_mib` | float | `allocated()` delta around `build_algorithm`. |
| `fwd_scratch_mib` | float | Peak − baseline within the forward call. |
| `recall@<k>`, `ndcg@<k>` | float | Quality, identical across the bs rows of one `(algo, params, k)` cell. NaN when `--skip-quality`. |
| `precision@<k>`, `mrr@<k>` | float | Additive quality columns (the metrics were always computed; now emitted). |
| `extra.params` | dict[str, str] | Per-algo `algo_params` for traceability. |
| `extra.gpu`, `extra.torch`, `extra.commit` | str | Additive provenance columns (GPU name, torch version, short git commit) for cross-machine result pooling. |

Cells whose `(algo, filter_kind)` pair is unsupported per
`SUPPORTED_FILTER_KINDS` are skipped up front with a log line — no stub
row. Construction failures no longer skip silently; they kill the run.

> **Checked-in results predate part of this schema.** The files under
> [`evaluation/results/`](../../evaluation/results/) were produced before
> `precision@k`, `mrr@k`, and the `extra.gpu` / `extra.torch` /
> `extra.commit` provenance columns were emitted; their rows carry only
> `recall@k` / `ndcg@k` and `extra: {params: …}`. Code that joins on the
> full schema must tolerate missing keys, or the campaign needs a rerun.
> `extra.commit` is best-effort: `git rev-parse` run from the harness
> directory, falling back to `"unknown"`.

## How to run

The repo is a uv workspace ([root pyproject](../../pyproject.toml));
`evaluation/` shares a single `.venv` with `retrieve/` at the workspace
root. `uv run` from inside `evaluation/` discovers the workspace root
automatically.

Named campaigns:

```bash
cd evaluation
uv run run-evaluation --eval-type filter --resume
uv run run-evaluation --eval-type quality --resume       # + stage-results per config
uv run run-evaluation --eval-type param-sweeps --resume
```

One config:

```bash
uv run run-evaluation config/goodreads/d128-filter.yaml --resume
```

One filter cell only (fastest iteration):

```bash
uv run evaluate \
    --config config/goodreads/d128-filter.yaml \
    --algo linr_v3 \
    --output results/goodreads/d128-filter/linr_v3.json \
    --filter-kind clause --sweep c0_genre
```

### Sanity checks to run after a sweep

1. GPU `median_ms(bs=8)` < `8 × median_ms(bs=1)` (sub-linear scaling —
   the proof point of the GPU implementations).
2. Quality columns identical across the bs rows of the same
   `(algo, params, k)` cell.
3. Two reruns with the same seed: quality columns byte-identical;
   latency columns within ~5%.

## Extending

### Add a new algorithm

1. Add `evaluation/retrieval/algos/<name>.py` with an `AlgoBase`
   subclass: build your `retrieve` layers in `__init__`, then call
   `self._finalize(*modules, filter_mod=...)` **as the last statement**
   — it wires `algo_modules` and applies the canonical
   `torch.compile(dynamic=True, mode="reduce-overhead")`. Keep
   `forward(q, qa_narrow=None) -> (ids, scores)` in the file — the
   per-algo forwards are the readable specification of each cascade.
2. Register it in
   [`algos/__init__.py`](../../evaluation/retrieval/algos/__init__.py):
   import, add to `ALGORITHMS`, add its filter-kind set to
   `SUPPORTED_FILTER_KINDS`, and add a `build_algorithm` branch that
   unpacks the params you need. `backend` should thread through to the
   underlying `retrieve` module (every current algo does).
3. Add it to the `algorithms:` list in any config that should sweep it,
   plus an `algo_params` entry if it takes knobs.
4. No driver changes — the schema columns are written uniformly.

### Add a new config (new checkpoint)

Copy an existing YAML, update `checkpoint:` (or `query_emb_path:`) and
any catalog-size-driven knobs (`silvertorch.n_lists/n_probe`). The
model loader reads hyperparams from `<ckpt-dir>/config.json`, so the
YAML never carries `embedding_dim` etc. (legacy 500M checkpoints
without one get the `D128_DROP05_DEFAULTS` fallback in
[`encode.py`](../../evaluation/retrieval/encode.py)).

### Add a new dataset

The driver dispatches on config shape, not a `dataset:` field, so a new
dataset is a question of producing the on-disk artifacts. The dataset
CLIs live in [`evaluation/eval_datasets/`](../../evaluation/eval_datasets/)
and produce, for the yambda layout:

```
data/<dataset>/
├── item_id_map.json
├── train.parquet
├── val.parquet
└── test.parquet
```

For the arxiv layout (no SASRec, pre-encoded text):

```
data/<dataset>/
├── item_id_map.json
├── papers.parquet
├── heldout.parquet
├── content/                   # default; varies via cfg.content_subdir
│   ├── text_emb.pt            # item-side, "search_document: " prefix
│   ├── text_emb.meta.json     # prefix drift is asserted at load
│   ├── query_emb.pt           # query-side, "search_query: " prefix
│   └── query_emb.meta.json
└── eval_split.parquet         # optional — only needed for filter sweeps
```

For filter sweeps either layout adds:

```
data/<dataset>/
├── item_attrs_narrow.pt
├── item_attrs_wide.pt           # DEAD: never loaded by load_filter_assets; wide-shelf query columns are explicitly skipped
├── clause_is_reverse_narrow.pt
├── eval_split.parquet
├── ... (per-clause vocab JSONs)
└── <gt_subdir>/                 # auto-built by the driver, oracle cache
```

Attr/reverse paths in the YAML resolve cwd-relative first, then
`data_dir`-relative — a missing path **raises** listing both candidates
(the old silent basename fallback that could load the wrong same-named
tensor is gone).

The CLIs are console scripts: `uv run yambda prep ...`,
`uv run arxiv all ...`, `uv run goodreads all ...`.

### HuggingFace I/O

[`eval_datasets/hf_io.py`](../../evaluation/eval_datasets/hf_io.py) is
the single source of truth for HF reads / writes: `EVAL_REPOS` maps
each dataset to its `pinkmeme/eval-<dataset>` HF dataset repo (eval
inputs + `checkpoints/<ckpt-id>/`), `RAW_REPOS` maps upstream raw
sources into `data/_raw/<source>/`.

```bash
uv run eval-fetch yambda-500m            # pull eval inputs to data/yambda-500m/
uv run eval-fetch arxiv-papers --dims d64,d128 --include-checkpoints
uv run eval-publish goodreads-work-id --dry-run
uv run eval-publish-checkpoint yambda-500m gsasrec-d128-drop0.5 --dry-run
```

The local data root resolves to `evaluation/data/` by default; override
with `RETRIEVE_DATA_ROOT=/some/path`.

## See also

- [architecture.md](architecture.md) — package layout and the
  `RetrievalModule` / `FilterModule` contracts.
- [kernels.md](kernels.md) — Triton and CUDA C++ kernel internals.
- [filtering.md](filtering.md) — filter API and the with-filters story.
- [datasets.md](datasets.md) — the ETL that produces the on-disk layout
  read here, and the SASRec training pipeline.
- [checkpoints.md](checkpoints.md) — the trained checkpoint inventory
  and the HF Hub workflow.
- [testing.md](testing.md) — the library correctness suite (separate
  from this perf harness).
