# Chapter 5. Протокол оценки (Evaluation Protocol) — Reference Notes

> **Working document.** English-language reference notes for the Russian-prose pass of Chapter 5 of the HSE MSc thesis. The chapter documents the in-process retrieval benchmark used to compare algorithm implementations in the open-source `torchretrieve` package against three datasets (Goodreads, arXiv, Yambda). **The harness described here is the infrastructure that delivers goal 2 of the thesis — the comprehensive empirical evaluation reported in Ch.6.** The Citation Policy from [00-thesis-plan.md](00-thesis-plan.md) is in effect: no Meta-affiliated work is cited. The in-repo symbol `silvertorch` (file names, class names, algorithm-name string in YAML) is retained as a legacy code-internal symbol; body prose always says "the co-designed IVF + INT8 + Bloom retriever" or "the co-designed retriever". Where the harness is dataset-aware (e.g. yambda vs filter suites) the chapter refers to "the yambda suite" and "the filter suite" — these are harness terms, not dataset descriptors.
>
> Cross-reference: [05-implementation.md](05-implementation.md) for the library implementation; this chapter consumes the same algorithm symbol names but documents the orchestration harness around them.

---

## §5.1 Архитектура харнесса (Harness architecture)

### Workspace and module layout

- The repository root is a `uv` workspace with two members; the harness lives in member 2 — `evaluation/`. Source: [pyproject.toml:14-15](../../pyproject.toml#L14-L15).
  ```toml
  [tool.uv.workspace]
  members = ["retrieve", "evaluation"]
  ```
- `evaluation/` is **not** published to PyPI. It depends on `retrieve` (the library under test) plus internal scaffolding for datasets and training.
- The evaluation harness modules live at [evaluation/retrieval/](../../evaluation/retrieval/). The complete public-symbol map (file → role) is:

  | File | LOC | Role |
  |---|---|---|
  | [config.py](../../evaluation/retrieval/config.py) | 129 | YAML schema: `EvalConfig`, `EncodeConfig`, `FilterCfg`, `FilterSweepCfg`; `load_eval_config(path)` |
  | [sweep.py](../../evaluation/retrieval/sweep.py) | 755 | Nested loop driver: `run_sweep → run_filter_kind → run_one_sweep → evaluate_cell` |
  | [bench_tools.py](../../evaluation/retrieval/bench_tools.py) | 459 | Timing (`measure_forward_cuda`, `measure_forward_cpu`), perf/quality passes, encode helpers |
  | [loaders.py](../../evaluation/retrieval/loaders.py) | 334 | Dataset loading dispatcher; SASRec encode; pre-encoded arXiv path; QA synthesis |
  | [queries_cache.py](../../evaluation/retrieval/queries_cache.py) | 114 | Disk cache for the SASRec encode pass |
  | [oracle.py](../../evaluation/retrieval/oracle.py) | 136 | Filtered ground-truth (brute-force FullScan), disk cache |
  | [metrics.py](../../evaluation/retrieval/metrics.py) | 171 | Recall@K, Precision@K, MRR@K, NDCG@K + per-query accumulator |
  | [results_io.py](../../evaluation/retrieval/results_io.py) | 34 | JSON readers (`load_rows`, `load_results`) |
  | [algos/__init__.py](../../evaluation/retrieval/algos/__init__.py) | 131 | `ALGORITHMS` tuple, `BACKEND_CAPABLE_ALGOS`, `build_algorithm` factory |
  | [algos/linr_v1.py](../../evaluation/retrieval/algos/linr_v1.py) | 49 | `LinrV1Algo` (covers `triton_knn` and `linr_v1_filter_mask`) |
  | [algos/linr_v2.py](../../evaluation/retrieval/algos/linr_v2.py) | 48 | `LinrV2Algo` (filter required) |
  | [algos/linr_v3.py](../../evaluation/retrieval/algos/linr_v3.py) | 92 | `LinrV3Algo` (1-bit prefilter + fp32 rerank cascade) |
  | [algos/linr_v4.py](../../evaluation/retrieval/algos/linr_v4.py) | 50 | `LinrV4Algo` (single-stage INT8 dense) |
  | [algos/silvertorch.py](../../evaluation/retrieval/algos/silvertorch.py) | 108 | `SilvertorchAlgo` (the co-designed IVF + INT8 + Bloom retriever) |
  | [algos/torch_knn.py](../../evaluation/retrieval/algos/torch_knn.py) | 31 | `TorchKnnAlgo` (FullScan reference) |
  | [algos/filter.py](../../evaluation/retrieval/algos/filter.py) | 73 | `build_filter`, `make_mask` (filter-module factory + mask helper) |
  | [algos/_helpers.py](../../evaluation/retrieval/algos/_helpers.py) | 23 | `collect_modules` (per-cell GPU cleanup helper) |
  | [cli/run_evaluation.py](../../evaluation/retrieval/cli/run_evaluation.py) | 309 | Top-level orchestrator: spawns per-algo subprocesses, manages logs |
  | [cli/evaluate.py](../../evaluation/retrieval/cli/evaluate.py) | 104 | Per-algo CLI: one process = one algorithm's cells; writes one JSON |
  | [cli/stage_results.py](../../evaluation/retrieval/cli/stage_results.py) | 75 | Post-processor: concatenates per-algo JSONs, renames dir → `.perkernel/` |
  | [cli/upload_results.py](../../evaluation/retrieval/cli/upload_results.py) | 194 | HuggingFace upload helper (stages + writes README) |

- The system-doc source of truth for this chapter is [docs/system/evaluation.md](../system/evaluation.md). The thesis chapter is the publication-quality re-presentation; the system doc is the live engineering reference.

### CLI tree and process model

Three CLI entry points, in order of containment:

1. `uv run run-evaluation` — top-level orchestrator. Resolves a list of YAML configs, spawns one subprocess per `(config, algorithm)` pair, tees output to logs, and (optionally) runs the staging step after each config. Source: [cli/run_evaluation.py:267-309](../../evaluation/retrieval/cli/run_evaluation.py#L267-L309).
2. `uv run evaluate` — single-algorithm runner. Loads the YAML, pins determinism globals, calls `run_sweep`, writes one JSON to the configured output path. Source: [cli/evaluate.py](../../evaluation/retrieval/cli/evaluate.py).
3. `uv run stage-results` — post-processor. After all per-algo JSONs exist under `<output>/`, concatenates them into `<output>.json`, renames the original directory to `<output>.perkernel/`, and copies the source YAML to `<output>.yaml`. Source: [cli/stage_results.py](../../evaluation/retrieval/cli/stage_results.py).

The legacy entry point `evaluation/run_per_algo.sh` (referenced in [docs/system/evaluation.md:22-23](../system/evaluation.md#L22-L23)) is a thin shell loop superseded by `run-evaluation`; the orchestrator is now the canonical entry point. The system doc and the CLI naming are in transition — see Writer's notes.

**Why one subprocess per algorithm?** Per-algorithm process isolation is a deliberate design choice, documented at [cli/evaluate.py:10-12](../../evaluation/retrieval/cli/evaluate.py#L10-L12) and [docs/system/evaluation.md:28-31](../system/evaluation.md#L28-L31):

> Each invocation is a fresh Python process, which is the point — torch compile / Triton autotune / CUDA-graph private pools that survive `torch._dynamo.reset()` get cleared between algos by the OS.

The single shared cost between subprocesses is the SASRec query encode, which is cached on disk by [queries_cache.py](../../evaluation/retrieval/queries_cache.py) so the N subprocesses don't pay N × encode cost (see §5.3).

### Suite definitions

The orchestrator exposes named **eval-types** that bundle configurations. Source: [cli/run_evaluation.py:35-54](../../evaluation/retrieval/cli/run_evaluation.py#L35-L54).

| Eval-type | Configs | Stage step | Suite (per-row) |
|---|---|---|---|
| `filter` | `config/{arxiv,goodreads}/d{64,128,256}-filter.yaml` (6 files) | No | `filter` |
| `quality` | `config/{arxiv,goodreads}/d{64,128,256}-quality.yaml` (6 files) | Yes | `filter` for filter cells, `yambda` for unfiltered cells (see below) |
| `param-sweeps` | `config/deep_sweeps/arxiv-d128-silvertorch.yaml`, `config/deep_sweeps/goodreads-d128-linr_v3.yaml` | Yes | `filter` (deep sweeps run over filter cells) |

A **suite** label is the value of the `suite` field in each output row. The label is determined at runtime by the config's `filters:` field, not by the eval-type or by the dataset name. Source: [sweep.py:72](../../evaluation/retrieval/sweep.py#L72):

```python
suite = "yambda" if cfg.filters is None else "filter"
```

Conventions:
- `suite="yambda"` — for every YAML where `filters:` is null. In current practice this is exactly the Yambda configurations (no filter benchmark on Yambda). The label name is historical and slightly misleading; arXiv `*-quality.yaml` configs also have `filters:` set to null but the suite label they emit is **`filter`** … actually, in the current configs, arXiv and Goodreads `*-quality.yaml` files omit the `filters:` block entirely, so they produce `suite="yambda"` rows. **TODO: clarify with author** — the "suite" naming conflates "filter-or-not" with "Yambda-or-not"; the writer should call out that `suite="yambda"` means "no-filter evaluation," not "data was Yambda".
- `suite="filter"` — every cell that ran inside a `clause` or `bloom` filter loop.

Yambda configurations live at [evaluation/config/yambda-500m/](../../evaluation/config/yambda-500m/) (`d64-quality.yaml`, `d128-quality.yaml`, `d256-quality.yaml`) and [evaluation/config/yambda-5b/](../../evaluation/config/yambda-5b/) (`d64-quality.yaml`, `d128-quality.yaml`). They are not bundled into any eval-type and run on demand from the orchestrator's positional-config interface.

### Definition of a cell

A **cell** is the atomic unit of measurement. It is the seven-tuple:

$$\text{cell} = (\text{filter\_kind},\ \text{sweep},\ \text{algo},\ \text{params},\ k,\ \text{batch\_size},\ \text{backend})$$

For each cell the harness emits exactly one row in the output JSON. The loop nesting (outermost to innermost) is defined by [sweep.py:8-13](../../evaluation/retrieval/sweep.py#L8-L13):

```
run_sweep                    # pin globals, warm GPU, dispatch
└─ run_filter_kind           # subsample users, build filter modules
   └─ run_one_sweep          # synth qa, build/load oracle, iterate algos
      └─ evaluate_cell       # build algo, quality, prewarm, per-bs perf rows
```

The cross-product unrolled by the loops:
1. `filter_kind ∈ {none, clause, bloom}` — selected from `cfg.filters` (or the synthetic `("none", FilterSweepCfg(name="full_scan"))` cell on yambda configs). Source: [sweep.py:108-122](../../evaluation/retrieval/sweep.py#L108-L122).
2. `sweep ∈ cfg.filters[filter_kind].sweeps` — named subset of active narrow clauses, e.g. `c0_genre`, `all4`. Source: [sweep.py:161](../../evaluation/retrieval/sweep.py#L161).
3. `algo ∈ cfg.algorithms` — algorithm name string. Source: [sweep.py:360](../../evaluation/retrieval/sweep.py#L360).
4. `backend ∈ active_backends` — only iterated for algos in `BACKEND_CAPABLE_ALGOS`; others emit one row regardless. Source: [sweep.py:361-364](../../evaluation/retrieval/sweep.py#L361-L364).
5. `params ∈ expand_param_combos(cfg.algo_params.get(algo, [{}]))` — each dict is one explicit parameter combination; combos that fail `is_valid_combo` (e.g. `silvertorch` with `n_probe > n_lists`) are skipped. Source: [sweep.py:365-368, 730-745](../../evaluation/retrieval/sweep.py#L365-L368).
6. `k ∈ cfg.ks` — top-K cutoff. Source: [sweep.py:369](../../evaluation/retrieval/sweep.py#L369).
7. `batch_size ∈ cfg.batch_sizes` — perf-pass batch size; quality is invariant to bs and computed once per `(algo, params, k)` cell and replicated across rows. Source: [sweep.py:464](../../evaluation/retrieval/sweep.py#L464); rationale at [bench_tools.py:347-353](../../evaluation/retrieval/bench_tools.py#L347-L353).

Algorithms that can't run on the requested `filter_kind` raise `ValueError` at construction; the driver catches and silently skips that cell (no stub row). Source: [sweep.py:517-546](../../evaluation/retrieval/sweep.py#L517-L546). Example: `linr_v2` requires a filter; `silvertorch` with `filter_kind="clause"` is fine in the codepath but skipped at the YAML level on configs that don't request it.

### Per-cell execution sequence

Inside `evaluate_cell` ([sweep.py:400-503](../../evaluation/retrieval/sweep.py#L400-L503)) the per-cell sequence is:

1. **Reset CUDA state for cell** — `torch.cuda.synchronize`, `empty_cache`, `reset_peak_memory_stats`. Source: [sweep.py:509-514](../../evaluation/retrieval/sweep.py#L509-L514).
2. **Record `mem_before`** — `torch.cuda.memory_allocated()` in MiB. Source: [sweep.py:425](../../evaluation/retrieval/sweep.py#L425).
3. **Build algo** — `build_algorithm(name, item_embs, k, …, backend)` returns the `nn.Module` (or `TorchKnnAlgo`); failure → `ValueError` → cell skipped. Source: [sweep.py:427-438](../../evaluation/retrieval/sweep.py#L427-L438).
4. **Compute `index_mem`** — `cuda_allocated_mib() − mem_before`. Source: [sweep.py:440-442](../../evaluation/retrieval/sweep.py#L440-L442).
5. **Quality pass** — one call to `_run_quality` (which dispatches to `quality_pass_cached`) producing `(recall, ndcg)` for this `k`. Source: [sweep.py:444-459](../../evaluation/retrieval/sweep.py#L444-L459).
6. **Autotune prewarm** — call the algo's `forward` once per batch size before any bs is timed, defending against a last-mile Triton autotune config leaking into the timing window. Source: [sweep.py:461, 606-631](../../evaluation/retrieval/sweep.py#L606-L631).
7. **Per-`bs` perf loop** — for each `bs ∈ cfg.batch_sizes` call `perf_pass_cached(algo_obj, queries_f, batch_size=bs, …)` and append one row to the cell's output. Source: [sweep.py:463-500](../../evaluation/retrieval/sweep.py#L463-L500).
8. **Release algo** — `algo_obj.algo_modules.clear()` + `empty_cache`. Note `modules.clear()` is required because `for m in modules: del m` only drops the loop variable, not the list's references, leaking the index into the next cell's `mem_before`. Source: [sweep.py:502, 720-724](../../evaluation/retrieval/sweep.py#L720-L724); cross-reference [docs/system/evaluation.md:311-313](../system/evaluation.md#L311-L313).

The per-cell memory-snapshot ordering is summarised verbatim in [docs/system/evaluation.md:300-309](../system/evaluation.md#L300-L309):

```
sync → empty_cache → reset_peak_memory_stats → mem_before
build_algorithm
sync → index_mem = allocated() - mem_before
quality_pass
for bs in batch_sizes: perf_pass
algo_obj.modules.clear(); del algo_obj; empty_cache
```

A `torch._dynamo.reset()` + `empty_cache()` is also issued at the end of each filter sweep (one level above the cell), to drop dynamo compile cache + CUDA-graph private pools that would otherwise OOM the next sweep on large catalogues (e.g. arXiv-d256). Cost: ~30–60 s of recompile at the start of the next sweep. Source: [sweep.py:186-192](../../evaluation/retrieval/sweep.py#L186-L192).

### Orchestrator detail: `run_evaluation.py`

- Argument parsing — [cli/run_evaluation.py:181-210](../../evaluation/retrieval/cli/run_evaluation.py#L181-L210). Flags: `--eval-type`, positional `configs`, `--resume` (skip cells whose JSON exists and is non-empty), `--force` (overwrite without prompting), `--stage` (run staging after each config), `--logdir` (default `results/_runlogs`), `--` (forward extra args to `evaluate` subprocess).
- Work plan — [cli/run_evaluation.py:213-230](../../evaluation/retrieval/cli/run_evaluation.py#L213-L230). Builds work triples `(cfg, out_dir, [(algo, json_path), ...])`; validates that outputs don't already exist unless `--resume`/`--force`.
- Per-config loop — [cli/run_evaluation.py:233-264](../../evaluation/retrieval/cli/run_evaluation.py#L233-L264). For each `(algo, json_path)`: spawn `uv run evaluate --config <cfg> --algo <algo> --output <json_path> [<extra>]`. Tee output to per-config `.log`, `full.log`, and stdout. If `stage=True`, spawn `uv run stage-results <cfg>` after all algos finish.
- GPU name detection — [cli/run_evaluation.py:61-69](../../evaluation/retrieval/cli/run_evaluation.py#L61-L69). Invokes `nvidia-smi --query-gpu=name --format=csv,noheader`; falls back to `(no nvidia-smi)`. Result is logged to the SUMMARY at session start.
- `LogSinks` tee — [cli/run_evaluation.py:111-163](../../evaluation/retrieval/cli/run_evaluation.py#L111-L163). Writes to `<logdir>/full.log` (all lines), `<logdir>/SUMMARY.txt` (campaign summary lines only), `<logdir>/<cfg_name>.log` (per-config), and `<logdir>/current.log` symlink (always points at the active per-config log).

The SUMMARY format is illustrated by [evaluation/results/_runlogs/SUMMARY.quality-deep.txt](../../evaluation/results/_runlogs/SUMMARY.quality-deep.txt) — see §5.6 (Environment) for a verbatim quote.

### Writer's notes — §5.1

- The "harness" word in Russian: предложение использовать "стенд оценки" или "испытательная система" (харнесс is informal). The reference doc here uses "harness" throughout.
- Diagram opportunity (one of the chapter's required visuals — see "Visual deliverables" near the end of this file): a flow diagram showing the cell ↔ algorithm ↔ sweep ↔ metrics ↔ JSON pipeline. A four-level nested-box diagram mirroring the `run_sweep → run_filter_kind → run_one_sweep → evaluate_cell` containment is the natural rendering.
- Doc/code drift to flag: [docs/system/evaluation.md](../system/evaluation.md) refers to `evaluation/conf/` and `evaluation/retrieval/evaluate.py`, but the current code paths are `evaluation/config/` (note the `config/` rename) and `evaluation/retrieval/cli/evaluate.py` (now namespaced under `cli/`). The writer should use the current paths; the system doc is partially stale.
- The `suite` field name is overloaded (see "Suite definitions" above). Recommend the writer either (a) describe the value semantics explicitly when the field is introduced, or (b) propose `suite_kind ∈ {filtered, unfiltered}` in a Future Work / minor-cleanup callout in Ch.7.
- Naming clarification (verified, no longer a TODO): the system doc at [docs/system/evaluation.md:210](../system/evaluation.md#L210) shows `algo.modules` but every code site uses `algo.algo_modules` — confirmed by `grep -rn` over [algos/](../../evaluation/retrieval/algos/) (10 hits) and [sweep.py:722](../../evaluation/retrieval/sweep.py#L722). The chapter should use `algo_modules`; system doc is locally stale at that one line.
- Naming clarification (verified): the deep-sweeps eval-type CLI flag value is `param-sweeps` ([cli/run_evaluation.py:46](../../evaluation/retrieval/cli/run_evaluation.py#L46)) while the on-disk directory is `deep_sweeps/`. Chapter prose should use "deep sweeps"; the CLI alias is a one-line footnote at first mention.

---

## §5.2 Метрики (Metrics)

The harness implements four standard top-K information-retrieval metrics, all defined in [metrics.py](../../evaluation/retrieval/metrics.py). All per-query functions return a `[B]` tensor; the harness accumulates per-query values across all batches in the evaluation pass and then averages them.

### Target representation

Inputs to every metric:
- `candidate_ids: Tensor[B, K_max]` — retrieved item IDs from the algorithm; `-1` signals "no item at this rank" (algorithms with filtered short fills use `-1` padding).
- `targets: Tensor[B, T]` — ground-truth item IDs; padded to a common `T` with `-1`.
- `num_targets: Tensor[B]` — count of valid (non-padding) targets per query.

Source: [metrics.py:1-8 (module docstring)](../../evaluation/retrieval/metrics.py#L1-L8). The dual-padding convention (both sides can carry `-1`) requires care: the harness defines a hits mask helper that rejects spurious `-1 == -1` matches.

### Hits mask helper

Implementation: [metrics.py:18-36](../../evaluation/retrieval/metrics.py#L18-L36). Returns a `[B, k]` boolean tensor: `True` at position $(b, i)$ iff `candidate_ids[b, i]` equals some target in `targets[b, :]` **and** both operands are non-padding.

```python
def _hits_mask(candidate_ids: Tensor, targets: Tensor, k: int) -> Tensor:
    topk = candidate_ids[:, :k]                                        # [B, k]
    eq = topk.unsqueeze(2) == targets.unsqueeze(1)                     # [B, k, T]
    valid = (topk.unsqueeze(2) != -1) & (targets.unsqueeze(1) != -1)
    return (eq & valid).any(dim=2)
```

### Recall@K — formal definition

For query $b$ with retrieved top-$K$ set $R_b^K \subseteq \mathcal{I}$ and relevant set $G_b \subseteq \mathcal{I}$:

$$\mathrm{Recall@K}(b) = \frac{|R_b^K \cap G_b|}{\max(1, |G_b|)}$$

The denominator clamp at $1$ handles the corner case of $|G_b| = 0$ (queries with no targets) and returns $0$ recall for those rows rather than dividing by zero.

The arithmetic mean over the evaluation set $\mathcal{Q}$:

$$\mathrm{Recall@K} = \frac{1}{|\mathcal{Q}|} \sum_{b \in \mathcal{Q}} \mathrm{Recall@K}(b)$$

For single-target leave-last-out evaluation (yambda, goodreads, arxiv default), $|G_b| = 1$ and Recall@K equals **hit rate**.

Implementation: [metrics.py:39-54](../../evaluation/retrieval/metrics.py#L39-L54).

```pseudocode
function recall_at_k(candidate_ids, targets, num_targets, k):
    hits ← hits_mask(candidate_ids, targets, k)     # [B, k] bool
    n_hits ← sum(hits, dim=1).float()               # [B]
    return n_hits / max(1, num_targets.float())     # [B]
```

### Precision@K — formal definition

$$\mathrm{Precision@K}(b) = \frac{|R_b^K \cap G_b|}{K}$$

The denominator is the constant cutoff $K$, not the per-query target count. Implementation: [metrics.py:57-69](../../evaluation/retrieval/metrics.py#L57-L69).

Precision@K is computed and recorded by the accumulator but is **not currently emitted** as a top-level row field — only `recall@k` and `ndcg@k` make it into the JSON schema (see §5.7). The metric remains computed because `accumulate_metrics` iterates over the entire `_METRIC_FNS` registry; the omission from the schema is a column choice in `_make_perf_row`, not a metric absence. Writer should note that the harness *could* expose Precision@K and MRR@K with a one-line schema change.

### MRR@K — formal definition

For query $b$, let $r_b^K$ be the rank (1-indexed) of the first relevant item in the retrieved top-$K$, or undefined if none of the top-$K$ are relevant:

$$\mathrm{MRR@K}(b) = \begin{cases} 1 / r_b^K & \text{if a relevant item appears in top-}K \\ 0 & \text{otherwise} \end{cases}$$

Implementation: [metrics.py:72-87](../../evaluation/retrieval/metrics.py#L72-L87). The harness uses `hits.float().argmax(dim=1) + 1` to extract the first-`True` index (argmax breaks ties by returning the lowest index, which is exactly the first hit). A `found = hits.any(dim=1)` mask zeros out rows with no hits.

```pseudocode
function mrr_at_k(candidate_ids, targets, num_targets, k):
    hits ← hits_mask(candidate_ids, targets, k)
    found ← any(hits, dim=1)                       # [B]
    rank ← argmax(hits.float(), dim=1) + 1         # [B], 1-indexed
    return (1.0 / rank.float()) * found.float()    # [B]
```

### NDCG@K — formal definition

Position-discount: $D(i) = 1 / \log_2(i + 1)$ for rank $i \in \{1, 2, \dots, K\}$.

Discounted Cumulative Gain at cutoff $K$ for query $b$:

$$\mathrm{DCG@K}(b) = \sum_{i=1}^{K} \mathrm{hit}_b(i) \cdot D(i)$$

where $\mathrm{hit}_b(i) = 1$ if the item at rank $i$ in the retrieved list is relevant, else $0$. (Binary relevance.)

Ideal DCG per query — assumes the ideal ranking places the first $\min(|G_b|, K)$ ranks as hits:

$$\mathrm{IDCG@K}(b) = \sum_{i=1}^{\min(|G_b|, K)} D(i)$$

Normalised:

$$\mathrm{NDCG@K}(b) = \frac{\mathrm{DCG@K}(b)}{\max(\mathrm{IDCG@K}(b), \varepsilon)}$$

with $\varepsilon = 10^{-8}$ guarding zero-IDCG rows.

Implementation: [metrics.py:90-114](../../evaluation/retrieval/metrics.py#L90-L114).

```pseudocode
function ndcg_at_k(candidate_ids, targets, num_targets, k):
    hits      ← hits_mask(candidate_ids, targets, k)            # [B, k]
    positions ← arange(1, k + 1).float()                        # [k]
    discounts ← 1.0 / log2(positions + 1)                       # [k]
    dcg       ← sum(hits.float() * discounts, dim=1)            # [B]
    n_rel     ← min(num_targets, k).float()                     # [B]
    ideal     ← positions ≤ n_rel.unsqueeze(1)                  # [B, k] bool
    idcg      ← sum(ideal.float() * discounts, dim=1)           # [B]
    return dcg / max(idcg, 1e-8)                                # [B]
```

[CITE: Järvelin, K. & Kekäläinen, J. (2002). "Cumulated gain-based evaluation of IR techniques." ACM Transactions on Information Systems, 20(4), 422–446. — KEEP; non-Meta authors (University of Tampere).]

### Per-query accumulation and batch reduction

The harness accumulates per-query metric values across the entire evaluation stream (chunked at `QUALITY_BATCH_SIZE = 16` rows per `forward`), then takes the arithmetic mean over all per-query values.

`accumulate_metrics` ([metrics.py:125-156](../../evaluation/retrieval/metrics.py#L125-L156)):
- Iterates over `_METRIC_FNS = {"recall": recall_at_k, "precision": precision_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}` and over each `k ∈ ks`.
- Computes the per-query `[B]` tensor for each `(metric, k)`, moves to CPU as Python floats, and appends to a dict keyed by `f"{metric}@{k}"`.
- The accumulator persists across batches; the next call extends the same lists.

`finalize_metrics` ([metrics.py:159-171](../../evaluation/retrieval/metrics.py#L159-L171)):
- For each key, `mean(values) = sum(values) / len(values)`, or `0.0` if the list is empty.

```pseudocode
function accumulate_metrics(candidate_ids, targets, num_targets, ks, accum):
    if accum is None: accum ← defaultdict(list)
    for k in ks:
        for (name, fn) in _METRIC_FNS:
            values ← fn(candidate_ids, targets, num_targets, k)   # [B]
            accum["{name}@{k}"].extend(values.cpu().tolist())
    return accum

function finalize_metrics(accum):
    return { key: mean(values) for (key, values) in accum.items() }
```

Storage on CPU as Python floats bounds GPU memory regardless of the eval-set size.

### Quality pass driver

`quality_pass_cached` ([bench_tools.py:332-381](../../evaluation/retrieval/bench_tools.py#L332-L381)) is the consumer of the metric helpers. Behaviour:
- Streams the cached query tensor through `forward` in chunks of size `QUALITY_BATCH_SIZE = 16` (constant defined at [bench_tools.py:324](../../evaluation/retrieval/bench_tools.py#L324)).
- Applies the optional `skip_mask` to drop users whose synthesised QA was all `-1` for the active clauses (chunk-by-chunk).
- For filter cells, the `targets` passed in are the cached oracle top-K (see §5.4), not the held-out test items.
- Returns `(recall@k, ndcg@k)` as Python floats.

**Why `QUALITY_BATCH_SIZE = 16` (rather than 64).** From the inline comment at [bench_tools.py:324-329](../../evaluation/retrieval/bench_tools.py#L324-L329):

> Was 64; reduced because `PrefilterKNN[backend="torch"]` on arxiv/d256 (N≈3M, dim=256) materialises `[B, P, D]` fp32 = ~110 GiB at bs=64 on loose clause filters (e.g. c3_nversions, P≈1.7M items) and OOMs an 80 GB card. bs=16 keeps the same allocation shape as the perf-pass bs=16 rows which already fit. The 4× chunking reduction lengthens the quality stream proportionally but it's a small fraction of total wall-clock.

This contradicts the older [docs/system/evaluation.md:243](../system/evaluation.md#L243) (which still says 64) — flag for the writer.

**Per-row replication of quality across batch sizes.** Because quality is invariant to `bs` (same per-row scoring math), the harness runs the quality pass **once per `(algo, params, k)` cell** and attaches the same `(recall, ndcg)` floats to every `bs` row of the cell. Source: [sweep.py:444-459, 463-500](../../evaluation/retrieval/sweep.py#L444-L500) and rationale at [bench_tools.py:347-353](../../evaluation/retrieval/bench_tools.py#L347-L353).

### Citation candidates — metrics

- [CITE: Manning, C., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. — KEEP; standard textbook reference for Recall, Precision, ranking metrics. Non-Meta authors (Stanford / Yahoo at time of writing).]
- [CITE: Järvelin, K. & Kekäläinen, J. (2002). "Cumulated gain-based evaluation of IR techniques." ACM TOIS 20(4):422-446. — KEEP for NDCG. Non-Meta (University of Tampere).]
- [CITE: Voorhees, E.M. (1999). "The TREC-8 Question Answering Track Report." Proc. TREC-8. — KEEP for MRR. NIST, non-Meta.]
- [CITE: Croft, W.B., Metzler, D., Strohman, T. (2010). *Search Engines: Information Retrieval in Practice*. Pearson. — OPTIONAL; another textbook for IR fundamentals; non-Meta.]

### Writer's notes — §5.2

- The chapter should foreground Recall@K and NDCG@K (the two metrics in the schema) and footnote MRR@K and Precision@K (computed but not emitted).
- The dual-padding convention (`-1` on both sides) is non-obvious and worth a sentence in prose: "The hits mask explicitly rejects `-1 == -1` matches because algorithms with short-fill outputs and ground-truth padding both use `-1` as the sentinel."
- The single-target case in this corpus: yambda (last item left out per user; T=1), goodreads (likewise), arxiv (single relevant heldout per query; T=1). Hit-rate ≡ Recall@K in all three.
- The multi-target generalisation: the harness supports `T > 1` (oracle paths can have multiple non-padding targets per query when the filter passes ≥ K_GT items); the metrics are written to handle it. Quote the docstring at [metrics.py:1-8](../../evaluation/retrieval/metrics.py#L1-L8).
- For the IDCG derivation, mention that the harness uses binary-relevance IDCG (assumes all relevant items are equally relevant). Non-binary relevance is not used in this thesis.
- [TODO: clarify with author] — is there an HSE/GOST convention requiring a specific metric formulation style (with $r_i$ vs $\text{rel}_i$ notation, base-2 vs base-$e$ log)? Default to base-2 throughout (matches the code).

---

## §5.3 Query generation

### Three dispatch modes

The harness adapts to three on-disk shapes, dispatched by config-field combinations rather than an explicit dataset field. Source: [docs/system/evaluation.md:37-44](../system/evaluation.md#L37-L44) (slightly stale but the dispatch is the same in the current code).

| `checkpoint` | `query_emb_path` | Mode | Datasets |
|---|---|---|---|
| set | unset | Encode queries via SASRec on demand | yambda (500m, 5b), goodreads |
| unset | set (or defaulted to `<content_subdir>/query_emb.pt`) | Load pre-encoded text embeddings | arXiv |
| set | unset, `filters: null` | Encode + no filter loop | yambda |
| set | unset, `filters: {clause: …, bloom: …}` | Encode + filter sweeps | goodreads |

The dispatching function is `load_item_and_queries` ([loaders.py](../../evaluation/retrieval/loaders.py)) called via `load_or_cache_queries` at [cli/evaluate.py:68-70](../../evaluation/retrieval/cli/evaluate.py#L68-L70).

### Query embedding cache

The shared encode cost is amortised by a disk cache. Source file: [queries_cache.py](../../evaluation/retrieval/queries_cache.py).

**Where:** `<ckpt-dir>/encoded_queries_<split>.pt` (e.g. `data/goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id/encoded_queries_test.pt`). Source: [queries_cache.py:48](../../evaluation/retrieval/queries_cache.py#L48).

**What is cached:** the blob is a dict bundling four CPU tensors plus three key components.
```python
{
    "item_embs": Tensor[N_items, D],       # CPU-side; moved to device on load
    "queries":   Tensor[N_users, D],       # CPU-side
    "targets":   Tensor[N_users, T_max],   # CPU-side, padded with -1
    "n_targets": Tensor[N_users],          # CPU-side
    "ckpt_mtime": float,                   # cache-key component 1
    "max_seq_length": int,                 # cache-key component 2
    "users_limit": int | None,             # cache-key component 3
}
```
Source: [queries_cache.py:98-109](../../evaluation/retrieval/queries_cache.py#L98-L109).

**Cache-key components and invalidation:**
1. `ckpt_mtime = ckpt_path.stat().st_mtime` — Unix epoch seconds of the checkpoint file. Detects checkpoint re-training or replacement.
2. `max_seq_length = cfg.encode.max_seq_length` — SASRec history-truncation length. Different value → different query embeddings.
3. `users_limit = cfg.users_limit` — if set, applied **before** caching so cache size scales with the limited set, not the full split. Critical for 100k+ user datasets like goodreads (313k test users).

The cache is valid iff all three match (`==`). Source: [queries_cache.py:53-67](../../evaluation/retrieval/queries_cache.py#L53-L67).

```python
if cache_path.exists():
    blob = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    if (
        blob.get("ckpt_mtime") == ckpt_mtime
        and blob.get("max_seq_length") == max_seq
        and blob.get("users_limit") == users_limit
    ):
        # hit; return cached tensors (item_embs → device, others stay CPU)
        ...
    # miss → fall through to recompute and overwrite
```

**Disk-budget guard.** Recently added (commit `2ea3dae`, "eval harness: queries_cache adds users_limit to cache key + disk-budget guard (skips write rather than fill disk on yambda-5b)"). Source: [queries_cache.py:86-110](../../evaluation/retrieval/queries_cache.py#L86-L110).

```python
est_bytes = sum(t.element_size() * t.numel()
                for t in (item_embs, queries, targets, n_targets))
free_bytes = shutil.disk_usage(str(cache_path.parent)).free
budget_bytes = max(0, int(free_bytes * _CACHE_FREE_FRACTION) - _CACHE_RESERVE_BYTES)
if est_bytes > budget_bytes:
    logger.warning("skipping cache write: est={:.1f}GB > budget={:.1f}GB ...")
else:
    torch.save({...}, str(cache_path))
```

Constants at [queries_cache.py:32-33](../../evaluation/retrieval/queries_cache.py#L32-L33):
- `_CACHE_FREE_FRACTION = 0.7` — write at most 70% of current free space.
- `_CACHE_RESERVE_BYTES = 4 * 2**30` — always keep at least 4 GB free.

When the budget would be exceeded the cache write is **skipped** (not aborted), and subsequent algos re-encode from scratch. This is the desired failure mode on the 5b dataset, where the encoded blob can exceed the per-subprocess disk budget.

### SASRec encode path (yambda, goodreads)

`load_sasrec_embeddings` ([loaders.py](../../evaluation/retrieval/loaders.py)) constructs a `GSASRec` model from the checkpoint, runs `predict_last()` over every user in the test split, and returns `(item_embs, queries, targets, n_targets, ckpt_path)`.

- Model construction: reads `<checkpoint_parent>/config.json` if present (modern checkpoints carry one via `GSASRecConfig.save`); falls back to `D128_DROP05_DEFAULTS` for legacy 500m checkpoints. Source: [bench_tools.py:230-260](../../evaluation/retrieval/bench_tools.py#L230-L260).
- Encode batch size / num_workers / max_seq_length: configured via `cfg.encode` ([config.py:26-30](../../evaluation/retrieval/config.py#L26-L30)). Independent of perf-pass batch size.
- Output: `queries[N_users, D]` is the last-item prediction vector per user; `targets[N_users, T]` is the held-out test item(s) padded with `-1`; `n_targets[N_users]` is the per-user valid-target count.

### Pre-encoded arXiv path

`load_pre_encoded_arxiv` ([loaders.py](../../evaluation/retrieval/loaders.py)) loads:
- `<data_dir>/<content_subdir>/text_emb.pt` → item embeddings (one row per arXiv paper, prefixed with `"search_document: "` at encode time; produced by the `arxiv encode` CLI).
- `<data_dir>/<content_subdir>/query_emb.pt` → query embeddings (held-out heldout-arXiv passages, prefixed with `"search_query: "`; produced by `arxiv encode_queries`).
- `<data_dir>/<content_subdir>/text_emb.meta.json`, `query_emb.meta.json` → metadata (encoder model id, prefix used, dimension).

ArXiv embeddings ship at three dimensions: `content_d64/`, `content_d128/`, `content/` (= d=256). The `content_subdir` config field picks one; the `gt_subdir` field must vary in lockstep (oracle scores depend on item embeddings which depend on dim). Source: [config.py:76-83](../../evaluation/retrieval/config.py#L76-L83).

The encoder is `nomic-ai/nomic-embed-text-v1.5` (cross-ref Ch.3 §3.2). The chapter introduces the arXiv encode pipeline but its detailed description is owned by Ch.3.

### `users_limit` semantics

[config.py:101-105](../../evaluation/retrieval/config.py#L101-L105):
> Optional cap on the number of users for ALL cells (quality and filter alike). Goodreads has 313k test users which makes the bs=1 quality stream the wall-clock bottleneck; cap to e.g. 50000 to speed runs up. Leave null to use the full split.

Applied:
- On SASRec encode: post-encode trim to the first `users_limit` users; cache is then keyed on this number. Source: [queries_cache.py:75-85](../../evaluation/retrieval/queries_cache.py#L75-L85).
- On the orchestrator at [sweep.py:201-220 (`_apply_users_limit`)](../../evaluation/retrieval/sweep.py#L201-L220).

Per the goodreads filter configs, `users_limit: 10000` is the canonical value. Source: [evaluation/config/goodreads/d128-filter.yaml:22](../../evaluation/config/goodreads/d128-filter.yaml#L22).

### Writer's notes — §5.3

- The three-component cache key is a non-trivial design choice; flag in prose: "The `users_limit` invariant on the cache means changing the user cap silently invalidates the cache, even though the encode would be a strict prefix of the previous one. The implementation prioritises correctness over partial reuse."
- The disk-budget guard is recent and load-bearing for yambda-5b. Should appear as a one-line callout under "implementation challenges on large datasets" if the writer wants a Ch.7-flavoured note.
- ArXiv's pre-encoded path skips the cache entirely ([queries_cache.py:44-45](../../evaluation/retrieval/queries_cache.py#L44-L45)). Quote: "Cache hit → no model load. Cache miss → load_sasrec_embeddings runs the full encode pass and the result is persisted before returning."
- [TODO: clarify with author] — were the yambda-5b runs actually completed with the cache, or did they re-encode per subprocess due to budget exceedance? Inspect the per-config logs to confirm.
- The dispatch table in [docs/system/evaluation.md:37-44](../system/evaluation.md#L37-L44) is slightly off (lists "filters: null → no filter loop (yambda)" but in practice every quality config has `filters: null` regardless of dataset). The writer should describe dispatch as "two orthogonal axes — encode mode (checkpoint vs pre-encoded) and filter mode (no filter vs filter sweeps)".

---

## §5.4 Candidate sets and ground truth

### Two ground-truth regimes

Quality is computed against two distinct ground-truth sources depending on the cell:

1. **Held-out test items** (yambda; filter cells with `filter_kind="none"`) — the per-user `targets` tensor produced by the SASRec encode pass (last-item leave-out for yambda/goodreads; held-out abstract for arXiv). This is the standard recsys/IR ground truth.
2. **Filtered FullScan oracle** (filter cells with `filter_kind ∈ {clause, bloom}`) — a per-sweep brute-force top-`K_GT` over the catalogue, restricted to items passing the **exact** clause predicate.

Dispatch lives in `_run_quality` ([sweep.py:549-603](../../evaluation/retrieval/sweep.py#L549-L603)):

```python
if cfg.filters is None or filter_kind == "none":
    return quality_pass_cached(algo_obj, queries_f, targets_f, n_targets_f, ...)
assert oracle_topk is not None
ot_k = oracle_topk[:, :k].contiguous()
nt_k = (ot_k != -1).sum(dim=1).clamp(max=k)
zero_target = nt_k == 0
combined_skip = zero_target if skip_mask is None else (skip_mask | zero_target)
return quality_pass_cached(algo_obj, queries_f, ot_k, nt_k, ..., skip_mask=combined_skip)
```

The `nt_k` computation deserves a callout: when a tight filter passes fewer than $K$ items, the oracle pads the tail with $-1$. Using a flat denominator of $K$ in Recall@K would under-count perfect runs (e.g. 17 real items in top-100 → 0.17 instead of 1.0) and fold zero-target rows into a 0-recall mean. Source comment at [sweep.py:586-590](../../evaluation/retrieval/sweep.py#L586-L590).

### Why a filtered oracle at all

Filter sweeps deliberately restrict the candidate catalogue. The held-out targets are drawn from the unfiltered split and frequently fall **outside** the filtered catalog (a user's held-out book may be in a genre the c0_genre filter excludes, for example). Using the held-out targets to score filtered retrieval would tank recall regardless of algorithm quality. The oracle solves this by recomputing ground truth restricted to the filter. Source: [oracle.py:1-12 (module docstring)](../../evaluation/retrieval/oracle.py#L1-L12).

### Oracle algorithm

Implementation: `compute_filtered_oracle` ([oracle.py:25-86](../../evaluation/retrieval/oracle.py#L25-L86)).

```pseudocode
function compute_filtered_oracle(item_embs, queries, qa_narrow_sweep,
                                  skip_mask, filter_mod, K_GT,
                                  batch_size=64, device):
    n_users ← queries.shape[0]
    out ← full((n_users, K_GT), -1, dtype=int64)
    keep_idx ← non-zero rows of (¬skip_mask)
    if keep_idx is empty: return out

    item_embs_t ← item_embs.t().contiguous()
    n_total ← item_embs.shape[0]
    K_eff ← min(K_GT, n_total)

    for batch in chunks(keep_idx, batch_size=64):
        q   ← queries[batch].to(device)
        qa  ← qa_narrow_sweep[batch].to(device)   if qa_narrow_sweep else None
        mask ← filter_mod.evaluate_mask(qa)        if filter_mod else None
        scores ← q @ item_embs_t                   # [B, N_items]
        if mask is not None:
            scores ← scores.masked_fill(¬mask, -inf)
        scores[:, 0] ← -inf                         # mask item-0 padding row
        topk ← torch.topk(scores, K_eff, dim=1)
        topk_ids ← where(isfinite(topk.values), topk.indices, -1)
        out[batch, :K_eff] ← topk_ids.cpu()
    return out
```

Three subtleties:
1. **Item-0 is always masked.** `item_embs` row 0 is the padding row (cross-ref [05-implementation.md](05-implementation.md) §4.7 / §4.8). The oracle forces `scores[:, 0] = -inf` so item 0 cannot be a top-K candidate. Source: [oracle.py:71](../../evaluation/retrieval/oracle.py#L71).
2. **Short fills get `-1`, not low-index items.** When the filter passes fewer than $K_{\mathrm{eff}}$ items, the bottom slots tie at $-\infty$ and `torch.topk` returns lowest-indexed items by tie-breaking convention (0, 1, 2, …). These are forced back to `-1` via `torch.where(isfinite(topk.values), topk.indices, -1)`. This prevents spurious matches against the algorithms' own `-1` padding. Source: [oracle.py:73-81](../../evaluation/retrieval/oracle.py#L73-L81).
3. **Skip mask propagated.** Rows in `skip_mask=True` are not scored at all; their output row stays `-1`. Source: [oracle.py:46-49](../../evaluation/retrieval/oracle.py#L46-L49).

### Exact filter even on Bloom sweeps

The `filter_mod` argument to `compute_filtered_oracle` **must** be an `ExactAttributeFilter` even when the algorithm under test uses a `BloomFilter`. Source: [oracle.py:9-11](../../evaluation/retrieval/oracle.py#L9-L11):

> `filter_mod` MUST be an *exact* mask source (`ExactAttributeFilter`). Bloom's false positives must NOT leak into the ground truth — bloom-suite runs build a separate exact filter over the same attrs at the call site.

Construction is in `_build_filter_modules` ([sweep.py:230](../../evaluation/retrieval/sweep.py#L230)) — for `filter_kind="bloom"` configurations the harness builds the per-backend Bloom filter for the algorithms and a separate exact filter for the oracle pass.

### Disk cache for the oracle

`load_or_build_oracle` ([oracle.py:89-133](../../evaluation/retrieval/oracle.py#L89-L133)):
- Cache path: `<data_dir>/<gt_subdir>/gt_topk_<sweep_name>.pt`. Example: `data/goodreads-work-id/gt/gt_topk_c0_genre.pt`.
- Stale detection: shape mismatch `(n_users, K_GT)`. The common cause of mismatch is changing `content_subdir` between runs (item embeddings differ → oracle scores differ).
- On miss: recompute and overwrite. On hit: load to CPU and return.

The cache is per-sweep, not per-algorithm: every algorithm in the lineup reads the same oracle file for a given sweep. Cache build cost is paid once per sweep, the first time `run_one_sweep` reaches it.

### Why `K_GT = max(cfg.ks)` not just `K`

The harness builds the oracle at the maximum K in the config (`K_GT = max(cfg.ks)`) and slices to `:k` for each individual recall calculation. Source: [sweep.py:71, 584-585](../../evaluation/retrieval/sweep.py#L71). This avoids rebuilding the oracle per K cutoff: one full-scan + topk pass amortises across all `cfg.ks`.

### Cross-reference: filter primitives

The filter modules used both by the oracle and by the algorithms themselves are documented in [05-implementation.md](05-implementation.md) §4.7. The bench-side filter factory is `build_filter` at [algos/filter.py](../../evaluation/retrieval/algos/filter.py), wrapping the library's `ExactAttributeFilter` and `BloomFilter` classes. The bench-side helper `make_mask(filter_mod, qa_narrow)` ([algos/filter.py](../../evaluation/retrieval/algos/filter.py)) routes per-batch query attrs through `filter_mod.evaluate_mask` and forces `mask[:, 0] = False` so reverse clauses don't admit the padding item.

### Sweep QA synthesis

For each sweep (`FilterSweepCfg(name=…, active_clauses=[…])`), the harness synthesises a per-query attribute tensor:
- Source: `build_sweep_qa` ([loaders.py](../../evaluation/retrieval/loaders.py)).
- For each query, retains the values of `qa_narrow_all` only for clauses in `sweep.active_clauses`; all other clauses are set to `-1` ("always pass" for `ExactAttributeFilter`; "no bits queried" for `BloomFilter`).
- A `skip_mask` is emitted marking users whose synthesised QA is all `-1` for the active clauses (no genuine predicate to enforce).

This lets a single source-of-truth `qa_narrow_all[N_users, C_narrow]` tensor (loaded from `data/<dataset>/eval_split.parquet`) drive multiple named sweeps cheaply.

### Writer's notes — §5.4

- The chapter should distinguish "ground truth" from "oracle" in prose: ground truth = the human-curated/leave-last-out target; oracle = the brute-force computed top-K under a filter. Both serve the same scoring role but for different evaluation regimes.
- The `K_GT` design (max K, slice to smaller K) deserves a one-line justification — "saves $|\text{ks}|$ full-scan passes per sweep at the cost of one larger topk".
- Visual deliverable opportunity: a small Venn / set diagram showing (catalogue, filter-mask, held-out target, oracle top-K) — the "why a filtered oracle" subsection benefits from a figure.
- [TODO: clarify with author] — the oracle uses `item_embs_t = item_embs.t().contiguous()` (transpose). At very large N (e.g. 15M synth arxiv), this allocates a second copy on device. Confirm whether this has been a memory issue in practice or whether it's been a non-issue at the catalogue sizes used.
- The text claim that algorithms have a `-1` padding sentinel is cross-referenced in §4.7 of Ch.4. The writer should land the convention in one of the two chapters (Ch.4 §4.7 is the natural place) and reference it from the other.

---

## §5.5 Sweep dimensions

### Top-level dimensions

| Dimension | Default values | Where set | Notes |
|---|---|---|---|
| `ks` | `[100, 500]` (dataclass default); `[100, 200, 400]` (quality configs); `[100, 500, 1000]` (filter configs) | `cfg.ks` ([config.py:87](../../evaluation/retrieval/config.py#L87)) | One row per K |
| `batch_sizes` | `[1, 8, 16]` (dataclass default); `[1]` (quality configs); `[1, 8, 16]` (filter + deep-sweep configs) | `cfg.batch_sizes` ([config.py:88](../../evaluation/retrieval/config.py#L88)) | One row per bs; quality replicated across bs |
| `backends` | `["triton"]` (dataclass default); `["triton", "torch"]` (every shipped config) | `cfg.backends` ([config.py:97](../../evaluation/retrieval/config.py#L97)) | Only iterated for algos in `BACKEND_CAPABLE_ALGOS` |
| `seed` | `0` (uniform across all shipped configs) | `cfg.seed` ([config.py:89](../../evaluation/retrieval/config.py#L89)) | See §5.6 for what the seed drives |
| `algorithms` | per-config (see lineup table below) | `cfg.algorithms` | Subset of `ALGORITHMS` tuple |
| `algo_params` | per-config (see deep-sweep tables) | `cfg.algo_params` | List-of-dicts per algo; each dict = one explicit combo |
| `users_limit` | `null` (yambda + quality configs); `10000` (filter + deep-sweep configs) | `cfg.users_limit` ([config.py:105](../../evaluation/retrieval/config.py#L105)) | Caps user count uniformly |

The cross-product `(filter_kind, sweep, algo, params, k, batch_size, backend)` defines the cell space; see §5.1 for the loop nesting.

### Algorithm lineup

`ALGORITHMS = ("torch_knn", "triton_knn", "linr_v1_filter_mask", "linr_v3", "linr_v4", "linr_v2", "silvertorch")` at [algos/__init__.py:44-52](../../evaluation/retrieval/algos/__init__.py#L44-L52). The 7 names map to 6 distinct algorithm classes (one alias).

| `algo` string | Class | File | Backend-capable | Notes |
|---|---|---|---|---|
| `torch_knn` | `TorchKnnAlgo` | [algos/torch_knn.py](../../evaluation/retrieval/algos/torch_knn.py) | No | Reference exhaustive IP scan via `FullScanKNN`; mask post-filter on filtered cells; not `nn.Module`, no `torch.compile`. |
| `triton_knn` | `LinrV1Algo` (alias) | [algos/linr_v1.py](../../evaluation/retrieval/algos/linr_v1.py) | Yes (both backends run same torch path) | Yambda-config alias for the same class as `linr_v1_filter_mask`. |
| `linr_v1_filter_mask` | `LinrV1Algo` | [algos/linr_v1.py](../../evaluation/retrieval/algos/linr_v1.py) | Yes | Dense `q @ E.T` + optional mask + topk. fp16 storage, fp32 reduction. |
| `linr_v3` | `LinrV3Algo` | [algos/linr_v3.py](../../evaluation/retrieval/algos/linr_v3.py) | Yes | Two-stage cascade: `OneBitKNN(k=candidate_pool)` → `PrefilterKNN(k)` rescore at fp32. Params: `candidate_pool`, `v3_seed`. |
| `linr_v4` | `LinrV4Algo` | [algos/linr_v4.py](../../evaluation/retrieval/algos/linr_v4.py) | Yes | Single-stage INT8 dense via `PostfilterKNNInt8` (`torch._int_mm` IMMA on Ampere+). |
| `linr_v2` | `LinrV2Algo` | [algos/linr_v2.py](../../evaluation/retrieval/algos/linr_v2.py) | Yes | Exact filtered top-K via `PrefilterKNN(candidate_ids=filter.evaluate_indices(qa))`. **Filter required**; raises on `filter_kind="none"`. Recall=1.0 by construction. |
| `silvertorch` | `SilvertorchAlgo` | [algos/silvertorch.py](../../evaluation/retrieval/algos/silvertorch.py) | Yes | The co-designed IVF + INT8 + Bloom retriever. Three filter modes routed through one class. Params: `n_lists`, `n_probe`, `n_iter`, `m_bits`, `k_hash`, `seed`. |

`BACKEND_CAPABLE_ALGOS = frozenset({"triton_knn", "linr_v1_filter_mask", "linr_v2", "linr_v3", "linr_v4", "silvertorch"})` at [algos/__init__.py:57-59](../../evaluation/retrieval/algos/__init__.py#L57-L59). `torch_knn` is the sole exception (one row per cell regardless of backend).

`build_algorithm` ([algos/__init__.py:62-121](../../evaluation/retrieval/algos/__init__.py#L62-L121)) is the factory; it raises `ValueError` for the single construction-time eligibility rule (`linr_v2` with `filter_kind="none"`). Silvertorch on reverse-clause sweeps is skipped at the harness level rather than in the factory.

### Filter sweeps — clause and bloom

Filter configurations declare named sweeps under `filters.clause.sweeps` and `filters.bloom.sweeps`. Each sweep names a subset of active narrow clauses.

**Goodreads filter sweeps** (from [evaluation/config/goodreads/d128-filter.yaml:25-47](../../evaluation/config/goodreads/d128-filter.yaml#L25-L47)):
- `clause`: 6 sweeps — `c0_genre`, `c1_lang_reverse`, `c2_format`, `c3_year`, `c0c1`, `all4`.
- `bloom`: 3 sweeps — `c0_genre`, `c2_format`, `c3_year`. (`c1_lang_reverse` is excluded because `BloomFilter` is forward-only — no NOT predicate.)
- Bloom parameters: `m_bits: 1024`, `k_hash: 5`.

**ArXiv filter sweeps** (from [evaluation/config/arxiv/d128-filter.yaml:26-46](../../evaluation/config/arxiv/d128-filter.yaml#L26-L46) — d64/d128/d256 share the same filter block):
- `clause`: 5 sweeps — `c0_maincat`, `c2_year`, `c3_nversions`, `c0c2`, `all4`. (No `c1` because the language clause on goodreads was reverse-only; arXiv uses a different schema.)
- `bloom`: 5 sweeps — same five names as `clause`. (`c1` is omitted symmetrically; BloomFilter could not handle reverse clauses anyway.)
- Bloom parameters: `m_bits: 1024`, `k_hash: 5`.
- Note: an earlier draft of these notes claimed "Bloom only" — that was incorrect; arXiv filter configs run both clause and bloom, just with no `c1` (language-reverse) clause.

### Deep-sweep parameter grids

Two deep-sweep configurations, each restricted to one algorithm and one dimension.

**1. `arxiv-d128-silvertorch.yaml`** ([evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml](../../evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml)):
- Algorithm: `silvertorch` only.
- 10 explicit `algo_params` combos: 2 cluster-count layouts × 5 `n_probe` values.

  | Layout | `n_lists` | `n_probe` values | Comment |
  |---|---|---|---|
  | A (paper-faithful $\sqrt{N}$) | 1664 | 4, 8, 32, 128, 256 | ~1800-item clusters; P ranges from ≈6.5k to ≈415k |
  | B (small clusters) | 8192 | 4, 8, 32, 128, 256 | ~365-item clusters; P ranges from ≈1.3k to ≈84k |

  Fixed across all combos: `n_iter: 10`, `m_bits: 1024`, `k_hash: 5`, `seed: 0`.
- Filter sweeps: 5 bloom sweeps — `c0_maincat`, `c2_year`, `c3_nversions`, `c0c2`, `all4`.
- `users_limit: 10000`, `ks: [100, 200, 400]`, `batch_sizes: [1, 8, 16]`, `backends: [triton, torch]`.

**2. `goodreads-d128-linr_v3.yaml`** ([evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml](../../evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml)):
- Algorithm: `linr_v3` only.
- 5 explicit `algo_params` combos: `candidate_pool ∈ {2000, 4000, 8000, 16000, 32000}` with fixed `v3_seed: 0`.
- Filter sweeps: same 6 clause + 3 bloom as the filter config.
- `users_limit: 10000`, `ks: [100, 200, 400]`, `batch_sizes: [1, 8, 16]`, `backends: [triton, torch]`.

### Cell-count examples — verified row counts

The cross-product is verified against the actual `evaluation/results/**/*.json` artifacts. Direct count of rows per combined JSON ([DATA: evaluation/results/]):

| Config | JSON path | Rows | Composition |
|---|---|---|---|
| `yambda-500m/d{64,128,256}-quality.yaml` | `yambda/500m-d{64,128,256}.json` | 24 each | 4 algos × 2 backends × 1 bs × 3 K |
| `yambda-5b/d{64,128}-quality.yaml` | `yambda/5b-d{64,128}.json` | 24 each | 4 algos × 2 backends × 1 bs × 3 K |
| `arxiv/d{64,128,256}-quality.yaml` | `arxiv/d{64,128,256}-quality.json` | 24 each | 4 algos × 2 backends × 1 bs × 3 K |
| `goodreads/d{64,128,256}-quality.yaml` | `goodreads/d{64,128,256}-quality.json` | 24 each | 4 algos × 2 backends × 1 bs × 3 K |
| `arxiv/d{64,128,256}-filter.yaml` | `arxiv/d{64,128,256}-filter.json` | 900 each | (4 algos × 2 bk + linr_v2 × 2 bk) × 3 bs × 3 K × (5 clause + 5 bloom) = 8×90 + 2×90 = 900 |
| `goodreads/d{64,128,256}-filter.yaml` | `goodreads/d{64,128,256}-filter.json` | 810 each | (4 algos × 2 bk + linr_v2 × 2 bk) × 3 bs × 3 K × (6 clause + 3 bloom) = 8×81 + 2×81 = 810 |
| `deep_sweeps/arxiv-d128-silvertorch.yaml` | `deep_sweeps/arxiv-d128-silvertorch.json` | 900 | 1 algo × 2 bk × 3 bs × 3 K × 10 params × 5 sweeps |
| `deep_sweeps/goodreads-d128-linr_v3.yaml` | `deep_sweeps/goodreads-d128-linr_v3.json` | 810 | 1 algo × 2 bk × 3 bs × 3 K × 5 params × 9 sweeps |
| **Total** | 19 files | **7,104 rows** | |

Method: a small Python pass over the JSON files counts rows and the unique tuple of `(impl, backend, batch_size, k)` per file. All 17 quality+filter configs and both deep sweeps produced exactly the expected counts — the harness is consistent and complete across the published campaign.

### Writer's notes — §5.5

- The Sweep-dimensions table above is one of the mandatory visual deliverables.
- The chapter should emphasise that the `algorithms` list in a YAML is the truth-source for which classes run; the `ALGORITHMS` tuple in the registry is the truth-source for which strings are valid.
- The `triton_knn`/`linr_v1_filter_mask` alias is historical and slightly confusing. Quote the docstring comment at [algos/__init__.py:21-24](../../evaluation/retrieval/algos/__init__.py#L21-L24) so the writer understands.
- The deep-sweep parameter grids matter for Ch.6 §6.3 — the writer should cross-reference these tables when discussing the recall-vs-latency Pareto curves.
- The 10-combo silvertorch deep sweep is described as "two cluster-count layouts" (A: $\sqrt{N}$-paper-faithful, B: small clusters). Pull this language into prose verbatim.
- The d64 and d256 variants of the silvertorch deep sweep are **not** in `config/deep_sweeps/` — only d128 is exercised at the parameter-grid level. `git log --all -p -S "deep_sweeps/arxiv-d64-silvertorch"` and the corresponding d256 query return zero matches — these configs have never existed in repo history. The intent is implicit in the YAML comment ([config/deep_sweeps/arxiv-d128-silvertorch.yaml:26](../../evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml#L26)): the paper-faithful $n_\mathrm{lists} \approx \sqrt{N} = 1664$ layout is a function of catalogue size $N$, not dimension $D$, so the deep sweep is dimension-agnostic at the IVF level; the marginal information from running the same sweep at d64 / d256 is the latency curve. The choice to only run d128 is a compute-budget cut. The writer can present this as "deep sweeps at d128 as the canonical dimension; d64 and d256 latency curves left as future work" — cross-reference Ch.7.

---

## §5.6 Окружение и hardware (Environment and hardware)

### GPU and software stack

Recorded in [evaluation/results/_runlogs/SUMMARY.quality-deep.txt](../../evaluation/results/_runlogs/SUMMARY.quality-deep.txt) at the head of the most recent campaign:

```
started at:  2026-05-23T09:52:29Z
host:        fd4a94de96b9
gpu:         NVIDIA A100-SXM4-80GB
eval-type:   (ad-hoc)
stage:       True
```

- **GPU**: NVIDIA A100-SXM4-80GB. Ampere (GA100), 80 GB HBM2e, FP32 peak ≈ 19.5 TFLOPS, TF32/FP16 tensor-core peak ≈ 312 TFLOPS, INT8 IMMA peak ≈ 624 TOPS, dp4a INT8 CUDA-core throughput ≈ 78 TOPS. Used for every run logged in [evaluation/results/](../../evaluation/results/).
- **GPU detection**: `nvidia-smi --query-gpu=name --format=csv,noheader`. Source: [cli/run_evaluation.py:61-69](../../evaluation/retrieval/cli/run_evaluation.py#L61-L69). Falls back to `(no nvidia-smi)` if absent.
- **Host**: `fd4a94de96b9` — a container hostname. Hardware sits on a rented Lambda/Coreweave/etc. node; identity isn't load-bearing for reproducibility.
- **Library versions** (resolved from [uv.lock](../../uv.lock)):
  - **PyTorch**: `torch==2.10.0+cu128` from the index `https://download.pytorch.org/whl/cu128` ([uv.lock:2461-2463](../../uv.lock#L2461-L2463)).
  - **Triton**: `triton==3.6.0` from PyPI ([uv.lock:2583-2584](../../uv.lock#L2583-L2584)).
  - **CUDA toolkit**: the `+cu128` wheel tag corresponds to CUDA 12.8 runtime (paired with PyTorch 2.10.0).
  - The library's declared bounds at [retrieve/pyproject.toml:9-10](../../retrieve/pyproject.toml#L9-L10) are `torch>=2.4,<3` and `triton>=3.0`; the actual resolved versions sit above these floors.
- **CUDA driver version**: not auto-captured in the SUMMARY. [TODO: clarify with author — to be definitive, capture from `nvidia-smi --query-gpu=driver_version --format=csv,noheader` at the time of the published run; the rented A100 host driver is whatever the cloud provider ships, typically a recent R555+ for CUDA 12.8 compatibility.]
- **Campaign wall-time**: the most recent end-to-end run logged in [SUMMARY.quality-deep.txt](../../evaluation/results/_runlogs/SUMMARY.quality-deep.txt) covers 8 configurations in **15 248 s ≈ 4 h 14 m** of GPU time, broken down as: arxiv quality (d64/d128/d256) 100/106/200 s; goodreads quality (d64/d128/d256) 2 166/2 430/2 932 s; deep sweeps (arxiv-d128-silvertorch / goodreads-d128-linr_v3) 4 012/3 302 s. Goodreads quality is the wall-time long pole (full 313k user test stream at bs=1).

### Determinism settings

Three layers of determinism are enforced at the start of each `evaluate` subprocess.

**Layer 1 — torch seeds** ([cli/evaluate.py:60-64](../../evaluation/retrieval/cli/evaluate.py#L60-L64)):

```python
torch.manual_seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
```

- `torch.manual_seed(cfg.seed)` seeds CPU and default CUDA device.
- `torch.cuda.manual_seed_all` seeds all CUDA devices (safe for multi-GPU).
- All shipped configs use `seed: 0`.

**Layer 2 — precision pinning** ([bench_tools.py:61-73](../../evaluation/retrieval/bench_tools.py#L61-L73)):

```python
def pin_precision_globals() -> None:
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
```

Called from `run_sweep` at the top of every suite ([sweep.py:68](../../evaluation/retrieval/sweep.py#L68)). The rationale (from the docstring): "Without this, anything earlier in the process (an upstream import, an unrelated model load) could flip `allow_tf32` and silently change both numerics and throughput. Eval rows must be comparable across runs, so we pin to the strict path."

A more specific rationale from [docs/system/evaluation.md:168-175](../system/evaluation.md#L168-L175):

> Keeps the oracle (cuBLAS `q @ E_t`) and the algos in the same precision so exact-mask paths (`linr_v1_filter_mask`) don't show ~1e-3 recall drift on narrow filters where top-K boundaries land within TF32's 10-bit mantissa band.

**Layer 3 — perf-pass query pool generator** ([bench_tools.py:409](../../evaluation/retrieval/bench_tools.py#L409)):

```python
g = torch.Generator(device="cpu").manual_seed(seed)
rows = torch.randint(0, n, (n_pool, batch_size), generator=g)
```

The same query-batch indices are sampled across reruns with the same seed. On filter cells, `qa_narrow` is sampled from the same row indices so the filter shape matches the queries. See §5.7 (perf pass) below for the full multi-query-pool design.

**Per-algorithm seeds**: independent of the global seed:
- `silvertorch.seed` — IVF cluster-assignment seed (passed to `KMeansTorch`).
- `linr_v3.v3_seed` — Sign-OPORP projection seed (passed to `OneBitKNN`).
- `linr_v2`, `linr_v4`, `linr_v1_filter_mask`, `torch_knn` — paramless (no additional seed).

These are wired through `algo_params[algo][seed]` ([algos/__init__.py:112-117](../../evaluation/retrieval/algos/__init__.py#L112-L117)).

### Determinism guarantees

From [docs/system/evaluation.md:315-326](../system/evaluation.md#L315-L326):

> Quality columns must be **byte-identical** across reruns with the same seed. Latency may drift within ~5% due to clock noise, NVML thermal state, and (for Triton autotune) JIT cache state.

- **Quality reproducibility**: every reproducibility test would pin `recall@k` and `ndcg@k` columns to bit-identical floats across two `uv run evaluate` calls with the same YAML and the same seed.
- **Latency drift**: median/p20/p80 latencies can drift ~5% due to (a) GPU clock noise and thermal throttling, (b) Triton JIT autotune cache state warming differently across runs.
- **Memory drift**: `peak_mem_mib` / `index_mem_mib` / `fwd_scratch_mib` are more stable than latency (allocator state plus a small jitter from `do_bench`'s internal buffers) but not bit-exact across runs.

### Compile cache management

`torch._dynamo.config.recompile_limit = 64` ([cli/evaluate.py:58](../../evaluation/retrieval/cli/evaluate.py#L58)) — guards against excessive recompilation within a single algorithm's cells. The per-algo process isolation ensures the Dynamo cache starts fresh for each algo (a new Python process resets it).

`torch._dynamo.reset()` + `torch.cuda.empty_cache()` at the end of each sweep ([sweep.py:190-192](../../evaluation/retrieval/sweep.py#L190-L192)) — drops the compile cache + CUDA-graph private pools between sweeps. Cost ~30-60 s of recompile at the start of the next sweep; benefit is avoiding OOM on large-N configs (observed on arxiv/d256).

### GPU pre-warm

Called once per `evaluate` subprocess at suite start ([bench_tools.py:76-92](../../evaluation/retrieval/bench_tools.py#L76-L92)):

```python
def warm_gpu_once(device: torch.device) -> None:
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)
    for _ in range(3):
        (a @ b).sum().item()  # .item() forces a sync per iter
    torch.cuda.synchronize()
    del a, b
```

Rationale: "The very first GPU op in a process pays cuBLAS-handle init, kernel-module load, and pinned-memory allocator setup — typically 0.5–1.5 s. Per-cell warmup loops absorb this on whichever cell happens to run first, which is unfair to that cell. Run a tiny mat-mul once at top-of-suite so the cost is paid outside any measured window." Source: [bench_tools.py:77-83](../../evaluation/retrieval/bench_tools.py#L77-L83).

### Why per-cell warmup is not enough

Layered with the global pre-warm, per-cell autotune prewarm is also done before the perf loop:

`_autotune_prewarm` ([sweep.py:606-631](../../evaluation/retrieval/sweep.py#L606-L631)) calls `algo_obj(q, …)` once per batch size before any `bs` is timed. Rationale from the docstring: "Per-bs warmup inside `perf_pass_cached` is supposed to populate Triton's autotune cache, but a slow last-mile autotune config has been observed leaking into the timing window (median collapsing to a single ~1.5 s sample on the first cell). Belt-and-suspenders defense."

### Writer's notes — §5.6

- Table of fixed-version dependencies is mandatory; pull from `uv.lock`. [TODO: clarify with author] for exact versions.
- The "byte-identical quality, ±5% latency" guarantee is the load-bearing reproducibility claim of the chapter — should be quoted in the abstract / introduction.
- The chapter should note the experimental setup is **single-node, single-GPU**. Multi-GPU is out of scope (cross-ref Ch.7 §7.1).
- Triton autotune is non-deterministic across runs (it picks the fastest config seen during JIT compilation, but the "fastest" can shift with clock noise). The harness ships single offline-tuned `DEFAULT_CONFIG` per kernel (cross-ref [05-implementation.md](05-implementation.md) §4.6) to remove this source of variance — the per-cell perf time still has jitter but the kernel-config dimension is removed.
- A100 SKU resolved (no longer a TODO): the SUMMARY records `NVIDIA A100-SXM4-80GB`. That is the SXM4 form factor (NVLink-attached, not PCIe) with 80 GB of HBM2e — the same SKU as DGX-A100 nodes and most cloud A100 offerings (Lambda, Coreweave, AWS p4de).

---

## §5.7 Result schema

### One row per cell

Every cell in the sweep cross-product emits exactly one row. The row constructor is `_make_perf_row` ([sweep.py:634-679](../../evaluation/retrieval/sweep.py#L634-L679)):

```python
return {
    "suite": suite,
    "cell": f"{filter_kind}_{sweep_name}_{backend}_bs{bs}_k{k}",
    "filter_kind": filter_kind,
    "sweep": sweep_name,
    "impl": algo,
    "backend": backend,
    "device": "cpu" if is_cpu else "cuda",
    "seed": seed,
    "batch_size": bs,
    "k": k,
    "n_users_kept": n_kept,
    "median_ms": med,
    "p20_ms": p20,
    "p80_ms": p80,
    "peak_mem_mib": peak,
    "index_mem_mib": index_mem,
    "fwd_scratch_mib": scratch,
    f"recall@{k}": recall,
    f"ndcg@{k}": ndcg,
    "extra": {"params": {str(pk): str(pv) for pk, pv in params.items()}},
}
```

Algorithms that fail `build_algorithm` with `ValueError` emit **no row** for the cell (no stub, no NaN). See [sweep.py:517-546](../../evaluation/retrieval/sweep.py#L517-L546).

### Field-by-field schema

| Field | Type | Source | Notes |
|---|---|---|---|
| `suite` | `str` | [sweep.py:72](../../evaluation/retrieval/sweep.py#L72) | `"yambda"` (no filters in config) or `"filter"`. See §5.1 caveat about naming. |
| `cell` | `str` | [sweep.py:660](../../evaluation/retrieval/sweep.py#L660) | Stable join key: `f"{filter_kind}_{sweep_name}_{backend}_bs{bs}_k{k}"`. Note: does **not** include `impl`, `params`, or `seed`. Use the full multi-column key for cross-row joins when those vary. |
| `filter_kind` | `str` | sweep loop | `"none"`, `"clause"`, or `"bloom"`. |
| `sweep` | `str` | `FilterSweepCfg.name` | E.g. `c0_genre`, `c0c1`, `all4`, `full_scan` (synthetic on yambda). |
| `impl` | `str` | inner loop | Algorithm name (`linr_v1_filter_mask`, `silvertorch`, …). |
| `backend` | `str` | inner loop | `"triton"` or `"torch"`. For `torch_knn` always the first listed backend. |
| `device` | `str` | `"cpu" if algo_obj.is_cpu else "cuda"` | `"cuda"` for every shipped run; `"cpu"` reserved for future CPU baselines. |
| `seed` | `int` | `cfg.seed` | Currently `0` for every shipped run. |
| `batch_size` | `int` | per-`bs` loop | First-class column for cross-`bs` joins. |
| `k` | `int` | `cfg.ks` loop | First-class column. |
| `n_users_kept` | `int` | [sweep.py:339](../../evaluation/retrieval/sweep.py#L339) | Users after `skip_mask` (drops users whose synthesised QA is all `-1` for active clauses) and after `users_limit`. |
| `median_ms` | `float` | `perf_pass_cached` → `do_bench(quantiles=[0.5])` | Latency over the multi-query pool. |
| `p20_ms` | `float` | same | 20th percentile. |
| `p80_ms` | `float` | same | 80th percentile. |
| `peak_mem_mib` | `float` | `measure_forward_cuda` peak window | `max_memory_allocated()` over `MEM_REPS=16` reps within the memory window. 0 for CPU rows. |
| `index_mem_mib` | `float` | `cuda_allocated_mib() - mem_before` | Delta around `build_algorithm` — marginal cost of the index (excludes shared `item_embs` and transient build-time scratch). 0 for CPU rows. |
| `fwd_scratch_mib` | `float` | `max(0, peak - baseline) / (1024*1024)` | Peak minus pre-call baseline within the forward call. 0 for CPU rows. |
| `recall@{k}` | `float` | `quality_pass_cached` → `finalize_metrics` | Mean per-query Recall@k. Identical across all `bs` rows of the same `(algo, params, k)` cell. NaN when `--skip-quality`. |
| `ndcg@{k}` | `float` | same | Mean per-query NDCG@k. Same replication rule. |
| `extra.params` | `dict[str, str]` | `_make_perf_row` | Per-algo `algo_params` for traceability. **All values are stringified** (`int → str`, `float → str`). |

### Sample row — yambda quality

From `evaluation/results/yambda/500m-d128.json`:

```json
{
  "suite": "yambda",
  "cell": "none_full_scan_triton_bs1_k100",
  "filter_kind": "none",
  "sweep": "full_scan",
  "impl": "linr_v1_filter_mask",
  "backend": "triton",
  "device": "cuda",
  "seed": 0,
  "batch_size": 1,
  "k": 100,
  "n_users_kept": 45932,
  "median_ms": 0.7097,
  "p20_ms": 0.7086,
  "p80_ms": 0.7446,
  "peak_mem_mib": 1370.00,
  "index_mem_mib": 456.00,
  "fwd_scratch_mib": 0.0,
  "recall@100": 0.1486,
  "ndcg@100": 0.1028,
  "extra": {"params": {}}
}
```
[DATA: evaluation/results/yambda/500m-d128.json]

### Sample row — filtered cell with parameters

From `evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json` (truncated):

```json
{
  "suite": "filter",
  "cell": "clause_c0_genre_triton_bs1_k100",
  "filter_kind": "clause",
  "sweep": "c0_genre",
  "impl": "linr_v3",
  "backend": "triton",
  "device": "cuda",
  "seed": 0,
  "batch_size": 1,
  "k": 100,
  "n_users_kept": 9859,
  "median_ms": 0.3608,
  "p20_ms": 0.3580,
  "p80_ms": 0.3630,
  "peak_mem_mib": 696.89,
  "index_mem_mib": 206.77,
  "fwd_scratch_mib": 0.0,
  "recall@100": 0.7521,
  "ndcg@100": 0.8136,
  "extra": {"params": {"candidate_pool": "2000", "v3_seed": "0"}}
}
```
[DATA: evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json]

### How latency is measured

The latency triple `(median_ms, p20_ms, p80_ms)` comes from `triton.testing.do_bench` with `quantiles=[0.5, 0.2, 0.8]`. Source: [bench_tools.py:154-156](../../evaluation/retrieval/bench_tools.py#L154-L156). The harness wraps `do_bench` with auto-extending sample count:

```python
cur_rep = float(rep_ms)               # start at 200 ms
while True:
    warmup_ms = max(50.0, cur_rep * 0.5)
    median, p20, p80 = ttesting.do_bench(
        fn, quantiles=[0.5, 0.2, 0.8], rep=cur_rep, warmup=warmup_ms
    )
    if median <= 0 or cur_rep >= max_rep_ms:    # max 3000 ms
        break
    if cur_rep / median >= min_samples:         # need 30 samples
        break
    cur_rep = min(max_rep_ms, median * min_samples * 1.2)
```

Source: [bench_tools.py:149-161](../../evaluation/retrieval/bench_tools.py#L149-L161). The rationale: "kernels whose single-call latency exceeds `rep_ms` collapse to one sample and return `median == p20 == p80` — a cold-start fingerprint that's indistinguishable from a real measurement". Source: [bench_tools.py:110-116](../../evaluation/retrieval/bench_tools.py#L110-L116).

Constants ([bench_tools.py:33-43](../../evaluation/retrieval/bench_tools.py#L33-L43)):
- `WARMUP_ITERS = 20`
- `DEFAULT_REP_MS = 200.0`
- `MIN_SAMPLES = 30`
- `MAX_REP_MS = 3000.0`
- `MEM_REPS = 16`

### Multi-query pool

The latency is **not** measured against a single fixed query batch — that collapses p20/p80 for IVF-style algorithms because one query traverses one cluster. Source: [bench_tools.py:398-402](../../evaluation/retrieval/bench_tools.py#L398-L402) and [docs/system/evaluation.md:276-294](../system/evaluation.md#L276-L294).

Pool construction ([bench_tools.py:409-420](../../evaluation/retrieval/bench_tools.py#L409-L420)):

```python
g = torch.Generator(device="cpu").manual_seed(seed)
n = queries.shape[0]
if skip_mask is not None:
    keep_idx = (~skip_mask.bool()).nonzero(as_tuple=False).reshape(-1)
    rows_local = torch.randint(0, keep_idx.numel(), (n_pool, batch_size), generator=g)
    rows = keep_idx[rows_local.reshape(-1)].reshape(n_pool, batch_size)
else:
    rows = torch.randint(0, n, (n_pool, batch_size), generator=g)
pool = queries[rows.reshape(-1)].reshape(n_pool, batch_size, -1).to(device).contiguous()
```

Default `n_pool = 4096`. The pool is fixed-seed and round-robin'd into the timed function:

```python
counter = {"i": 0}
def perf_fn():
    idx = counter["i"] % n_pool
    counter["i"] += 1
    q = pool[idx]
    ...
    return forward(q, **kw)
```

Source: [bench_tools.py:431-444](../../evaluation/retrieval/bench_tools.py#L431-L444).

### How memory is measured

Three memory columns per row:

1. **`peak_mem_mib`** — `torch.cuda.max_memory_allocated()` in MiB, captured over the dedicated memory window. The window runs `MEM_REPS = 16` iterations of `fn()` after warmup, keeping the last output alive (a `keep_alive` list) so the peak captures result + scratch. Source: [bench_tools.py:136-147](../../evaluation/retrieval/bench_tools.py#L136-L147). The window does **not** use `do_bench` because `do_bench` allocates a ~256 MiB L2 cache-buster each call, which would dominate the reported transient peak.
2. **`index_mem_mib`** — `cuda_allocated_mib() - mem_before` taken around `build_algorithm` (after `_reset_cuda_state_for_cell` cleared the pool and reset the peak counter). Captures the marginal cost of the algorithm's index, excluding shared `item_embs` and transient build-time scratch the caching allocator has since released. Source: [sweep.py:425, 442](../../evaluation/retrieval/sweep.py#L425) and [bench_tools.py:219-226](../../evaluation/retrieval/bench_tools.py#L219-L226).
3. **`fwd_scratch_mib`** — `max(0, peak - baseline) / (1024*1024)` where `baseline = allocated_bytes()` is captured just before the memory window. Captures the transient peak inside the forward call, distinct from the persistent index footprint. Source: [bench_tools.py:138-147, 166](../../evaluation/retrieval/bench_tools.py#L138-L166).

Importantly, no `empty_cache()` is called between warmup and timing: this is intentional to preserve pool reuse, which matches the steady-state behaviour of long-running query streams. Source: [bench_tools.py:118-121, 132-135](../../evaluation/retrieval/bench_tools.py#L118-L135).

### Planned schema extensions (post-defense, not yet present)

The following fields are referenced by the Chapter 6 plot catalog ([docs/thesis/results-data/recipes/plot_catalog.md](../../docs/thesis/results-data/recipes/plot_catalog.md)) and would unlock additional figures/tables. Status: **deferred** — not currently emitted by `bench_tools.py` or recorded in any JSON. Listed here so Ch.5 documents the gap honestly rather than implying the data exists.

| Field | Type | Source measurement | Unlocks |
|---|---|---|---|
| `throughput_qps` | float | `n_queries_total / total_wall_clock_s` directly over the timing window | A2, A4, B4, G2 — true Recall–QPS Pareto (eliminates the `1/median_ms` rate-statistic fudge) |
| `mean_ms` | float | arithmetic mean per-batch over warm runs | sanity-check against `median_ms`; rough QPS approximation `B/mean_ms` |
| `p99_ms` (and ideally `p999_ms`) | float | empirical quantile of the per-call distribution | E1 violin plots; any online-serving SLO claim |
| `build_time_s` | float (or dict of phase floats) | wall-clock for `build_algorithm`, optionally split into (encode / quantize / cluster / assemble) | F4 build-time bar chart; ann-benchmarks "Recall vs build time" |
| `topk_ids_jaccard_vs_torch` | float | `|triton_topk ∩ torch_topk| / |union|`, target >0.999 | F1 / G4 stronger parity guarantee than `recall_abs_diff` alone |
| kernel-level timing | dict | per-Triton-kernel ms + occupancy from Nsight Compute | F3 kernel breakdown table |
| per-query latency vector | list[float] | full distribution, not just median/quantiles | E1 violin plots; per-query paired permutation tests without re-running |

Each addition is independent: extending `bench_tools.py` for any single one is a self-contained change. Re-running the sweep is the bottleneck — order of operations should be: (1) `throughput_qps` + `mean_ms` + `p99_ms` first (cheapest re-run), (2) `build_time_s` (requires per-phase instrumentation), (3) Jaccard + per-query latency vector (requires the harness to keep top-k index outputs and timing logs across calls), (4) Nsight kernel profiling (manual one-off per kernel, not a bulk re-run).

### Writer's notes — §5.7

- This subsection's schema table is one of the mandatory visual deliverables.
- Show one sample row in Russian prose; the table above can stay in English code-block form.
- The `extra.params` field is stringified at construction; this is a UX choice that prevents type confusion in JSON readers but loses round-trip fidelity (consumers must `int(s)` / `float(s)` themselves). Worth a brief mention.
- The `cell` field excludes `impl`, `params`, and `seed` from its identifier. For analysis joins across algorithms/params/seeds, the full multi-column key is required. Worth noting because it's a common gotcha.
- The harness reports latencies and never fails a build; correctness lives in `retrieve/tests/` and gates CI separately (cross-ref [05-implementation.md](05-implementation.md) §4.9). Source: [docs/system/evaluation.md:59-60](../system/evaluation.md#L59-L60).
- [TODO: clarify with author] — should the chapter formalise the schema as a JSON Schema document in an appendix? The current code is the truth-source.
- The "Planned schema extensions" sub-table above is the honest disclosure of what's MISSING. Mirror it in Ch.7 (Limitations) as a "future-work" entry: "the harness can be extended with seven additional fields (throughput_qps, mean_ms, p99_ms, build_time_s, topk_ids_jaccard_vs_torch, per-query latency vector, kernel-level timing) to remove the current limitations on §6.2.9 (latency-at-fixed-recall), §6.5.3 (K sensitivity), §6.7.5 (kernel breakdown), §6.7.6 (build time), and §6.8.1 (seed stability)."

---

## §5.8 Layout результатов (Results layout)

### Directory tree (current state, May 2026)

```
evaluation/results/
├── _runlogs/
│   └── SUMMARY.quality-deep.txt
├── yambda/
│   ├── 500m-d{64,128,256}.json          # combined (staged)
│   ├── 500m-d{64,128,256}.yaml          # YAML config snapshot
│   ├── 500m-d{64,128,256}.perkernel/    # per-algo originals
│   ├── 5b-d{64,128}.json
│   ├── 5b-d{64,128}.yaml
│   └── 5b-d{64,128}.perkernel/
├── goodreads/
│   ├── d{64,128,256}-{quality,filter}.json
│   ├── d{64,128,256}-{quality,filter}.yaml
│   └── d{64,128,256}-{quality,filter}.perkernel/
├── arxiv/
│   ├── d{64,128,256}-{quality,filter}.json
│   ├── d{64,128,256}-{quality,filter}.yaml
│   └── d{64,128,256}-{quality,filter}.perkernel/
└── deep_sweeps/
    ├── arxiv-d128-silvertorch.{json,yaml}
    ├── arxiv-d128-silvertorch.perkernel/
    ├── goodreads-d128-linr_v3.{json,yaml}
    └── goodreads-d128-linr_v3.perkernel/
```

Verified by `ls evaluation/results/` and subdirectories.

### Per-algo vs combined JSONs

The `.perkernel/` convention separates the two layouts:

- **Runtime per-algo writes** — each `uv run evaluate --algo <name>` writes one JSON to `cfg.output/<name>.json`. The directory `cfg.output` is initially a directory of per-algo files. Source: [cli/evaluate.py:93-97](../../evaluation/retrieval/cli/evaluate.py#L93-L97).
- **Post-stage combined** — `uv run stage-results <yaml>` concatenates `cfg.output/*.json` into a single `<cfg.output>.json` (one level up, with `.json` suffix), then renames `cfg.output/` → `<cfg.output>.perkernel/`, and copies the YAML to `<cfg.output>.yaml`. Source: [cli/stage_results.py](../../evaluation/retrieval/cli/stage_results.py).

Example state before staging:
```
results/goodreads/d128-filter/
  ├── linr_v1_filter_mask.json
  ├── linr_v2.json
  ├── linr_v3.json
  ├── linr_v4.json
  └── silvertorch.json
```

After staging:
```
results/goodreads/d128-filter.json           # combined (~810 rows)
results/goodreads/d128-filter.yaml           # YAML snapshot
results/goodreads/d128-filter.perkernel/     # per-algo originals retained
  ├── linr_v1_filter_mask.json
  ├── linr_v2.json
  ├── linr_v3.json
  ├── linr_v4.json
  └── silvertorch.json
```

The per-algo files are retained (not deleted) so a failed algorithm can be re-run individually and re-staged without re-running the cohort.

### Runlogs

[`evaluation/results/_runlogs/`](../../evaluation/results/_runlogs/) is structured as follows, but **only the campaign-summary file is checked into the repository** — per-config logs are git-ignored after the workspace restructure commit (`79696c8`). What runs locally:
- `SUMMARY.quality-deep.txt` — campaign summary (one line per config + start/finish stamps). **Checked in** ([evaluation/results/_runlogs/SUMMARY.quality-deep.txt](../../evaluation/results/_runlogs/SUMMARY.quality-deep.txt)). Verbatim example reproduced in §5.6.
- `full.log` — concatenated full output of every subprocess invocation. **Local-only.**
- `<cfg_name>.log` — per-config log (e.g. `arxiv_d128-quality.log`). One per YAML. **Local-only.**
- `current.log` — symlink updated to point at the active per-config log; useful for `tail -f` during a run. **Local-only.**

The naming convention `<cfg_name>` (e.g. `arxiv_d128-quality`) is produced by `_cfg_name` ([cli/run_evaluation.py:72-77](../../evaluation/retrieval/cli/run_evaluation.py#L72-L77)) and follows the rule `config/arxiv/d128-quality.yaml → arxiv_d128-quality`.

Naming convention from [cli/run_evaluation.py:72-77](../../evaluation/retrieval/cli/run_evaluation.py#L72-L77):
```python
def _cfg_name(cfg: Path) -> str:
    """`config/arxiv/d64-filter.yaml` -> `arxiv_d64-filter`."""
    rel = str(cfg.resolve().relative_to(EVAL_DIR).with_suffix(""))
    if rel.startswith("config/"):
        rel = rel[len("config/"):]
    return rel.replace("/", "_")
```

### Per-row analysis path: `load_results`

The canonical read function is `load_results(dir)` ([results_io.py:25-31](../../evaluation/retrieval/results_io.py#L25-L31)):
```python
def load_results(results_dir: str | Path) -> list[dict]:
    """Return the concatenation of every <results_dir>/*.json in name order."""
    rows: list[dict] = []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            rows.extend(json.load(f))
    return rows
```

Used by analysis notebooks and plotting code; concatenates across either per-algo (under `.perkernel/`) or combined-staged files. Output is a flat `list[dict]` ready for Pandas: `pd.DataFrame(load_results(...))`.

The resume-check variant is `load_rows(path)` ([results_io.py:15-22](../../evaluation/retrieval/results_io.py#L15-L22)) — loads one file and returns `None` on missing/unparseable/not-a-list. Used by the orchestrator to decide whether `(config, algo)` is already complete (`--resume` flag).

### Upload to HuggingFace

The `upload_results.py` script ([cli/upload_results.py](../../evaluation/retrieval/cli/upload_results.py)) stages and uploads combined JSONs to a HuggingFace dataset. The relevant constants and entry points:

- `REPO_ID = "pinkmeme/retrieval-filter-evals-2026-05-23"`, `REPO_TYPE = "dataset"`, `PRIVATE = False` ([cli/upload_results.py:27-29](../../evaluation/retrieval/cli/upload_results.py#L27-L29)). The dataset is therefore **public**; the canonical URL is `https://huggingface.co/datasets/pinkmeme/retrieval-filter-evals-2026-05-23`. The README's "Schema reference" link ([cli/upload_results.py:116](../../evaluation/retrieval/cli/upload_results.py#L116)) currently points at the earlier-dated `pinkmeme/retrieval-filter-evals-2026-05-20` — minor cross-repo drift that the writer may want to footnote if both URLs end up in the thesis.
- `_find_combined_jsons()` ([cli/upload_results.py:32-44](../../evaluation/retrieval/cli/upload_results.py#L32-L44)) walks `evaluation/results/` and excludes `*.perkernel/` and `_*` directories.
- `_classify(rel_path)` ([cli/upload_results.py:47-59](../../evaluation/retrieval/cli/upload_results.py#L47-L59)) buckets results for the README:
  ```python
  if parts[0] == "deep_sweeps":  return "deep_sweeps"
  if parts[0] == "yambda":       return "yambda"
  if "filter" in name:           return "filter"
  if "quality" in name:          return "quality"
  return "other"
  ```
- `stage_all()` copies combined JSONs + YAMLs to a fresh `upload_staging/` directory and gathers per-file statistics (row count, unique impls, yaml-present flag).
- `write_readme(stats)` produces a Markdown README per HF convention with the four sections `filter / quality / yambda / deep_sweeps`.
- `upload(dry_run)` invokes `HfApi.create_repo` + `upload_folder`.

CLI: `uv run upload-results [--dry-run]`.

### Writer's notes — §5.8

- The `.perkernel/` convention should be explained early in the section — it's the load-bearing distinction between staged and unstaged outputs and a reader will encounter it before any of the consumers.
- The HuggingFace upload is a Ch.7 / Conclusion-flavoured artefact (the open-source release) — appropriate to mention here but not to detail. The Conclusion / abstract should cite the published HF dataset path.
- The HF dataset URL is confirmed public (`PRIVATE = False`). Recommended citation: `https://huggingface.co/datasets/pinkmeme/retrieval-filter-evals-2026-05-23`. Author should confirm URL stability for the defence date (HF repos can be renamed).
- Visual deliverable opportunity: a tree diagram of the results layout (one of the chapter's required visuals could be combined with §5.1 architecture).
- The `SUMMARY.quality-deep.txt` example is short enough to reproduce verbatim in the chapter; it's a useful concrete grounding for the campaign-execution narrative.

---

## Visual deliverables (specifications, not rendered)

Per the Ch.5 contract in [00-thesis-plan.md](00-thesis-plan.md):

### Visual 1 — Flow diagram (cell → algorithms → sweep → metrics → JSON)

**Type:** Block diagram (TikZ recommended; PlantUML or Mermaid acceptable in source).

**Content:** Four nested stages mirroring the loop structure in [sweep.py:8-13](../../evaluation/retrieval/sweep.py#L8-L13):

```
┌─ run_sweep ────────────────────────────────────────────────────────────┐
│  pin_precision_globals   warm_gpu_once   resolve K_GT, suite           │
│ ┌─ run_filter_kind (per filter_kind ∈ {none, clause, bloom}) ────────┐ │
│ │  _apply_users_limit   _build_filter_modules (per backend)          │ │
│ │ ┌─ run_one_sweep (per FilterSweepCfg) ─────────────────────────┐   │ │
│ │ │  build_sweep_qa (skip_mask)   load_or_build_oracle           │   │ │
│ │ │ ┌─ evaluate_cell (per algo × backend × params × k) ────────┐ │   │ │
│ │ │ │  reset_cuda  build_algo  _run_quality  prewarm           │ │   │ │
│ │ │ │  per bs: perf_pass_cached → _make_perf_row → append      │ │   │ │
│ │ │ └──────────────────────────────────────────────────────────┘ │   │ │
│ │ └──────────────────────────────────────────────────────────────┘   │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         list[dict] → JSON
```

Driving data: code references in [sweep.py](../../evaluation/retrieval/sweep.py). Output: one diagram per chapter.

### Visual 2 — Sweep dimensions table

Source: §5.5 above. Two-column table: dimension name × values per config family. Lift from the §5.5 "Top-level dimensions" table; reformat to LaTeX `tabular` or `longtable`.

### Visual 3 — Result schema table

Source: §5.7 above. Four-column table: field × type × source × notes. Lift from the §5.7 "Field-by-field schema" table; reformat to LaTeX `tabular` (likely with `longtable` because it has 20+ rows).

---

## Visual deliverables for Chapter 6 (Results)

The full figure & table specifications, including catalog IDs, axes, source CSVs, and Pareto / faceting conventions, live in the dedicated catalog at [docs/thesis/results-data/recipes/plot_catalog.md](../../docs/thesis/results-data/recipes/plot_catalog.md). The notes file [07-results.md](07-results.md) annotates each figure stub with its catalog ID (e.g. "Fig 6.2.6 = A1").

This summary section is the table-of-contents view; consult the catalog or 07-results.md for plot-rendering details.

| ID | Title | Section | Source | Status |
|---|---|---|---|---|
| **A1** | Recall–latency Pareto (online serving, bs=1), faceted 3×3 | §6.2.6 | `all_results_long.csv` (filter_kind=none, bs=1, k=100) | upgrade existing `pareto_quality_3x3.png` |
| **A2** | Recall–latency Pareto (batched, bs=16), faceted 3×3 | §6.2.7 | same CSV, bs=16 | new |
| **A3** | Recall–memory Pareto | §6.3.1 | same CSV + derived `bytes_per_vec` | new |
| **A4** | Recall × latency × memory 3D bubble (executive summary) | §6.3.2, §6.9.1 | same CSV + derived `bytes_per_vec` | new |
| **B1** | Silvertorch (n_lists × n_probe) heatmap with iso-contours | §6.5.1 | `deep_sweep_silvertorch_arxiv_d128.csv` | upgrade `6_3-silvertorch-arxiv-d128-heatmap.png` |
| **B2** | LinR V3 candidate_pool curves; planned bit-budget axis | §6.5.2 | `deep_sweep_linr_v3_goodreads_d128.csv` | upgrade existing; bit-budget axis requires new sweep |
| **B3** | K sensitivity curve (K ∈ {10, 100, 1000}) | §6.5.3 | quality JSONs (K=10 missing) | **blocked on K=10 re-run** |
| **B4** | Batch-size scaling (bs ∈ {1, 4, 16, 64, 256, 1024}) | §6.5.4 | perf-pass JSONs (bs >16 missing) | **blocked on extended bs sweep** |
| **C1** | Dataset N scaling (Yambda 500m vs 5b) | §6.6.1 | `all_results_long.csv` (yambda rows) | yambda-5b-d256 missing |
| **C2** | Dimensionality scaling (d64/d128/d256) | §6.6.2 | `all_results_long.csv` | derivable |
| **D1** | Recall–latency Pareto stratified by selectivity bucket | §6.4.6 | filter JSONs + selectivity buckets | new (filter-DiskANN/ACORN-style) |
| **D2** | Recall vs selectivity curves (twin-panel) | §6.4.5 | filter JSONs | upgrade existing `goodreads_d128_filter_recall.png` |
| **D3** | Per-clause selectivity distribution (workload characterization) | §6.1.1 | `goodreads_clause_c*.csv` + new arxiv equivalents | partial; arXiv CSVs missing |
| **E1** | Latency distribution (violin/box per algo) | (§6.7 sidebar) | per-query latency vector (planned) | **blocked on schema ext** |
| **E2** | Seed stability bar chart | §6.8.1 | multi-seed re-run (planned) | **blocked on multi-seed re-run** |
| **E3** | Per-query selectivity CDF | §6.1.2 | same as D3 + per-query joint pass-rate | TBD-script-stub |
| **F1** | Backend parity scatter (with Jaccard) | §6.7.1 | `backend_parity_recall.csv`; Jaccard col TBD | partial |
| **F2** | Backend speedup distribution, faceted by (algo, bs) | §6.7.2 | `backend_speedup.csv` | upgrade `backend_speedup_hist.png` |
| **F3** | Kernel-level breakdown (Nsight Compute) | §6.7.5 | new (Nsight one-off) | **blocked on profiling pass** |
| **F4** | Build / index-construction time stacked bar | §6.7.6 | new (`build_time_s` schema ext) | **blocked on schema ext** |
| **G1** | Headline summary table | §6.2.8 | `all_results_long.csv` + derived `bytes_per_vec` | new |
| **G2** | Latency-at-fixed-recall table | §6.2.9 | derived; needs K=10 column | **blocked on K=10 re-run** |
| **G3** | Recall-at-fixed-latency table | §6.2.10 | derivable from current data | new |
| **G4** | Backend parity table (with Jaccard) | §6.7.3 | `backend_parity_recall.csv` + Jaccard col | partial |
| **G5** | Speedup matrix (Triton/torch) | §6.7.4 | `backend_speedup.csv` | derivable |
| **G6** | Memory breakdown table | §6.3.3 | extend `d128_quality_memory.csv` to all dims; add `ratio_vs_fp32` | upgrade |
| **G7** | Cross-over points table | §6.4.7 | derived from §6.4 tables | new |
| **G8** | SASRec quality ceiling | §6.1.3 | `sasrec_quality_ceiling.csv` | extend (yambda-d256 missing) |
| **G9** | Reproducibility / hardware disclosure | §6.8.2 | `SUMMARY.quality-deep.txt` + `uv.lock` | new |
| **G10** | Mirror LiNR paper Tables 3/4 | §6.8.3 | `all_results_long.csv` | partial (K=2000, p95 missing) |

**Best-practice guidelines** that apply across all Ch.6 figures (drawn from ANN-benchmarks, CAGRA, SCANN, DiskANN, HNSW, FAISS-GPU, BIG-ANN 2023, Filtered-DiskANN, ACORN, SIGIR significance testing):

1. **Pareto envelopes only** — plot the Pareto-optimal frontier solidly; fade or omit dominated points.
2. **Log Y for throughput/latency; linear X for recall**, zoomed to operational range (0.5–1.0 or 0.8–1.0).
3. **Brute-force as a reference line** at `recall=1.0`, not a point.
4. **Color = algorithm; marker shape = backend or dim; size = batch_size.** Consistent legend across all chapter figures.
5. **Single-query (bs=1) and batched (bs≥16) regimes get separate plots** (CAGRA precedent — different algorithms win in each).
6. **Report median + p20/p80; for online-serving claims, add p99 tail** (requires schema extension).
7. **Recall@10 + Recall@100 + Recall@1000 in appendix tables**; main figures fix K=100 (LinR paper's choice).
8. **Significance testing**: per-query paired t-test OR Fisher permutation test. Avoid Wilcoxon (poor power). Report p-value + bootstrap 95% CI.
9. **Backend parity**: report Jaccard agreement of top-k sets (target >0.999), not only `recall_abs_diff`.
10. **Workload characterization before system numbers** (D3, E3 first, then Pareto).
11. **bytes per vector**, not raw `index_mem_mib`, for cross-dim comparison.
12. **Russian-language captions** (GOST); algorithm names like `linr_v3` may stay in English.

**Throughput / QPS convention.** All Ch.6 Pareto plots use `median_ms` (per-batch latency) on X. The catalog deliberately avoids deriving QPS from `median_ms` because `1/median` (or `B/median`) conflates a robust statistic with a sum-based rate. A direct `throughput_qps` field is listed under "Planned schema extensions" below — once added, A1 / A2 / B4 and table G2 supersede to throughput-on-X equivalents.

---

## Citation candidates summary

**KEEP — Non-Meta literature** (all introduced in this chapter):

| Citation | Where used | Affiliation status |
|---|---|---|
| Manning, Raghavan & Schütze 2008 — *Introduction to Information Retrieval* | §5.2 (Recall, Precision, IR fundamentals) | Stanford / Yahoo Research / U. Stuttgart — non-Meta ✓ |
| Järvelin & Kekäläinen 2002 — "Cumulated gain-based evaluation of IR techniques" | §5.2 (NDCG formula) | University of Tampere — non-Meta ✓ |
| Voorhees 1999 — "TREC-8 QA Track Report" | §5.2 (MRR) | NIST — non-Meta ✓ |
| Tillet, Kung, Cox 2019 — "Triton: an intermediate language and compiler for tiled neural network computations" (MAPL) | §5.7 (mention of `triton.testing.do_bench` is a code reference; the citation supports Triton-the-framework context) | Harvard / Anthropic (Tillet now) — non-Meta ✓ |
| Aumüller, Bernhardsson & Faithfull 2017 — "ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms" (SISAP) | OPTIONAL — §5.4 supports the "filtered FullScan oracle" methodology as standard practice | IT University of Copenhagen / Spotify — non-Meta ✓ |

**DROPPED — none.** This chapter introduces no Meta-affiliated citations.

**NOT cited in body prose:** the SilverTorch paper (Meta) — per Citation Policy, the in-repo `silvertorch` symbol is retained as a code-internal name; the algorithm itself is presented in this thesis as a second bundled retriever in the `retrieve` framework, assembled from classical primitives, and cites only the component primitives (cross-ref Ch.1 §1.7 and Ch.2).

---

## Open questions / TODOs (consolidated)

### Resolved (folded into the body above)

The original draft of these notes listed a number of items as TODOs. Those that could be answered by reading code, configs, lock files, or git history without running the GPU stack have been resolved and folded into the relevant subsections. Recap for the writer:

- **Version pins** (§5.6) — resolved from [uv.lock](../../uv.lock): `torch==2.10.0+cu128`, `triton==3.6.0`, CUDA toolkit 12.8 (per wheel tag).
- **Per-config row counts** (§5.5) — verified by direct count of all 19 combined JSONs (7 104 total rows; per-config table inlined in §5.5).
- **ArXiv filter "Bloom only" claim** (§5.5) — corrected: arXiv filter configs ship 5 clause + 5 bloom sweeps (no `c1`, by analogy with goodreads' reverse-only c1 exclusion).
- **HF dataset URL** (§5.8) — confirmed public: `https://huggingface.co/datasets/pinkmeme/retrieval-filter-evals-2026-05-23`.
- **`algo.modules` vs `algo.algo_modules`** (§5.1) — code uses `algo_modules` at 10 sites; system doc has a single stale line. Use `algo_modules`.
- **`param-sweeps` vs `deep_sweeps`** (§5.1) — `param-sweeps` is the CLI eval-type flag value; `deep_sweeps/` is the directory and the recommended prose term.
- **d64 / d256 silvertorch deep sweeps** (§5.5) — never existed in git history; deliberate scope cut (paper-faithful $\sqrt{N}$ layout is dimension-agnostic at the IVF level).
- **A100 SKU** (§5.6) — confirmed `A100-SXM4-80GB` (SXM4 form factor, 80 GB HBM2e).
- **Campaign wall-time** (§5.6) — 8-config run in 4 h 14 m (15 248 s); goodreads quality dominates.

### Still-open (require author input)

1. [TODO clarify with author] — Naming of `suite` field: rename to `suite_kind ∈ {filtered, unfiltered}` in a Ch.7 minor-cleanup callout, or leave the historical name and footnote? (Author preference required because this is a forward-looking change, not a verifiable fact.)
2. [TODO clarify with author] — Yambda-5b cache behaviour at runtime: did the disk-budget guard trigger? The per-config logs are git-ignored so this can't be verified from disk; an author note or live re-run would close this.
3. [TODO clarify with author] — Whether to formalise the result schema as a JSON Schema document in an appendix (Appendix D candidate in the thesis-plan structure).
4. [TODO clarify with author] — CUDA driver version on the rented A100 host at the time of the published run (needed for full HSE-style reproducibility table).
5. [TODO clarify with author] — Whether to update [docs/system/evaluation.md](../system/evaluation.md) inline or to footnote the live doc/code drifts (`evaluation/conf/` → `config/`, `QUALITY_BATCH_SIZE=64` → `16`, `MEM_REPS=5` → `16`, `run_per_algo.sh` → `uv run run-evaluation`, `algo.modules` → `algo_modules`, `evaluate.py` → `cli/evaluate.py`). The chapter uses the current code throughout.

---

## Sources consulted

### Repository code
- [pyproject.toml](../../pyproject.toml) — workspace definition
- [retrieve/pyproject.toml](../../retrieve/pyproject.toml) — library package definition (PyTorch/Triton version constraints)
- [evaluation/retrieval/config.py](../../evaluation/retrieval/config.py)
- [evaluation/retrieval/sweep.py](../../evaluation/retrieval/sweep.py)
- [evaluation/retrieval/bench_tools.py](../../evaluation/retrieval/bench_tools.py)
- [evaluation/retrieval/loaders.py](../../evaluation/retrieval/loaders.py) (function map only)
- [evaluation/retrieval/queries_cache.py](../../evaluation/retrieval/queries_cache.py)
- [evaluation/retrieval/oracle.py](../../evaluation/retrieval/oracle.py)
- [evaluation/retrieval/metrics.py](../../evaluation/retrieval/metrics.py)
- [evaluation/retrieval/results_io.py](../../evaluation/retrieval/results_io.py)
- [evaluation/retrieval/algos/__init__.py](../../evaluation/retrieval/algos/__init__.py)
- [evaluation/retrieval/algos/linr_v1.py](../../evaluation/retrieval/algos/linr_v1.py) (function-map level via Explore agent)
- [evaluation/retrieval/algos/linr_v2.py](../../evaluation/retrieval/algos/linr_v2.py) (function-map level)
- [evaluation/retrieval/algos/linr_v3.py](../../evaluation/retrieval/algos/linr_v3.py) (function-map level)
- [evaluation/retrieval/algos/linr_v4.py](../../evaluation/retrieval/algos/linr_v4.py) (function-map level)
- [evaluation/retrieval/algos/silvertorch.py](../../evaluation/retrieval/algos/silvertorch.py) (function-map level)
- [evaluation/retrieval/algos/torch_knn.py](../../evaluation/retrieval/algos/torch_knn.py) (function-map level)
- [evaluation/retrieval/algos/filter.py](../../evaluation/retrieval/algos/filter.py) (function-map level)
- [evaluation/retrieval/algos/_helpers.py](../../evaluation/retrieval/algos/_helpers.py) (function-map level)
- [evaluation/retrieval/cli/run_evaluation.py](../../evaluation/retrieval/cli/run_evaluation.py)
- [evaluation/retrieval/cli/evaluate.py](../../evaluation/retrieval/cli/evaluate.py)
- [evaluation/retrieval/cli/stage_results.py](../../evaluation/retrieval/cli/stage_results.py) (function-map level)
- [evaluation/retrieval/cli/upload_results.py](../../evaluation/retrieval/cli/upload_results.py)

### Configurations
- [evaluation/config/](../../evaluation/config/) — directory listing
- [evaluation/config/yambda-500m/d128-quality.yaml](../../evaluation/config/yambda-500m/d128-quality.yaml)
- [evaluation/config/yambda-5b/d128-quality.yaml](../../evaluation/config/yambda-5b/d128-quality.yaml)
- [evaluation/config/goodreads/d128-filter.yaml](../../evaluation/config/goodreads/d128-filter.yaml)
- [evaluation/config/arxiv/d128-quality.yaml](../../evaluation/config/arxiv/d128-quality.yaml) (via agent)
- [evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml](../../evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml)
- [evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml](../../evaluation/config/deep_sweeps/goodreads-d128-linr_v3.yaml) (via agent)

### Results artifacts (for schema verification)
- [evaluation/results/](../../evaluation/results/) — directory tree
- [evaluation/results/_runlogs/SUMMARY.quality-deep.txt](../../evaluation/results/_runlogs/SUMMARY.quality-deep.txt)
- [evaluation/results/yambda/500m-d128.json](../../evaluation/results/yambda/500m-d128.json) (first rows)
- [evaluation/results/goodreads/d128-filter.json](../../evaluation/results/goodreads/d128-filter.json) (first rows)
- [evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json](../../evaluation/results/deep_sweeps/goodreads-d128-linr_v3.json) (first rows, via agent)

### Documentation
- [docs/system/evaluation.md](../system/evaluation.md) — primary engineering reference, partially stale (see TODOs)
- [docs/thesis/00-thesis-plan.md](00-thesis-plan.md) — Ch.5 contract
- [docs/thesis/05-implementation.md](05-implementation.md) — cross-referenced for algorithm internals
- [retrieve/README.md](../../retrieve/README.md) — public-API overview (light cross-reference)

### Tests not consulted in detail (recommended cross-reference for the writer)
- [retrieve/tests/correctness/](../../retrieve/tests/correctness/) — parity tests gating the per-algorithm equivalence claims (cross-ref §5.6 reproducibility guarantee).
