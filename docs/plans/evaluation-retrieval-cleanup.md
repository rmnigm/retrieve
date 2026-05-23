# Refactor `evaluation/retrieval/` for cleanliness and conciseness

## Context

The `evaluation/retrieval/` package has accumulated small bits of defensiveness, single-use helpers, and inconsistent CLI frameworks over many incremental changes. The goal is a pruning pass: same functionality, less ceremony. Scope is `evaluation/retrieval/**` only — `evaluation/config/`, `evaluation/training/`, and `evaluation/datasets/` are out of scope.

The bar throughout is **prune, don't restructure**. The existing structure (run_sweep → run_filter_kind → run_one_sweep → evaluate_cell decomposition in `sweep.py`; the algo-class-per-file shape in `algos/`) is the intended pattern and stays. We are removing dead branches, redundant guards, and helpers that earn nothing.

## Changes

### `algos/` — algo plugins

**`algos/_helpers.py` → delete.**
- `collect_modules(*base, filter_mod=...)` is a 2-line wrapper over `[*base, filter_mod] if filter_mod else list(base)`. Inline at each call site. The docstring's circular-import justification doesn't hold: `_helpers` has no dependency on `algos/__init__`.

**`algos/torch_knn.py` — normalize to `nn.Module`.**
- Inherit `nn.Module` and call `super().__init__()` to match the other algos. Do **not** add `.compile()` — `FullScanKNN` is the cupy baseline and compile gives nothing; the divergence on `.compile()` stays, only the type hierarchy is normalized.

**`algos/silvertorch.py` — collapse the 3-branch SilverTorch construction.**
Lines 49–99 instantiate `SilverTorch(...)` three times. Replace with one constructor call whose kwargs are computed up front:
- `filter_kw = {"bloom": dict(filter="bloom", m_bits=m_bits, k_hash=k_hash), "clause": dict(filter="exact"), "none": {}}[filter_kind]`
- One `if filter_kind in ("bloom", "clause") and item_attrs_narrow is None: raise ValueError(...)` guard (currently duplicated 2×).
- `register_index(item_embs, item_clause_attrs=item_attrs_narrow)` when filtered, plain `register_index(item_embs)` otherwise.

In `forward` ([silvertorch.py:103-107](../../evaluation/retrieval/algos/silvertorch.py#L103-L107)) drop the bare `assert qa_narrow is not None`; the `query_clause_attrs=qa_narrow` kwarg is already conditional on `self._filter_kind`, so just pass it directly.

**`algos/linr_v{1,2,3,4}.py` — leave structure as-is; only inline `collect_modules`.**
Do **not** introduce a `LinrAlgoBase`. The "shipped pattern" is one explicit class per algo. Only change is the `algo_modules = collect_modules(...)` line → inline list expression.

**`algos/__init__.py` — keep `build_algorithm` as-is.**
The explicit if/elif chain is the readable form; it handles per-algo param unpacking and named aliases (`triton_knn` ≡ `linr_v1_filter_mask`). A registry dict would not improve clarity. Drop `collect_modules` from `__all__` once the helper is gone.

**`algos/filter.py` — replace the if-chain in `build_filter` with a `match`.**
The three `if filter_kind == "..."` arms (lines 39–53) translate cleanly into `match filter_kind:` with the unknown case as the default. Pure cosmetic but it removes the trailing `raise ValueError(f"unknown filter_kind: ...")` branch and makes the closed set explicit.

### `retrieval/*.py` — top-level files

**`bench_tools.py`:**
- Inline `allocated_bytes()` and `peak_bytes()` — each called once internally; the inlined form `(0 if not torch.cuda.is_available() else int(torch.cuda.memory_allocated()))` is one line at the two call sites in `measure_forward_cuda`.
- Keep `cuda_allocated_mib()` (called from `sweep.evaluate_cell` for the `index_mem` delta — it's the public surface).
- The `kw: dict = {}; if qa_narrow is not None: kw["qa_narrow"] = ...` pattern in `quality_pass_cached` ([bench_tools.py:375-378](../../evaluation/retrieval/bench_tools.py#L375-L378)) and `perf_pass_cached`'s inner `perf_fn` ([bench_tools.py:437-444](../../evaluation/retrieval/bench_tools.py#L437-L444)) stays — `forward` is an arbitrary callable and we cannot pass `qa_narrow=None` blindly.

**`loaders.py`:**
- `resolve_path` ([loaders.py:38-47](../../evaluation/retrieval/loaders.py#L38-L47)): drop the redundant `if p.is_absolute() and p.exists()` branch — `p.exists()` alone covers it.
- Move `EXPECTED_DOC_PREFIX` / `EXPECTED_QUERY_PREFIX` ([loaders.py:31-32](../../evaluation/retrieval/loaders.py#L31-L32)) into `assert_arxiv_prefixes` as local constants (only used there).
- `__all__` is fine as-is.

**`sweep.py`:**
- `expand_param_combos` ([sweep.py:730-735](../../evaluation/retrieval/sweep.py#L730-L735)) returns `[dict(combo) for combo in combos]` — only caller is in the same file ([sweep.py:365](../../evaluation/retrieval/sweep.py#L365)) and doesn't mutate. Inline as `cfg.algo_params.get(algo, [{}])`; delete the function. Drop from `__all__`.
- `is_valid_combo` — only called once internally; not externally referenced. Push the silvertorch n_probe/n_lists check into `SilvertorchAlgo.__init__` as a `ValueError` (which `_try_build_algo` already catches as a skip). Delete `is_valid_combo`; drop from `__all__`.
- `_build_filter_modules` ([sweep.py:285-293](../../evaluation/retrieval/sweep.py#L285-L293)): the nested `if filter_mods:` / `if any_mod is not None:` is overdefensive — `filter_mods` is non-empty whenever `cfg.filters is not None` (line 253 early-returns the empty case). Collapse to one guard.
- The `kw: dict = {}; if qa_n_sweep is not None: kw["qa_narrow"] = ...` pattern in `_autotune_prewarm` ([sweep.py:627-630](../../evaluation/retrieval/sweep.py#L627-L630)) stays — same reason as `bench_tools.py`.

**`oracle.py`, `queries_cache.py`, `results_io.py`, `metrics.py`, `config.py`:** leave untouched. These are at the right size and already concise.

### `cli/` — entry points

**`cli/stage_results.py` → click.**
Replace `sys.argv[1]` parsing and bare `print(..., file=sys.stderr)` exits with click. The script becomes one `@click.command()` with one `--config` argument; failure modes (`out_dir` missing, no JSONs) become `click.echo(..., err=True); sys.exit(N)`.

**`cli/upload_results.py` → click.**
Replace `argparse.ArgumentParser` with `@click.command()` carrying `--dry-run` and `--clean-stage`. Body is unchanged.

**`cli/run_evaluation.py` — keep argparse.**
This script's argparse handles `--`-passthrough to forward extra args to `evaluate`, the mutually-exclusive `--resume`/`--force` group, and the `--eval-type` ↔ positional configs xor. Click expresses these but with more ceremony than argparse here. Leave argparse.

**Shared raw-yaml loader.** Add a small helper to `retrieval/config.py`:
```python
def load_eval_config_raw(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}
```
Then `load_eval_config` uses it internally, and `run_evaluation._load_cfg` / `stage_results.main` consume it. Removes the duplicated 3-line YAML+filter pattern.

**Single-use helpers in `run_evaluation.py`** (`_gpu_name`, `_cfg_name`, `_format_dur`, `_expand_configs`): keep. Their names document intent at the call sites; inlining would hurt readability. This is the one place where named helpers earn their keep.

### Stale exports

Audit `__all__` lists in every file after the prune:
- `sweep.__all__`: drop `expand_param_combos`, `is_valid_combo` (functions deleted).
- `algos.__init__.__all__`: drop `collect_modules` (function deleted).

## Critical files to modify

| File | Action |
|---|---|
| [evaluation/retrieval/algos/_helpers.py](../../evaluation/retrieval/algos/_helpers.py) | **Delete** |
| [evaluation/retrieval/algos/__init__.py](../../evaluation/retrieval/algos/__init__.py) | Drop `collect_modules` import + export |
| [evaluation/retrieval/algos/torch_knn.py](../../evaluation/retrieval/algos/torch_knn.py) | Inherit `nn.Module`, inline helper |
| [evaluation/retrieval/algos/linr_v1.py](../../evaluation/retrieval/algos/linr_v1.py) | Inline helper |
| [evaluation/retrieval/algos/linr_v2.py](../../evaluation/retrieval/algos/linr_v2.py) | Inline helper |
| [evaluation/retrieval/algos/linr_v3.py](../../evaluation/retrieval/algos/linr_v3.py) | Inline helper |
| [evaluation/retrieval/algos/linr_v4.py](../../evaluation/retrieval/algos/linr_v4.py) | Inline helper |
| [evaluation/retrieval/algos/silvertorch.py](../../evaluation/retrieval/algos/silvertorch.py) | Collapse 3-branch ctor, drop `assert`, inline helper |
| [evaluation/retrieval/algos/filter.py](../../evaluation/retrieval/algos/filter.py) | `match` in `build_filter` |
| [evaluation/retrieval/bench_tools.py](../../evaluation/retrieval/bench_tools.py) | Inline `allocated_bytes`/`peak_bytes` |
| [evaluation/retrieval/loaders.py](../../evaluation/retrieval/loaders.py) | Simplify `resolve_path`, localize prefix constants |
| [evaluation/retrieval/sweep.py](../../evaluation/retrieval/sweep.py) | Inline `expand_param_combos`, move silvertorch check into algo, collapse defensive `if`s, drop `__all__` entries |
| [evaluation/retrieval/config.py](../../evaluation/retrieval/config.py) | Add `load_eval_config_raw` helper |
| [evaluation/retrieval/cli/stage_results.py](../../evaluation/retrieval/cli/stage_results.py) | Convert to click, use shared raw loader |
| [evaluation/retrieval/cli/upload_results.py](../../evaluation/retrieval/cli/upload_results.py) | Convert to click |
| [evaluation/retrieval/cli/run_evaluation.py](../../evaluation/retrieval/cli/run_evaluation.py) | Use shared raw loader |

## Verification

There are no tests under `evaluation/`; verification is end-to-end:

1. **Import smoke test** — `uv run python -c "from retrieval.cli import evaluate, run_evaluation, stage_results, upload_results; from retrieval import sweep, bench_tools, loaders, oracle, config, metrics; from retrieval.algos import build_algorithm, build_filter, ALGORITHMS"` from `/workspace/retrieve/evaluation/`. Catches deleted-but-still-imported symbols.

2. **CLI help** — `uv run evaluate --help`, `uv run run-evaluation --help`, `uv run stage-results --help`, `uv run upload-results --help`. Confirms click migration didn't break entry-point resolution.

3. **One-cell run** — pick the smallest yambda or arxiv config (e.g. `config/yambda-500m/d128.yaml` if it exists, else `config/arxiv/d64-filter.yaml`) and run `uv run evaluate --config <cfg> --algo linr_v1_filter_mask --output /tmp/refactor_check.json --skip-quality`. The `--skip-quality` flag bypasses oracle building so the run is fast. Confirm: exits 0, JSON contains rows, no warnings about missing helpers.

4. **Algo coverage** — repeat (3) once each for `torch_knn` (the nn.Module-ified algo), `silvertorch` (the collapsed-ctor algo), and `linr_v3` (the most complex algo, to catch helper-inline regressions). Compare row counts and recall@k/ndcg@k against a pre-refactor JSON if one is on disk.

5. **`stage-results` round-trip** — after step 4, run `uv run stage-results <cfg>` and confirm the `.perkernel` dir + combined `.json` + `.yaml` copy appear. The CLI rewrite is the highest-risk change.

6. **Static check** — `uv run python -m compileall evaluation/retrieval/` to catch syntax errors in unexercised branches.

If any of (3)–(5) emit a different row count or different metric values than a pre-refactor baseline on the same seed, that's a functional regression — investigate before considering the refactor done.
