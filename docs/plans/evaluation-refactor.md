# Evaluation harness refactor — readability + leanness

> **Status: IMPLEMENTED (2026-07-06); library test gates passed on A100 2026-09-02, harness gates pending** (see the roadmap, Open work 1). All phases E1-E8 are committed
> on branch `refactor/kernels-eval`. This document is now a **record of intent**, not a work
> queue. The as-built harness is documented in
> [../system/evaluation.md](../system/evaluation.md), which is the maintained reference — this
> plan's "Ground truth: actual architecture" section has been superseded by it.
>
> It stays here (rather than in [archive/](archive/)) only until
> [refactor-validation-handoff.md](refactor-validation-handoff.md) signs off. Archive it once
> validation passes.
>
> Line references are against commit `2b1ff80` and are **stale**; several files named here
> (`bench_tools.py`, `algos/torch_knn.py`, `datasets/`) no longer exist — that is the point of
> the plan, not an error in it.
>
> Companion plans: [kernels-layers-design.md](kernels-layers-design.md) (library side),
> [future-work-and-research.md](future-work-and-research.md) (post-refactor ideas).
> Does not conflict with [torch-export-refactor.md](torch-export-refactor.md) or
> [live-update-api.md](live-update-api.md); where they touch the same files, this plan defers
> to their decisions (noted inline).

## Context

The eval harness grew organically across the yambda → goodreads → arxiv campaigns and the
per-algo-subprocess rework. The bones are good — `sweep.py` is already decomposed by loop
level, the measurement methodology is careful and documented — but three kinds of debt
accumulated:

1. **Dead weight**: a broken-and-unused algo, an unreachable CPU-timing path, identity-function
   indirection, three unused dependencies, and a one-shot campaign upload script.
2. **Parameter-threading**: the driver threads 15–20 keyword args through five call levels.
   [silvertorch-reverse-clause-wrapper-fix.md](archive/silvertorch-reverse-clause-wrapper-fix.md)
   §Changes documents the cost concretely: adding **one** field (`clause_is_reverse`) required
   touching **five** function signatures in `sweep.py` alone.
3. **Doc drift**: [docs/system/evaluation.md](../system/evaluation.md) describes a layout
   (`evaluate.py`, `run_per_algo.sh`, `conf/`, `algo.modules`, `QUALITY_BATCH_SIZE = 64`) that
   no longer exists.

### Ground truth: actual architecture (2026-07-03)

Since the system doc is stale, this is the as-built map an implementing agent should trust:

- **Entry points** ([evaluation/pyproject.toml:24-35](../../evaluation/pyproject.toml)):

  | script | target | role |
  |---|---|---|
  | `evaluate` | `retrieval.cli.evaluate:main` | one algo × one config → one JSON |
  | `run-evaluation` | `retrieval.cli.run_evaluation:main` | orchestrator; spawns `uv run evaluate` per (config, algo) subprocess; `--eval-type {filter,quality,param-sweeps}` presets; tee'd run logs under `results/_runlogs/` |
  | `stage-results` | `retrieval.cli.stage_results:main` | per-algo JSONs → combined `<name>.json` + `.perkernel/` + `.yaml` |
  | `upload-results` | `retrieval.cli.upload_results:main` | staged results → HF dataset repo (campaign-bound, see E1.5) |
  | `yambda` / `arxiv` / `goodreads` | `datasets.*:main` | dataset ETL CLIs |
  | `eval-fetch` / `eval-publish` / `eval-publish-checkpoint` | `datasets.hf_io:*` | HF dataset/checkpoint sync |
  | `upload-checkpoints` | `training.upload_checkpoints:main` | trainer-side HF upload |

- **Driver flow**: `cli/evaluate.py::main` → seeds + TF32 off + `recompile_limit=64` →
  `queries_cache.load_or_cache_queries` (SASRec encode, disk-cached at
  `<ckpt-dir>/encoded_queries_<split>.pt`, keyed on `(ckpt_mtime, max_seq_length,
  users_limit)`) → `loaders.load_query_attrs` (filter configs only) → `sweep.run_sweep` →
  JSON dump to `--output`.
- **Loop nest** (`sweep.py`): `run_sweep` → per `(filter_kind, FilterCfg)` from
  `_select_filter_iter` (yambda synthesizes a single `("none", full_scan)` cell) →
  `run_filter_kind` (users_limit subsample → `_build_filter_modules`: one `FilterModule` per
  backend + an always-exact oracle filter) → per sweep `run_one_sweep` (`build_sweep_qa` →
  `load_or_build_oracle` → per `(algo, backend, params, k)` `evaluate_cell`) →
  `evaluate_cell` (reset CUDA state → `_try_build_algo` → `index_mem` delta → `_run_quality` →
  `_autotune_prewarm` → per-bs `perf_pass_cached` → `_make_perf_row` → `_release_algo`).
- **Row schema**: one dict per `(filter_kind, sweep, algo, backend, params, k, bs)` cell —
  `suite, cell, filter_kind, sweep, impl, backend, device, seed, batch_size, k, n_users_kept,
  median_ms, p20_ms, p80_ms, peak_mem_mib, index_mem_mib, fwd_scratch_mib, recall@K, ndcg@K,
  extra.params` ([sweep.py:646-691](../../evaluation/retrieval/sweep.py#L646-L691)).
- **Measurement invariants** (must survive this refactor unchanged): TF32 pinned off; one-shot
  GPU warmup outside any measured window; warmup **before** peak-memory reset; memory window
  without do_bench's L2-buster; timing auto-extends `rep_ms` until ≥ `MIN_SAMPLES=30` samples
  (cap `MAX_REP_MS=3000`); perf pool of `n_pool=4096` fixed-seed query batches; quality
  streamed at `QUALITY_BATCH_SIZE=16`; per-sweep `torch._dynamo.reset()` + `empty_cache()`
  ([sweep.py:191-197](../../evaluation/retrieval/sweep.py#L191-L197)); per-algo process
  isolation via the orchestrator.
- **Algos** (`retrieval/algos/`): thin `nn.Module` wrappers over `retrieve` layers; duck-typed
  protocol `(algo_modules: list[nn.Module], is_cpu: bool, __call__(q, qa_narrow=None) → (ids,
  scores))`; each compiled with `torch.compile(dynamic=True, mode="reduce-overhead")` in
  `__init__`; registry + factory in `algos/__init__.py`.

## Out of scope

- **No behavioral changes to measurement methodology** (warmup counts, do_bench budgets,
  memory-window ordering, pool sampling). Rows for existing columns must be reproducible:
  quality columns byte-identical, latency within ~5% noise.
- **`datasets/` CLI internals** (goodreads.py / arxiv.py bucketing, ETL) — domain plumbing,
  largely non-duplicated (the shared numeric core already lives in
  `datasets/common.py`). Only the package-name hazard
  (E1.6) touches this tree.
- **Stage 4a schema extensions themselves** (`throughput_qps`, `p99_ms`, `build_time_s`, …) —
  they stay in [00-roadmap.md](00-roadmap.md) Stage 4a; **but** E5 restructures return types so
  landing them later is a one-field change, not another 5-signature thread.
- **Training pipeline** (`training/`) — read but healthy; no changes beyond the import-path
  effects of E5.1.
- **Library (`retrieve/`) changes** — see [kernels-layers-design.md](kernels-layers-design.md).

## Conventions for the implementing agent

- Work phase by phase; each phase is one reviewable PR.
- Before phase 1, capture a **golden run**: `cd evaluation && uv run evaluate --config
  config/goodreads/d128-filter.yaml --algo linr_v3 --output /tmp/golden.json --filter-kind
  clause --sweep c0_genre`. After every phase, rerun to `/tmp/phaseN.json` and diff: quality
  columns byte-identical, same `cell` keys, latency within noise. (Requires GPU + fetched
  goodreads data; if no GPU is available, the CPU-only unit tests from E8 are the gate and the
  golden diff runs once at the end on a GPU box.)
- `uv run ruff check .` clean after every phase; do not reformat untouched code.
- Never rename a JSON row field that already exists — downstream thesis plotting scripts
  consume them. New fields are additive.

---

## Phase E1 — Delete dead weight

Cheapest, zero-risk leanness wins. Each sub-item is independently landable; land as one PR of
small commits.

### E1.1 Remove `TorchKnnAlgo` (broken *and* unused)

**Evidence.** `algos/torch_knn.py:14` is a
**plain class** (not `nn.Module`) with a `forward` method but no `__call__`. The harness
invokes algo objects directly:
`bench_tools.py:378` `topk_ids, _ = forward(q,
**kw)` where `forward` *is* the algo instance, and
[sweep.py:642](../../evaluation/retrieval/sweep.py#L642) `algo_obj(q, **kw)`. Enabling
`torch_knn` in any config therefore raises `TypeError: 'TorchKnnAlgo' object is not callable`
at the first quality batch. No YAML under `evaluation/config/` lists it (verified by grep across
all 19 configs), and
[cli/upload_results.py:145](../../evaluation/retrieval/cli/upload_results.py#L145) documents it
was "dropped from quality YAMLs in favor of `linr_v1_filter_mask` (also exact) + `linr_v4`".

**Edits.**

1. Delete `evaluation/retrieval/algos/torch_knn.py`.
2. [algos/__init__.py](../../evaluation/retrieval/algos/__init__.py): remove the import (line
   43), the `"torch_knn"` entry in `ALGORITHMS` (line 46), the `build_algorithm` branch (lines
   84-85), and the `torch_knn` mention in the module docstring.
3. `BACKEND_CAPABLE_ALGOS` (lines 58-60) now equals "every registered algo" — delete the set;
   in `run_one_sweep` replace

   ```python
   algo_backends = (
       backends if algo in BACKEND_CAPABLE_ALGOS else [backends[0]]
   )
   ```

   ([sweep.py:368-370](../../evaluation/retrieval/sweep.py#L368-L370)) with `for backend in
   backends:` and drop the exception note from the docstring (lines 333-335).
4. Remove the stale `FullScanKNN` cross-reference in
   [docs/system/evaluation.md](../system/evaluation.md) when E8 rewrites it.

**Note for the future:** if a strict fp32 exact reference is ever wanted again, resurrect it as
an `nn.Module` wrapper over `FullScanKNN` in one commit — record this in the commit message,
not in code. The oracle does **not** depend on this class
(`compute_filtered_oracle` inlines its own matmul,
[oracle.py:54-84](../../evaluation/retrieval/oracle.py#L54-L84)).

### E1.2 Remove the unreachable CPU-timing path

**Evidence.** After E1.1 every algo hard-codes `is_cpu = False`
(linr_v1.py:28, linr_v2.py:26, linr_v3.py:41, linr_v4.py:31, silvertorch.py:33), so
`measure_forward_cpu` (`bench_tools.py:170-216`,
47 lines) is unreachable, and every `is_cpu` branch is dead.

**Edits** (exhaustive touch list):

| site | change |
|---|---|
| `bench_tools.py:170-216` | delete `measure_forward_cpu` |
| `bench_tools.py:384-395` | drop `is_cpu` param from `perf_pass_cached`; delete the `if is_cpu: return measure_forward_cpu(...)` tail (lines 446-447) |
| [sweep.py:450-452](../../evaluation/retrieval/sweep.py#L450-L452) | `index_mem = cuda_allocated_mib() - mem_before` unconditionally |
| [sweep.py:475-484](../../evaluation/retrieval/sweep.py#L475-L484) | drop `is_cpu=algo_obj.is_cpu` kwarg |
| [sweep.py:494](../../evaluation/retrieval/sweep.py#L494) | `_make_perf_row`: drop `is_cpu` param; hard-code `"device": "cuda"` (line 677) |
| [sweep.py:632](../../evaluation/retrieval/sweep.py#L632) | `_autotune_prewarm`: drop the `algo_obj.is_cpu or` clause |
| `algos/*.py` (5 files) | delete the `is_cpu = False` class attr |
| [algos/__init__.py:6](../../evaluation/retrieval/algos/__init__.py#L6) | drop `is_cpu` from the protocol docstring |

Keep the `device` **column** in the row schema (hard-coded `"cuda"`) — downstream analysis
filters on it and removing a column violates the additive-schema rule.

### E1.3 Inline `expand_param_combos`

[sweep.py:742-747](../../evaluation/retrieval/sweep.py#L742-L747) is an identity wrapper
(copies each dict). Its docstring's "used by … external tests" claim is false — the only test
file is `tests/test_silvertorch_algo_reverse.py`. Replace the call site
([sweep.py:372](../../evaluation/retrieval/sweep.py#L372)):

```python
# before
for params in expand_param_combos(cfg.algo_params.get(algo, [{}])):
# after
for params in cfg.algo_params.get(algo, [{}]):
    params = dict(params)   # defensive copy; combos are reused across k/bs loops
```

Delete the function and its `__all__` entry. **Keep `is_valid_combo`**
([sweep.py:750-757](../../evaluation/retrieval/sweep.py#L750-L757)) — it encodes a real
constraint (`n_probe > n_lists` skip).

### E1.4 Drop unused dependencies

[evaluation/pyproject.toml:17-21](../../evaluation/pyproject.toml) declares `torchvision`,
`matplotlib`, `einops` — **zero** imports anywhere under `evaluation/` (verified:
`grep -rn "import torchvision|import matplotlib|import einops|from torchvision|from matplotlib|from einops"`
returns nothing). Remove all three plus the `torchvision = { index = "pytorch-cu128" }` source
pin (line 52). `wandb` (used in `training/train_sasrec.py` only) and `sentence-transformers`
(used in `datasets/arxiv.py` only) stay. Run `uv sync` and commit the lockfile delta.

### E1.5 Quarantine the campaign-bound upload script

[cli/upload_results.py](../../evaluation/retrieval/cli/upload_results.py) hard-codes
`REPO_ID = "pinkmeme/retrieval-filter-evals-2026-05-23"` (line 27), a dated commit message
(line 176), and hand-written campaign README prose (lines 112-127, 142-147).

**Edits.**

1. `--repo-id` becomes a **required** argparse flag; delete the module constant.
2. Commit message becomes `f"Upload results ({dt.date.today().isoformat()})"`.
3. The hard-coded `sections` descriptions (lines 122-127) move behind a
   `--notes-file <md>` optional flag (appended verbatim to the README); the generated part
   keeps only the per-kind tables derived from `stats`. The "Notes" bullets (lines 142-147)
   are campaign-specific — into the notes file of that campaign, not code.
4. Add `--private/--public` flag replacing the `PRIVATE = False` constant.

Alternative (if not worth the effort): move the file to `evaluation/scripts/` with a
"campaign-specific — edit before reuse" header and drop the console script. Either resolution
is acceptable; the current state — a dated one-shot masquerading as reusable infra — is not.

### E1.6 Rename the `datasets` package (name collision with HF `datasets`)

`packages = ["datasets", "retrieval", "training"]`
([pyproject.toml:42](../../evaluation/pyproject.toml)) installs a top-level `datasets` package
that shadows HuggingFace `datasets` in this venv. `sentence-transformers` (a declared dep)
imports HF `datasets` opportunistically; if that path is hit, it imports *our* package and
fails with a confusing AttributeError far from the cause.

**Mechanics** (~30 files, purely mechanical):

1. `git mv evaluation/datasets evaluation/eval_datasets`.
2. pyproject: `packages = ["eval_datasets", "retrieval", "training"]`; console-script targets
   `datasets.yambda:main` → `eval_datasets.yambda:main` (×5 entries); ruff
   `known-first-party` list.
3. `grep -rn "from datasets\.\|import datasets" evaluation/` and rewrite (known importers:
   `retrieval/` does **not** import it; `datasets/hf_io.py` self-references via relative
   paths — check `_repo_root()` at
   `hf_io.py:89` uses `__file__`, unaffected).
4. Console-script *names* (`yambda`, `arxiv`, `goodreads`, `eval-fetch`, …) are unchanged —
   no doc or muscle-memory impact.

Take it now; the failure mode is a time bomb and the rename is grep-mechanical.

---

## Phase E2 — Context objects for the driver (the big readability lever)

**Problem, concretely**: `run_filter_kind` takes 15 parameters
([sweep.py:128-146](../../evaluation/retrieval/sweep.py#L128-L146)), `run_one_sweep` 19
([sweep.py:307-328](../../evaluation/retrieval/sweep.py#L307-L328)), `evaluate_cell` 20
([sweep.py:408-431](../../evaluation/retrieval/sweep.py#L408-L431)), `_run_quality` 14
([sweep.py:561-577](../../evaluation/retrieval/sweep.py#L561-L577)). Nearly all are
pass-through. The reverse-clause fix threaded one tensor through five signatures.

### E2.1 New module `retrieval/context.py`

```python
"""Context dataclasses threaded through the sweep driver.

SweepContext is built once per run (post users_limit). FilterAssets is built
once per filter_kind; its per-sweep fields are stamped by run_one_sweep via
dataclasses.replace. Both are frozen: a new input = a new field here, visible
to every loop level at once.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from retrieval.config import EvalConfig
from retrieve.interfaces import Backend, FilterModule


@dataclass(frozen=True)
class SweepContext:
    cfg: EvalConfig
    algorithms: tuple[str, ...]          # explicit — cli no longer mutates cfg
    item_embs: Tensor                    # [N, D] on device
    queries: Tensor                      # [U, D] cpu, post users_limit
    targets: Tensor                      # [U, T] cpu
    n_targets: Tensor                    # [U]    cpu
    qa_narrow_all: Tensor | None         # [U, C] cpu or None
    device: torch.device
    gt_dir: Path
    k_gt: int                            # max(cfg.ks)
    suite: str                           # "yambda" | "filter"
    backends: tuple[Backend, ...]
    skip_quality: bool
    sweep_filter: str | None             # --sweep CLI narrow


@dataclass(frozen=True)
class FilterAssets:
    """Per-filter_kind modules and attrs; per-sweep fields default empty."""
    filter_mods: dict[Backend, FilterModule | None]
    oracle_filter: FilterModule | None
    item_attrs_narrow: Tensor | None
    clause_is_reverse: Tensor | None
    n_clauses: int
    # --- stamped per sweep by run_one_sweep ---
    qa_n_sweep: Tensor | None = None
    skip_mask: Tensor | None = None
    oracle_topk: Tensor | None = None
    n_kept: int = 0

    def for_sweep(self, *, qa_n_sweep, skip_mask, oracle_topk, n_kept) -> "FilterAssets":
        return replace(self, qa_n_sweep=qa_n_sweep, skip_mask=skip_mask,
                       oracle_topk=oracle_topk, n_kept=n_kept)

EMPTY_ASSETS = FilterAssets(filter_mods={}, oracle_filter=None,
                            item_attrs_narrow=None, clause_is_reverse=None, n_clauses=0)
```

### E2.2 Signature collapse

Target signatures (bodies keep their current logic — this phase moves arguments, not
behavior):

```python
def run_sweep(ctx: SweepContext) -> list[dict]: ...
def run_filter_kind(filter_kind: str, fcfg: FilterCfg, ctx: SweepContext) -> list[dict]: ...
def run_one_sweep(sweep: FilterSweepCfg, filter_kind: str,
                  ctx: SweepContext, assets: FilterAssets) -> list[dict]: ...
def evaluate_cell(algo: str, params: dict, k: int, backend: Backend,
                  sweep: FilterSweepCfg, filter_kind: str,
                  ctx: SweepContext, assets: FilterAssets) -> list[dict]: ...
def _run_quality(algo_obj: RetrievalAlgo, k: int, desc: str,
                 ctx: SweepContext, assets: FilterAssets) -> tuple[float, float]: ...
def _autotune_prewarm(algo_obj: RetrievalAlgo, ctx: SweepContext, assets: FilterAssets) -> None: ...
```

**Migration steps** (mechanical; do in this order so each commit compiles):

1. Add `context.py`; build `SweepContext` in `cli/evaluate.py::main` — this is where
   `_apply_users_limit` moves to (rename to `apply_users_limit`, export from `loaders.py`),
   so the context always holds post-limit tensors. One invariant replaces the
   `queries` vs `queries_f` naming split. Note `queries_cache.load_or_cache_queries` *also*
   trims to `users_limit` before caching
   ([queries_cache.py:75-85](../../evaluation/retrieval/queries_cache.py#L75-L85)) — after this
   move the second trim in the driver becomes a no-op guard for the cache-miss arxiv path;
   keep it (idempotent) and note it in the docstring.
2. `cli/evaluate.py` stops mutating cfg
   ([cli/evaluate.py:54](../../evaluation/retrieval/cli/evaluate.py#L54) `cfg.algorithms =
   [algo]`): pass `algorithms=(algo,)` into the context instead.
3. Rewrite `run_sweep` to consume `ctx` (its body keeps `pin_precision_globals()` /
   `warm_gpu_once`, the gt_dir mkdir, and the filter-kind loop; `K_GT`/`suite`/`gt_dir` move
   into context construction).
4. `_build_filter_modules(filter_kind, fcfg, ctx) -> FilterAssets` — same body, returns the
   dataclass instead of the 5-tuple ([sweep.py:242-248](../../evaluation/retrieval/sweep.py#L242-L248)).
5. `run_one_sweep` stamps per-sweep fields:

   ```python
   qa_n_sweep, skip_mask = build_sweep_qa(sweep, filter_kind, ctx.qa_narrow_all, assets.n_clauses)
   n_kept = int((~skip_mask).sum().item()) if skip_mask is not None else ctx.queries.shape[0]
   oracle_topk = None
   if ctx.cfg.filters is not None and filter_kind != "none" and not ctx.skip_quality:
       oracle_topk = load_or_build_oracle(ctx.gt_dir, sweep.name, ctx.queries.shape[0], ctx.k_gt,
                                          item_embs=ctx.item_embs, queries=ctx.queries,
                                          qa_narrow_sweep=qa_n_sweep, skip_mask=skip_mask,
                                          oracle_filter=assets.oracle_filter, device=ctx.device)
   sweep_assets = assets.for_sweep(qa_n_sweep=qa_n_sweep, skip_mask=skip_mask,
                                   oracle_topk=oracle_topk, n_kept=n_kept)
   ```

6. `evaluate_cell` and helpers read `ctx.*` / `assets.*`; the `filter_mod` for the current
   backend is `assets.filter_mods.get(backend)`.

**Guard rails**: tensors inside a frozen dataclass are still mutable — that is fine (they are
treated read-only by convention already); the dataclass freeze is about *fields*, i.e. the
threading topology. Do not add `slots=True` (harmless here but blocks
`dataclasses.replace`-with-inheritance patterns some tools use; keep it simple).

## Phase E3 — Typed algo protocol + declarative cell eligibility

### E3.1 `RetrievalAlgo` Protocol

The driver types algo objects as `Any`
([sweep.py:527-538](../../evaluation/retrieval/sweep.py#L527-L538), `_run_quality`,
`_autotune_prewarm`, `_release_algo`). Add to `algos/_helpers.py`:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class RetrievalAlgo(Protocol):
    algo_modules: list[nn.Module]
    def __call__(self, q: Tensor, qa_narrow: Tensor | None = None) -> tuple[Tensor, Tensor]: ...
```

Annotate: `build_algorithm` return, `_try_build_algo`/successor, `_run_quality`,
`_autotune_prewarm`, `_release_algo`, `quality_pass_cached(forward: RetrievalAlgo | Callable…)`,
`perf_pass_cached`. (`is_cpu` is gone after E1.2.) `runtime_checkable` lets the factory assert
`isinstance(obj, RetrievalAlgo)` in one place.

### E3.2 Capability table instead of ValueError-as-control-flow

Today `build_algorithm` raises `ValueError` for incompatible cells and `_try_build_algo`
catches **all** `ValueError`s at debug level
([sweep.py:544-558](../../evaluation/retrieval/sweep.py#L544-L558)). Failure mode: a genuine
bad argument — a typo'd param key, an int cast failing, a library-level shape ValueError from
`SilverTorch.register_index` (e.g. `k > n_probe × max_cluster_size`,
[main.py:130-134](../../retrieve/src/retrieve/layers/silvertorch/main.py#L130-L134)) — is
silently logged as "combo isn't applicable" and the cell vanishes from the results with no
error. This has real silent-data-loss potential in a sweep campaign.

**Edits.**

1. In `algos/__init__.py`:

   ```python
   SUPPORTED_FILTER_KINDS: dict[str, frozenset[str]] = {
       "triton_knn":          frozenset({"none", "clause", "bloom"}),
       "linr_v1_filter_mask": frozenset({"none", "clause", "bloom"}),
       "linr_v2":             frozenset({"clause", "bloom"}),   # candidate source IS the filter
       "linr_v3":             frozenset({"none", "clause", "bloom"}),
       "linr_v4":             frozenset({"none", "clause", "bloom"}),
       "silvertorch":         frozenset({"none", "clause", "bloom"}),
   }

   def supports(algo: str, filter_kind: str) -> bool:
       return filter_kind in SUPPORTED_FILTER_KINDS[algo]
   ```

2. `run_one_sweep` checks `supports(algo, filter_kind)` before descending into
   `(backend, params, k)` and logs one `logger.info("skipping {}: unsupported filter_kind {}")`
   per skip.
3. Delete `_try_build_algo`'s `except ValueError` — construction errors now propagate and kill
   the run loudly (which is what you want mid-campaign; the orchestrator records rc≠0 and the
   resume flag lets you continue after fixing).
4. `build_algorithm` keeps its defensive raises (now genuinely exceptional) — e.g.
   `linr_v2` without `filter_mod` stays a hard error.
5. Retire the eligibility prose in the module docstring
   ([algos/__init__.py:21-25](../../evaluation/retrieval/algos/__init__.py#L21-L25)) in favor
   of the table.

### E3.3 `FilterKind` literal alias

`filter_kind: str` is string-compared in ≥8 places (`sweep.py`, `algos/__init__.py`,
`algos/filter.py`, `loaders.py:325`, `oracle` call sites). Add next to the `Backend` import in
[retrieval/config.py](../../evaluation/retrieval/config.py):

```python
FilterKind = Literal["none", "clause", "bloom"]
```

and annotate through. Pure typing; pyright catches typos. The YAML side stays a plain str key
(config dicts are keyed by it); `_select_filter_iter` narrows on load.

## Phase E4 — `AlgoBase` for the wrapper boilerplate

All five algo wrappers repeat the same construction tail
([linr_v1.py:41-45](../../evaluation/retrieval/algos/linr_v1.py#L41-L45),
[linr_v2.py:39-43](../../evaluation/retrieval/algos/linr_v2.py#L39-L43),
[linr_v3.py:59-65](../../evaluation/retrieval/algos/linr_v3.py#L59-L65),
[linr_v4.py:42-46](../../evaluation/retrieval/algos/linr_v4.py#L42-L46),
[silvertorch.py:107-108](../../evaluation/retrieval/algos/silvertorch.py#L107-L108)). The
`torch.compile(dynamic=True, mode="reduce-overhead")` invocation — the single most load-bearing
line in the harness (it is what makes one cudagraph capture per algo forward) — is duplicated
five times.

**New base** (in `algos/_helpers.py`, replacing `collect_modules`):

```python
class AlgoBase(nn.Module):
    """Shared tail for eval algo wrappers.

    Subclasses build their retrieve-layer modules in __init__ and then call
    _finalize(...) exactly once as the LAST statement — it wires the
    algo_modules cleanup list and applies the one canonical torch.compile
    call (dynamic=True + reduce-overhead → single cudagraph capture of the
    full forward: filter + index + cascade)."""

    filter_mod: FilterModule | None

    def _finalize(self, *modules: nn.Module, filter_mod: FilterModule | None) -> None:
        self.filter_mod = filter_mod
        mods = list(modules)
        if filter_mod is not None:
            mods.append(filter_mod)
        self.algo_modules = mods
        self.compile(dynamic=True, mode="reduce-overhead")
```

**Per-file change** (linr_v1 shown; the other four are isomorphic):

```python
class LinrV1Algo(AlgoBase):
    def __init__(self, item_embs, k, *, filter_mod=None, backend="triton"):
        super().__init__()
        self.idx = PostfilterKNN(k=k, backend=backend).to(item_embs.device)
        self.idx.register_index(item_embs)
        self._finalize(self.idx, filter_mod=filter_mod)

    def forward(self, q, qa_narrow=None):
        return self.idx(q, mask=make_mask(self.filter_mod, qa_narrow))
```

linr_v3 calls `self._finalize(self.stage1, self.stage2, filter_mod=filter_mod)`; silvertorch
calls `self._finalize(self.idx, filter_mod=None)`.

**Do not** move `forward` logic into the base — the per-algo forwards are the readable
specification of each cascade (especially
[linr_v3.py:67-89](../../evaluation/retrieval/algos/linr_v3.py#L67-L89), whose comments carry
the cudagraph-stitching rationale) and must stay visible in their files. Delete
`collect_modules` once all five are migrated; `_release_algo`
([sweep.py:732-736](../../evaluation/retrieval/sweep.py#L732-L736)) is unchanged (it consumes
`algo_modules`).

## Phase E5 — Measurement module split + structured results

### E5.1 Split `bench_tools.py` by concern

459 lines mixing four concerns, with a stale module docstring ("The single driver in `evaluate.py` imports from here" — no such file) and a buried cross-package
dependency (`retrieval` → `training`,
`bench_tools.py:29-30`).

| new module | contents (moved verbatim) | imports allowed |
|---|---|---|
| `retrieval/measure.py` | `WARMUP_ITERS, DEFAULT_REP_MS, MIN_SAMPLES, MAX_REP_MS, MEM_REPS`, `allocated_bytes`, `peak_bytes`, `pin_precision_globals`, `warm_gpu_once`, `measure_forward_cuda`, `cuda_allocated_mib` | torch, triton.testing only — **no** model/training imports, ever (enforce with a comment + review) |
| `retrieval/encode.py` | `D128_DROP05_DEFAULTS`, `load_model_for_eval`, `encode_queries` | `training.model`, `training.evaluate` (the coupling now lives in exactly one file) |
| `retrieval/passes.py` | `QUALITY_BATCH_SIZE`, `quality_pass_cached`, `perf_pass_cached` | measure, metrics |

`queries_cache.py` and `loaders.py` switch their imports to `encode.py`; `sweep.py` to
`measure.py` + `passes.py`. Delete `bench_tools.py` after the moves (no re-export shim — this
is a private harness, the three importers are all in-repo:
`loaders.py:28`, `queries_cache.py` (indirect), `sweep.py:32-38`).

Fix the stale comment pair while moving: the `QUALITY_BATCH_SIZE = 16` rationale
(`bench_tools.py:324-329`) is current and
moves with the constant, but `quality_pass_cached`'s docstring still claims "Default
``batch_size=64`` cut per-cell wall ~50×"
(`bench_tools.py:349-353`) — reword to
reference the constant instead of a literal.

### E5.2 `PerfStats` / `QualityStats` instead of tuples

`measure_forward_cuda` and `perf_pass_cached` return positional 5-tuples, unpacked at
[sweep.py:475](../../evaluation/retrieval/sweep.py#L475) `med, p20, p80, peak, scratch = …`.
Roadmap Stage 4a wants `throughput_qps`, `mean_ms`, `p99_ms`, per-query latency vectors —
every tuple extension breaks all unpack sites.

```python
# retrieval/measure.py
@dataclass(frozen=True)
class PerfStats:
    median_ms: float
    p20_ms: float
    p80_ms: float
    peak_mib: float
    transient_mib: float
    # Stage 4a fields land here (mean_ms, p99_ms, throughput_qps, samples) —
    # additive, no call-site churn.

    def as_row_fields(self) -> dict[str, float]:
        return {"median_ms": self.median_ms, "p20_ms": self.p20_ms, "p80_ms": self.p80_ms,
                "peak_mem_mib": self.peak_mib, "fwd_scratch_mib": self.transient_mib}
```

```python
# retrieval/passes.py — quality returns all four computed metrics
@dataclass(frozen=True)
class QualityStats:
    recall: float
    ndcg: float
    precision: float
    mrr: float
```

**Why QualityStats**: [metrics.py:150-154](../../evaluation/retrieval/metrics.py#L150-L154)
already computes recall/precision/mrr/ndcg per batch; the harness then discards precision and
mrr (`bench_tools.py:381`). Emit them —
`_make_perf_row` adds `precision@{k}` / `mrr@{k}` columns (additive, Stage 4a-friendly, zero
extra compute). `_make_perf_row` slims to
`row = {**fixed_fields, **stats.as_row_fields(), f"recall@{k}": q.recall, …}`.

**Reproducibility check**: recall/ndcg values must be byte-identical to pre-refactor; the new
columns are additive only.

## Phase E6 — Oracle cache: content fingerprint, not shape check

**Problem.** [oracle.py:114-124](../../evaluation/retrieval/oracle.py#L114-L124) invalidates the
cached oracle only on `(n_users, K_GT)` **shape** mismatch. Same-shape content changes — a
different `content_subdir` dim off the same `data_dir`, regenerated attrs, a retrained
checkpoint — silently reuse a stale oracle. This burned the thesis once (the goodreads
stale-cache incident; [00-roadmap.md](00-roadmap.md) Stage 4b item 7 calls the rerun a
**publication blocker**), and the current mitigation is a *convention*: a YAML comment warning
that `gt_subdir` "MUST vary by dim"
([config/goodreads/d128-filter.yaml:4-7](../../evaluation/config/goodreads/d128-filter.yaml)).

**Fix.** Store the oracle as a dict with a content fingerprint; validate on load.

```python
# oracle.py
_FP_SAMPLE_ROWS = 64

def _oracle_fingerprint(item_embs: Tensor, queries: Tensor,
                        qa_narrow_sweep: Tensor | None, k_gt: int) -> str:
    """Cheap deterministic content hash: shapes + dtypes + a fixed row sample.

    Sampling (vs hashing 3M×256 fp32 fully) keeps this <10 ms; linspace rows
    catch dim changes, re-encodes, attr regens, and checkpoint swaps — any of
    which perturb sampled bytes. Not adversarially robust; doesn't need to be."""
    h = hashlib.sha256()
    tensors = [item_embs, queries] + ([qa_narrow_sweep] if qa_narrow_sweep is not None else [])
    for t in tensors:
        h.update(repr((tuple(t.shape), str(t.dtype))).encode())
        idx = torch.linspace(0, t.shape[0] - 1, steps=min(_FP_SAMPLE_ROWS, t.shape[0])).long()
        h.update(t[idx].detach().float().cpu().contiguous().numpy().tobytes())
    h.update(str(k_gt).encode())
    return h.hexdigest()
```

`load_or_build_oracle` changes:

```python
gt_path = gt_dir / f"gt_topk_v3_{sweep_name}.pt"      # v3: dict blob with fingerprint
fp = _oracle_fingerprint(item_embs, queries, qa_narrow_sweep, K_GT)
if gt_path.exists():
    blob = torch.load(str(gt_path), map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and blob.get("fingerprint") == fp:
        return blob["topk"]
    logger.warning("stale/legacy oracle at {} (fingerprint mismatch); recomputing", gt_path)
oracle_topk = compute_filtered_oracle(...)
torch.save({"topk": oracle_topk, "fingerprint": fp, "k_gt": K_GT}, str(gt_path))
```

Notes:

- Bump the filename prefix to `gt_topk_v3_` so v2 bare-tensor caches are simply ignored (and
  can be deleted); do not attempt in-place migration.
- `weights_only=True` works for dict-of-tensor+str blobs.
- Keep the per-dim `gt_subdir` convention as cheap defense-in-depth, but correctness no longer
  depends on it — downgrade the YAML comment from "MUST" to "keeps caches tidy per dim".
- The fingerprint must be computed from the **post-users_limit** tensors the oracle is built
  from (they are, in `run_one_sweep`'s scope), so changing `users_limit` also invalidates —
  today that happens to work via the `n_users` shape check; the fingerprint preserves it.
- This retires roadmap Stage 4b item 7's "fix described in oracle.py:113" placeholder.

## Phase E7 — Config / loader hygiene

### E7.1 `resolve_path` basename fallback

[loaders.py:38-47](../../evaluation/retrieval/loaders.py#L38-L47): a non-existent
`attrs_path: subdir/foo.pt` silently resolves to `data_dir/foo.pt` — **basename only** — which
is exactly how a wrong attr tensor gets loaded without an error. Replace:

```python
def resolve_path(data_dir: Path, path_str: str) -> Path:
    """cwd-relative if it exists; else data_dir-relative (full relative path,
    not basename); else raise listing both candidates."""
    p = Path(path_str).expanduser()
    if p.exists():
        return p
    candidate = (data_dir / p).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"{path_str!r}: tried {p.resolve()} and {candidate}")
```

Check the shipped YAMLs still resolve (they use `data/<dataset>/x.pt` paths that exist
cwd-relative when run from `evaluation/` — they do).

### E7.2 One YAML-strip implementation

Three copies of the `_`-prefixed anchor-key strip: `config.py:113`,
`run_evaluation.py:83` (`_load_cfg`), `stage_results.py:33-34`. Add to `config.py`:

```python
def load_raw_config(path: Path) -> dict:
    """yaml.safe_load + drop `_`-prefixed anchor scratch keys. The one place
    this rule lives; load_eval_config builds the dataclass on top."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}
```

`load_eval_config` uses it; `_load_cfg` and `stage_results.main` import it and read
`output`/`algorithms` from the raw dict as today.

### E7.3 Guard `output: null`

`_load_cfg` does `Path(data["output"])`
([run_evaluation.py:85](../../evaluation/retrieval/cli/run_evaluation.py#L85)) — a config
using the documented `output: null` default dies with a bare `TypeError`. Raise
`SystemExit(f"{path}: config must set output: (a directory) to be orchestrated")` instead.

### E7.4 Docstring sweep (no behavior)

- [config.py:1-13](../../evaluation/retrieval/config.py#L1-L13): references
  `eval_goodreads_retrieval.py`, `eval_arxiv_retrieval.py`,
  `docs/plans/goodreads-filter-eval.md` — none exist; rewrite against the current driver.
  Same for the field comments at lines 98-99.
- [algos/filter.py](../../evaluation/retrieval/algos/filter.py),
  [queries_cache.py](../../evaluation/retrieval/queries_cache.py),
  [results_io.py](../../evaluation/retrieval/results_io.py): current — leave.
- `_autotune_prewarm`'s docstring ([sweep.py:624-630](../../evaluation/retrieval/sweep.py#L624-L630))
  says it defends against "Triton's autotune cache" — the library moved to offline-tuned
  `DEFAULT_CONFIG`s (no runtime autotune) in roadmap Stage 1. The prewarm is still needed (it
  now covers **JIT compile + cudagraph capture** per batch size), but the rationale text is
  stale; rewrite: "Call forward once per batch size before timing so Triton JIT compilation
  and the per-shape cudagraph capture happen outside the timed window."

## Phase E8 — Tests + system-doc rewrite

### E8.1 CPU-only unit tests (land FIRST, before E1)

The eval package has exactly one test. These lock current behavior and make every phase
falsifiable without a GPU. New files under `evaluation/retrieval/tests/`:

```
tests/
├── test_metrics.py          # pure functions, hand-built tensors
├── test_sweep_helpers.py    # build_sweep_qa, is_valid_combo, (later) supports()
├── test_oracle.py           # padding semantics + (later) fingerprint invalidation
└── test_config.py           # load_eval_config round-trip
```

Sketches (implementing agent expands):

```python
# test_metrics.py
def test_hits_mask_ignores_padding():
    # -1 == -1 must NOT count as a hit (metrics.py:29-36)
    cand = torch.tensor([[5, -1, 3]]); tgt = torch.tensor([[-1, 3]])
    nt = torch.tensor([1])
    assert recall_at_k(cand, tgt, nt, k=3).item() == 1.0          # 3 hits; -1 doesn't
    assert recall_at_k(cand, tgt, nt, k=2).item() == 0.0          # -1 slot isn't a hit

def test_ndcg_idcg_clamps_to_k(): ...
def test_recall_denominator_is_num_targets_not_k(): ...           # sweep.py:600-603 rationale

# test_sweep_helpers.py
def test_build_sweep_qa_masks_inactive_clauses():
    qa = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    out, skip = build_sweep_qa(FilterSweepCfg(name="s", active_clauses=[1]), "clause", qa, 4)
    assert (out[:, [0, 2, 3]] == -1).all() and (out[:, 1] == qa[:, 1]).all()
    assert not skip.any()

def test_build_sweep_qa_skip_mask_all_inactive(): ...
def test_build_sweep_qa_out_of_range_clause_raises(): ...

# test_oracle.py  (CPU device works: compute_filtered_oracle only needs torch)
def test_oracle_pads_short_rows_with_minus_one():
    # 3 items, filter admits 1 → K_GT=2 rows end with -1, not index-0 junk
    ...
def test_oracle_fingerprint_invalidates_on_content_change():   # after E6
    ...

# test_config.py
def test_underscore_keys_stripped(): ...
def test_filters_block_upgraded_to_dataclasses(): ...
def test_encode_defaults(): ...
```

Also port the goodreads YAMLs' shape into a `tests/data/mini.yaml` fixture so config tests
don't read the real configs.

### E8.2 Rewrite `docs/system/evaluation.md`

After E1–E5 land, rewrite from the Ground-truth section above. Current doc's concrete lies to
fix: `evaluation/retrieval/evaluate.py` (now `cli/evaluate.py` + `sweep.py` + `loaders.py` +
`oracle.py`), `run_per_algo.sh` (now `run-evaluation`), `evaluation/conf/` (now
`evaluation/config/`), `algo.modules` (now `algo_modules`), `QUALITY_BATCH_SIZE = 64` (now 16),
missing `backend` column + `backends:` fan-out, missing `--eval-type` presets and
`stage-results`/`upload-results`, missing `users_limit` in the cache key, missing
`gt_topk_v2_`→`v3` naming and fingerprint, "Seven algorithm names … one duplicate alias"
(now six after E1.1), the `is_cpu` perf-primitive selection note.

## Additional smaller improvements (fold into nearest phase, or skip)

Catalogued during the audit; none blocks the phases above.

- **JSONL streaming writes** (crash resilience): `run_sweep` accumulates all rows in memory and
  `cli/evaluate.py` writes once at the end — an OOM in the last cell of a 6-hour config loses
  everything. Cheap fix: `evaluate.py` opens `<output>.partial.jsonl`, `evaluate_cell` appends
  rows as produced, final success renames to the JSON list format (or keep JSONL and teach
  `results_io.load_rows` both formats). Combine with an orchestrator `--resume` that can then
  skip *cells*, not just whole algos. Medium effort; do after E2 when rows flow through one
  choke point.
- **tqdm vs orchestrator logs**: the quality/oracle tqdm bars
  (`bench_tools.py:361`,
  [oracle.py:58](../../evaluation/retrieval/oracle.py#L58)) write control characters into the
  orchestrator's tee'd logs. Pass `disable=not sys.stderr.isatty()` (or
  `TQDM_DISABLE`-aware) to both.
- **`torch.load(..., weights_only=False)`** in
  [queries_cache.py:54](../../evaluation/retrieval/queries_cache.py#L54): the blob is
  dict-of-tensors + scalars, so `weights_only=True` works and removes a pickle-execution
  surface on a shared box. Same in `loaders.load_filter_assets` (already plain tensors;
  add the flag).
- **`EVAL_TYPES` preset lists** ([run_evaluation.py:35-54](../../evaluation/retrieval/cli/run_evaluation.py#L35-L54))
  hard-code config paths that must be hand-synced with `evaluation/config/`. Fine at this
  scale; alternative is `sorted(glob("config/*/d*-filter.yaml"))` at call time — take it only
  if a preset drifts again.
- **`_gpu_name()` / provenance in rows**: the orchestrator logs GPU name to SUMMARY.txt but
  rows carry no hardware/provenance fields. Adding `extra.gpu`, `extra.torch`, `extra.commit`
  to `_make_perf_row` is a 5-line change that future-proofs cross-machine result pooling
  (thesis appendix wants this). Additive columns — allowed.
- **`is_valid_combo` placement**: after E3.2 it is the only remaining eligibility helper;
  move next to `SUPPORTED_FILTER_KINDS` in the registry so all eligibility logic lives in one
  module.
- **Naming**: `linr_v1_filter_mask` vs alias `triton_knn` (a *pure-torch* full scan, despite
  the name — the Triton kernel was removed;
  [linr_v1.py:3-7](../../evaluation/retrieval/algos/linr_v1.py#L3-L7)). Renaming breaks YAML
  lineage + downstream result joins, so **don't rename**; instead make the registry docstring
  the canonical explanation and add `extra.impl_note` only if confusion recurs.

## Verification

1. **Golden-run diff** (see Conventions): quality columns byte-identical, same `cell` keys,
   latency within ~5%, plus the new additive columns after E5.2.
2. **Unit tests** green without GPU: `cd evaluation && uv run pytest retrieval/tests/ -v`.
3. **Oracle fingerprint test**: build on synthetic data → mutate one item embedding → assert
   recompute; restore → assert cache hit (E8.1/test_oracle.py).
4. **Grep gates**: zero hits for `expand_param_combos`, `is_cpu`, `measure_forward_cpu`,
   `TorchKnnAlgo`, `BACKEND_CAPABLE_ALGOS`; `torchvision|matplotlib|einops` absent from
   pyproject; exactly one site strips `_`-prefixed YAML keys; no `import bench_tools` left.
5. **Orchestrator smoke**: `uv run run-evaluation config/goodreads/d128-filter.yaml --resume
   -- --skip-quality --sweep c0_genre` completes and writes per-algo JSONs (exercises the E7.3
   guard, subprocess flow, staging untouched).
6. `uv run ruff check` clean; pyrefly/pyright (retrieve has `pyrefly.toml`; if evaluation gains
   one, keep it green).

## Critical files

| File | Phases |
|---|---|
| [evaluation/retrieval/sweep.py](../../evaluation/retrieval/sweep.py) | E1.2-3, E2, E3, E5.2 |
| `evaluation/retrieval/context.py` | E2 — CREATE |
| `evaluation/retrieval/bench_tools.py` | E1.2, E5.1 — split into `measure.py` / `encode.py` / `passes.py`, then DELETE |
| [evaluation/retrieval/algos/__init__.py](../../evaluation/retrieval/algos/__init__.py) | E1.1, E3 |
| `evaluation/retrieval/algos/torch_knn.py` | E1.1 — DELETE |
| [evaluation/retrieval/algos/_helpers.py](../../evaluation/retrieval/algos/_helpers.py) | E3.1, E4 |
| algos/{linr_v1,linr_v2,linr_v3,linr_v4,silvertorch}.py | E1.2, E4 |
| [evaluation/retrieval/oracle.py](../../evaluation/retrieval/oracle.py) | E6 |
| [evaluation/retrieval/loaders.py](../../evaluation/retrieval/loaders.py) | E2.1, E7.1 |
| [evaluation/retrieval/config.py](../../evaluation/retrieval/config.py) | E3.3, E7.2, E7.4 |
| [evaluation/retrieval/cli/evaluate.py](../../evaluation/retrieval/cli/evaluate.py) | E2 |
| [evaluation/retrieval/cli/run_evaluation.py](../../evaluation/retrieval/cli/run_evaluation.py) | E7.2, E7.3 |
| [evaluation/retrieval/cli/stage_results.py](../../evaluation/retrieval/cli/stage_results.py) | E7.2 |
| [evaluation/retrieval/cli/upload_results.py](../../evaluation/retrieval/cli/upload_results.py) | E1.5 |
| [evaluation/pyproject.toml](../../evaluation/pyproject.toml) | E1.4, E1.6 |
| `evaluation/retrieval/tests/…` | E8.1 — CREATE (4 files) |
| [docs/system/evaluation.md](../system/evaluation.md) | E8.2 rewrite |

## Sequencing + effort

| step | phases | est. diff | risk |
|---|---|---|---|
| 1 | E8.1 tests | +300 | none (new files) |
| 2 | E1 deletions | −250 / +30 | none |
| 3 | E2 context objects | ±400 | mechanical but wide; golden-diff gate |
| 4 | E3 + E4 | ±150 | low |
| 5 | E5 split + stats | ±250 | low; additive columns |
| 6 | E6 oracle fingerprint | ±80 | low; forces one oracle rebuild per sweep on first run |
| 7 | E7 hygiene | ±60 | none |
| 8 | E8.2 doc rewrite | doc only | none |

After landing, trim this plan per repo convention ([00-roadmap.md](00-roadmap.md)
"Cleanup status").
