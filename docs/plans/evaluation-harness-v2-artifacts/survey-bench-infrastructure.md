# Benchmark & sweep infrastructure: design research for the retrieve GPU bench harness

Scope: how mature microbenchmark, ML/GPU benchmark, sweep-orchestration and
results-storage tools are engineered, with citations, aimed at a single-machine
~700-cell, ~24h GPU retrieval benchmark campaign (JSONL-per-cell + parquet
latency vectors, `bench run/campaign/report` CLI, subprocess isolation,
resume-by-key, minimal abstractions).

---

## §1 Timing methodology: the actual consensus, and where tools disagree

### 1.1 The shared skeleton

Every serious microbenchmarking tool (Criterion.rs, Google Benchmark,
nanobench, pytest-benchmark, asv, torch.utils.benchmark) implements the same
three phases, differing only in the knobs:

1. **Calibrate** how much work constitutes one measured "iteration" or
   "round" so that timer-resolution noise is a small fraction of the
   measured signal.
2. **Warm up** (untimed) to reach a steady state (caches, JIT, clocks,
   allocators) before recording anything.
3. **Repeat the measurement** N times and report a **distribution**, not a
   single number — because runtimes are frequently heavy-tailed / skewed,
   so mean is fragile and median/IQR (or the whole distribution) is what's
   actually trustworthy.

Concretely:

- **Google Benchmark**: iterates until "CPU time is greater than the
  minimum time, or wall-clock time is 5x minimum time," with `iterations`
  bounded to `[1, 1e9]`. Default `--benchmark_min_time=0.5s`.
  `--benchmark_min_warmup_time` (default 0.0s) adds an explicit, discarded
  warmup phase. Repeating the whole benchmark via
  `--benchmark_repetitions=N` (default 1) is the tool's mechanism for
  getting a *distribution across repetitions* — inside each repetition,
  only one point estimate exists (a single mean of many iterations); the
  documented statistics (mean/median/stddev/cv) are computed **across
  repetitions**, not across the iterations inside one repetition.
  [User Guide](https://google.github.io/benchmark/user_guide.html),
  [tools.md](https://github.com/google/benchmark/blob/main/docs/tools.md)
- **Criterion.rs**: default `sample_size = 100` measurements, `warm_up_time
  = 3s`, `measurement_time = 5s` (targets, not hard iteration counts —
  iteration counts per sample are chosen adaptively, scaling roughly
  linearly, `d, 2d, 3d, ... Nd`). It then runs **bootstrap resampling with
  `nresamples = 100,000`** over the measured samples to build confidence
  intervals for mean, std-dev, median, MAD and the linear-regression slope
  estimate of the runtime, at `confidence_level = 0.95`.
  [Criterion struct docs (docs.rs)](https://docs.rs/criterion/latest/criterion/struct.Criterion.html)
- **nanobench**: default **11 epochs**; per-epoch iteration count is
  calibrated by doubling until roughly one epoch's worth of time passes,
  then *re-measuring that same count a second time* before trusting it
  (protects against a single preempted/lucky calibration run). It reports
  the **median over epochs**, plus `err%` (a median-absolute-percentage
  measure), and flags a row unstable (wavy-dash marker) once `err% > 5%`,
  suggesting the fix is to raise `minEpochIterations`, not to average
  harder. Zero-time and clock-tied rounds are dropped as uninformative.
  [nanobench tutorial](https://github.com/martinus/nanobench/blob/master/src/docs/tutorial.rst)
- **pytest-benchmark**: two-level structure — a **round** (whose size is
  fixed once, during calibration, by growing iteration count until the
  round duration is ~10× the timer resolution) and a fixed/adaptive number
  of **rounds** (`--benchmark-min-rounds`, default 5; `--benchmark-max-time`
  caps total time; or `--benchmark-precision=X` for a **margin-of-error
  stopping rule**: keep running rounds until the CI half-width on the mean
  is below X at the configured confidence, subject to min/max round
  bounds). Reported stats include mean, median, stddev, IQR, min/max, and
  two independent outlier counts: **stddev outliers** (beyond mean ± 1σ)
  and **IQR outliers** (Tukey fences on Q1/Q3).
  [Calibration](https://pytest-benchmark.readthedocs.io/en/latest/calibration.html),
  [Glossary](https://pytest-benchmark.readthedocs.io/en/latest/glossary.html)
- **asv**: per-benchmark, stores `stats_ci_99_a`/`stats_ci_99_b` (a 99% CI
  pair), `stats_q_25`/`stats_q_75`, `stats_number` and `stats_repeat`, plus
  the raw `samples` array — i.e. it keeps both the summary statistic *and*
  the raw samples so post-hoc reanalysis is possible.
  [asv result schema / dev docs](https://asv.readthedocs.io/en/stable/dev.html)
- **torch.utils.benchmark.Timer.blocked_autorange()**: explicitly designed
  as a fusion of `timeit.Timer.autorange` (pick a loop count) and
  `.repeat` (repeat and keep all samples) — the stated reason is that
  plain `autorange` "discards ~75% of measurements" by using a single
  large timed block; `blocked_autorange` grows `block_size` during warmup
  until per-invocation timer overhead is under ~0.1% of the block's total
  time, then repeats *that* block many times, keeping every block's timing
  in the resulting `Measurement`. `Timer` also auto-syncs CUDA and pins
  `torch.set_num_threads`.
  [torch.utils.benchmark README](https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/README.md),
  [docs](https://docs.pytorch.org/docs/stable/benchmark_utils.html)

### 1.2 Why "don't report the mean alone" is near-universal

- nanobench reports median specifically because "initial outliers will be
  filtered away automatically" by a robust statistic — no separate warmup
  phase is even required by design.
  [nanobench README/tutorial](https://github.com/martinus/nanobench)
- pytest-benchmark and Criterion both keep the *raw* per-round/per-sample
  data and outlier counts rather than collapsing to one number, precisely
  so a fat right tail doesn't get silently absorbed into a "clean" mean.
- Practitioner consensus (easyperf.net, echoing Kalibera & Jones 2013,
  "Rigorous Benchmarking in Reasonable Time," ISMM'13) is: for ≥30 samples
  with small standard error, compare means; for few samples or skewed
  (typically right-skewed, compute-bound) distributions, use
  minimum/median because "the mean can be spoiled by outliers," and always
  visualize the distribution (box/violin) rather than trust one scalar. A
  **bimodal distribution is itself a signal the measurement setup is
  broken** (e.g., turbo-boost transitions, background noise), not
  something to average through.
  [Easyperf: Comparing performance measurements](https://easyperf.net/blog/2019/12/30/Comparing-performance-measurements),
  [Kalibera & Jones, KAR](https://kar.kent.ac.uk/33611/45/p63-kaliber.pdf)
- Criterion's own noise model makes this explicit as a first-class knob:
  `noise_threshold` (default 0.01 = 1%) means "changes smaller than 1%
  will be ignored" when comparing to a baseline, and `significance_level`
  (default 0.05) is "the probability that two measurements of identical
  code will be considered 'different' due to noise" — i.e. Criterion
  treats every regression/improvement claim as a hypothesis test with an
  explicit false-positive budget, not a raw delta.
  [docs.rs Criterion struct](https://docs.rs/criterion/latest/criterion/struct.Criterion.html)

### 1.3 Outlier handling: flag, don't drop

- **Criterion**: modified Tukey's-fences classification into *mild*
  (beyond Q1/Q3 ± 1.5×IQR) and *severe* (± 3×IQR) outliers. Outliers are
  **not removed** from the sample — they're reported as a warning
  ("N (X%) high mild, M (Y%) high severe" etc.) so the user decides.
- **pytest-benchmark**: same idea, dual criteria (stddev-based and
  IQR/Tukey-based), both counts surfaced, nothing dropped.
- **nanobench**: robust-by-construction (median), with an explicit
  instability flag rather than outlier removal.

None of the five tools researched silently discards outliers from the
reported statistic — the consistent design is *detect, surface, let the
robust statistic (median) absorb it, and warn*, never quietly delete data
points to make a number look cleaner.

### 1.4 Regression detection between runs

- **Criterion `--baseline`/`--save-baseline`/`--load-baseline`**: named,
  on-disk baselines (`target/criterion/<bench>/<baseline>/`) with a
  bootstrap-based two-sample comparison (Welch-like t-test on bootstrap
  resamples of the difference of means) gated by `noise_threshold` and
  `significance_level`.
  [Command-line options](https://bheisler.github.io/criterion.rs/book/user_guide/command_line_options.html)
  There is also a companion CLI, **critcmp**
  ([BurntSushi/critcmp](https://github.com/BurntSushi/critcmp)), that
  diffs two saved Criterion baselines outside of `cargo bench` itself —
  i.e. regression comparison is treated as a separate, composable tool
  operating on the stored JSON, not baked only into the runner.
- **asv**: regression detection is a batch/offline pass (`asv publish`)
  over the full commit-indexed history, modeled as **piecewise-constant +
  noise** (a changepoint/step-detection problem, `asv.step_detect`), which
  is a fundamentally different framing from Criterion's "compare exactly
  two runs" — asv is designed to answer "which commit introduced this
  regression" across a whole git history, Criterion to answer "did *this*
  change regress relative to *that* saved baseline."
  [asv dev docs](https://asv.readthedocs.io/en/stable/dev.html)
- **Firefox Perfherder**: regressions are detected online, per new data
  point, via a T-test against a sliding window, requiring the change to
  "sustain" over the next ~12 data points before an alert fires (reduces
  false positives from one noisy run) — alerts on the same revision are
  grouped into one alert summary for a human ("performance sheriff") to
  triage.
  [Mozilla Performance Sheriffing docs](https://firefox-source-docs.mozilla.org/testing/perfdocs/perf-sheriffing.html)

### 1.5 Points of open disagreement across tools

- **What "N" should scale**: Google Benchmark scales *iterations inside one
  repetition* to hit a time budget, then optionally repeats the whole
  benchmark N times for statistics across repetitions. Criterion scales
  iterations similarly but treats every sample as a statistic-bearing
  unit directly. pytest-benchmark explicitly separates "rounds" (the
  statistical unit) from "iterations" (the calibration unit to beat timer
  resolution) — these look similar but are philosophically different
  units of repetition, and porting numbers between tools without knowing
  which axis you're reading is a common bug source.
- **Bootstrap vs asymptotic CIs**: Criterion always bootstraps (100k
  resamples); Google Benchmark and pytest-benchmark, by contrast, use
  simple sample stddev/CI formulas — nobody except Criterion (and, in the
  systems-research literature, Kalibera & Jones-style tooling) bootstraps
  by default. There is no consensus that bootstrapping is required;
  it's treated as "more correct, more expensive," and most tools skip it.
- **Discard vs keep raw data**: asv, pytest-benchmark's JSON store, and
  Criterion's baseline files all keep the *raw* per-sample vector
  alongside the summary; Google Benchmark's default JSON keeps raw
  per-iteration rows only if you don't pass `--benchmark_report_aggregates_only`
  — i.e. there's a real, common failure mode of throwing away the raw
  vector and keeping only the aggregate, which later blocks any
  re-analysis with a different statistic. This is directly relevant to us
  since we already plan to keep a parquet of per-call latency vectors —
  that decision matches the tools that got this right (asv, Criterion,
  pytest-benchmark), not the ones that default to aggregates-only.

---

## §2 Sweep tooling comparison

| Tool | Grid declaration | Run keying / dedup | Results retrieval | Adoption cost |
|---|---|---|---|---|
| **Hydra multirun** (+ basic/Optuna/Ax sweepers) | `--multirun key=a,b,c other=1,2` (basic sweeper does full cross-product); Optuna/Ax sweepers replace the cross-product with a search algorithm over a declared space (YAML `sweeper` config) | Directory-name keyed: `hydra.sweep.dir` (default `multirun/<date>/<time>`) + per-job `hydra.sweep.subdir`, usually built from `job.override_dirname` (the sorted `key=value,key2=value2` string of that job's overrides, with configurable separators and an `exclude_keys` list e.g. to drop `seed` from the dirname). No built-in "skip if this exact override-set already ran" — dedup across restarts is on you. [Multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), [workdir](https://hydra.cc/docs/configure_hydra/workdir/) | Each job's own output dir has its own logs/config; no built-in dataframe view — you `glob` the sweep dir and load configs+outputs yourself (or use Optuna's own `optimization_results.yaml` for the tuned case). [Optuna sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/) | **Low** for the basic (cross-product) sweeper — it's an argument-parsing/directory-naming convention on top of your own script; **medium** if you adopt Optuna/Ax sweepers since then you also inherit their storage/algorithm machinery |
| **submitit** | Not a sweep declarer itself — a job-submission library. You build the list of param combos yourself (e.g. `itertools.product`) and call `executor.submit(fn, params)` per combo, or `executor.map_array` for a batch. Ships `LocalExecutor`/`DebugExecutor` (in-process, for exactly our single-machine case) alongside `SlurmExecutor`, unified under `AutoExecutor`. [submitit repo](https://github.com/facebookincubator/submitit), [structure.md](https://github.com/facebookincubator/submitit/blob/main/docs/structure.md) | None built in — `Job` objects are `concurrent.futures`-style futures with their own job-id and pickled result/exception on disk; you key runs yourself | `job.result()` returns whatever your function returned (arbitrary Python object) — no dataframe abstraction | **Low**: essentially a `concurrent.futures`-shaped subprocess/Slurm launcher. Directly useful for our "per-algo subprocess isolation" need if the harness ever needs a queueing layer, but overkill if a simple `subprocess.run` loop already does the job on one machine |
| **W&B Sweeps** | YAML: `method: grid\|random\|bayes`, `parameters: {name: {values: [...]}}` [sweep-config-keys](https://docs.wandb.ai/models/sweeps/sweep-config-keys) | Server-side; agents pull the next un-run config from the sweep controller. **Known weak point**: multiple GitHub issues (2021–2024) report duplicate runs when multiple agents start near-simultaneously, and resuming a grid sweep after deleting/failed runs does **not** reliably relaunch the missing configs — dedup/resume for grid mode is documented as buggy, not just under-documented. [wandb/wandb#3522](https://github.com/wandb/wandb/issues/3522), [#6594](https://github.com/wandb/wandb/issues/6594), [#1787](https://github.com/wandb/wandb/issues/1787) | Full web UI + `wandb.Api().runs()` → pandas-friendly | **High**: requires an account/server (cloud or self-hosted), a running agent process, network dependency for a benchmark that's supposed to run unattended for 24h on one box |
| **MLflow Tracking** | No grid primitive — you write your own loop and call `mlflow.start_run()` per cell; `mlflow.start_run(nested=True)` under a parent run groups a sweep as parent/children. [Understanding Parent and Child Runs](https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/part1-child-runs/) | None automatic; you key by whatever params/tags you log yourself and query them back to detect "already done" | `mlflow.search_runs()` returns a **pandas DataFrame** directly — closest of all these tools to "give me a dataframe of everything" | **Medium**: needs a tracking server or at least a local `mlruns/` dir + the mlflow package; parent/child run model and its UI are more machinery than a JSONL file, but the params/metrics/tags/artifacts model is exactly the shape we want and it auto-captures **git commit, branch, dirty flag, source file** as system tags (`mlflow.source.git.commit`, `.git.branch`, `.git.dirty`, `.git.repoURL`) which is exactly our provenance need. [mlflow_tags.py](https://github.com/mlflow/mlflow/blob/master/mlflow/utils/mlflow_tags.py) |
| **Sacred** | Python `@ex.config` "config scopes" (plain functions whose locals become the config) + `ex.observers.append(...)`; grid sweeps are typically driven externally (e.g. a shell loop or `sacred`+`labwatch`), not a first-class Sacred feature | Observers assign each run an `_id`; no built-in cross-run dedup | Observers (Mongo/file/S3/etc.) log config+result; querying back is whatever the observer backend supports (e.g. MongoDB queries) | **Medium-high**: decorator-based config-injection model, a whole Ingredient/Observer object model — this is exactly the "config object model" our brief says to avoid. [Sacred docs](https://sacred.readthedocs.io/), [Observers](https://sacred.readthedocs.io/en/latest/observers.html) |
| **DVC experiments** | `dvc.yaml` pipeline stages + `params.yaml`; sweep via CLI: `dvc exp run -S 'train.min_split=8,64' -S 'train.n_est=range(100,500,100)' --queue` expands to the full cross-product and enqueues it (Hydra-style choice/range syntax supported). [exp run docs](https://doc.dvc.org/command-reference/exp/run) | **Content-addressed**: DVC hashes each stage's declared deps/params/code (MD5) and **skips re-running a stage whose hash is unchanged** — this is real, working, automatic idempotent-resume, not just a convention. [Internal Files](https://doc.dvc.org/user-guide/project-structure/internal-files) | `dvc exp show` / `dvc exp diff` render a table across queued experiments; can export to CSV/JSON | **Medium**: requires adopting DVC's pipeline/params/cache model (a real dependency, git-integrated, its own cache dir) — good hashing semantics, but a bigger footprint than we want for a single evaluation repo that already has its own CLI |
| **Optuna** (as used via Hydra sweeper or standalone) | Program-defined search space (`trial.suggest_*`) rather than a static grid; grid is possible via `GridSampler` | `storage='sqlite:///...'` + `load_if_exists=True` gives real persistent/resumable studies, and **distributed** optimization is supported by pointing multiple processes at the same SQLite/RDB backend — but no explicit trial-level dedup feature was found in the docs searched. [Saving/Resuming with RDB](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html), [Distributed](https://optuna.readthedocs.io/en/v2.2.0/tutorial/004_distributed.html) | `study.trials_dataframe()` → pandas | **Medium**: worth considering only if/when we add a Bayesian search phase over hyperparameters; pure overkill for an exhaustive 700-cell grid |

### Verdict for our case (single machine, ~700 cells, ~24h, minimal abstractions)

- **Adopt the *pattern*, not the *tool*, from Hydra**: `key=value,key2=value2`
  deterministic naming for a cell (their `override_dirname` idea) is a good
  model for our resume-by-key hash — but running actual Hydra adds a config
  object model (OmegaConf, `@hydra.main`, config groups) we don't need.
  **[cheap to imitate the naming convention, don't adopt the framework]**
- **submitit's `LocalExecutor`/`AutoExecutor` split is worth studying** as
  a design precedent for "the same code path runs locally now and on a
  cluster later without touching call sites" — relevant if the harness
  ever needs multi-machine campaigns, but out of scope now given
  single-A100 usage.
- **W&B Sweeps and Sacred are both overkill and, per the linked GitHub
  issues, W&B's grid-resume/dedup story is actively unreliable** — this
  directly undercuts adopting it for something we need to trust for
  unattended 24h runs.
- **MLflow's *provenance tag set* is worth copying verbatim** even without
  adopting MLflow itself: git commit, branch, dirty flag, repo URL are
  exactly the fields a mature harness captures automatically, and they're
  cheap to write as a JSON block per JSONL record.
- **DVC's content-hash-based stage skip is the right mental model for our
  "resume-by-key"** feature: hash the *cell's* full identity (dataset,
  dim, filter_kind, sweep, algo, backend, params, seed, **and code
  version**) and treat a matching hash as "already done" — this is
  materially stronger than a naive "does this JSONL contain this name yet"
  check because it also invalidates cache when the algorithm code itself
  changes.

---

## §3 Results storage: schema shape and provenance checklist

### 3.1 JSONL vs parquet vs sqlite — what the field actually does

- **"Results are files in the repo" is a first-class, load-bearing
  pattern**, not a toy shortcut: **ClickBench** stores
  `system/results/YYYYMMDD/*.json`, one JSON per hardware/DB configuration,
  explicitly so the whole result set can be queried with
  `clickhouse-local: SELECT * FROM '*/results/*/*.json'` — i.e. the
  directory-of-JSON-files *is* the database, queried lazily, and plots are
  generated from it on demand. [ClickBench](https://github.com/ClickHouse/ClickBench)
- **DuckDB's db-benchmark** goes even flatter: results land as rows
  appended to `time.csv` / `logs.csv` in the repo, reviewed and merged via
  normal PR review, then a static report is regenerated —
  "spreadsheet-as-database," and it works at their scale.
  [db-benchmark](https://github.com/duckdblabs/db-benchmark),
  [DuckDB blog: Benchmarking Ourselves over Time](https://duckdb.org/2024/06/26/benchmarks-over-time)
- **asv**: one JSON file per `(machine, commit, env)` triple
  (`results/<machine>/<hash>-pyX.Y-depA-depB.json`), with a
  separate `benchmarks.json` (metadata: source code hash, params,
  param_names, "version") — i.e. **schema/definition metadata is
  versioned separately from the numeric results**, and a benchmark's
  "version" is by default the hash of its own source code, so a changed
  benchmark implementation is detected as a discontinuity rather than
  silently compared against old semantics. [asv dev docs](https://asv.readthedocs.io/en/stable/dev.html)
- **pytest-benchmark**: JSON per run under
  `.benchmarks/<platform-interpreter-arch>/000N_<hash>_<timestamp>.json`,
  carrying both `machine_info` and `commit_info` blocks alongside the
  stats — this pairing (hardware fingerprint + code fingerprint,
  co-located with the numbers) recurs in every mature tool below.
- **Conbench**: explicitly designed as a *language-independent* results
  schema — benchmarks in any language POST a JSON result to an API; besides
  the numeric result it always collects **machine info (architecture, CPU,
  L1d/L1i/L2/L3 cache sizes)** and a project-configurable "context" block,
  precisely so cross-machine/cross-language comparisons don't silently
  conflate different hardware. [Conbench](https://github.com/conbench/conbench),
  [Introducing Conbench](https://ursalabs.org/blog/announcing-conbench/)
- **Codespeed / Firefox Perfherder**: both use a normalized relational
  model (Codespeed: Django models `Executable / Revision / Project /
  Branch / Environment / Benchmark / Result`, with `Result` carrying
  optional stddev/min/max) — the "Environment" entity is mandatory and
  must pre-exist, i.e. **you cannot log a result without first declaring
  what machine it ran on**. [Codespeed](https://github.com/tobami/codespeed)

### 3.2 Recommended schema shape for our harness

Given our record is "one cell = one JSONL line, append the moment it
finishes," the converged shape across the tools above is a flat record
with clearly separated blocks:

```jsonc
{
  "schema_version": 1,                 // asv/pytest-bench pattern: version the record shape itself
  "cell_key": "arxiv-d128-nofilter-sweepA-silvertorch-triton-...-seed0",  // deterministic, hash-stable
  "cell_key_hash": "sha256:...",       // DVC-style content hash of the full cell identity, for resume-by-key
  "provenance": {
    "git_commit": "b7f4e22...",
    "git_dirty": false,                // MLflow's mlflow.source.git.dirty — never citable if true (rule 2)
    "git_branch": "development",
    "hostname": "...",
    "gpu": "A100-SXM4-80GB",
    "driver_version": "...",
    "cuda_version": "12.x",
    "torch_version": "2.10.0+cu128",
    "triton_version": "3.6.0",
    "python_version": "3.11.x",
    "clocks_locked_mhz": 1410,          // our own rule 1 requirement, made explicit in the record
    "config_hash": "sha256:..."         // hash of the resolved YAML config, catches silent config drift
  },
  "identity": {"dataset": "...", "dim": 128, "filter_kind": "...", "sweep": "...",
               "algo": "silvertorch", "backend": "triton", "params": {...}, "seed": 0},
  "started_at": "...", "finished_at": "...", "duration_s": ...,
  "build": {"time_s": ..., "index_memory_bytes": ...},
  "quality": {"recall@10": ..., "ndcg@10": ..., ...},
  "latency": {                          // summary only; raw vectors live in the parquet sidecar
    "(k=10,bs=1,mode=single)": {"mean_us": ..., "median_us": ..., "p50": ..., "p90": ..., "p99": ...,
                                  "stddev_outliers": 0, "iqr_outliers": 2, "n": 500}
  },
  "status": "ok"                        // | "failed" | "partial" — see §4
}
```

The **provenance block is the single highest-value cheap addition** — every
tool surveyed that people actually trust for cross-run comparison
(asv, pytest-benchmark, MLflow, Conbench, Codespeed) puts hardware +
code-version fields directly on the result record, not in a separate
"campaign metadata" file the reader has to cross-reference.

### 3.3 Provenance field checklist mature harnesses converge on

From asv's `machine.json` (`machine, os, arch, cpu, num_cpu, ram`)
+ pytest-benchmark's `machine_info`/`commit_info` + MLflow's git tags
+ Conbench's machine/cache-hierarchy block + Google Benchmark's `context`
object (date, host_name, num_cpus, mhz_per_cpu, cpu_scaling_enabled,
cache sizes, library version, `json_schema_version`), the recurring
fields are:

- **Code identity**: commit hash, dirty-tree flag, branch — every tool
  that has this treats a dirty tree as a hard "cannot trust this number"
  signal, matching our own rule 2 ("not yet validated").
- **Hardware identity**: CPU/GPU model, core/SM count, cache sizes,
  memory size, and (Google Benchmark specifically) whether CPU frequency
  scaling is enabled — the GPU analogue is exactly our "clocks locked"
  flag, which naive harnesses (that don't come from an HPC/systems
  background) routinely omit.
- **Toolchain/driver identity**: compiler/interpreter version (asv stores
  Python version + full `requirements` dict); our analogue is
  torch/triton/CUDA/driver versions.
- **Config identity**: a hash of the fully-resolved config, so two runs
  claiming "same params" can be checked byte-for-byte, not just by eyeballing
  YAML.
- **Timing/run metadata**: timestamp (asv stores commit date, not just
  wall-clock time of the run, so time-series analyses are keyed correctly),
  duration, and (Google Benchmark, pytest-benchmark) load/CPU-scaling state
  at run time, which naive harnesses forget and then can't explain outliers
  months later.
- **Schema version**: asv (`version: 2`), Google Benchmark
  (`json_schema_version`), pytest-benchmark (implicit via package version
  in the filename) — all version their record schema explicitly so a
  future reader/report tool can branch on it instead of guessing.

### What naive harnesses forget (explicitly, per the above)

1. **The dirty-git-tree flag.** Recording the commit hash without also
   recording whether the tree was clean makes "reproduce this number"
   impossible.
2. **Whether frequency/clock scaling was active.** Google Benchmark's
   `cpu_scaling_enabled` field exists because unpinned clocks make
   cross-run comparison meaningless — our GPU analogue (`nvidia-smi -lgc`)
   is already a stated hard rule in CLAUDE.md but should be a **recorded
   field**, not just an operational habit, or a future reader can't tell
   which historical runs had it.
3. **Raw per-sample data**, when only the aggregate is kept (Google
   Benchmark's aggregates-only mode is opt-in and a real footgun if
   someone flips it without realizing what's lost).
4. **A schema version field**, so record-shape evolution doesn't silently
   corrupt older records when the reporting code changes.
5. **The benchmark/algorithm code's own version/hash**, independent of the
   repo commit — asv's `benchmarks.json` treats a changed benchmark
   *definition* as a new "version" so historic numbers aren't silently
   compared across a redefinition of what's being measured. Directly
   relevant to us: if an algorithm's kernel changes, the cell's stored
   result should not be silently treated as comparable to the pre-change
   result just because the YAML config hash is unchanged.

---

## §4 Resume / robustness patterns

- **Content hashing for idempotent skip (DVC)**: hash every declared
  dependency (code, params, inputs) per unit of work; unchanged hash ⇒
  skip. This is the most rigorous version of "resume by key" found — the
  key isn't just "did we run cell X" but "did we run cell X with this
  exact code+config," which additionally catches the case where an
  algorithm's implementation changed silently.
  [DVC internal files](https://doc.dvc.org/user-guide/project-structure/internal-files)
- **Append-only log + one-file-per-unit as the crash boundary
  (ClickBench, db-benchmark, asv)**: each unit of work (a hardware config,
  a query, a commit×machine×env triple) writes to its **own** file; a
  crash mid-run corrupts at most the one file/line currently being
  written, never the whole results set. This maps directly onto our
  planned "append one JSONL record the moment a cell finishes" — the
  per-cell file/record boundary *is* the crash-recovery unit, and it's the
  same design every one of these systems converged on independently.
- **Atomic write, not atomic append, for the record itself**: JSONL
  durability write-ups converge on "write to temp file, then rename" (or
  O_APPEND with a single `write()` syscall for a line short enough to be
  atomic on the target filesystem) as the way to guarantee a reader never
  observes a half-written JSON object; a corrupted last line should be
  treated as `status: crashed`/discardable on next read, not as fatal to
  the whole file. General pattern discussed in JSONL-recovery contexts
  (agent-resume style checkpointing: "if a process dies mid-run, resuming
  finds the existing checkpoint and resumes from where it left off";
  corruption is detected because the last JSONL line fails to parse).
- **Marker/state files for "in progress" vs "done"**: DVC's `dvc.lock`
  records the hash-state after a *successful* stage run, so a
  partially-run stage simply doesn't get a lock entry and is correctly
  retried — the presence of a valid, hash-matching lock entry is the
  "done" marker, not a separate boolean flag that could itself be
  half-written.
- **Detecting a partially-written record**: none of these tools trust "the
  file exists" as proof of completion — asv's result files and DVC's lock
  entries both encode enough content (hash, full stats block) that a
  reader can distinguish "fully written good record" from "empty/truncated
  file" by parse failure or missing required fields, which is the
  simplest robust check available to us (JSON parse fails or a required
  field like `status`/`finished_at` is absent ⇒ treat as not-done, re-run).
- **Two-pass verification against transient noise (nanobench)**: calibrate
  once, then re-measure the same iteration count a second time before
  trusting it — not directly a crash-recovery pattern, but the same
  "don't trust a single observation" discipline applies to detecting a
  flaky/partial GPU run (e.g., a cell that "succeeded" but produced a
  wildly discontinuous latency distribution should be flagged for rerun,
  not silently appended).

---

## §5 Specific recommendations for our harness

1. **[cheap] Report median + IQR/MAD alongside mean for every latency
   distribution, and store both outlier counts (stddev-based and
   Tukey/IQR-based) per (k, batch_size, mode) point**, following
   pytest-benchmark/Criterion/nanobench. Do **not** collapse to mean-only
   in the JSONL record — this is cheap because we already plan the
   parquet of raw per-call vectors; computing median/IQR from it at
   write-time is a few lines.
2. **[cheap] Add a `schema_version` field to the JSONL record now**, before
   the first campaign, per asv/Google Benchmark/pytest-benchmark
   convention. Retrofitting this after records exist is the classic mistake;
   it costs nothing to add on day one.
3. **[cheap] Put a full provenance block on every record** (git commit +
   dirty flag + branch, GPU model, driver/CUDA/torch/triton versions,
   `clocks_locked_mhz`, config hash) rather than a separate "campaign
   metadata" file. This directly operationalizes CLAUDE.md rule 2 ("not
   yet validated" until gates pass) — a `git_dirty: true` field lets
   `bench report` refuse to cite a run automatically instead of relying on
   a human remembering.
4. **[cheap] Adopt DVC-style content hashing for resume-by-key**: hash
   `(cell identity fields, resolved config, algorithm code
   version/commit)` rather than just the cell's declared name. A naive
   "is this cell name already in the JSONL" resume check (which the brief's
   phrasing "resume-by-key" suggests we're leaning toward) will silently
   treat a cell as done even if the underlying kernel code changed since
   the last partial campaign — this is the single most important
   correction this research surfaces relative to our stated plan.
5. **[cheap] One JSONL line = one crash-recovery unit, written via
   temp-file-then-rename or single-syscall append**, and treat a JSON
   parse failure on the last line as "not done, rerun" rather than a fatal
   error for the whole file. This matches ClickBench/db-benchmark/asv's
   independently-converged per-unit-file design and our own plan; no
   change needed except making the atomicity discipline explicit in the
   implementation (verify Python's `open(..., 'a')` + `write()` for a
   single JSON line is atomic on the target filesystem, or use
   write-temp-then-`os.replace`).
6. **[medium] For GPU timing, follow triton.testing.do_bench's structure
   for any custom kernel-level micro-timing we add**: estimate cost with a
   handful of untimed reps, derive `n_warmup`/`n_repeat` from a target
   warmup/measurement time budget (not fixed iteration counts, since our
   5 algorithms will have wildly different per-call costs), flush the L2
   cache before each timed rep with a same-or-larger-than-L2 zeroed
   buffer, and time with `torch.cuda.Event(enable_timing=True)` pairs plus
   `torch.cuda.synchronize()`, never wall-clock `time.time()`, for anything
   inside a single algorithm call.
   [triton/testing.py](https://github.com/openai/triton/blob/main/python/triton/testing.py)
   Do **not** L2-flush for the end-to-end/latency-at-batch-size cells that
   are meant to reflect realistic serving conditions (see point 8) — L2
   flushing is for isolated kernel comparison, not for a metric meant to
   represent real query latency.
7. **[medium] Adopt MLPerf's open-loop vs closed-loop distinction for our
   latency sweep, if we don't already have it**: our brief lists "latency
   at several (k, batch_size, mode) points" — check that at least one
   `mode` is **open-loop** (queries arrive on a schedule independent of
   whether the previous one finished, MLPerf's Server scenario) rather
   than only closed-loop (issue-next-after-previous-completes, MLPerf's
   SingleStream/MultiStream). Closed-loop-only latency numbers
   systematically understate tail latency under real concurrent load,
   which is exactly the gap MLPerf's scenario split exists to close.
   [inference_rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
   This is `[medium]` because it may require a small addition to the
   latency-measurement code (a request-issue scheduler), not just a
   reporting change.
8. **[cheap] Report p90/p99 (not just mean/median) for every latency
   point, explicitly**, mirroring MLPerf's percentile-target framing
   (90th for SingleStream, 99th for MultiStream) — cheap because it's the
   same quantile-computation code already needed for point 1.
9. **[medium] Consider a `bench report --baseline <path-to-old-jsonl>`
   command modeled on Criterion's `--baseline`/`critcmp` pattern**: a
   noise-thresholded (e.g., 1–2%, tunable) per-cell comparison between two
   JSONL result sets, flagging regressions/improvements beyond the
   threshold, rather than a human eyeballing two files. This is `[medium]`
   (not `[cheap]`) because a defensible regression test needs at least a
   two-sample comparison with a stated tolerance, which is a small but
   real piece of statistics code — but it can reuse the same summary
   stats already computed per point 1, and is exactly the kind of
   standalone, composable tool `critcmp` demonstrates (a separate command
   operating on stored JSON, not new machinery baked into `bench run`).
10. **[expensive, explicitly NOT recommended] Do not adopt Hydra, W&B
    Sweeps, MLflow, Sacred, or DVC's pipeline/params model wholesale** for
    orchestrating the 700-cell grid. Every one of them either requires
    infrastructure we don't need (a server, an agent process, a config
    object model with decorators/OmegaConf) or provides grid
    orchestration whose dedup/resume story is documented as unreliable in
    practice (W&B's grid-resume bugs, §2). Our plan (flat CLI +
    subprocess isolation + resume-by-key) already matches what ClickBench
    and DuckDB's db-benchmark do at comparable or larger scale — the
    research explicitly supports keeping this minimal, and going further
    (adding any of these frameworks) would be adding abstraction the
    brief already correctly rejects.
11. **[cheap] Correction to the stated plan**: "resume-by-key" as
    currently phrased in the brief doesn't mention a code-version
    component. Recommendation 4 above is the one place this research
    says our stated design is incomplete, not merely under-specified —
    make the cell-identity hash include an algorithm/kernel code version,
    not just dataset/dim/filter/sweep/algo/backend/params/seed.

---

## §6 What I could not verify

- **Criterion.rs's exact bootstrap CI computation** (which quantiles of
  the 100,000 bootstrap resamples are reported as the CI bounds, and
  whether it's a percentile bootstrap or BCa) — the public docs pages
  fetched describe the *existence* and *count* of resamples and the
  confidence-level default (0.95) but not the exact interval-construction
  formula; would need to read the `criterion-plot`/`criterion-stats` Rust
  source directly to confirm.
- **Google Benchmark's exact JSON schema for an *aggregate* entry** (I
  confirmed the per-iteration entry shape and the existence of
  mean/median/stddev/cv aggregates, but did not get a full worked JSON
  example of an aggregate row with its distinguishing `run_type` value
  and `aggregate_name` field from the fetched page — the tools.md/user
  guide pages summarized rather than quoted the full aggregate JSON
  block).
- **W&B Sweeps' current (2026) state of the grid-resume/dedup bugs** — the
  GitHub issues found range from 2021–2024; I could not confirm whether
  these are still open/unfixed in the current wandb release, only that
  they were reported and that the pattern (duplicate/missing runs on
  grid resume) recurred across multiple years, which is itself a signal
  worth weighting even if any single issue has since been patched.
- **Optuna's trial-level deduplication behavior** for `GridSampler`
  specifically (whether re-running with `load_if_exists=True` skips
  already-evaluated grid points) — search results covered
  study-level persistence/resume and distributed optimization but not
  this specific point; would need to check `GridSampler` source or a
  more targeted doc page.
- **Conbench's and Codespeed's exact stored JSON schema** (field-by-field)
  — I confirmed their architectural posture (language-independent JSON
  POST for Conbench; normalized Django models with a mandatory
  Environment entity for Codespeed) but did not pull a full example
  payload/schema for either.
- **Kalibera & Jones (2013) methodological details** (their specific
  statistical test for steady-state/warmup-end detection) — confirmed the
  paper's existence, venue, and that it uses visualization-based
  techniques (autocorrelation/lag/run-sequence plots) rather than
  automated changepoint detection, but did not fetch the full PDF to
  extract the precise statistical procedure.
- I did not independently verify current `pip`/`cargo` version numbers for
  any of these tools (e.g., current Criterion.rs or pytest-benchmark
  release), since the design principles rather than exact current
  version numbers were the object of this research.
