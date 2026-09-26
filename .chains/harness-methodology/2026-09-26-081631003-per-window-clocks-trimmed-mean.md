---
chain: "harness-methodology"
branch: "main"
nextStep: "Orchestrator: review commit 8fbf330 on dev/harness-methodology (worktree /scratch/wt/harness-method), merge into staging and push. Before D1, one short GPU run of `bench run` on one cell, when the GPU is free, to confirm window_sm_mhz is populated and the nvidia-smi call between windows does not visibly change window 2/3 medians."
created: "2026-09-26T08:16:31Z"
---

# Harness methodology: per-window clocks, trimmed mean, audits

## Request
A small CPU-only worker step before roadmap D1. It hardens `evaluation/bench/measure.py`'s
measurement methodology with 5 items from an external survey of benchmarking practice
(triton do_bench, FlashInfer, cuVS). The step does not change what `unstable` means.

## Work completed (commit 8fbf330, local only, not pushed)
- **Item 1, clock samples per window.** `measure.latency` samples `clocks()["sm_mhz"]` right after
  each window's sync and stores them as `window_sm_mhz` (list, in the same order as
  `window_medians_ms`). `sm_mhz = window_sm_mhz[-1]` is the same sample as before. `unstable` and
  `clocks_drift` do not read the new list. It is in `run.PERF_STAT_KEYS` (null entries) and in
  `records._PERF_SKIP` (kept out of flat.csv, like `window_medians_ms`).
- **Item 2, trimmed mean.** `stats` gains `trimmed_mean_ms`: the sorted window with `n // 10`
  calls dropped from each end. With n < 10 it equals the plain mean. `test_stats_matches_numpy_reference` checks it
  against `np.sort(x)[50:450].mean()`.
- No schema bump: both fields are additive, and readers use `.get`. flat.csv columns are
  dynamic, so `perf_trimmed_mean_ms` appears there automatically.
- Docs: `docs/system/evaluation.md` changed in §5 (stat list, clock sample) and in "Reading the
  numbers" (clock bullet rewritten, plus new bullets: no L2 flush, one sync per window, no
  sub-launch-floor flag, trimmed_mean_ms). The perf-entry table was updated. In
  `docs/decisions.md` Harness, the clocks bullet now says "after every timing window".

## Audited, already correct, not changed
- **Item 4, sync in the loop.** `evaluation/bench/measure.py:207-228` `_time_calls` calls
  `torch.cuda.synchronize()` at :220 (before the window) and :226 (after it). The loop at
  :222-225 only records events and calls `fn()`, with no per-call sync. The per-call value is
  `s.elapsed_time(e)`, read after the final sync.
- **Item 3, sub-10 µs.** The smallest `min_ms` across all 2106 perf entries in
  `evaluation/results/` (archive included) is 0.196 ms (silvertorch triton graph, bs=1,
  goodreads-d128), 20× above the floor. So the harness gets no flag.
  The kernel microbenchmarks (`docs/artifacts/q3/roofline.py` launch timing,
  `kernel-opt/bench_kernels.py`) *can* be sub-10 µs per call. They time one event pair per
  window over many iterations, which amortizes timer overhead. They are frozen artifacts,
  so they were not touched.
- **Item 5, L2.** `measure.latency` does not flush L2. It relies on the pool rotation (this
  was already in the module docstring and §5). Now it is also stated in "Reading the numbers".
  roofline.py and bench_kernels.py do not flush either. Both already sample clocks once per window.

## Unverified / risks
- No GPU run was done (CPU-only worker, GPU in use by others). One risk: the `nvidia-smi`
  subprocess between windows (~tens of ms of GPU idle) could let the clock sag before
  windows 2 and 3. It is expected to be negligible against about 2 s windows, but that is not
  measured. Before this change the same call happened only after window 3.
- `ruff format --check evaluation` reports 15 pre-existing files (report.py, upload.py,
  etl/*, hub.py, train.py, some tests) that would be reformatted. None of them are in this diff,
  and the 5 changed files are formatted. Harness pytest: 238 passed, 1 skipped. Link checker: 0.

## Deliberately left as future ideas
- Adaptive iteration count (torch.utils.benchmark-style auto-calibration).
- Using `window_sm_mhz` in `unstable`/`clocks_drift`, or showing it in report.py. Either
  changes a flag's meaning, so it needs a user decision.
- An opt-in cold-L2 (flushed) variant for bandwidth-bound cells.
- Recall-bucketed / Pareto reporting in report.py (a separate step).
