# Artifacts — evaluation harness v2

Raw material behind [../evaluation-harness-v2.md](../evaluation-harness-v2.md).

| file | what it is |
|---|---|
| [survey-ann-ir-benchmarks.md](survey-ann-ir-benchmarks.md) | How ann-benchmarks, big-ann-benchmarks (incl. the NeurIPS'23 filtered track), VectorDBBench, MTEB / BEIR / `ir_measures`, cuVS `raft-ann-bench` and FAISS's `benchs/` store results, declare sweeps, isolate runs, cache ground truth and time queries. §3 lists the ten conventions that recur across ≥3 of them; §4 grades each against our plan; §5 lists what the survey could not verify from primary sources. |
| [a1_golden_run.sh](a1_golden_run.sh) | The A1 GPU lane, five stages: the WP-0 golden cells (11 processes, one per `(dataset, algo, backend)` — goodreads d128 clause/`c0_genre` x 5 algos x {triton, torch}, arxiv d128 clause/`c0_maincat` x silvertorch x {triton}), then handoff steps 4, 7, 6 and 5. Locks the SM clock, logs per cell, resumes by skipping cells whose JSON exists, and refuses step 5 above a wall-time budget. Cell outputs and their README live in [../../../evaluation/golden/](../../../evaluation/golden/README.md); everything else lands in `a1/`. |
| [a1_step6_diff.py](a1_step6_diff.py) | Handoff step 6's pass criteria as code: joins the `main`-side and branch-side cells on `(filter_kind, sweep, impl, backend, batch_size, k)` — never the colliding `cell` string — and requires the quality columns byte-identical, the new columns additive, and the `cell` key sets equal. Latency deltas are printed, not gated. |
| [c4_gate.py](c4_gate.py) | Roadmap C4 / §6 WP-4's gate as code: one or more harness-v2 JSONL files against `evaluation/golden/*.json`. Per cell: (1) `quality.oracle` recall / ndcg within 1e-6 of golden at every `k` (the v2 cell at the algo defaults stands for the golden one); (2) `graph` `median_ms` within 5 % at the same `(k, bs)`, both SM clocks printed — the golden's 1140 MHz from `_logs/clocks.csv`, the run's per-variant `perf[].sm_mhz` — and clock-normalised, labelled, when they differ by > 2 % (§7's fallback); (3) every graph entry measured (a `cudagraph_skips` null entry fails; `not_capturable` only on `official`); (4) exact algos' `jaccard_vs_first@100 == 1.0`, `official` ≥ 0.99; (5) no `unstable` record; plus coverage, `status: ok`, and the official cell's `cache_plans: false`. Exit 1 on any FAIL. Tested by `evaluation/retrieval/tests/test_c4_gate.py` on synthetic records against the real golden files. Gate (6), kill-and-resume, is manual. |
| [a1_step5_compare.py](a1_step5_compare.py) | Handoff step 5's +-5 % per-kernel gate: reads both sides' `tune-kernels --json-out` blobs and joins them by shape position, since `main` writes `per_bucket` keyed by an int P for one kernel where the branch writes `per_regime` keyed by a label string. |
| [survey-bench-infrastructure.md](survey-bench-infrastructure.md) | The same question for general benchmarking infrastructure: Criterion.rs, Google Benchmark, nanobench, pytest-benchmark, airspeed velocity, MLPerf Inference's LoadGen scenarios, `torch.utils.benchmark`, Hydra / submitit / W&B Sweeps / MLflow / Sacred / DVC, and the results-as-data conventions of ClickBench, db-benchmark, Conbench and Codespeed. |

Both were produced by literature/repo search, not by running anything;
claims carry URLs and each report ends with an explicit "could not
verify" section. The findings that were **accepted into the plan** are in
[../evaluation-harness-v2.md](../evaluation-harness-v2.md) §8 — read that
first; these files are the evidence, not the instruction.
