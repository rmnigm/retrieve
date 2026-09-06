# Artifacts — evaluation harness v2

Raw material behind [../evaluation-harness-v2.md](../evaluation-harness-v2.md).

| file | what it is |
|---|---|
| [survey-ann-ir-benchmarks.md](survey-ann-ir-benchmarks.md) | How ann-benchmarks, big-ann-benchmarks (incl. the NeurIPS'23 filtered track), VectorDBBench, MTEB / BEIR / `ir_measures`, cuVS `raft-ann-bench` and FAISS's `benchs/` store results, declare sweeps, isolate runs, cache ground truth and time queries. §3 lists the ten conventions that recur across ≥3 of them; §4 grades each against our plan; §5 lists what the survey could not verify from primary sources. |
| [survey-bench-infrastructure.md](survey-bench-infrastructure.md) | The same question for general benchmarking infrastructure: Criterion.rs, Google Benchmark, nanobench, pytest-benchmark, airspeed velocity, MLPerf Inference's LoadGen scenarios, `torch.utils.benchmark`, Hydra / submitit / W&B Sweeps / MLflow / Sacred / DVC, and the results-as-data conventions of ClickBench, db-benchmark, Conbench and Codespeed. |

Both were produced by literature/repo search, not by running anything;
claims carry URLs and each report ends with an explicit "could not
verify" section. The findings that were **accepted into the plan** are in
[../evaluation-harness-v2.md](../evaluation-harness-v2.md) §8 — read that
first; these files are the evidence, not the instruction.
