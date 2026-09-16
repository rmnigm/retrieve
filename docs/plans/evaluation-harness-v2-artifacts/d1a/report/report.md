# `bench report`

- generated: `2026-09-16T09:33:34Z`  from `/tmp/claude-0/-workspace-retrieve/7c1995ca-a1a3-4b78-9907-2c0de7f4d6fc/scratchpad/snap`
- records: **72** {'ok': 72}, cells flagged unstable (window spread or clock drift): 23
- schema_version: [2]  |  code_version: ['0e6778056238de3c921cd2a14beec28b98a1d41c']
- runs (commit, branch): [('5fd05a6', 'dev/d1a-campaign')]
- gpu: ['NVIDIA A100-SXM4-80GB']  host: ['96ef99fba44c']  window: 2026-09-15T22:39:06+00:00 .. 2026-09-16T08:09:01+00:00

## Citability (CLAUDE.md rule 2)

**NOT CITABLE.** Every artifact carries the marker. Reasons:

- no --gate given: these records come from no green roadmap gate
- record(s) produced on dev/d1a-campaign — CLAUDE.md rule 2: harness numbers from a branch are not paper material

These records predate the D1 campaign; they come from C4's gate run and C5's one-cell check and are evidence about the harness, not results.

## Selection used by the tables

`--dim 128` `--k 100` `--bs 1` `--mode eager` `--backend triton` `--compare-bs 16` `--budget-ms [0.5, 1.0, 2.0, 5.0]`; batch-scaling dataset: `goodreads` (chosen by coverage).

## Coverage

| dataset | dim | suite | filter | sweep | algo | backend | params | seeds | status |
|---|---|---|---|---|---|---|---|---|---|
| goodreads | 128 | filter | bloom | c0_genre | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c0_genre | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c0_genre | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c0_genre | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c0_genre | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c0_genre | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c0_genre | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c0_genre | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c2_format | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c2_format | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c2_format | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c2_format | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c2_format | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c2_format | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c2_format | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c2_format | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c3_year | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c3_year | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c3_year | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c3_year | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c3_year | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c3_year | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c3_year | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | bloom | c3_year | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | all4 | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | all4 | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | all4 | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | all4 | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | all4 | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | all4 | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | all4 | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | all4 | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0c1 | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0c1 | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0c1 | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0c1 | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0c1 | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0c1 | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0c1 | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0c1 | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c1_lang_reverse | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c1_lang_reverse | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c1_lang_reverse | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c1_lang_reverse | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c1_lang_reverse | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c1_lang_reverse | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c1_lang_reverse | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c1_lang_reverse | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c2_format | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c2_format | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c2_format | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c2_format | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c2_format | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c2_format | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c2_format | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c2_format | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c3_year | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c3_year | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c3_year | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c3_year | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c3_year | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c3_year | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c3_year | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c3_year | linr_v4 | triton | {} | [0] | {'ok': 1} |

## Failed cells (excluded from every number)

none

## Partial records (marked `*`)

none

## Unstable perf variants (marked `†`): 37

- goodreads linr_v1_filter_mask/triton k=100 bs=1 eager: spread 8.4%
- goodreads linr_v1_filter_mask/triton k=500 bs=1 eager: spread 21.9%
- goodreads linr_v1_filter_mask/triton k=1000 bs=8 eager: spread 19.8%
- goodreads linr_v1_filter_mask/triton k=500 bs=16 eager: spread 16.4%
- goodreads linr_v2/triton k=100 bs=1 eager: spread 9.4%
- goodreads linr_v2/triton k=1000 bs=1 eager: spread 12.8%
- goodreads linr_v2/triton k=500 bs=1 eager: spread 17.3%
- goodreads linr_v2/triton k=1000 bs=1 eager: spread 8.0%
- goodreads linr_v2/triton k=1000 bs=1 eager: spread 9.4%
- goodreads linr_v2/triton k=1000 bs=1 eager: spread 8.7%
- goodreads linr_v2/triton k=100 bs=1 eager: spread 10.4%
- goodreads linr_v2/triton k=1000 bs=1 eager: spread 10.1%
- goodreads linr_v2/triton k=100 bs=1 eager: spread 5.6%
- goodreads linr_v3/triton k=500 bs=1 eager: spread 19.4%
- goodreads linr_v3/triton k=1000 bs=1 eager: spread 17.1%
- goodreads linr_v3/triton k=500 bs=1 eager: spread 27.7%
- goodreads linr_v3/triton k=100 bs=1 eager: spread 6.9%
- goodreads linr_v3/triton k=100 bs=1 eager: spread 14.6%
- goodreads linr_v3/triton k=500 bs=1 eager: spread 6.7%
- goodreads linr_v3/triton k=100 bs=1 eager: spread 8.1%
- goodreads linr_v3/triton k=500 bs=1 eager: spread 15.0%
- goodreads linr_v3/triton k=1000 bs=1 eager: spread 11.7%
- goodreads linr_v3/triton k=100 bs=8 eager: spread 9.1%
- goodreads linr_v3/triton k=500 bs=8 eager: spread 5.1%
- goodreads linr_v3/torch k=100 bs=1 eager: spread 9.9%
- goodreads linr_v3/torch k=1000 bs=1 eager: spread 5.4%
- goodreads linr_v3/torch k=100 bs=1 eager: spread 9.9%
- goodreads linr_v3/torch k=500 bs=1 eager: spread 21.2%
- goodreads linr_v3/torch k=1000 bs=1 eager: spread 8.1%
- goodreads linr_v4/triton k=1000 bs=1 eager: spread 20.4%
- goodreads linr_v4/triton k=500 bs=8 eager: spread 8.9%
- goodreads linr_v4/triton k=500 bs=1 eager: spread 12.1%
- goodreads linr_v4/triton k=1000 bs=1 eager: spread 8.3%
- goodreads linr_v4/triton k=500 bs=8 eager: spread 9.2%
- goodreads linr_v4/triton k=100 bs=1 eager: spread 11.9%
- goodreads linr_v4/triton k=500 bs=1 eager: spread 18.6%
- goodreads linr_v4/triton k=1000 bs=1 eager: spread 11.1%

## Artifacts

- `figures/fig-batch-scaling.png`
- `figures/fig-deep-sweep.png`
- `figures/fig-latency-violin.png`
- `figures/fig-pareto-goodreads.png`
- `figures/fig-qps-recall.png`
- `flat.csv`
- `methodology.tex`
- `tables/tab-backend_parity.tex`
- `tables/tab-batch_scaling.tex`
- `tables/tab-memory.tex`
- `tables/tab-paper_comparison.tex`
- `tables/tab-pareto_goodreads.tex`
- `tables/tab-recall_at_budget.tex`
- `tables/tab-recall_nofilter.tex`

## Clock estimator

Latency artifacts use the per-variant under-load `perf[].sm_mhz`. Idle samples (`env.sm_mhz_idle`, and the schema-1 `env.sm_mhz`, which is a whole-run median dominated by idle) are provenance only and are never compared with an under-load sample — the error that made 92 of 99 of C4's latency rows appear to fail.

