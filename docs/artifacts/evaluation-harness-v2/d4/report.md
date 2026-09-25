# `bench report`

- generated: `2026-09-15T20:57:27Z`  from `/tmp/claude-0/-workspace-retrieve/7c1995ca-a1a3-4b78-9907-2c0de7f4d6fc/scratchpad/all`
- records: **24** {'ok': 24}, cells flagged unstable (window spread or clock drift): 21
- schema_version: [1, 2]  |  code_version: ['0fe440d4bc9013097737e225af59b71ccec94a6d']
- runs (commit, branch): [('71430d7', 'dev/c5-harness-split'), ('afc2ab9', 'dev/c4-gate-rerun')]
- gpu: ['NVIDIA A100-SXM4-80GB']  host: ['96ef99fba44c']  window: 2026-09-15T08:50:50+00:00 .. 2026-09-15T13:54:29+00:00

## Citability (CLAUDE.md rule 2)

**NOT CITABLE.** Every artifact carries the marker. Reasons:

- no --gate given: these records come from no green roadmap gate
- record(s) produced on dev/c4-gate-rerun, dev/c5-harness-split — CLAUDE.md rule 2: harness numbers from a branch are not paper material

These records predate the D1 campaign; they come from C4's gate run and C5's one-cell check and are evidence about the harness, not results.

## Selection used by the tables

`--dim 128` `--k 100` `--bs 1` `--mode eager` `--backend triton` `--compare-bs 16` `--budget-ms [0.5, 1.0, 2.0, 5.0]`; batch-scaling dataset: `goodreads` (chosen by coverage).

## Coverage

| dataset | dim | suite | filter | sweep | algo | backend | params | seeds | status |
|---|---|---|---|---|---|---|---|---|---|
| arxiv | 128 | filter | clause | c0_maincat | silvertorch | official | {"n_probe": 24} | [0] | {'ok': 1} |
| arxiv | 128 | filter | clause | c0_maincat | silvertorch | official | {"n_probe": 32} | [0] | {'ok': 1} |
| arxiv | 128 | filter | clause | c0_maincat | silvertorch | torch | {"n_probe": 24} | [0] | {'ok': 1} |
| arxiv | 128 | filter | clause | c0_maincat | silvertorch | torch | {"n_probe": 32} | [0] | {'ok': 1} |
| arxiv | 128 | filter | clause | c0_maincat | silvertorch | triton | {"n_probe": 24} | [0] | {'ok': 1} |
| arxiv | 128 | filter | clause | c0_maincat | silvertorch | triton | {"n_probe": 32} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v1_filter_mask | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v1_filter_mask | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v2 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v2 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v3 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v3 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v4 | torch | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | linr_v4 | triton | {} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | silvertorch | official | {"n_probe": 24} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | silvertorch | official | {"n_probe": 32} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | silvertorch | torch | {"n_probe": 24} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | silvertorch | torch | {"n_probe": 32} | [0] | {'ok': 1} |
| goodreads | 128 | filter | clause | c0_genre | silvertorch | triton | {"n_probe": 24} | [0, 1, 2] | {'ok': 3} |
| goodreads | 128 | filter | clause | c0_genre | silvertorch | triton | {"n_probe": 32} | [0, 1, 2] | {'ok': 3} |

## Failed cells (excluded from every number)

none

## Partial records (marked `*`)

none

## Unstable perf variants (marked `†`): 29

- arxiv silvertorch/triton k=500 bs=1 graph: spread 14.9%
- arxiv silvertorch/triton k=1000 bs=1 eager: spread 6.9%
- arxiv silvertorch/triton k=1000 bs=1 graph: spread 14.3%
- arxiv silvertorch/triton k=1000 bs=8 eager: spread 5.4%
- arxiv silvertorch/triton k=100 bs=16 eager: spread 6.8%
- arxiv silvertorch/triton k=1000 bs=1 eager: spread 5.3%
- arxiv silvertorch/torch k=500 bs=1 eager: spread 5.2%
- arxiv silvertorch/official k=500 bs=1 eager: spread 6.0%
- goodreads linr_v3/triton k=100 bs=1 eager: spread 22.8%
- goodreads linr_v3/triton k=500 bs=1 eager: spread 5.1%
- goodreads silvertorch/triton k=100 bs=1 graph: spread 14.8%
- goodreads silvertorch/triton k=1000 bs=8 eager: spread 26.6%
- goodreads silvertorch/official k=500 bs=1 eager: spread 7.0%
- goodreads silvertorch/triton k=1000 bs=1 graph: spread 16.2%
- goodreads silvertorch/triton k=100 bs=1 graph: spread 9.1%
- goodreads silvertorch/triton k=100 bs=1 eager: spread 14.0%
- goodreads silvertorch/triton k=100 bs=1 graph: spread 22.3%
- goodreads silvertorch/triton k=500 bs=1 eager: spread 13.7%
- goodreads silvertorch/triton k=500 bs=1 graph: spread 11.4%
- goodreads silvertorch/triton k=1000 bs=1 eager: spread 13.8%
- goodreads silvertorch/triton k=1000 bs=1 graph: spread 9.6%
- goodreads silvertorch/triton k=1000 bs=8 eager: spread 14.1%
- goodreads silvertorch/triton k=100 bs=16 eager: spread 12.7%
- goodreads silvertorch/triton k=1000 bs=16 eager: spread 13.4%
- goodreads silvertorch/triton k=100 bs=1 eager: spread 13.5%
- goodreads silvertorch/triton k=100 bs=1 graph: spread 11.4%
- goodreads silvertorch/triton k=500 bs=1 eager: spread 13.8%
- goodreads silvertorch/triton k=1000 bs=1 eager: spread 6.4%
- goodreads silvertorch/triton k=1000 bs=8 eager: spread 11.8%

## Artifacts

- `figures/fig-batch-scaling.png`
- `figures/fig-deep-sweep-arxiv-c0_maincat-silvertorch-n_probe.png`
- `figures/fig-deep-sweep-goodreads-c0_genre-silvertorch-n_probe.png`
- `figures/fig-latency-violin.png`
- `figures/fig-pareto-arxiv.png`
- `figures/fig-pareto-goodreads.png`
- `figures/fig-qps-recall.png`
- `flat.csv`
- `methodology.tex`
- `tables/tab-backend_parity.tex`
- `tables/tab-batch_scaling.tex`
- `tables/tab-memory.tex`
- `tables/tab-paper_comparison.tex`
- `tables/tab-pareto_arxiv.tex`
- `tables/tab-pareto_goodreads.tex`
- `tables/tab-recall_at_budget.tex`
- `tables/tab-recall_nofilter.tex`

## Clock estimator

Latency artifacts use the per-variant under-load `perf[].sm_mhz`. Idle samples (`env.sm_mhz_idle`, and the schema-1 `env.sm_mhz`, which is a whole-run median dominated by idle) are provenance only and are never compared with an under-load sample — the error that made 92 of 99 of C4's latency rows appear to fail.

