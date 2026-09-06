# Golden baseline — the old harness, frozen

These JSONs are the **reference quality numbers of the pre-v2 evaluation
harness**, produced by roadmap step **A1**
([../../docs/plans/00-roadmap.md](../../docs/plans/00-roadmap.md)), which
executes WP-0 of
[../../docs/plans/evaluation-harness-v2.md](../../docs/plans/evaluation-harness-v2.md)
§6 and closes steps 4–7 of
[../../docs/plans/refactor-validation-handoff.md](../../docs/plans/refactor-validation-handoff.md).

The harness is being rewritten (roadmap Phase C). The rewrite changes the
measurement protocol *and* the output schema, so there is exactly one
thing that must not change: the quality columns. These files are that
check. C4's gate is `recall@k` / `ndcg@k` equal to the numbers here
within `1e-6` for every `(algo, backend, k)`, with tie order under the
`k_max` slice the only permitted difference. Nothing else about this
directory is load-bearing — the latency columns are recorded for context
(they are cold-L2 `do_bench` medians of CUDA-graph replay, the thing v2
replaces; see H §1 verdicts 1–2), not as a target.

## The cells

One process per `(dataset, algo, backend)` — the process boundary H §8.2 K
settles on, so no dynamo cache or CUDA-graph pool outlives the backend
under test. Each process writes 9 rows: `ks = [100, 500, 1000]` ×
`batch_sizes = [1, 8, 16]`.

| file | dataset | filter | algo | backend |
|---|---|---|---|---|
| `goodreads-d128-c0_genre-linr_v1_filter_mask-{triton,torch}.json` | goodreads-work-id, d128 | clause / `c0_genre` | linr_v1_filter_mask | triton, torch |
| `goodreads-d128-c0_genre-linr_v2-{triton,torch}.json` | " | " | linr_v2 | triton, torch |
| `goodreads-d128-c0_genre-linr_v3-{triton,torch}.json` | " | " | linr_v3 | triton, torch |
| `goodreads-d128-c0_genre-linr_v4-{triton,torch}.json` | " | " | linr_v4 | triton, torch |
| `goodreads-d128-c0_genre-silvertorch-{triton,torch}.json` | " | " | silvertorch | triton, torch |
| `arxiv-d128-c0_maincat-silvertorch-triton.json` | arxiv-papers, d128 | clause / `c0_maincat` | silvertorch | triton |

11 files. `_logs/` holds one stdout/stderr log per cell plus
`provenance.txt` (commit, host, UTC, `nvidia-smi`, torch/triton
versions), kept so a number can be traced back to the run that produced
it.

`_main/` holds the same 11 cells run from a throwaway worktree off
`main` — the other half of handoff step 6, below. It is **gitignored**:
the diff report is the artifact worth keeping, and the JSONs are
reproducible from that worktree.

## A1 is more than the golden cells

Roadmap A1 also closes steps 4-7 of
[../../docs/plans/refactor-validation-handoff.md](../../docs/plans/refactor-validation-handoff.md),
whose status blockquote records steps 1 and 4-7 as never having run (no
dataset was on the box). The runbook does all of it in five stages:

| stage | what | artifacts |
|---|---|---|
| `golden` | the 11 cells above | `evaluation/golden/*.json` |
| `step4` | `TORCH_LOGS=graph_breaks` compile smoke on `linr_v3 --skip-quality`; break reasons diffed against `main`, since "no *new* breaks" is only decidable against it | `.../a1/step4/` |
| `step7` | `run-evaluation --resume` orchestrator smoke: killed as soon as the first algo finishes, resumed, must skip what completed | `.../a1/step7/` |
| `step6` | the same 11 cells from the `main` worktree + the quality diff | `_main/`, `.../a1/step6/report.md` |
| `step5` | per-kernel `tune-kernels` +-5 % gates on both sides, behind a wall-time estimate | `.../a1/step5/` |

(`.../a1/` is
[../../docs/plans/evaluation-harness-v2-artifacts/a1/](../../docs/plans/evaluation-harness-v2-artifacts/README.md).)

**Why `main` needed patching.** Step 6 diffs `main` against the refactor
track, and the `users_limit` bug below made `main` unable to run a
goodreads filter cell at all. The handoff's caveat offers two ways out:
either the error fires identically on both sides, or both sides run with
`users_limit: null`. The second is unaffordable — a 313k-query exact
filtered oracle — so the same three-line fix is ported onto a throwaway
branch off `main` (`tmp/main-users-limit-fix`, never merged) and the two
sides stay comparable at 10k queries. Both `main` and this branch keep
their own oracle cache file (`gt_topk_v2_*` vs `gt_topk_v3_*`), so
neither run can read the other's ground truth; they do share the
`encoded_queries_test.pt` cache, whose blob format is identical, so both
sides see the same queries.

## Only `triton` and `torch` are golden — cuda/cute are void

H §6 WP-0's own text asks for arxiv `silvertorch` on `--backend cuda
cute`. **That is void**, per H's amendment of 2026-09-05 and roadmap
Phase B: Meta's official kernels become the reference backend
(`backend="official"`) and the CUDA C++ and CuTe DSL backends are deleted
once the official parity gate (B2) is green. A golden column for a
backend that will not exist would be checked against nothing, so it was
not produced. The backend axis of the v2 harness is
`triton | torch | official`; `official` gets its own gate in C4 (jaccard
≥ 0.99 vs `triton` on one goodreads cell), not a golden comparison here.

## Exact commands

Run from the repository root on the A100 box, with `evaluation/data`
pointing at the dataset root and clocks locked
(`sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1410`). The script that
does all of it, including the clock lock/unlock, the per-cell logs and a
skip-if-present resume:

```bash
bash docs/plans/evaluation-harness-v2-artifacts/a1_golden_run.sh
```

Stages are selectable (`STAGES="golden step4 step7 step6 step5"`), the
`main` worktree is `MAIN_WORKTREE=/workspace/wt/main-golden`, and step 5
refuses to run if its estimated wall time exceeds `STEP5_BUDGET_S`
(default 2 h for both sides).

The golden stage runs, from `evaluation/`, one invocation per cell:

```bash
# goodreads: 5 algos x {triton, torch}
uv run evaluate --config config/goodreads/d128-filter.yaml \
  --algo <linr_v1_filter_mask|linr_v2|linr_v3|linr_v4|silvertorch> \
  --backend <triton|torch> --filter-kind clause --sweep c0_genre \
  --output golden/goodreads-d128-c0_genre-<algo>-<backend>.json

# arxiv: silvertorch, triton only
uv run evaluate --config config/arxiv/d128-filter.yaml \
  --algo silvertorch --backend triton \
  --filter-kind clause --sweep c0_maincat \
  --output golden/arxiv-d128-c0_maincat-silvertorch-triton.json
```

The working directory matters: both configs' `data_dir` and filter
`attrs_path` are cwd-relative (`retrieval.loaders.resolve_path`).

## Provenance

| what | value |
|---|---|
| harness code state | `0129e25` — `fix(A1): users_limit row-count in load_query_attrs`, on `dev/a1-golden` off `development` |
| box | A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, Python 3.11 |
| clocks | SM locked to 1410 MHz for every cell |
| exact HEAD, host and UTC of the run | `_logs/provenance.txt` |
| run record | appended to [evaluation-harness-v2.md](../../docs/plans/evaluation-harness-v2.md) and to the harness half of [refactor-validation-handoff.md](../../docs/plans/refactor-validation-handoff.md) |

`0129e25` carries the one behavioural change these numbers depend on:
before it, `load_query_attrs` required `eval_split.height == n_queries`
while `queries_cache` had already trimmed the queries to
`users_limit: 10000` against a 313,178-row `eval_split.parquet`, so every
goodreads filter cell raised before running. `users_limit` is a prefix,
so the attrs are now trimmed to the same prefix — quality stays
comparable with anything the old harness ever produced on the arxiv path,
where the count matched and the trim is an identity.

## What these files are not

- **Not a latency reference.** See above; use them for quality only.
- **Not citable on their own** (CLAUDE.md rule 2): they are the input to
  C4's gate, and the paper's numbers come from the D1 campaign on the v2
  harness.
- **Not the old campaign outputs.** Those live in `../results/` and are
  pre-oracle-fix; v2 moves them to `results/archive/`.
