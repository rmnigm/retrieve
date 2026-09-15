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

| stage | what | result 2026-09-06 | artifacts |
|---|---|---|---|
| `golden` | the 11 cells above | **11/11 green**, 99 rows, 35.9 min | `evaluation/golden/*.json` |
| `step4` | `TORCH_LOGS=graph_breaks` compile smoke on `linr_v3 --skip-quality`; break reasons diffed against `main`, since "no *new* breaks" is only decidable against it | **pass** — zero breaks on the branch, so the comparison is vacuous; the `main` leg did not run | `.../a1/step4/` |
| `step7` | `run-evaluation --resume` orchestrator smoke: killed as soon as the first algo finishes, resumed, must skip what completed | **pass** — exit 0, 1 resume skip, 5/5 JSONs | `.../a1/step7/` |
| `step6` | the same 11 cells from the `main` worktree + the quality diff | **deferred** (user: heavy evals later) | `_main/`, `.../a1/step6/report.md` |
| `step5` | per-kernel `tune-kernels` +-5 % gates on both sides, behind a wall-time estimate | **deferred** (same); estimate 332 points/side, ~44 min both | `.../a1/step5/` |

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

## Only `triton` and `torch` are golden — the deleted backends are void

H §6 WP-0's own text asks for arxiv `silvertorch` on the two hand-written
SilverTorch backends of the time. **That is void**, per H's amendment of
2026-09-05 and roadmap Phase B: Meta's official kernels became the
reference backend (`backend="official"`) and those two backends were
deleted at B4 once the official parity gate (B2) was green. A golden
column for a backend that no longer exists would be checked against
nothing, so it was not produced. The backend axis of the v2 harness is
`triton | torch | official`; `official` gets its own gate in C4 (jaccard
≥ 0.99 vs `triton` on one goodreads cell), not a golden comparison here.

## Exact commands

Run from the repository root on the A100 box, with `evaluation/data`
pointing at the dataset root. The plan asks for locked clocks
(`sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 1410`); **this container
cannot lock them** and neither the A1 run nor the re-derive did — H §7's
sampled-`sm_mhz` fallback is used instead. The script that does all of it,
including the per-cell logs, the clock sampler and a skip-if-present resume:

```bash
bash docs/plans/evaluation-harness-v2-artifacts/a1_golden_run.sh
# the 2026-09-15 re-derive ran this copy instead (GPU lock, /venvs/golden,
# private inductor cache, golden stage only):
STAGES=golden bash docs/plans/evaluation-harness-v2-artifacts/a1-rederive/a1_golden_rerun.sh
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

**These cells were re-derived on 2026-09-15** (roadmap eval-queue item 1).
The originals are A1's, produced 2026-09-06; they are still in git history at
`development`'s parent of the re-derive commit. Why they had to be redone, and
what moved, is below the table.

| what | value |
|---|---|
| cells produced at | `87a9b38` on the throwaway branch `tmp/golden-rederive` = `origin/dev/a1-golden` (`1ccdb27`, the old harness) with the **current library** swapped in (`git checkout 4f52972 -- retrieve/`; library tree `28fda5ae`). Every row carries `87a9b38` in `extra.commit`. The branch is never merged; it exists so the run can be repeated |
| library under test | `4f52972:retrieve` — deterministic k-means, `clause_compact` / `bloom_compact` as opaque custom ops, the O §14.7 `-1` id sentinel (the three C4 library fixes merged at `7a21095`) |
| harness | unchanged from A1: `origin/dev/a1-golden`. One porting change, typing only: `retrieve.interfaces.Backend` is gone from the current library (split into `LinrBackend` / `SilverTorchBackend` at B4), so the harness declares the old literal locally — copy at [`../../docs/plans/evaluation-harness-v2-artifacts/a1-rederive/harness_compat_backend.py`](../../docs/plans/evaluation-harness-v2-artifacts/a1-rederive/harness_compat_backend.py) |
| box | A100-SXM4-80GB, driver 580.159.04, nvcc 12.4, torch 2.10.0+cu128, triton 3.6.0, Python 3.11 |
| inputs | identical to A1: the same `encoded_queries_test.pt` blob (cache hit — the checkpoint's mtime was set to the blob's recorded `ckpt_mtime` so no re-encode could perturb the queries) and the same cached oracles `gt_d128/gt_topk_v3_{c0_genre,c0_maincat}.pt` (fingerprint hit). goodreads keeps 9,859 / 10,000 users, arxiv 10,000 / 10,000 — both as in A1 |
| clocks | **still not locked — this container cannot** (`nvidia-smi -lgc` denied, no `sudo`). H §7's fallback: sampled every 30 s into `_logs/clocks.csv`. **1155 MHz** median under load (A1: 1140), 1410 MHz peak, 210 MHz idle, 26-39 °C. `c4_gate.py --golden-sm-mhz` must be given **1155**, not its 1140 default. Quality is unaffected; **the latency columns are not clock-controlled**, and this run also shared the GPU with a second worker through `flock /workspace/gpu.lock` (serialised, but the thermal state between cells was not controlled) |
| wall time | 29.4 min of cell time for the 11 cells, 2.4-3.1 min each, 33 min end to end including lock waits |
| runbook | [`a1-rederive/a1_golden_rerun.sh`](../../docs/plans/evaluation-harness-v2-artifacts/a1-rederive/a1_golden_rerun.sh), `STAGES=golden` — a copy of [`a1_golden_run.sh`](../../docs/plans/evaluation-harness-v2-artifacts/a1_golden_run.sh) with the GPU lock, `uv run --no-sync` against `/venvs/golden`, a private inductor cache, and stages 4-7 dropped |
| exact HEAD, host and UTC of the run | `_logs/provenance.txt` |
| run record | [evaluation-harness-v2.md §11](../../docs/plans/evaluation-harness-v2.md) (A1's own record is §10) |
| quality diff vs A1 | [`a1-rederive/quality-diff.md`](../../docs/plans/evaluation-harness-v2-artifacts/a1-rederive/quality-diff.md) |

### Why they were re-derived

A1's cells came from the library *before* the three C4 fixes, so comparing
C4's harness output against them at `1e-6` would have measured the library
change, not the harness rewrite. The old harness was therefore run unchanged
against the new library; the harness, the data, the queries and the oracle are
all held fixed, so every delta is a library delta.

### What moved

**6 of 11 cells are bit-identical**: both `linr_v1_filter_mask` cells, both
`linr_v4` cells, `linr_v2-torch`, `linr_v3-torch`. **5 moved**, all by
≤ 1.6e-4 — above the `1e-6` gate tolerance, which is the whole reason this
re-derive had to happen:

| cell | max abs delta | direction |
|---|---|---|
| `goodreads-…-silvertorch-torch` | 1.57e-4 | up |
| `goodreads-…-silvertorch-triton` | 7.79e-5 | mixed |
| `arxiv-…-silvertorch-triton` | 3.50e-5 | mixed |
| `goodreads-…-linr_v3-triton` | 2.43e-5 | up |
| `goodreads-…-linr_v2-triton` | 5.07e-6 | mixed |

The headline: SilverTorch's `torch` and `triton` backends **now agree exactly**
on goodreads — all 9 rows, `recall` and `ndcg` delta `0.0`, where A1 had a
1.2-1.7e-4 gap it attributed to tie order. Deterministic k-means gives both
backends the same index.

**The roadmap's prediction for this run does not hold** and should not be
repeated: it expected SilverTorch-*triton* to move *down* on rows with fewer
than `k` survivors, with `torch` unchanged. Among the 9,859 **kept** goodreads
users every query has ≥ 1000 survivors — the 141 dropped users are exactly the
zero-survivor ones — so the `-1` sentinel cannot move a goodreads number at
all; and `torch` moved because deterministic k-means is backend-independent.
The analysis is in §11 of the run record.

## Three bugs the run found

WP-0 was budgeted as "commit a 3-line fix, run 11 cells"; it took three
attempts, and every failure was a real defect that only a golden run could
surface. Full account in the §9 record — in short: the `users_limit` row
count (`0129e25`); the fetched datasets being the pre-`3b1b5b3` 1-indexed
`[N+1, …]` artifacts, which crashed goodreads and *silently* misaligned
arxiv by one row (`cos(query, target)` 0.99 → 0.62) (`df6db40`); and K3's
`common.clause_pass` being a `NameError` under inductor, which failed every
compiled filter algo and blocked all eleven cells (`70bafc4`).

`0129e25` carries the first of the three behavioural changes these numbers
depend on:
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
