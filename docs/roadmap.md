---
title: roadmap
created: 2026-09-26
updated: 2026-10-11
type: summary
tags: [roadmap]
sources: [evaluation/config/suites.yaml, evaluation/bench/, infra/runpod/, docs/validation.md, docs/artifacts/hub-index.md]
---

# Roadmap: the open queue

The single ordered work queue: what is left to run or implement, in
dependency order. What already holds is in [validation](validation.md),
the standing decisions in [decisions](decisions.md), work that is not
queued in the [backlog](backlog.md). An agent starting a session reads
[AGENTS.md](../AGENTS.md), then this page, then only the wiki pages its
step names. Who executes a step is
[agent orchestration](contracts/agent-orchestration.md).

**The study** ([decisions](decisions.md#goal-and-scheduling)): Meta's
official SilverTorch code, adapted by us, against LiNR V1-V3. The queue
below finishes the adapted SilverTorch backend, measures it for the paper
(LiNR's numbers until then come from the final pass, `campaign-final`,
library `e16512f5`), lets the user write the paper, removes our Triton
SilverTorch reimplementation from the code, and ends with one full rerun
of everything at the last repo version.

**Rules.** No dates: every step is done, in the order given, and nothing
is optional. Do not start a step whose dependencies are open. When a step
finishes, its gate's row in [validation](validation.md) and the affected
`docs/system` page are updated, and the orchestrator **removes** the step
from this page. Step names (META-FORK, D1-G, ...) are stable identifiers
used in records, commits and code comments; they are not renumbered.

**Estimates** are A100-SXM4-80GB GPU-hours (the reference GPU for every
citable number); a *stream* is a slice of a step that can run on its own
GPU at the same time as the others (see
[Multi-GPU execution](#multi-gpu-execution)).

## Needs the user

- **Contact the original authors** (re-plan decision 7): the LinkedIn LiNR
  team and Meta's SilverTorch team — filter-set details, the V1/V2 setup,
  the SilverTorch paper's FPR inconsistency (0.067 % vs 0.00173 %) — and
  file the published-code defects (OF-3, OF-4, OF-6, OF-9, OF-10 in
  [deviations](paper/reproduction-deviations.md)) upstream as issues. The
  ECIR call asks what contact happened; replies take weeks.
- **The release decisions** in [release §5](paper/release-and-licenses.md#5-decisions-for-the-user)
  (goodreads checkpoints, Re-LAION vectors, the PubMed slice, the public
  May result repos, `eval-results` when public, GitHub during review).
- **Citability of a narrowed run.** A run with a narrowed mode set is
  recorded `status: partial` and reported NOT CITABLE; the user decides
  whether a deliberate, recorded narrowing reads differently.
- **A4**: merging `staging` into `main` is on hold until the user decides.
- **HF: squash `pinkmeme/eval-goodreads-work-id`** to reclaim 2.3 GB
  (`HfApi().super_squash_history(repo_id=..., repo_type="dataset")`).
- **`.git` history size.** The JSON/JSONL results left `HEAD`, but their
  bytes stay in history; shrinking `.git` needs a history rewrite (`git
  filter-repo`), which forces every clone and worktree to re-sync
  ([decisions](decisions.md#harness)).

## Running GPU work on a pod

GPU work runs on RunPod pods launched with
[`infra/runpod/pod.sh`](../infra/runpod/pod.sh) (`pod.sh up --gpu a100
--gpus N`); disks and environment in [storage](system/storage.md).

### Restore on a fresh pod

1. Checkout `staging` (the image clones it into `/workspace/retrieve`);
   the venv is `/venvs/retrieve`. A worker that needs a different
   environment gets its own venv (`UV_PROJECT_ENVIRONMENT=/venvs/<name> uv
   sync --all-packages --all-groups --extra official`): a worker that
   re-syncs the shared venv to its worktree breaks every running job.
   Meta's code as published runs from an `-O3` build
   (`scripts/build_official_o3.sh`; `env.official_build` records the loaded
   extension).
2. Data under `/data` (`RETRIEVE_DATA_ROOT`), with `evaluation/data ->
   /data` (a gitignored symlink) in every checkout and worktree. `eval-data
   fetch` only what the step reads: `arxiv-papers` (the license-fixed
   version: `item_attrs_narrow.pt` min = -1, `eval_split.parquet`
   `query_attrs_narrow` min = 0), `goodreads-work-id` with its checkpoints
   ([checkpoints](system/checkpoints.md#what-the-harness-reads)). PubMed is not on the Hub: rebuild the 10 M slice with
   `eval-data pubmed` ([datasets](system/datasets.md#pubmed); ~1 h
   download-bound `convert`, 17 GB), or copy it pod to pod. YFCC-10M is public: `eval-data yfcc
   all` ([datasets](system/datasets.md#yfcc10m), ~12.5 GB with the raw).
   LAION 30 M is not on the Hub either (pod-to-pod copies). The synth attrs
   are built on the pod (*eval-data synth-filter*, CPU, deterministic from
   its seed).
3. Results: the final aggregation needs the final pass's records:
   `bench fetch --path-in-repo campaign-final/<subtree> --results <dir>` per
   [hub-index](artifacts/hub-index.md) row.
4. Every GPU job: its own `TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>`
   (`bench run` keys a default by `code_version` when none is given,
   [evaluation](system/evaluation.md#inductor-cache)), `HF_HOME=/scratch/hf`.
   Run from `evaluation/` as `/venvs/retrieve/bin/python -m bench.cli …`,
   **not** `uv run`, which masks bench's exit code. `bench campaign`'s
   per-group timeout defaults to 48 h. A long
   job runs from one sequential driver script per GPU (one process group,
   no chained waiters, and no log line containing the text a watcher
   greps for). Every job's log carries the pod's `nvidia-smi -q -d CLOCK`
   and its `sm_mhz` histogram, uploaded with the results.

### Multi-GPU execution

On a pod with several GPUs, each job owns one GPU
(`CUDA_VISIBLE_DEVICES=<i>`), its own inductor cache dir and its own
`--out` tree or `--output` file; `aggregate` reads every
`<suite>/*.jsonl`, so per-stream files in one tree merge without a step.
Pin each job's CPU threads to its GPU's NUMA-local cores (`taskset -c`,
the affinity column of `nvidia-smi topo -m`): batch-1 latency is
launch-bound, so a neighbour's host load moves it. Record in the step's
validation row that neighbour GPUs were busy. Split streams by `--algo`,
then by `--filter-kind` / `--sweep` / `--seed`; keep the `silvertorch`
backends of one cell in **one** stream, in the suite's order: they share
a `_parity/<dataset, dim, algo>` spill and the second backend's
`jaccard_vs_first` compares against the first's. `bench campaign` has no
`--algo`, so a split stream is a `bench run`. M1's result (one timed job
per GPU of a two-GPU pod) is in
[validation](validation.md#campaign-v2-phase-v-not-yet-validated).

**Timings compare only within one box**
([decisions](decisions.md#harness)): every ratio is read inside one leg on
one pod.

### Stop rules

- **Noise gate.** Every timed comparison in an exhibit has a 95 % CI on its
  ratio that excludes 1.0, or it is reported as "no difference". No more
  than 5 repeats to force significance.
- **Budget gate.** If a step's GPU-h exceed 1.25× its estimate, report it
  and update the remaining estimates from the measured rate; the queue
  continues (user, 2026-10-08: the full grid runs).
- **Surprise gate.** A result that contradicts an earlier one (e.g. the
  adapted backend slower than Meta's published code on a cell its gate
  passed) stops the queue: rerun that one cell interleaved and report
  before anything else runs.

## The queue

The order is the user's (2026-10-11): numbers for the paper first, from
measurement legs on the adapted backend; the paper written on them; the
code cleaned; and **the full rerun of everything, pinned to the last repo
version, as the very last step**.

### 1. META-FORK: review, ACCEPT, tag

Meta's official SilverTorch code, adapted (`official-fork/`,
`backend="official-fork"`, branch `dev/meta-fork`): the chain fork-coder →
GPU gate runner → fork-review, one agent at a time (chain notes under
`.chains/meta-fork/`). What the adaptation changes and its parity gate:
[Meta as published vs adapted](paper/meta-published-vs-adapted.md). The
bar is the user's: bit-exact against Meta's published code, no host syncs,
CUDA-graph capturable with the compile gate, a clean API, faster than
Meta's `-O3` build. Being at least as fast as our Triton SilverTorch is
no longer a target (Triton is out of the study, controller 2026-10-11): no
new K fixes; the cells still slower than Triton are listed with their
cause, for the record.

- [ ] **fork-coder's done note**: what is in flight finished, every gate
  result, the Triton-relative cells with cause.
- [ ] **fork-review**: the branch against the plan's definition of done
  (X bit-exact, R, zero host syncs, capture, the compile gate, one clean
  API, licence and headers, docs). Fixes go back to fork-coder, then
  review again. **ACCEPT** ends the chain.
- [ ] **Then** (controller): tag `campaign-fork`; merge `dev/meta-fork`
  into `staging` (the `official-fork/` member, its validation section,
  `kernels.md`'s adapted-backend section).

*Gate*: fork-review's ACCEPT note, every gate row green in validation's
META-FORK section.

### 2. NUMBERS: adapted SilverTorch over the study grid

Needs 1. Measurement legs at the `campaign-fork` tag, uploaded as Hub
artifacts / `numbers-*` legs, that give the paper its numbers: SilverTorch
on `backend="official-fork"` over the study grid (`filter`, `deep`,
`synth`, `codesign`, `h2h`, tune-timed, `laion30m`, the N-sweep,
`seeds-c7`, the C6 ceiling checks, `bloomwidth` for C4). LiNR's numbers
come from `campaign-final` (not rerun here), except where an item below
runs it beside SilverTorch. Meta's code as published (`official`, `-O3`)
runs only in the cells that feed T3 and the parity spill (`h2h`,
`codesign`). Configuration: int32 score path, exact recheck of bloom
survivors on (C13); fp16 only as a reported variant where cheap
([decisions](decisions.md#goal-and-scheduling)). Recall targets reported:
0.90 and 0.95, 0.99 a stretch.

The grid plan the user approved (2026-10-11; plan
`.chains/results-draft/2026-10-11-163000000-grid-plan.md`), items 1-8 and
10 (item 9, d apart from the dataset, is not scheduled):

1. **Per-filter probe curves**: every filter sweep (real and synth),
   `n_probe` doubling from 8 to the cap `n_lists` / 4, k 100 and 1000,
   bs 1 / 16, in place of the fixed probe sets; the tuned point is
   reported with the recall each filter gets
   ([decisions](decisions.md#campaign-v2-user-2026-10-08), operating point).
2. **Enough clusters at 10-30 M** (user, 2026-10-11): PubMed, YFCC and
   LAION 10 M and LAION 30 M at `n_lists` {4096, 16384, 65536}, `n_probe`
   doubling to the cap `n_lists` / 4, with the k 1000 check per `n_lists`
   (whether the adapted top-k lifts the `n_probe` ≤ 1024 limit at k 1000
   that forced 4096 lists; where it does not, k 1000 stays at the largest
   `n_lists` that runs and the paper says so). The tuned point per size is
   picked from this sweep.
3. **The adapted code on the LAION N-sweep** (laion1m / 3m / 10m / 30m,
   d256; real `c0_domain`, `tags4` and the synth rates) with one `n_lists`
   rule across N (≈ 4√N to a power of two) and item 1's curves: the
   paper's N axis.
4. **Time the released configuration**: the shipped bloom default
   (configured width C12 + exact recheck C13, decided) is what the paper's
   SilverTorch bloom numbers use; Meta's semantics stay only in the "as
   published vs adapted" table.
5. **Interleaved SilverTorch vs LiNR**: SilverTorch and LiNR (V1, V2, the
   best per cell) interleaved in one process per cell group (ABAB rounds,
   as `seeds-c1`), so every SilverTorch / LiNR ratio has a paired CI.
6. **bs 64** (and 256 where LiNR fits at 10 M) on the real filters and the
   LAION sweep.
7. **LiNR V3 as a curve**: candidate pool 0.5 / 1 / 2 / 5 / 10 / 20 % on the
   real filters and the LAION sweep, at the default bits per d: the
   approximate-vs-approximate comparison at matched recall.
8. **Correlated filters**: `arxiv-corr-synth` gains LiNR V1 / V2 and the
   adapted SilverTorch.
10. **The papers' own k**: k 2048 on a subset (LiNR reports recall@2000;
    SilverTorch top-k 1024-4096).

The figures the paper leads with also get their points here (user,
2026-10-11): F1 / F2 at 10 M and 30 M with the low and middle pass rates
(0.001, 0.003, 0.03, 0.05) on uniform synth, the probe curve at 30 M, seeds
0-2 on the cells a T2 confidence interval rests on.

- Size the legs from the final pass's per-cell wall times (the method of
  [F-REPRO sizing](artifacts/f-repro/README.md)).
- After each leg: `bench upload --verify`, a [hub-index](artifacts/hub-index.md)
  row, the validation row.

*Gate*: every planned cell `ok`; eager-vs-graph ids equal on every
adapted cell; the int32 cells' scores `torch.equal` to the published code
where both ran. *Estimate*: not sized yet.

### 3. RESULTS: the results draft on the real numbers, the abstract fixed

Needs 2. Rebuild the results draft from NUMBERS' legs and LiNR's
`campaign-final` records (`build_results.py` under the results-draft
artifacts, on `dev/results-draft`), refresh [results](paper/results.md) and the
story page, restate C1-C7's status in [claims](paper/claims.md), and fix
the abstract's sentences ([abstract](paper/abstract.md)) to what the
numbers say. CPU.

### 4. PAPER: the user writes it

Needs 3. The user writes the paper (ECIR 2027 reproducibility format, 12
pages LNCS) on RESULTS' numbers, against the
[ECIR criteria](paper/ecir-criteria.md); agents support with tables,
figures and the `docs/paper/` sections on request (F2: the
[Meta as published vs adapted](paper/meta-published-vs-adapted.md) section
from T3 and the META-FORK gates, with CIs and paired tests). The final
tables are regenerated from FULL-RERUN's parquet (step 7).

### 5. RENAME: the adapted backend becomes SilverTorch's default

Needs 2 (beside 3-4, on its own branch). `backend="official-fork"` becomes
the default `SilverTorch` backend under a name decided in this step; the
package, distribution and op-namespace names (`silvertorch_fork`,
`silvertorch-fork`, `stfork`) likewise. Records keep the names they were
written with; the report maps them. Every copy of the old names in
`AGENTS.md`, `README.md`, `docs/` and `retrieve/docs/` follows (rule 4).

*Gate*: library and harness suites green, Meta's suite on the renamed
package, the compile gate, outputs `torch.equal` before / after on the
gate grid's int32 cells (a rename changes no behaviour).

### 6. TRITON-ST-DELETE: delete the Triton SilverTorch reimplementation

Needs 5 (rule 5: the adapted backend is accepted and NUMBERS measured it).
Tag `triton-silvertorch-final` on `staging` first, then delete
`SilverTorch`'s `backend="triton"` and `backend="torch"` paths: the
`codesigned_probe_score*` and `bloom_full_mask` kernels and their
`ops/reference` twins, SilverTorch-only host helpers, their parity /
correctness / compile tests (the official parity tests re-pointed at
adapted vs published), the harness arms and suite entries, the
`tune-kernels` entries, and their docs (`kernels.md`'s SilverTorch kernels
section, `architecture.md`, `filtering.md`, `testing.md`, `evaluation.md`,
`retrieve/docs/`). What SilverTorch's other backends share stays: phase 1
(`KMeans`, `quantize_int8*`, `csr_layout`, probe selection), the clause
kernels, the bloom hash (LiNR's `BloomFilter`), `two_level_topk` (LiNR V1).
LiNR's kernels stay.

*Gate*: library and harness suites green, `interfaces.DISPATCH` without
the deleted backends, `ruff`, links 0.

### 7. FULL-RERUN: everything, pinned to the last repo version (the very last step)

Needs 4 and 6. Tag the last repo version and rerun everything at it: LiNR,
the adapted SilverTorch (under its final name) and the grid of NUMBERS,
into one `results.parquet` with its recipe (a leg list per GPU like
[campaign-final](artifacts/campaign-final/README.md)) and a manifest that
names, per (dataset, suite, algo, backend), the accepted subtree. Then
D1-G's checks (eager-vs-graph ids, a byte-identical quality subset per
leg, batch scaling, `bench report --manifest`), T1 with C1-C7, T3 as Meta
as published vs adapted (its scorer column from `kernel_scopes`: H-SCOPE's
one GPU test on a real call is left). The paper's tables are regenerated
from this parquet; a number that moves against RESULTS' is reported, not
smoothed.

*Gate*: every planned cell `ok`, D1-G's four checks, 0 manifest gaps.
*Estimate*: sized at the step's start from NUMBERS' and the final pass's
wall times.

### 8. F4: package the artifacts

Needs 7. The results public, the PubMed slice or its build recipe with
checksums, a tagged `torchretrieve` release (LiNR plus the adapted
SilverTorch; the adapted code keeps Meta's Apache-2.0 licence and
headers), a Zenodo DOI including the adapted source and Meta's pinned
sdist, a one-command reproduction, an anonymised mirror
(anonymous.4open.science) for the double-blind review. Needs the user's
accounts and the release decisions.

## Dependencies

```
1 META-FORK ─> tag campaign-fork, merge ─> 2 NUMBERS ─> 3 RESULTS ─> 4 PAPER (user) ─┐
                                              └─> 5 RENAME ─> 6 TRITON-ST-DELETE ──┴─> 7 FULL-RERUN (last) ─> 8 F4
Needs the user: release decisions before F4
```
