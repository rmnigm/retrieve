---
title: roadmap
created: 2026-09-26
updated: 2026-10-08
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

**Rules.** No dates: every step is done, in the order given, and nothing
is optional. Do not start a step whose dependencies are open. When a step
finishes, its gate's row in [validation](validation.md) and the affected
`docs/system` page are updated, and the orchestrator **removes** the step
from this page. Step names (D1, H2, ...) are stable identifiers used in
records, commits and code comments; they are not renumbered.

**Estimates** are A100-SXM4-80GB GPU-hours (the reference GPU for every
citable number), from the per-cell wall times the record inventory measured
([artifact](artifacts/campaign-v2/README.md#b-measured-wall-time-per-cell-old-grid-3-bs--3-k--2-modes)); a *stream* is
a slice of a step that can run on its own GPU at the same time as the
others (see [Multi-GPU execution](#multi-gpu-execution)).

## Needs the user

- **Contact the original authors** (re-plan decision 7): the LinkedIn LiNR
  team and Meta's SilverTorch team — filter-set details, the V1/V2 setup,
  the SilverTorch paper's FPR inconsistency (0.067 % vs 0.00173 %) — and
  file the official-code defects (OF-3, OF-4, OF-6 in
  [deviations](paper/reproduction-deviations.md)) upstream as issues. The
  ECIR call asks what contact happened; replies take weeks.
- **The campaign budget, after the pilot.** Measured per-cell times put
  the campaign at roughly 130-250 GPU-h, not the re-plan's 70-90: the 10 M
  exact arms cost 0.4-2.4 GPU-h a cell and the torch arms 5-9× Triton
  ([inventory](artifacts/campaign-v2/README.md#b-measured-wall-time-per-cell-old-grid-3-bs--3-k--2-modes)).
  Nothing waits on it now (code comes first); the budget gate stops the
  queue before V-AX-SYNTH / V-YFCC if the pilot confirms it, and the user
  then chooses between the full grid and a trimmed one.
- **Citability of a narrowed run.** A run with a narrowed mode set is
  recorded `status: partial` and reported NOT CITABLE. The user decides
  once `bench report` runs on the campaign's output (D1-G).
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
2. Data under `/data` (`RETRIEVE_DATA_ROOT`), with `evaluation/data ->
   /data` (a gitignored symlink) in every checkout and worktree. `eval-data
   fetch` only what the step reads: `arxiv-papers` (the license-fixed
   version: `item_attrs_narrow.pt` min = -1, `eval_split.parquet`
   `query_attrs_narrow` min = 0), `goodreads-work-id` with its checkpoints
   ([checkpoints](system/checkpoints.md#what-the-harness-reads)). PubMed is not on the Hub: rebuild the 10 M slice with
   `eval-data pubmed` ([datasets](system/datasets.md#pubmed); ~1 h
   download-bound `convert`, 17 GB). YFCC-10M is public: `eval-data yfcc
   all` ([datasets](system/datasets.md#yfcc10m), ~12.5 GB with the raw).
   The synth attrs are built on the pod (*eval-data synth-filter*, CPU,
   deterministic from its seed).
3. Results: the remaining runs do not need the old records locally (the
   narrow scoping below never resumes against them), but the final uploads
   and the report do: `bench fetch --path-in-repo d1/<subtree> --results
   <dir>` per [hub-index](artifacts/hub-index.md) row, merged into
   `evaluation/results/<suite>/`.
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
then by `--filter-kind` / `--sweep` / `--seed`; keep `silvertorch`'s
`triton` and `official` backends in **one** stream, in that order: they
share a `_parity/<dataset, dim, algo>` spill and the second backend's
`jaccard_vs_first` compares against the first's. `bench campaign` has no
`--algo`, so a split stream is a `bench run`. M1 decides whether
multi-GPU numbers are comparable at all.

### Campaign code_version and reruns

Every campaign cell runs at the `campaign-v2` tag's code_version (the
tree hash of `retrieve/src/retrieve`), recorded in
*evaluation/campaign.yaml* (to be created by CV2-REPORT) ([decisions](decisions.md#campaign-v2-user-2026-10-08)).
New legs run into fresh result trees; old records are never re-stamped and
enter the paper only through the reuse rule and the manifest. A
correctness bug found mid-campaign stops every run; it is fixed and
re-tagged, only the arms its own gates prove changed are rerun, and the
manifest's `log` says so. After each step: `bench upload --verify`, a
[hub-index](artifacts/hub-index.md) row, the manifest and the validation
row updated, `bench report` on the cumulative tree, and a look at the
exhibit the step feeds; if that exhibit already settles its claim, the
claim's remaining runs are skipped and the reason logged.

### Stop rules

- **Pilot gate** (after V-PILOT): proceed to the arXiv and YFCC synth legs
  only if achieved pass rates are within 1 % relative of target, V1's
  `recall_oracle` is ≥ 0.99 at every p, and the curves are monotone where
  they must be (V1 latency roughly flat in p, postfilter recall falling
  with p). Otherwise stop and report.
- **Noise gate.** Every timed comparison in an exhibit has a 95 % CI on its
  ratio that excludes 1.0, or it is reported as "no difference". No more
  than 5 repeats to force significance.
- **Densify only on demand**: a parameter point is added only where an
  exhibit's curve has a kink or crossover between two measured points and
  the claim depends on where it is.
- **Budget gate.** If cumulative GPU-h exceed 1.25× the estimate at any
  step, stop and re-plan with the user.
- **Surprise gate.** A result that contradicts an earlier one (e.g. Triton
  slower than official end to end on the frozen code) stops the queue:
  rerun that one cell interleaved and report before anything else runs.

## Phase C: the campaign-v2 code batch (development pod, before any cell)

All of #1-#16 of the re-plan land before the freeze; nothing is cut. Four
workloads on `dev/cv2-<w>` off `dev/campaign-v2`, editing disjoint files,
at most two at once; the orchestrator merges green workloads into
`dev/campaign-v2`. Each workload's gates are in its brief; the headline
ones are below.

- [ ] **CV2-LIB** (library): #7 LiNR V4 and `PostfilterKNNInt8` out of the
  library (tag `linr-v4-final` first; golden V4 cells go); #14's library
  side, an item-range `evaluate_mask(q, start, end)` on both filters; #11
  L6, official int32 parity against Triton at d ∈ {64, 128, 192, 768}; #16
  filter-first tile skipping in the fused bloom/exact probe scorer,
  measured first (`torch.profiler`, kernel-only, interleaved) and kept only
  if it wins at low pass rate without a regression at p = 1.0. *Gates*:
  library suite green; L6 and the probe-scorer parity tests **bit-exact**;
  range mask `torch.equal` to the full mask's slice. **≈ 1 GPU-h** of
  testing.
- [ ] **CV2-DATA** (harness config and data): the harness side of #7; #1
  the synth builder *eval-data synth-filter*, `<ds>-synth.yaml` siblings
  and `filters.query_attrs`; the `suites.yaml` grid redesign (seeds
  {0, 1, 2} everywhere, bs {1, 16}, k {100, 1000}, kept sweeps, official on
  bloom/none only, per-dataset `n_lists` / `n_probe`, `filter` / `deep` /
  `synth` / `codesign` / `bloomwidth` / `n95` suites); #12 the SilverTorch
  torch-reference arm; #13 the `torch.compile(mode="max-autotune")` arm
  (the torch arms on goodreads and arXiv only until C3 is settled there);
  #15 `m_bits` / `k_hash` as build params; V3 `candidate_pool` as a
  fraction of passing items. *Gates*: synth CPU test (pass rate within 1 %
  relative, nesting, determinism); existing record keys byte-identical
  (**bit-exact**); expansion counts pinned. Merges before CV2-LIB (the
  harness imports `LiNRV4` until it lands).
- [ ] **CV2-CORE** (cell loop; after CV2-DATA): #2 quality cache
  (`seed_scope`); #3 `--interleave` with comparison groups, official
  `score_path` as a build param and the `h2h` suite; #4 per-query npz
  sidecars (uploaded); #5 per-(mode, bs, k) id hashes; #14 the chunked
  exact oracle and *bench oracle*; the per-job clock log. Record schema 4.
  *Gates*: cached quality equals fresh quality (**bit-exact**); chunked
  oracle equals the existing blobs on arXiv and PubMed up to counted
  exact-score ties (**bit-exact otherwise**); ABAB order and per-arm keys
  pinned.
- [ ] **CV2-REPORT** (beside CV2-CORE): #8 report fixes (dev/d5-report's
  postfilter label and the α rule, English labels, matched-recall rows and
  `n95`, the T1/T2/T3 and F1-F4 generators); #9 statistics (median of
  window medians, bootstrap CIs over seed × window and over queries, paired
  ratio CIs for interleaved groups); #6 the campaign manifest
  *evaluation/campaign.yaml*, read by `bench report --manifest`; #15's
  FPR / memory columns. *Gates*: every exhibit builds from a fixture tree
  with one schema-4 record of every arm; known-answer stats tests.
- [ ] **CV2-FREEZE** (needs the four above): on the integrated tree, the
  library suite, the harness suite, the golden harness gate (**bit-exact**
  against the H2 rerun, standing residuals unchanged), one `--skip-perf`
  smoke cell per suite per dataset (yfcc10m staged first), one timed
  interleaved smoke, `bench report` over a tree with one record of every
  arm; then the code_version goes into the manifest, `dev/campaign-v2`
  merges into staging once, and the orchestrator tags `campaign-v2`.
  **≈ 2 GPU-h.**

## Phase P: CPU work beside Phase C

The claims and the claim-to-evidence matrix are [claims](paper/claims.md);
the record inventory is [artifacts/campaign-v2](artifacts/campaign-v2/README.md).

- [ ] **Manifest reuse entries** (CPU, after CV2-REPORT): fill
  *evaluation/campaign.yaml*'s quality and perf entries for the existing
  records that pass the reuse rule
  ([inventory](artifacts/campaign-v2/README.md#c-which-records-pass-the-reuse-rule):
  arXiv `filter` / `deep` quality, timing where the clock criterion holds;
  SilverTorch-Triton timing only once CV2-LIB's #16 decision is known), and
  the claims' cell selectors ([claims](paper/claims.md)).
- [ ] **R-RES: the unexplained residuals**, time-boxed to 2 h of CPU:
  `linr_v2` 4.5e-4 and `linr_v3` 1.7e-5 against their golden JSONs, arXiv
  SilverTorch 2.0e-6 ([validation](validation.md#harness-gates)). Still
  unexplained after the box: reported as-is in the deviations table.

## Phase V: the campaign (pods the user creates; needs CV2-FREEZE)

GPU 0 takes all timed work from one sequential driver, cores pinned
NUMA-local; GPU 1 takes quality-only work (`--skip-perf`: oracle builds via
*bench oracle* (CV2-CORE), bloomwidth quality, the PubMed embedding check, the `n95`
probe). Timed work moves to a second GPU only if M1 passes. Exhibits:
T1 claims, T2 real-filter headline, T3 official vs Triton, F1 latency vs
pass rate, F2 recall vs pass rate, F3 Pareto, F4a bloom FPR/memory, F4b
co-design.

- [ ] **M1: multi-GPU interference control**, before any timed step on a
  multi-GPU pod: one arxiv `filter` cell at bs 1 and 16
  (`silvertorch`/triton, `linr_v2`) alone, then with every other GPU
  loaded, interleaved, cores pinned. *Gate*: loaded medians within each
  arm's own repeat noise; otherwise timed steps run one GPU at a time.
  **0.5 GPU-h** on a ≥ 2-GPU pod.
- [ ] **V-PILOT: goodreads synth** (uniform, 7 points), the pilot gate
  above decides the rest of the synth axis. **≈ 4 GPU-h**, GPU 0.
- [ ] **V-GR-FILTER: goodreads `filter` on E1c** (3 sweeps, 3 seeds, every
  arm); hub-index labels `d1/goodreads` "gSASRec, superseded". T2's
  goodreads row. **≈ 5 GPU-h**, GPU 0.
- [ ] **H2H-FINAL: official vs Triton on the release code** (`h2h` suite:
  goodreads E1c and arXiv, interleaved, 5 repeats, official `score_path`
  fp16 and int32; e2e, kernel-only, launches, memory, parity). The only
  T3 source; replaces b3 and the kernel-opt head-to-head (C7). **≈ 3
  GPU-h**, GPU 0.
- [ ] **V-CODESIGN: `codesign` on arXiv and goodreads**, interleaved
  partial/full, `n_probe` {8, 32, 128}, 3 sweeps, 3 seeds; replaces D1-B2
  and D1-D. F4b, C5. **≈ 2 GPU-h**, GPU 0.
- [ ] **D3: `bloomwidth`** on goodreads, arXiv and PubMed (`c0_mesh`):
  both blooms, `m_bits` 64-2048 × `k_hash` {3, 5}, quality-only plus one
  timed point per width at bs 16. F4a, C4. **≈ 3-5 GPU-h**, GPU 1 (the
  timed points on GPU 0).
- [ ] **V-AX-SYNTH: arXiv synth**, uniform 7 points, then the
  cluster-correlated variant (3 points) if the uniform sweep shows the IVF
  recall collapse at low p; arXiv's `n95`. F1/F2 3M panel. Needs V-PILOT.
  **≈ 11 GPU-h**, GPU 0.
- [ ] **V-GR-DEEP: goodreads `deep`**, trimmed (`n_lists` {1024, 4096},
  `n_probe` {8, 16, 32, 64, 128}, V3 pool fractions); replaces D1-C.
  F3; goodreads' `n95`. Needs V-PILOT. **≈ 6 GPU-h**, GPU 0/1.
- [ ] **V-YFCC: YFCC synth (5 points) and a small `deep`** (`n_lists`
  {4096, 16384}). F1/F2 10M panel; YFCC's `n95`. Needs V-AX-SYNTH.
  **≈ 18 GPU-h**, GPU 0/1.
- [ ] **V-SEEDS: arXiv and YFCC `filter` backfill**: seeds 1-2 and the
  `n95` column on the kept sweeps (arXiv's other-sweep records stay as
  they are). T2 with CIs. Needs V-AX-SYNTH and V-YFCC. **≈ 4 GPU-h**,
  GPU 0.
- [ ] **V-PUBMED: PubMed `filter` under the new grid** (replaces D1-E):
  the embedding identity check on GPU 1 first (`eval-data pubmed
  encode_queries`, `bench check`, one V1 cell equal to `d1/pubmed`'s
  quality, or stop), the `n95` probe suite (quality-only), then the 3
  kept sweeps, 3 seeds, every arm. T2's 768-d row. **≈ 14 GPU-h**, GPU 0.
- [ ] **D1-G: gate reruns and the report.** Needs every step above.
  Eager-vs-graph id identity from the stored hashes, a byte-identical
  quality subset per leg (`--force` into a scratch tree),
  `median_ms(bs=16) < 16 × median_ms(bs=1)`, then `bench report
  --manifest` over everything. **≈ 4 GPU-h**, both GPUs.

## Phase F: the paper

- [ ] **F2: the official-vs-reimplementation section** rewritten from
  H2H-final's numbers, with CIs and paired tests
  ([paper](paper/official-vs-reimplementation.md)). Needs H2H-FINAL and
  D1-G.
- [ ] **F4: package the artifacts**: the results public, the PubMed slice
  or its build recipe with checksums, a tagged `torchretrieve` release, a
  Zenodo DOI including the pinned official sdist, a one-command
  reproduction, an anonymised mirror (anonymous.4open.science) for the
  double-blind review. Needs D1-G and the user's accounts.
- [ ] **F5: write the paper** (ECIR 2027 reproducibility format, 12 pages
  LNCS): every table from `bench report`, the `docs/paper/` drafts
  rewritten against claims C1-C7 (they still cite retired plan labels).
  Needs everything above.

## Estimates

| step | A100 GPU-h | GPU | basis |
|---|---|---|---|
| CV2-LIB … CV2-FREEZE | ≈ 3 | development pod | tests and smokes |
| M1 | 0.5 | both | |
| V-PILOT | ≈ 8-14 | 0 | goodreads Triton cells ~65 s at the v2 grid; the torch arms ~400-460 s and dominate |
| V-GR-FILTER | ≈ 5 | 0 | ~114 cells; SilverTorch torch + compile ~3 of it |
| H2H-FINAL | ≈ 3 | 0 | |
| V-CODESIGN | ≈ 1 | 0 | `d1/arxiv-codesign`: 60 cells in 0.4 h |
| D3 bloomwidth | ≈ 3-5 | 1 (+0) | quality-only cells |
| V-AX-SYNTH | ≈ 15-45 | 0 | Triton ~2,000 s per pass point; V1/V2 torch at 3 points ×3 seeds is most of it |
| V-GR-DEEP | ≈ 3-6 | 0/1 | ~210 cells at ~50 s (`d1/arxiv-deep`: 873 cells in 36 h at 3 M) |
| V-YFCC | ≈ 60-120 | 0/1 | V1 / V2 / V3 cells 0.3-1.2 h each at the v2 grid; V2/V3 cost vs p at 10 M unmeasured |
| V-SEEDS | ≈ 8-12 | 0 | YFCC seeds 1-2: V2 and V3 ~1.1 h a cell |
| V-PUBMED | ≈ 20-25 | 0 (+1 for the checks) | V1 ~650 s, V2 ~1,300 s, SilverTorch-Triton ~700 s a cell; 3 sweeps × 3 seeds |
| D1-G | ≈ 4 | both | |

Total ≈ 130-250 GPU-h, against the re-plan's 70-90: the 10 M exact arms
and the torch arms were underestimated there (see "Needs the user"). The
SilverTorch/Triton perf-only rerun the re-plan held in reserve costs
0 GPU-h: the CSR probe layout and transposed bloom predate `72e5a90`, and
the one later library change (per-width probe tiles) moves tiles only at
`D_PAD > 256` ([validation](validation.md#library-gates), *Probe-scorer
tile per width*).

## Dependencies

```
CV2-DATA ─> CV2-LIB merge;  CV2-DATA ─> CV2-CORE;  CV2-REPORT ∥ CV2-CORE
CV2-LIB, CV2-DATA, CV2-CORE, CV2-REPORT ─> CV2-FREEZE (tag campaign-v2)
CV2-REPORT ─> manifest reuse entries ─> CV2-FREEZE
CV2-FREEZE ─┬─> V-PILOT ─┬─> V-AX-SYNTH ─> V-YFCC ─> V-SEEDS ─┐
            │            └─> V-GR-DEEP ─────────────────────────┤
            ├─> V-GR-FILTER, H2H-FINAL, V-CODESIGN, D3 ─────────┤
            └─> V-PUBMED ───────────────────────────────────────┴─> D1-G ─> F2, F4, F5
M1: before any timed step on a multi-GPU pod
```
