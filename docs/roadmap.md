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
  file the official-code defects (OF-3, OF-4, OF-6, OF-9, OF-10 in
  [deviations](paper/reproduction-deviations.md)) upstream as issues. The
  ECIR call asks what contact happened; replies take weeks.
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

**Exploration first, one clean pass last** (user, 2026-10-09,
[decisions](decisions.md#campaign-v2-user-2026-10-08)). While the library
is still improving, legs run at the current tag (now `campaign-v2.5`,
code_version `472f2fc68c697179b463d3b5a6b19ade194e6e2f` = v2.4 + ST-LANE +
V2-FILL + C5-OURS, Hub `campaign-v2.5/<dataset>-<suite>`; `campaign-v2.4` was
`d67d6263` (v2.3 + ST-SKIP128), `campaign-v2.3` was `1258a63e`
(v2.2 + ST-IDS + V2-HIGHP + V1-FUSE), `campaign-v2.2` was
`0d23c615`, `campaign-v2.1` `f01255f1`, `campaign-v2` `408b1188`) to see how
everything behaves and to make the charts. A new tag does not stop or
invalidate anything: every record keeps its code_version, and a change adds
rows to the **redo ledger** below only for the cells it actually changes.
The final repro pass (F-REPRO) reruns the ledger, or the whole grid at the
final tag, and D1-G gates it. After each leg: `bench upload --verify`, a
[hub-index](artifacts/hub-index.md) row, the validation row, a look at the
exhibit it feeds.

**Library under a running leg.** The shared venv `/venvs/retrieve` is an
editable install: every `bench` child imports `retrieve` from
`/workspace/retrieve`. Since H-PROVENANCE, `code_version` is the imported
package's tree, read per child, so a record is stamped with the code that
ran. A leg launched with an older harness still reads the launching
checkout: for those, `/workspace/retrieve`'s library stays fixed until the
leg ends. A new tag still moves a pod's checkout only between legs, so one
leg runs one library.

**Redo ledger** (cells whose code or parameters changed after they ran):

| records | changed by | redo |
|---|---|---|
| V2 + V3 Triton perf at `408b1188` (V-PILOT goodreads-synth, V-GR-FILTER) | V2-FIX-A (bit-exact; faster) | perf at the final tag, quality reused |
| SilverTorch eager perf at `408b1188` (H2H-FINAL, V-CODESIGN, D3 timed: all rerun at v2.1) | quantize fix (+≈30 µs eager, graph unchanged) | eager perf |
| goodreads SilverTorch `filter` + `synth` at `n_lists` 1024 | IVF-TUNE (goodreads 4096 / n95 64) | whole SilverTorch arms |
| arXiv `72e5a90` reuse entries, V2 / V3 perf half | V2-FIX-A | perf |
| PubMed SilverTorch Triton perf (D3 PubMed timed at v2.1; V-PUBMED's Triton arms) | ST-DLOOP (scores bit-exact) | Triton perf |
| SilverTorch Triton bloom perf at d128 / d192 with p < 1/256 (synth low-p bloom) | ST-SKIP128 | Triton bloom perf |
| V2 Triton perf at `D_PAD` ≥ 1024 run at v2.1 (V-PUBMED; d128 / d192 legs gain or are neutral) | V2-HIGHP | V2 perf |
| every SilverTorch Triton perf record before v2.3 (all widths; ≈ −3 µs at k 100 × n_probe 24, more at k 1000) | ST-IDS | Triton perf |
| V1 Triton perf before v2.3 | V1-FUSE | V1 perf |
| every official perf record before v2.6 (C5 codesign official, T3 h2h, D3 / bloomwidth official) | OFFICIAL-REWORK (our adapter's per-forward host work left the timed call) | official perf |
| SilverTorch Triton bloom perf before v2.5 (all widths; 0 to −10 %) | ST-LANE (bit-exact; none / exact SASS unchanged) | Triton bloom perf |
| V2 Triton perf at `D_PAD` ≤ 256 before v2.5 (incl. C1's crossover; 0.22-0.97 of v2.4) | V2-FILL (bit-exact) | V2 perf |
| graph-mode ids of SilverTorch records at `408b1188` | quantize fix | ids (D1-G's id gate) |

### Stop rules

- **Pilot gate** (passed 2026-10-08 on goodreads-synth, [validation](validation.md)):
  proceed to the arXiv and YFCC synth legs only if achieved pass rates are
  within 1 % relative of target where N·p ≥ 10⁴ (below that the binomial
  spread exceeds 1 %; goodreads p 0.001 is −1.01 % at N·p 797), V1's
  `recall_oracle` is ≥ 0.99 at every p, and the curves are monotone where
  they must be (V1 latency roughly flat in p, postfilter recall falling
  with p). Otherwise stop and report.
- **Noise gate.** Every timed comparison in an exhibit has a 95 % CI on its
  ratio that excludes 1.0, or it is reported as "no difference". No more
  than 5 repeats to force significance.
- **Densify only on demand**: a parameter point is added only where an
  exhibit's curve has a kink or crossover between two measured points and
  the claim depends on where it is.
- **Budget gate.** If a step's GPU-h exceed 1.25× its estimate, report it
  and update the remaining estimates from the measured rate; the queue
  continues (user, 2026-10-08: the full grid runs).
- **Surprise gate.** A result that contradicts an earlier one (e.g. Triton
  slower than official end to end on the frozen code) stops the queue:
  rerun that one cell interleaved and report before anything else runs.

## Phase P: CPU work before the campaign

The claims and the claim-to-evidence matrix are [claims](paper/claims.md);
the record inventory is [artifacts/campaign-v2](artifacts/campaign-v2/README.md).

## Phase V: the campaign (pods the user creates)

GPU 0 takes all timed work from one sequential driver, cores pinned
NUMA-local; GPU 1 takes quality-only work (`--skip-perf`: oracle builds via
*bench oracle*, bloomwidth quality, the PubMed embedding check, the `n95`
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
- [ ] **EXHIBITS: the exhibits and the checks, after every leg** (user,
  2026-10-09: use the GPUs to catch bugs, compare, reproduce the main takes).
  CPU: `bench fetch` every `campaign-v2/` and `campaign-v2.1/` leg, `bench
  report` (latest record per key; mixed code_versions labelled), and build
  T1 (claims table, verdict per C1-C7), T2, T3, F1, F2 (with the real-data
  per-query selectivity overlay from the dumps), F3, F4a, F4b; plus the D1-G
  checks on what exists (eager-vs-graph `ids_sha256`, bs 16 < 16 × bs 1,
  recall sanity vs exact arms). Each run: an artifact page for the user and
  a note naming new bugs, surprises and claim status. Pod 1, slot 2.
- [ ] **V-AX-CORR: the cluster-correlated synth variant** (re-plan P1; its
  trigger, the IVF recall collapse at low pass rate, shows on YFCC and
  PubMed). Builder: `eval-data synth-filter --correlated` (coarse k-means
  with round(1/p) centroids; item attribute = its cluster, a query's clause =
  its nearest centroid), `arxiv-corr-synth.yaml`, CPU test (pass rate, query
  side); then arXiv at p {0.01, 0.03, 0.1}, every synth arm. IVF's best case
  next to the uniform worst case (F2). Pod c, slot 2. **≈ 3 GPU-h.**
- [ ] **H-SCOPE follow-ups**: `kernel_scopes` is merged (harness, CPU
  test over every real kernel name). Left: the report's T3 scorer column
  reads `kernel_scopes`, and one GPU test on a real call (pod 1, ~2 min, in
  the H-QLOOP gate slot).
- [ ] **C5-OURS cells: co-design vs full mask inside our Triton backend**
  (idea #3; the `bloom_path=full` option is in `campaign-v2.5`). Run the
  `codesign` triton cells interleaved partial / full on goodreads + arXiv,
  plus one PubMed 10 M point (C5 is claimed at 20 M; the full mask's
  scratch grows with N). Separates the co-design idea from Meta's host-side
  partial path (V-PROF3). st-dloop, pod b, at `campaign-v2.5`, Hub
  `campaign-v2.5/<dataset>-codesign-ours`. **≈ 1-2 GPU-h.**
- [ ] **EXHIBITS ideas #1, #2, #6** (CPU, exhibits analyst on pod 1): a per-query
  pass-rate router (IVF / V2 / V1) as a counterfactual from the sidecars;
  filter-cluster alignment (GLS) to explain F2's real-vs-synth gap, with
  V-AX-CORR; Big-ANN-style QPS at recall 0.95 per selectivity band, and a
  reproduction defect ledger.
- [ ] **V-ROUTER: a local-pass-rate router as a measured arm** (EXHIBITS idea
  #2: IVF recall follows the local pass rate l_q, the share of a query's
  unfiltered top-100 that pass; Spearman 0.82 / 0.48 vs 0.38 / 0.30 for the
  global p; the counterfactual l_q router is within 5 % of the per-query
  oracle on goodreads). Harness arm `router`: a cheap unfiltered SilverTorch
  pre-probe (small n_probe, k 100) gives l_q per query, then exact (V2) below
  a threshold, IVF above; the pre-probe's cost is timed in the arm. Fit the
  threshold on goodreads, report it fixed on arXiv and PubMed (transfer).
  Gates: recall equal to its two branches per query (ids from the routed arm),
  harness suite. Beyond the original papers; F2 / T2 practical take. Pod 1's
  runner after V-GR-DEEP. **≈ 1-2 GPU-h.**
- [ ] **V-V3BITS (goodreads-synth done; goodreads' 48 cells deferred): V3 at LiNR's bit budget, next to our deviation** (C2 does
  not hold so far: recall −7-13 % at a 1 % pool, no gain at bs 1; our V3
  runs `k_bits` = D = 128 against LiNR's 512, LN-8). `k_bits` must divide D
  (`quantize.py` `_oporp_k_bits`), so at D 128 only {64, 128} exist without a
  library change: goodreads-synth and goodreads `filter` V3 at `k_bits`
  {64, 128}, pool {1 %, 5 %}, quality first, then the timed cells, to show
  the recall-vs-bits slope. LN-8 stays the campaign's setting. Pod 1.
  **≈ 1 GPU-h.**
- [ ] **D3: `bloomwidth`**: PubMed `bloomwidth-timed` reruns at `campaign-v2.2`
  (its v2.1 run tripped the surprise gate, ST-DLOOP); goodreads and arXiv done
  (timed at v2.1), PubMed `bloomwidth` (quality) done
  ([validation](validation.md), Hub `campaign-v2/arxiv-bloomwidth[-timed]`):
  both blooms, `m_bits` 64-2048 × `k_hash` {3, 5}, quality-only plus one
  timed point per width at bs 16. F4a, C4. **≈ 3-5 GPU-h**, GPU 1 (the
  timed points on GPU 0).
- [ ] **SYNTH-TRIM: a smaller synth grid, dense in the middle** (user,
  2026-10-10: the lowest rates add little, 0.1 / 0.2 / 0.5 matter more).
  arxiv-synth rates {0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}, yfcc10m-synth
  {0.01, 0.1, 0.2, 0.5, 1.0}; seed 0 only, seeds 0-2 at p 0.01 and 0.2
  (the variance check); SilverTorch clause `n_probe` {24, 64, 256, 1024}
  (YFCC {24, 256, 1024}). New attrs (`eval-data synth-filter --rates`),
  configs, `suites.yaml`, the YFCC chunk list regenerated; cell counts and
  GPU-h re-projected before V-AX-SYNTH starts. goodreads-synth gains 0.05 /
  0.2 / 0.5 at seed 0 (≈ 1-2 GPU-h) so the panels share the middle points.
  d-run, pod d, before V-AX-SYNTH.
- [ ] **V-AX-SYNTH: arXiv synth** on SYNTH-TRIM's grid (the correlated
  variant is V-AX-CORR). F1/F2 3M panel. **≈ 15-45 GPU-h**
  (the pilot measured 8.7 GPU-h at 0.8 M; re-projected ≈ 24-30 GPU-h, the
  torch arms run ≈ 385 s a cell at 3 M), GPU 0. Its timed cells run at
  `campaign-v2.1`, at IVF-TUNE's arXiv `n_lists`.
- [ ] **V-GR-DEEP: goodreads `deep`**, trimmed (`n_lists` {1024, 4096},
  `n_probe` {8, 16, 32, 64, 128}, V3 pool fractions); replaces D1-C.
  F3; goodreads' `n95`, then the goodreads `filter` n95 cells and the
  189 new goodreads-synth cells of the SilverTorch n_probe sweep (in
  suites.yaml; resume adds only the new cells; driver
  [v-gr-deep](artifacts/campaign-v2/v-gr-deep/driver.sh)). **≈ 6 GPU-h**,
  GPU 0/1.
- [ ] **V-YFCC: YFCC synth (SYNTH-TRIM's 5 points) and a small `deep`** (`n_lists`
  {4096, 16384}). F1/F2 10M panel; YFCC's `n95`. Runs alongside
  V-AX-SYNTH on another GPU (three pods, no fourth: user, 2026-10-08); a
  collapse found on arXiv adds YFCC points afterwards. **≈ 31-54 GPU-h**
  (EXHIBITS run 1; V3 at 10 M is most of the range: one timed V3 cell first).
  `deep` on pod 1; `synth` split in arm-group chunks (`bench run --algo`, one
  interleave unit never split), each taken by whichever of pod 1 and pod c
  frees first (V-AX-SYNTH is now ≈ 24-30 GPU-h on pod c).
- [ ] **V-SEEDS: arXiv and YFCC `filter`, the cells the manifest does not
  reuse.** Deferred to F-REPRO (user, 2026-10-10: the exploration needs
  numbers that decide the claims, not final ones; seed variance is a
  final-pass question). The arXiv half runs only if pod b has nothing
  claim-deciding queued. arXiv: 117 cells (V1-V3 `c3_nversions` seeds 0-2; SilverTorch
  triton; official bloom; `postfilter` α {1, 8}; SilverTorch torch, plain
  and compiled, for C3); V1-V3 on `c0_maincat` / `all4` are reused
  through the manifest (36 cells, [campaign.yaml](../evaluation/campaign.yaml)).
  YFCC: all 18 cells, seed 0 included (a manifest entry cannot name a
  seed). Plus the `n95` column on the kept sweeps. T2 with CIs. Needs
  V-AX-SYNTH and V-YFCC. **≈ 8-12 GPU-h**, GPU 0.
- [ ] **V-PUBMED: PubMed `filter` under the new grid** (replaces D1-E):
  the embedding identity check on GPU 1 first (`eval-data pubmed
  encode_queries`, `bench check`, one V1 cell equal to `d1/pubmed`'s
  quality, or stop), the `n95` probe suite (quality-only), then the 3
  kept sweeps, 3 seeds, every arm, now at `campaign-v2.1` and IVF-TUNE's values
  (its SilverTorch Triton perf goes on the redo ledger for ST-DLOOP). T2's 768-d row.
  **≈ 14 GPU-h**, GPU 0.
- [ ] **V-LAION30: a 30 M scale point at d256, `filter` only** (data staged
  through the generic ingest, `bench check` ok; the grid remains) (user,
  2026-10-10; [decisions](decisions.md#datasets)).
  Re-LAION-2B-en-research-safe (gated, auto-approved; captions + url /
  size / similarity / punsafe / pwatermark, no embeddings): the first two
  parquet parts (32.8 M rows), 30 M items + held-out caption queries;
  captions encoded by us with nomic-embed-text-v1.5 at its Matryoshka
  d256 (≈ 1.5 GPU-h). Filters from the metadata as tags: url domain
  ("search within a site"), size, similarity and punsafe / pwatermark
  buckets; a query's clause from its own tags. A new `eval-data` command
  + `laion30m.yaml` + CPU tests + `bench check`; then a reduced `filter`
  grid (SilverTorch triton, V1, V2, bs 1 / 16, 3 sweeps, seed 0) at the
  current tag, IVF sized for 30 M under the 25 % cap. Memory: 31 GB fp32
  items + 7.7 GB int8 codes (the item-chunked oracle holds no copy). Pod d, CPU
  now, GPU after V-AX-SYNTH. **≈ 10-15 GPU-h** (estimate, scaled from
  PubMed's 10 M cells).
- [ ] **V3-BITS-PUBMED: V3 at `k_bits` 256 on PubMed d768** (user,
  2026-10-10). LiNR's V3 used 512 bits on d128; our OPORP gives at most D
  bits (one per coordinate, LiNR's own §3.2 wording), so d128 / d192 run
  D bits and V-V3BITS's 64; at d768 we already run 768 bits, and 256
  (divides 768) brackets LiNR's 512 from below. PubMed `filter` kept
  sweeps, V3 triton, pool {1 %, 5 %}, seed 0. Pod b after V-SEEDS arXiv.
  **≈ 1-2 GPU-h.**
- [ ] **OFFICIAL-REWORK: the official backend run as Meta intends** (user,
  2026-10-10: "the algo speed should not suffer from the need to adapt the
  code"). Our adapter did per-forward host work inside every timed call
  (`queries_to_expressions` with a `.tolist()` sync, the CPU parse with the
  plan cache off for timing, an 826 MiB `pack_mask` intermediate in exact
  mode). Redesign: every index-side transform at `register_index`, every
  query-side transform in `prepare_queries` (outside the timed call,
  reported as `query_prep_ms` for every backend alike), a forward of Meta's
  ops plus minimal glue with zero host syncs of ours, Meta's in-kernel
  top-k if it has one, a packed exact-mask kernel, CUDA-graph capture
  wherever Meta's ops allow. Gates: ids + scores `torch.equal` against the
  current paths, library + harness suites, keep rule; tag `campaign-v2.6`;
  then C5 official and T3 h2h re-run claims-first. st-dloop, pod b; C5's
  verdict on Meta's code waits for it.
- [ ] **ROUTER-LIB: the router as a library method** (the library goal,
  user 2026-10-10; unassigned, after OFFICIAL-REWORK; WIP on `dev/router-lib`): the harness `router` arm's logic (unfiltered pre-probe
  → l_q → V2 below the threshold, SilverTorch above) as a `retrieve`
  module with its docs page; the harness arm calls it; gates: the arm's
  records unchanged (ids + scores `torch.equal`), library + harness suites.
  After V-ROUTER's goodreads fit.
- [ ] **REL-LIC: license audit and the public-release plan** (user,
  2026-10-10, ECIR Availability): per dataset (goodreads, arXiv, PubMed /
  MedCPT, YFCC-10M, Re-LAION, the synth attrs) whether the derived data may
  be redistributed or only its build recipe; which Hub repos go public at
  submission (results, datasets, checkpoints) and what each must not hold;
  a page under `docs/paper/`. Making anything public stays the user's
  call. CPU.
- [ ] **F-CRIT: the ECIR criteria checklist** (user, 2026-10-10): each
  question of the replicability-track criteria (reliability, impact,
  novelty, availability) mapped to the paper section, table or artifact
  that answers it, with gaps named; a new page ecir-criteria.md under docs/paper. F5
  writes against it.
- [ ] **F-REPRO: the final pass.** When the library stops changing: tag the
  final version, rerun the redo ledger (or the whole grid if the ledger is
  most of it) at that tag into `campaign-final/`, write *campaign.yaml* for
  one code_version. Then D1-G.
- [ ] **D1-G: gate reruns and the report.** Needs F-REPRO.
  Eager-vs-graph id identity from the stored hashes, a byte-identical
  quality subset per leg (`--force` into a scratch tree),
  `median_ms(bs=16) < 16 × median_ms(bs=1)`, then `bench report
  --manifest` over everything. **≈ 4 GPU-h**, both GPUs.

## Phase F: the paper

- [ ] **F2: the official-vs-reimplementation section** rewritten from
  H2H-final's numbers, with CIs and paired tests
  ([paper](paper/official-vs-reimplementation.md)). Needs H-PROFILE and
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
| M1 | 0.5 | both | |
| EXHIBITS | 0 | CPU | after every leg |
| V-AX-CORR | ≈ 3 | 0 | 3 points, arXiv |
| V1-FUSE | ≈ 1 | 0 | gates + keep-rule grid |
| V-V3BITS | ≈ 1-2 | 0 | goodreads |
| D3 bloomwidth | ≈ 3-5 | 1 (+0) | quality-only cells |
| V-AX-SYNTH | ≈ 15-45 | 0 | Triton ~2,000 s per pass point; V1/V2 torch at 3 points ×3 seeds is most of it |
| V-GR-DEEP | ≈ 3-6 | 0/1 | ~210 cells at ~50 s (`d1/arxiv-deep`: 873 cells in 36 h at 3 M) |
| V-YFCC | ≈ 31-54 | 0/1 | V1 / V2 / V3 cells 0.3-1.2 h each at the v2 grid; V2/V3 cost vs p at 10 M unmeasured |
| V-SEEDS | ≈ 8-12 | 0 | YFCC seeds 1-2: V2 and V3 ~1.1 h a cell |
| V-PUBMED | ≈ 20-25 | 0 (+1 for the checks) | V1 ~650 s, V2 ~1,300 s, SilverTorch-Triton ~700 s a cell; 3 sweeps × 3 seeds |
| V-LAION30 | ≈ 10-15 (+1.5 encode) | 0/1 | scaled from PubMed 10 M; 30 M d256 |
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
tag campaign-v2 ───────────────────────┬─> V-PILOT ─┬─> V-AX-SYNTH, V-YFCC ─> V-SEEDS ─┐
                                       │            └─> V-GR-DEEP ─────────────────────────┤
                                       ├─> V-CODESIGN, D3 ─────────────────────┤
                                       └─> V-PUBMED ───────────────────────────────────────┴─> D1-G ─> F2, F4, F5
M1: before any timed step on a multi-GPU pod
tag campaign-v2.1 and IVF-TUNE (both done): before every timed V-AX-SYNTH, V-YFCC, V-PUBMED, V-SEEDS cell
```
