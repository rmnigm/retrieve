---
title: decisions
created: 2026-09-26
updated: 2026-10-11
type: summary
tags: [decisions]
sources: [pyproject.toml, evaluation/config/suites.yaml, retrieve/src/retrieve/, evaluation/bench/, evaluation/training/]
contested: true
---

# Standing decisions and constraints

Decisions in force, with the reason each was taken. They were made by the
user or measured into place and are not reopened without the user. How the
code implements them is in the [system pages](index.md#system); the open
work is the [roadmap](roadmap.md). The two process contracts,
[agent orchestration](contracts/agent-orchestration.md) and
[coding guidelines](contracts/coding-guidelines.md), carry their own
decisions.

## Goal and scheduling

- **The goal is a reproducibility paper** on SilverTorch (Meta) and LiNR
  (LinkedIn): one correct harness, public datasets up to the papers' scale.
  **The SilverTorch arm is Meta's public code, forked and improved** (user,
  2026-10-10, roadmap META-FORK): the paper's improved baseline is a fork of
  `meta-recsys/silvertorch` with its measured bottlenecks removed, next to
  Meta's code as shipped. LiNR stays a reimplementation from the paper's
  text (it has no public code).
- **Our Triton SilverTorch reimplementation stays in the repo but is
  probably not in the paper** (user, 2026-10-10): it wins on fewer launches
  and syncs, not on its scorer, and at 30 M bs 64 Meta's kernels are level
  or ahead. No Triton code or result is deleted; the final pass still
  measures it for the record. New Triton kernel work is stopped (ST-XQ is
  cancelled).
- **No dates.** Every roadmap step is done, in order; nothing is optional
  or conditional on a deadline. The GPU is the bottleneck, so GPU steps
  are ordered first and CPU steps run beside them.
- **Nothing is citable until its gate passed** (CLAUDE.md rule 2).
  [validation.md](validation.md) says which gates have passed.
- **The study answers the ECIR replicability-track criteria, and ships a
  library plus a benchmark** (user, 2026-10-10): every step serves
  reliability (methodology, representative and tuned baselines, described
  parameters, the right portions replicated), impact, novelty (insights,
  baselines and measures the originals lack) or availability (code, data,
  documentation, no gap between paper and material); the checklist is
  roadmap F-CRIT. Beside the paper: an open-source library of the methods
  and an easy way to benchmark them on more datasets (roadmap
  H-ADDDATA).
- **Scope: model-based retrieval, i.e. retrieval as torch ops inside the
  model graph** (user, 2026-10-10), the frame both papers use (SilverTorch:
  retrieval as PyTorch model ops; LiNR: retrieval as a model on the GPU).
  The paper names it "model-based retrieval", not "ANN on GPU". Baselines are
  what a practitioner can put in that graph (generic torch, compiled
  torch, the torch reference backends); standalone ANN libraries (Faiss,
  cuVS, HNSW) are outside that frame and stay out
  ([Harness](#harness)); the paper states the frame and why.
- **V3's bit budget is one bit per coordinate** (user, 2026-10-10): LiNR's
  V3 benchmark used 512 bits on d128, and its §3.2 describes a 1-bit code
  "of the same dimension"; our OPORP gives at most D bits. The paper states
  that d128 / d192 run D bits (and 64 in V-V3BITS) where the dimension is
  below LiNR's budget; PubMed d768 runs 768 and 256 bits (roadmap
  V3-BITS-PUBMED), bracketing 512. No `k_bits` > D.
- **Venue: the ECIR 2027 reproducibility track format** (user, 2026-10-08):
  12 pages LNCS, double-blind. No deadline pressure ("do not care about
  deadlines, just do all the work; the most useful first"); nothing is cut
  for time.

## Campaign v2 (user, 2026-10-08)

- **T3 reports device time and end-to-end, as measured** (controller,
  2026-10-09, user away and delegating): official SilverTorch eager is
  host-bound (device kernels 0.18-0.31 ms of 1.1-1.85 ms end to end at v2.1;
  its windows read ~1140 MHz because the GPU idles, and its end-to-end
  moves with host load at the same device time), so clock-normalising would
  correct a symptom. T3 gives kernel-only and end-to-end side by side, as
  measured, with the host-bound mechanism stated; F4b (official vs official,
  interleaved) is reported as measured.
- **Exploration first, one final repro pass** (user, 2026-10-09): while the
  library improves, legs run at the current tag to collect behaviour,
  comparisons and charts; a library change does not stop or invalidate
  runs. Records keep their code_version; the roadmap's redo ledger lists
  only the cells a change actually moved; the final pass reruns them at the
  final tag before D1-G.

The campaign was re-planned backwards from the paper's claims (source: the
user's re-plan, kept in the orchestrator's handoff notes; the claims are
C1-C7, the exhibits T1-T3 and F1-F4). These rules govern every cell from
now on; the [roadmap](roadmap.md) holds the steps.

- **Grid and code changes after the pilot** (user, 2026-10-08): `synth`
  sweeps SilverTorch clause n_probe {24, 64, 128, 256, 512, 1024} on
  goodreads and arXiv and {24, 256, 1024} on YFCC (capped at the dataset's
  `n_lists`; bloom arms {24, 256}), replacing synth's n95 slots, so matched
  recall exists at low p; V2's batch-16 floor is profiled before arXiv
  synth (V2-PROF). Declined: V3 `k_bits` > D (LN-8 stays a stated
  deviation), an fp32-pinned compiled arm, an adaptive postfilter α.
- **Budget: the full grid runs** (user, 2026-10-08, after the measured
  estimate of ~130-250 GPU-h replaced the re-plan's 70-90): no trimmed
  grid. The budget gate (roadmap § Stop rules) reports an overrun and
  re-estimates; it does not stop the queue.
- **Claims drive cells.** Every planned cell maps to a figure or table row
  of the paper; a cell that maps to none is cut. A parameter value stays
  only if it moves a curve in an exhibit; if latency is monotone across
  three measured points, the axis is not densified; a claim already
  unambiguous on two datasets gets only the headline point on the third.
- **Code first, then one code_version.** All code changes of the batch
  (plan items #1-#16, P0 and P1, including the chunked oracle and the
  probe-scorer tile skipping) land, pass their gates and are merged before
  any campaign cell runs. Then the code is frozen at one tag,
  `campaign-v2`, whose library tree hash is the campaign's code_version,
  recorded in [`evaluation/campaign.yaml`](../evaluation/campaign.yaml). Allowed before the freeze:
  profiling, unit and parity tests, smoke cells into scratch trees (no
  campaign records). Not allowed during the campaign: feature or
  optimization work. A correctness bug found mid-campaign stops all runs;
  it is fixed, re-tagged, only the arms its own gates prove changed are
  rerun, and the manifest logs it.
  *Contested:* the later *Exploration first* (user, 2026-10-09, above)
  lets legs run at each new tag while the library improves, with a redo
  ledger instead of a freeze.
- **Record reuse rule.** An existing record enters the paper only if all
  three hold: its inputs match the final ones (the E1c encoder for
  goodreads, the license-fixed arXiv attrs); its arm's kernels are proven
  identical to the frozen code by that change's own gates; and its timing
  came from an interleaved comparison or from a run with fewer than 10 %
  of windows below the device's maximum SM clock (`env.sm_max_mhz`;
  the record's `env.frac_windows_below_max`). Quality and timing are
  judged separately (quality is largely reusable). Otherwise it is rerun
  or dropped from the exhibit. The manifest names, per (dataset, suite,
  algo, backend), the accepted code_version and Hub subtree for quality and
  for perf; `bench report` reads the manifest, not "the latest record".
- **Four datasets**: goodreads (E1c, 0.8M, d128), arXiv (3M, d128),
  YFCC-10M (10M, d192), PubMed (10M, d768). **OpenAlex is dropped** (same
  N, width and domain family as PubMed; the leg was E5;
  [backlog](backlog.md#datasets-not-in-the-study)). No point above 10M
  before submission; the chunked oracle is still built.
  *Contested:* LAION 30 M joins as a fifth, `filter`-only dataset (user,
  2026-10-10, [Datasets](#datasets)).
- **Three kept sweeps per dataset** in the headline grid (one high pass
  rate, one low, one conjunctive or reverse): goodreads `c0_genre`,
  `c1_lang_reverse`, `all4`; arXiv `c3_nversions` (0.444), `c0_maincat`
  (0.136), `all4` (0.0094); PubMed `c0_mesh` (0.0002),
  `c3_journal_reverse` (0.9993), `all5` (0.018); YFCC `tags_and`. arXiv
  records already run on the other sweeps are kept, not extended.
- **A synthetic selectivity suite** (`synth`): one nested, per-item uniform
  pass flag (`u_i < p`, generator seed 20261008) at
  p ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0} on goodreads, arXiv and
  YFCC (YFCC at {0.001, 0.01, 0.03, 0.1, 1.0}), on the real embeddings,
  as sibling dataset configs that never touch the real attrs. It is IVF's
  worst case (the filter is independent of the embedding), and p = 1.0 is
  the unfiltered point. *Contested:* the roadmap's SYNTH-TRIM (user,
  2026-10-10) sets ten rates and a smaller grid (arXiv {0.001, 0.01, 0.05,
  0.1, 0.2, 0.5, 1.0}, YFCC {0.01, 0.1, 0.2, 0.5, 1.0}, seed 0 with seeds
  0-2 at p 0.01 and 0.2, clause `n_probe` {24, 64, 256, 1024}).
  A cluster-correlated variant (arXiv, 3 points) runs
  only if arXiv's uniform sweep shows the IVF recall collapse at low p.
- **Grid**: batch sizes {1, 16} (bs 8 dropped), k {100, 1000} (500 dropped;
  1000 is the closest to LiNR's 2000), no `n_probe` 4 or 256 as fixed grid
  points (a dataset's tuned n95 may be 256: arXiv).
  **3 seeds {0, 1, 2} everywhere** (user): every suite, every dataset, every
  sweep; the deterministic arms (V1, V2, postfilter) compute quality once
  and reuse it across seeds, perf repeats per seed. Postfilter α ∈ {1, 8}.
  *Contested:* during exploration the extra seeds are deferred to the
  final pass (user, 2026-10-10; roadmap V-SEEDS, SYNTH-TRIM).
  *Final pass* (controller decision 2026-10-13 on the user's final-pass
  directive of 2026-10-10): seed 0 everywhere, seeds {0, 1, 2} only where a
  T2 confidence interval rests on them (`filter`); every SilverTorch cell at
  `n_probe` ≤ `n_lists` / 4; n_lists / n_probe from the tune
  ([evaluation](system/evaluation.md#ivf-tuning)).
- **Headline SilverTorch operating point**: the paper's `n_probe` 24 plus
  the matched-recall `n95` (the smallest `n_probe` reaching
  `recall_oracle@100` ≥ 0.95 on the dataset's median sweep), not {24, 32}:
  a fixed `n_probe` across 0.8M-10M compares different recall levels.
- **IVF tuned per dataset size** (user, 2026-10-08): `n_lists` and
  `n_probe` depend on N and are tuned, not fixed at the library default
  1024. The sweep runs on **one small and one big dataset only** (user):
  goodreads (0.8 M) and PubMed (10 M), `n_lists` ≈ √N and 4√N ({1024,
  4096} and {4096, 16384}), `n_probe` doubling as far as 0.95 needs (the
  "no `n_probe` 256" grid rule does not bind tuning cells). **Tuning is
  fast and separate from the paper sweeps** (user): quality-only, seed 0,
  bs 16, k 100, median sweep; its records are artifacts, never paper
  numbers, and the full sweeps run **once**, at the tuned values. Chosen
  point per swept dataset: each `n_lists`' smallest `n_probe` with
  `recall_oracle@100` ≥ 0.95, then the one scanning the fewest items
  (`n_probe` · N / `n_lists` + `n_lists`), ties to the smaller `n_lists`. From the two, a size rule
  (`n_lists` as a multiple of √N, `n_probe` as a fraction of `n_lists`)
  sets arXiv (3 M) and YFCC (10 M). The `filter` and `synth` SilverTorch
  arms of each dataset run at its `n_lists` with `n_probe` {24, n95}.
  **Cap** (user, 2026-10-09): tuning stops at `n_probe` = `n_lists`/4
  (25 % scanned); if 0.95 is not reached by then, the record says so with
  the recall reached, and the slot takes `n_lists`/4. n95 follows the
  filter's pass rate more than N (goodreads at pass 0.33: 1/64 of the
  lists; PubMed at 0.018: not reached at 1/16), so the size rule sets only
  `n_lists` (≈ 4√N to a power of two: goodreads 4096, arXiv 8192, YFCC
  and PubMed 16384); arXiv's and YFCC's n95 come from a **quick check**
  (user): that one `n_lists`, quality-only `n_probe` doubling on the
  median sweep, seed 0, same cap. Measured picks: goodreads 4096 / 64, arXiv 2048 / 256
  (2048-8192 all reach 0.95 at ≈ 12.7 % scanned), YFCC and PubMed 4096 /
  1024 = the cap, 0.95 not reached (0.72, 0.873): at k 1000 the probe
  scorers run `n_probe` ≤ 1024, which makes 4096 the largest `n_lists` whose
  25 % cap runs.
- **Official SilverTorch runs only on `none` and `bloom`.** Its clause and
  exact cells time our `pack_mask` adapter, not Meta's code; the existing
  official-clause records stay as an "adapter-bound upper bound" footnote.
- **Timing**: CUDA-graph is the headline mode, eager the secondary; the
  official backend is eager-only (its ops cannot be captured), said in T3.
  Every ratio claim (C1, C5, C7) is timed interleaved: the arms of one
  comparison run round-robin (ABAB) in one process, so clock drift cancels.
- **Baselines stay torch-importable; no Faiss** (user): the generic-torch
  postfilter, LiNR V1/V2 on the torch backend (the native floor), the
  SilverTorch torch-reference backend (unfused IVF) and a
  `torch.compile(mode="max-autotune")` arm of the torch-reference ops. The
  paper states the scope once in the setup and once in threats to
  validity.
- **LiNR V4 leaves the library** (user): it is ours, not LiNR's. The code
  stays under the git tag `linr-v4-final`.
- **H2H-final replaces b3 and the kernel-opt head-to-head** as the only
  source of T3 (official vs Triton): goodreads E1c and arXiv, release code,
  interleaved, 5 repeats, official in both `score_path` modes (fp16, the
  default the paper reports; int32, parity).
- **Execution**: code and GPU testing run on the 1×A100 development pod;
  the long evaluations run on pods the user creates, one sequential
  timed driver on GPU 0, quality-only work on GPU 1, and timed work on a
  second GPU only if M1 passes.

## Library

- **Meta's `meta-recsys/silvertorch` ops are the reference backend**,
  `backend="official"`, installed by the `official` extra and pinned to one
  commit (`21aa35e`, see [pyproject.toml](../pyproject.toml)). Bumping the
  pin is a deliberate change that reruns the official parity gate.
- **Triton is our implementation and gets the kernel effort.** The
  hand-written CUDA C++ and CuTe DSL SilverTorch backends were deleted once
  the official backend's parity gate was green; the git tag
  `cuda-cute-backends-final` holds them.
- **Meta's shape**: `retrieve.modules` (the `nn.Module`s and builders) and
  `retrieve.ops` (registered kernels, one namespace per backend: `triton`,
  `reference`, `official`), plus `retrieve.indexing` and
  `retrieve.functional`. Meta's modules and ops are imported and wired in,
  never copied. See [architecture](system/architecture.md).
- **LiNR V1-V3 are library modules**; the harness keeps only a name to
  class table. V4 leaves (Campaign v2 above).
- **Op names, module names, buffer names and op schemas are stable.** A
  state dict written by an earlier release loads into the current modules.
- **k-means++ is opt-in** (`kmeans_init="random"` is the default) until
  campaign numbers say otherwise. Seeding cost at 3M × 128 with 8,192
  lists is 9.5 s for k-means++ against 0.8 s random.
- **Stream compaction is deterministic**: survivors come out in ascending
  item order on both backends, and a rerun is byte-identical on the
  `[:counts]` prefix, the only part a kernel writes
  ([kernels](system/kernels.md)). Chosen over a wider quality tolerance
  because two golden cells could not reproduce themselves.
- **LiNR's exact scorers store items fp16 and return fp32 scores**
  (`PostfilterKNN`, `PrefilterKNN`, every backend): fp16 storage is the
  LiNR paper's, and keeps the item table at `N × D × 2` bytes (PubMed 10M
  × 768: 14.3 GiB, where an fp32 table is 28.6 GiB on top of the
  harness's own fp32 copy). fp32 scores are what the exact-algorithm gate
  needs: fp16 scores gave `recall_oracle@1000` 0.956 on YFCC-10M, fp32
  scores 0.993 (gate 0.99), an fp32 table 1.0. Cost: the `[B, N]` score
  buffer doubles (+610 MiB at B=16 over 10M items). An fp32 table is the
  next step if a dataset's storage rounding alone breaks the gate. On
  CPU, which has no `out_dtype=` matmul, the operands are cast to fp32
  instead ([kernels](system/kernels.md#score-conventions),
  [artifact](artifacts/l1-l2/README.md)).
- **Int8 quantization uses one global scale**, as the SilverTorch paper
  does.
- **Bloom hashes are keyed on `(clause_idx, value)`**, a deviation from
  the paper that stops equal values in different clauses from colliding
  ([filtering](system/filtering.md#bloom-hash-keys-clause_idx-value)).
- **A library fix reruns only the arms it changes** (user). The records'
  resume key includes the library tree hash (`code_version`), so any edit
  under `retrieve/src/retrieve/` would make `bench campaign --resume`
  rerun everything. Instead the fix's own gates decide which arms it
  changes; those are rerun by narrow `bench run`s, and no record is
  re-stamped ([policy](validation.md#code_version-policy)).

## Harness

- **Timings compare only within one box** (controller, 2026-10-10): the pods
  share the GPU model (A100-SXM4-80GB) but not the driver (570.195 / 570.172
  / 595.91), CPU (EPYC 7763 / Xeon 8470 / EPYC 7742) or power limit (400 / 500
  / 400 W). Every ratio or crossover is read inside one leg on one box; no
  claim compares records from two pods; F-REPRO runs on one pod, whose driver
  / CPU / power limit the provenance page records. The size of the box effect
  is not measured: the first estimate (V2 graph +15-28 % on pod d) also
  crossed a data change (synth tables widened from 7 to 10 clauses, which
  every clause kernel paid for until CLAUSE-SKIP), so it is withdrawn.
- **The baseline is generic torch** (user): a dense matmul over the whole
  item table on the GPU, `torch.topk(K)`, then drop the ids that fail the
  filter, losing candidates from K. It is what a practitioner writes
  without a retrieval library. Beside it, the torch-importable arms of
  Campaign v2 above. No library or ANN baselines (Faiss, HNSW, cuVS) are in
  the study ([backlog](backlog.md#baselines-outside-the-study)).
- **Three packages, one dependency direction**: `bench` → `training` →
  `eval_datasets`, enforced by `tests/test_dependency_direction.py`. The
  library retrieves; the harness measures.
- **`triton` is the algorithms' campaign backend**, because it is the
  fastest arm on every algorithm measured (eager median torch/triton
  4.16× on V1, 7.27× on V2, 10.19× on V3). `silvertorch` runs
  `[triton, official]`, because that comparison is the paper; the torch
  arms return only as the baselines of Campaign v2 above (V1/V2 torch on
  three synth pass rates, the SilverTorch torch reference at the headline
  point), not as a backend matrix.
- **No unfiltered `quality` suite.** The synth suite's p = 1.0 point is
  the unfiltered cell on goodreads, arXiv and YFCC.
- **Modes: `eager` everywhere, `graph` on `triton`**; graph is the
  headline (Campaign v2 above). Graph-only was rejected: the official ops
  cannot be captured. A run with a narrowed mode set records
  `status: partial`, which the report treats as not citable; whether a
  deliberate, recorded narrowing should read differently is open (see the
  roadmap).
- **Clocks cannot be locked** in the container. Records carry the SM clock
  sampled under load after every timing window and an `unstable` flag
  (window spread over 5 %). A batch-size-1 comparison narrower than about
  21 % is noise.
- **Query preparation is outside the timed forward, for every arm** (user,
  2026-10-10): each arm's query-side filter encoding (official plans, bloom
  signatures) runs in its `prepare_queries` before timing and
  is recorded as `query_prep_ms`
  ([evaluation](system/evaluation.md#query-preparation)). It replaces the
  earlier rule that timed official forwards paid the expression parse
  (`cache_plans=False`).
- **Results storage** (user): no results in git. A run appends
  JSONL to a local, gitignored results tree (one `write` + `fsync` per cell,
  and resume reads it back without the network); a finished leg is
  aggregated into Parquet (`results.parquet`, one row per perf entry — the
  table every report reads) and published with its JSONL and samples to the
  private Hub repo `pinkmeme/eval-results` (`bench upload`, `bench fetch`
  back). Raw outputs behind a documented finding go to the same repo under
  `artifacts/<plan>/`; what is neither cited nor needed for re-derivation is
  dropped. Git keeps code, prose, gate reports, the golden cells (a test
  fixture) and the sha256 of each Hub manifest. The `.git` history still
  holds the removed files; rewriting it is a separate decision.

## Datasets

- **The study's datasets: semantic search on text plus one recsys
  dataset** (user): arXiv, YFCC-10M, PubMed + MedCPT, and Goodreads. Each
  has real filters and an open or local query encoder.
- **Dropped**: OpenAlex and Semantic Scholar SPECTER2 (user, 2026-10-08;
  [backlog](backlog.md#datasets-not-in-the-study)); Amazon Reviews 2023; Cohere Wikipedia (scale without
  meaningful filters, closed query encoder); yambda-500m and yambda-5b
  (user: no attributes, so no filtered cells); KuaiRand-27K (user: does
  not fit the study). Their checkpoints and trainer results stay as they
  are.
- **LAION as a 30 M scale point, `filter` only** (user, 2026-10-10;
  roadmap V-LAION30): Re-LAION-2B-en-research-safe captions encoded by us
  (nomic-embed-text-v1.5, Matryoshka d256, a native width of that
  encoder, so "No PCA" holds), metadata tags as filters. 30 M only: no
  100 M point and no fp16-items harness change. At d256 the fp32 items
  are 31 GB; the item-chunked oracle adds no item copy
  ([validation](validation.md#harness-gates), G-oracle).
- **The router stays only if it is on the Pareto front** (user, 2026-10-10):
  between IVF and exact search — recall above IVF's and latency below exact
  search's, at the same batch size and mode — on PubMed 10 M (the large
  dataset; LAION is not used for it). If it is not on the front there, it is
  dropped from the paper and the library (ROUTER-LIB is not built).
  **Verdict (2026-10-11): dropped** — on PubMed 10 M it is slower than
  exact V2 at bs 1 and collapses onto IVF at p ≈ 1; its two bs-16 passes are
  within 2 % of the front (Hub `campaign-v2.5/pubmed-router`). The harness arm
  is deleted (ROUTER-DROP); its records stay on the Hub.
- **YFCC exact gate: an fp16-storage allowance at k 100** (user, 2026-10-10).
  The §2.4 gate (`recall_oracle@k_max ≥ 0.99` for V1 / V2) stays as is
  everywhere else. For the fp16-stored 10 M YFCC catalogs (`yfcc10m`,
  `yfcc10m-synth`) at `k_max` 100, the threshold is a per-dataset value set
  from a quality-only probe, and only if the probe shows the residual is
  fp16 item storage: V1 = V2 = the library's exact torch path on the same
  inputs, and an fp32 item table gives 1.0. The value, the probe and its
  numbers are recorded in validation.md; @1000 keeps 0.99.
- **No PCA.** Every dataset runs at its encoder's native width (YFCC 192,
  PubMed 768); each dataset contributes one width, and the dim ablation is
  dropped.
- **PubMed runs as a 10M slice**, not the full ~36M: the harness holds
  items fp32 on the device (110 GB for the full catalog), while at 10M the
  items, codes and attributes fit in about 40 GB, and 10M matches the
  papers' pool and YFCC's size.
- **YFCC runs clause filters only**, no bloom, so bloom false positives
  cannot spoil the cross-check against the shipped ground truth.

## Sequential encoder

The current trainer's checkpoints replace the published gSASRec ones as the history encoder behind the sequential
benchmarks.
- **Output contract unchanged.** Item embeddings are `[N, D]` from an embedding table, query
  embeddings are `[B, D]` from `encode.py`, and scoring is the dot product. D stays 64, 128 or
  256.
- **The encoder is the gSASRec body** (`SASRecBlock`, the published architecture), with the
  E1c recipe as the trainer's defaults (user). The trainer reads item ids and
  positions only; a `timestamps` column in the data is ignored.
- **Two losses** (user): gBCE with per-position negatives, the published gSASRec
  baseline (one shared vector per step collapses it,
  [evidence](artifacts/seqrec-encoder/gate-b-yambda-d64/README.md)); and sampled softmax on
  L2-normalized embeddings with logQ (temperature 0.05, in-batch positives plus shared uniform
  negatives, expected-count correction, positive uncorrected), the default since E1c
  ([recipe search](validation.md#recipe-search-yambda-500m-d64)).
- **HSTU was tried and dropped.** The softmax-attention HSTU body with time-bucket bias (E2a,
  E2c) did not beat the gSASRec body with logQ (E1c); the user dropped it and its code
  (`HSTUBlock`, `use_time`, `hidden_dim`). Pointwise (softmax-free) attention was
  never pursued ([recipe search](validation.md#recipe-search-yambda-500m-d64)).
- **Success is measured on the same test file with the same eval code.** A new encoder must
  beat the published gSASRec checkpoint of the same D, re-scored on today's
  `trainer/test.parquet`, on test NDCG@10 **and** R@100, on both yambda-500m and
  goodreads-work-id, with one shared recipe. The stored `eval_quality.json` numbers are not the
  bar: yambda's were scored on a test split that is no longer on disk
  ([bars](validation.md#bars)).
- **Runs** (user): the E1c recipe (ffn 4×D) at d64, d128 and d256 on yambda-500m and
  goodreads-work-id, and at d64 on KuaiRand; only D, epochs, patience and eval cadence
  vary ([final models](validation.md#final-models-the-e1c-recipe)).
- **KuaiRand is final at d64, refit on train + val** (`train_on_val=true`, 4 epochs, the
  train-only run's best epoch + 1). The refit, because next-day clicks drift and the val day is
  otherwise never trained on ([temporal drift](validation.md#kuairand-temporal-drift)). d128 was
  not run ([OOM](validation.md#final-models-the-e1c-recipe)). KuaiRand is out of the study
  (Datasets above); the refit stands as a trainer result only.
- **The harness uses the E1c checkpoints** (user): goodreads `sasrec-ssm-logq-d{dim}` (yambda-500m's
  config reads its own, but yambda is out of the study). Records made on a gSASRec checkpoint are kept as that experiment, not deleted and not
  mixed with the new ones; the golden baseline stays on `gsasrec-d128-drop0.5-id`, because it compares
  two harnesses on fixed inputs ([what must be redone](validation.md#encoder-switch-evals-to-redo)).
- **Out of scope for this line:** a LLaMA block, row-wise Adagrad, a bf16 table,
  FuXi-style channels and multi-GPU.

## Environment

- **GPU work runs on RunPod pods** (AGENTS.md rule 1), launched with
  `infra/runpod/pod.sh`; A100-SXM4-80GB is the reference GPU for every
  citable number. One job per GPU at a time; each job has its own
  `TORCHINDUCTOR_CACHE_DIR` ([roadmap](roadmap.md#multi-gpu-execution)).
  The orchestrator runs on the user's laptop or on a pod.
- **Disks**: the repository on `/workspace`, everything large on the
  ephemeral container disk ([storage](system/storage.md)).
- **`ncu` is blocked**; kernel attribution uses `torch.profiler`.
- **The sequential-encoder results were measured on an H100 80GB HBM3 pod**; every number
  records the GPU name.
