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
code implements them is in the [system pages](../docs/index.md#system); the open
work is the [roadmap](roadmap.md). The two process contracts,
[agent orchestration](contracts/agent-orchestration.md) and
[coding guidelines](contracts/coding-guidelines.md), carry their own
decisions.

## Goal and scheduling

- **The study: Meta's official SilverTorch code, adapted, against LiNR**
  (user, 2026-10-11). The paper is a reproducibility study of SilverTorch
  (Meta) and LiNR (LinkedIn) on public datasets, one correct harness, up to
  the papers' scale. Its two arms:
  - **SilverTorch** is Meta's officially published code
    (`meta-recsys/silvertorch`, synced to `22c2007`), adapted by us with a
    listed set of improvements (host syncs removed, CUDA-graph capturable,
    one-call search, `m_bits` honoured, an exact recheck of bloom
    survivors as an option that is off in the study, kernel tuning; each
    change with its gate). In prose it is
    "Meta's official SilverTorch code, adapted", never "a fork". In code it
    is still the `official-fork/` workspace member, package
    `silvertorch_fork`, `backend="official-fork"`, until roadmap
    [RENAME](roadmap.md#5-rename-the-adapted-backend-becomes-silvertorchs-default).
  - **LiNR** is V1-V3, our implementation from the paper's text (LiNR has
    no public code). Its kernels stay, whatever they are written in.

  The comparison of the two is the paper.
- **Our Triton SilverTorch reimplementation is out of the paper and out of
  the released code** (user, 2026-10-11): `SilverTorch(backend="triton")`
  and `backend="torch"` (our Algorithm 1 kernels and their eager
  reference) are neither a result nor a shipped backend. Until its
  deletion step the code stays in the repository, recorded and unused; it
  is deleted under CLAUDE.md rule 5 only after the adapted backend's review
  ACCEPT and its NUMBERS legs, behind the tag `triton-silvertorch-final`
  ([roadmap](roadmap.md#6-triton-st-delete-delete-the-triton-silvertorch-reimplementation)).
  Its records stay on the Hub (`campaign-final` and earlier), never
  deleted, never cited. No new Triton SilverTorch kernel work.
- **Meta's code as published (`backend="official"`, pinned `21aa35e`,
  built `-O3`) is the parity reference, not an arm.** The adapted code is
  gated bit-exact against it, and the paper's "Meta as published vs
  adapted" table ([paper](paper/meta-published-vs-adapted.md)) reports
  what the adaptation changed. Meta's code as published is not a separate
  arm of the study.
- **The adapted arm's score path is Meta's default, fp16** (user,
  2026-10-13; the earlier "int32 score path" record, the controller's, is
  withdrawn): every study number of the adapted code (curves, timed legs,
  bloom legs, ceiling split) ran the fp16 path, the library default, and is
  reported as such; no reruns. The int32 path stays only as the C7 parity
  ablation (bit-exact against Meta's published code).
- **Filtering in the study** (user, 2026-10-12; this replaces the earlier
  record that the C13 recheck was on by default "by controller and user",
  which the user does not stand behind): SilverTorch is evaluated with its
  own fast probabilistic filter, the bloom, **with false positives
  allowed** and **no recheck in timed code** (`OfficialConfig.recheck`
  off); the question is how much the bloom's false positives cost recall,
  and recall is graded against the exact filtered oracle, so a false
  positive in the top-k counts as a miss. LiNR is evaluated with its exact
  clause filter only. SilverTorch's exact clause mode is our adaptation (not in
  Meta's CUDA path) and leaves the head-to-head. The C13 recheck stays in
  the code as an option, off in the study; the library default changes at
  RENAME. Reverse (NOT) clauses have no bloom form, so those filters are
  LiNR-exact only.
- **LiNR with a bloom filter is not in the study** (user, 2026-10-13): LiNR
  has no bloom in its paper, and our bloom for it (the library's
  `BloomFilter`) is not reported, compared or mentioned in the paper.
  The bloom legs time SilverTorch's bloom against LiNR V1 / V2 with the
  exact filter only. The library keeps the code; nothing here measures it.
- **False-positive claims only on SilverTorch's own bloom** (user,
  2026-10-13): measured on the adapted code's bloom index with no recheck,
  untimed; a recheck exists only in grading.
- **D1-G is green for the final pass** (user, 2026-10-11): the user
  accepted the 25 listed, explained failures of the batching check
  (`median_ms(bs=16) < 16 × median_ms(bs=1)`) without loosening it; the
  checks and the entries are in
  [validation](validation.md#campaign-v2-phase-v) (D1-G).
- **NUMBERS gate green** (controller ACCEPT, 2026-10-12), on the D1-G
  model, with the user's rulings: LiNR V2 at bs 256 on
  PubMed `all5` reported as TIMEOUT (no result at 80 GB; `c3_journal_reverse` finished at 2.55 s a batch and is reported); sub-millisecond cells unstable in
  every run kept and flagged.
- **LiNR's recall is end-task recall, ours is fidelity to exact search**
  (user, 2026-10-12): LiNR reports product recall (recall@2000 against its
  labels) with a protocol not published in enough detail to rebuild, so no
  LiNR recall claim is called reproduced or not; our numbers are stated
  under our metric (share of the exact filtered top-k returned), with the
  Goodreads correlation to end-task recall beside them. The adapted SilverTorch numbers of the NUMBERS legs
  are citable ([validation](validation.md#numbers-adapted-silvertorch-over-the-study-grid)).
- **Paper numbers, and the full rerun last** (user, 2026-10-11): the paper
  is written on measurement legs first: SilverTorch on the adapted backend
  over the study grid (roadmap
  [NUMBERS](roadmap.md#2-numbers-adapted-silvertorch-over-the-study-grid)),
  LiNR from `campaign-final`. The full rerun of everything (LiNR, the
  adapted SilverTorch, the grid), pinned to the last repo version, into one
  parquet with its recipe, D1-G, T1 and T3, is the very last step
  ([FULL-RERUN](roadmap.md#7-full-rerun-everything-pinned-to-the-last-repo-version-the-very-last-step));
  the paper's final tables are regenerated from it.
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
  2026-10-09, user away and delegating): Meta's code as published is
  host-bound in eager (device kernels 0.18-0.31 ms of 1.1-1.85 ms end to
  end at v2.1; its windows read ~1140 MHz because the GPU idles, and its
  end-to-end moves with host load at the same device time), so
  clock-normalising would correct a symptom. T3 (now Meta as published vs
  adapted, [claims](paper/claims.md)) gives kernel-only and end-to-end side
  by side, as measured, with the host-bound mechanism stated; F4b
  (co-design, partial vs full on one backend, interleaved) is reported as
  measured.
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
  goodreads and arXiv (capped at the dataset's
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
- **Datasets**: goodreads (E1c, 0.8M, d128), arXiv (3M, d128),
  PubMed (10M, d768); YFCC-10M left the study on 2026-10-13
  ([Datasets](#datasets)). **OpenAlex is dropped** (same
  N, width and domain family as PubMed; the leg was E5;
  [backlog](backlog.md#datasets-not-in-the-study)). No point above 10M
  before submission; the chunked oracle is still built.
  *Contested:* LAION 30 M joins as a fifth, `filter`-only dataset (user,
  2026-10-10, [Datasets](#datasets)).
- **Three kept sweeps per dataset** in the headline grid (one high pass
  rate, one low, one conjunctive or reverse): goodreads `c0_genre`,
  `c1_lang_reverse`, `all4`; arXiv `c3_nversions` (0.444), `c0_maincat`
  (0.136), `all4` (0.0094); PubMed `c0_mesh` (0.0002),
  `c3_journal_reverse` (0.9993), `all5` (0.018). arXiv
  records already run on the other sweeps are kept, not extended.
- **A synthetic selectivity suite** (`synth`): one nested, per-item uniform
  pass flag (`u_i < p`, generator seed 20261008) at
  p ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0} on goodreads and arXiv,
  on the real embeddings,
  as sibling dataset configs that never touch the real attrs. It is IVF's
  worst case (the filter is independent of the embedding), and p = 1.0 is
  the unfiltered point. *Contested:* the roadmap's SYNTH-TRIM (user,
  2026-10-10) sets ten rates and a smaller grid (arXiv {0.001, 0.01, 0.05,
  0.1, 0.2, 0.5, 1.0}, seed 0 with seeds
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
  ([evaluation](../docs/system/evaluation.md#ivf-tuning)).
- **SilverTorch is reported at its tuned point per dataset size** (user,
  2026-10-11): the matched-recall point from the IVF tune (below), with the
  recall each filter and k actually reaches, read off per-filter probe
  curves (roadmap NUMBERS item 1). The paper's `n_probe` 24 is one
  reference row only, not the headline: a fixed `n_probe` across 0.8-30 M
  compares different recall levels. Recall targets reported: 0.90 and 0.95,
  0.99 a stretch (user, 2026-10-11). At 10-30 M the cluster count is swept
  (`n_lists` {4096, 16384, 65536}, `n_probe` doubling to the 25 % cap, the
  k 1000 check per `n_lists`) and the tuned point per size picked from it
  (user, 2026-10-11; roadmap NUMBERS item 2).
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
  sets arXiv (3 M). The `filter` and `synth` SilverTorch
  arms of each dataset run at its `n_lists` with `n_probe` {24, n95}.
  **Cap** (user, 2026-10-09): tuning stops at `n_probe` = `n_lists`/4
  (25 % scanned); if 0.95 is not reached by then, the record says so with
  the recall reached, and the slot takes `n_lists`/4. n95 follows the
  filter's pass rate more than N (goodreads at pass 0.33: 1/64 of the
  lists; PubMed at 0.018: not reached at 1/16), so the size rule sets only
  `n_lists` (≈ 4√N to a power of two: goodreads 4096, arXiv 8192,
  PubMed 16384); arXiv's n95 comes from a **quick check**
  (user): that one `n_lists`, quality-only `n_probe` doubling on the
  median sweep, seed 0, same cap. Measured picks: goodreads 4096 / 64, arXiv 2048 / 256
  (2048-8192 all reach 0.95 at ≈ 12.7 % scanned), PubMed 4096 /
  1024 = the cap, 0.95 not reached (0.873): at k 1000 the probe
  scorers run `n_probe` ≤ 1024, which makes 4096 the largest `n_lists` whose
  25 % cap runs. These picks hold for the final pass's records; for the
  adapted backend at 10-30 M the `n_lists` sweep under the operating
  point above replaces them.
- **Meta's code as published runs only on `none` and `bloom`.** Its
  clause and exact cells time our `pack_mask` adapter, not Meta's code; the
  existing official-clause records stay as an "adapter-bound upper bound"
  footnote. The adapted code runs `exact` inside its own search op (its
  CHANGES row C9), so the study's SilverTorch arm has all three filter
  modes.
- **Timing**: CUDA-graph is the headline mode, eager the secondary. The
  adapted backend captures (its compile and capture gates); Meta's code as
  published is eager-only (its ops cannot be captured), so "Meta as
  published vs adapted" compares eager with eager and says so.
  Every ratio claim (C1, C5, C7) is timed interleaved: the arms of one
  comparison run round-robin (ABAB) in one process, so clock drift cancels.
- **Baselines stay torch-importable; no Faiss** (user): the generic-torch
  postfilter, LiNR V1/V2 on the torch backend (the native floor) and a
  `torch.compile(mode="max-autotune")` arm of LiNR's torch-reference ops.
  The SilverTorch torch-reference backend (unfused IVF) leaves with the
  Triton reimplementation it mirrors (Goal above). The paper states the
  scope once in the setup and once in threats to validity.
- **LiNR V4 leaves the library** (user): it is ours, not LiNR's. The code
  stays under the git tag `linr-v4-final`.
- **T3 is Meta as published vs adapted**, from the `h2h` cells (NUMBERS,
  then FULL-RERUN)
  (the adapted backend against `backend="official"` at `-O3`, interleaved,
  both `score_path` modes: fp16, Meta's default; int32, parity). b3,
  the kernel-opt head-to-head and H2H-FINAL (official vs our Triton) are
  history: not in the paper.
- **Execution**: code and GPU testing run on the 1×A100 development pod;
  the long evaluations run on pods the user creates, one sequential
  timed driver on GPU 0, quality-only work on GPU 1, and timed work on a
  second GPU only if M1 passes.

## Library

- **SilverTorch ships as Meta's official code, adapted** (Goal above):
  the `official-fork/` uv workspace member (Apache-2.0, Meta's `LICENSE`
  and file headers kept, each changed file marked, every change a row of
  its `CHANGES.md` with its gate in [validation](validation.md)), reached
  through `SilverTorch(backend="official-fork")` and the `official` extra.
  Our phase 1 (k-means, int8 codes, probe selection) stays ours, as for
  `backend="official"`.
- **Meta's `meta-recsys/silvertorch` as published is the parity
  reference**, `backend="official"`, installed by the `official` extra and
  pinned to one commit (`21aa35e`, see [pyproject.toml](../pyproject.toml)).
  Bumping the pin is a deliberate change that reruns the official parity
  gate.
- **Our kernel effort goes to LiNR and to the adapted SilverTorch code.**
  The hand-written CUDA C++ and CuTe DSL SilverTorch backends were deleted
  once the official backend's parity gate was green (tag
  `cuda-cute-backends-final`); the Triton SilverTorch backend follows the
  same path (Goal above).
- **Meta's shape**: `retrieve.modules` (the `nn.Module`s and builders) and
  `retrieve.ops` (registered kernels, one namespace per backend: `triton`,
  `reference`, `official`), plus `retrieve.indexing` and
  `retrieve.functional`. Meta's code never enters `retrieve/`: the
  published version installs from its pin, the adapted version is its own
  workspace member. See [architecture](../docs/system/architecture.md).
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
  ([kernels](../docs/system/kernels.md)). Chosen over a wider quality tolerance
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
  instead ([kernels](../docs/system/kernels.md#score-conventions),
  [artifact](artifacts/l1-l2/README.md)).
- **Int8 quantization uses one global scale**, as the SilverTorch paper
  does.
- **Bloom hashes are keyed on `(clause_idx, value)`**, a deviation from
  the paper that stops equal values in different clauses from colliding
  ([filtering](../docs/system/filtering.md#bloom-hash-keys-clause_idx-value)).
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
- **`triton` is LiNR's campaign backend**, because it is the fastest arm
  on every LiNR algorithm measured (eager median torch/triton 4.16× on V1,
  7.27× on V2, 10.19× on V3). `silvertorch` runs the adapted backend
  (`official-fork`), with `official` (Meta as published) beside it only
  where a cell feeds the parity gate or T3 (Goal above). The torch arms
  return only as the baselines of Campaign v2 above (V1/V2 torch on three
  synth pass rates), not as a backend matrix. The final pass's
  `silvertorch` `triton` and `torch` arms are records, not results.
- **No unfiltered `quality` suite.** The synth suite's p = 1.0 point is
  the unfiltered cell on goodreads and arXiv (and LAION at 1-30 M).
- **Modes: `eager` everywhere, `graph` on every capturable arm** (LiNR
  `triton`, the adapted SilverTorch backend); graph is the headline
  (Campaign v2 above). Graph-only was rejected: Meta's ops as published
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
  ([evaluation](../docs/system/evaluation.md#query-preparation)). It replaces the
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
  dataset** (user): arXiv, PubMed + MedCPT, and Goodreads, plus Re-LAION
  at 1-30 M (below). Each has real filters and an open or local query
  encoder.
- **YFCC-10M is dropped, with no substitute** (user, 2026-10-13). Its
  vectors are the Big-ANN filtered track's uint8 CLIP image descriptors,
  whose shipped ground truth is squared L2; they are not trained for inner
  product, and under inner product their distribution is not a real one
  (after L2 normalisation a random pair's cosine is 0.97). Every YFCC
  result is unusable: yfcc10m and yfcc10m-synth recall, SilverTorch
  curves, timed cells, the int8 ceiling split, the fp16-storage gate
  allowance (HB-6), the precision analysis, the LiNR timeouts. The oracle
  check against its shipped GT (HB-2) and its four lines of the META-FORK
  parity grid go too: the paper never names YFCC. The 10 M point is
  PubMed and LAION 10 M, real and synthetic. Its ETL, configs and Hub copy
  stay; it is in no suite. A PubMed synthetic-filter leg joins FULL-RERUN.
- **Dropped**: OpenAlex and Semantic Scholar SPECTER2 (user, 2026-10-08;
  [backlog](backlog.md#datasets-not-in-the-study)); Amazon Reviews 2023; Cohere Wikipedia (scale without
  meaningful filters, closed query encoder); yambda-5b (no attributes, and
  its checkpoints are gSASRec, not E1c); KuaiRand-27K (user: does not fit
  the study). Their checkpoints and trainer results stay as they are.
- **Yambda-500m in, unfiltered** (user, 2026-10-13): how recall against the
  exact top-k tracks end-task recall (held-out Listen+ events), across
  dims. SilverTorch adapted only, `filter_kind none` (Yambda has no
  attributes), an n_lists / n_probe sweep at d64 / d128 / d256 on the E1c
  checkpoints, the first 10,000 test users; an exact scan as the
  reference. The abstract's "five public datasets" counts it in place of
  YFCC.
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
- **No PCA.** Every dataset runs at its encoder's native width (PubMed
  768, LAION 256); each dataset contributes one width, and the dim ablation is
  dropped.
- **PubMed runs as a 10M slice**, not the full ~36M: the harness holds
  items fp32 on the device (110 GB for the full catalog), while at 10M the
  items, codes and attributes fit in about 40 GB, and 10M matches the
  papers' pool size.

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
- **The harness uses the E1c checkpoints** (user): goodreads and yambda-500m `sasrec-ssm-logq-d{dim}`. Records made on a gSASRec checkpoint are kept as that experiment, not deleted and not
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
  ephemeral container disk ([pod storage](environment/storage.md)).
- **`ncu` is blocked**; kernel attribution uses `torch.profiler`.
- **The sequential-encoder results were measured on an H100 80GB HBM3 pod**; every number
  records the GPU name.
