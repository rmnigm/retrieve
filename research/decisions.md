---
title: decisions
created: 2026-09-26
updated: 2026-10-17
type: summary
tags: [decisions]
sources: [pyproject.toml, evaluation/config/suites.yaml, retrieve/src/retrieve/, evaluation/bench/, evaluation/training/]
---

# Standing decisions and constraints

Decisions in force, with the reason each was taken. They were made by the
user or measured into place and are not reopened without the user. How the
code implements them is in the [system pages](../docs/index.md#system); the
open work is the [roadmap](roadmap.md). The two process contracts,
[agent orchestration](contracts/agent-orchestration.md) and
[coding guidelines](contracts/coding-guidelines.md), carry their own
decisions.

## Goal and scheduling

- **The study: Meta's official SilverTorch code, adapted, against LiNR**
  (user, 2026-10-11): a reproducibility study of SilverTorch (Meta) and LiNR
  (LinkedIn) on public datasets, one correct harness, up to the papers'
  scale. The paper is kept outside this repository. Its two arms:
  - **SilverTorch** is Meta's published code (`meta-recsys/silvertorch`,
    synced to `22c2007`), adapted by us with a listed set of improvements
    (host syncs removed, CUDA-graph capturable, one-call search, `m_bits`
    honoured, an exact recheck of bloom survivors as an option that is off
    in the study, kernel tuning; each change with its gate). In prose it is
    "Meta's official SilverTorch code, adapted", never "a fork". In code it
    is the `silvertorch/` workspace member under Meta's own names: package
    `silvertorch`, ops `torch.ops.st.*`, `SilverTorch(backend="cuda")`, the
    only backend (RENAME, user 2026-10-16). Records keep the names they were
    written with (`official-fork`, `official`); the readers map them.
  - **LiNR** is V1-V3, our implementation from the paper's text (LiNR has
    no public code).
- **Our Triton SilverTorch reimplementation is out of the study and deleted
  from the code** (user, 2026-10-11), with our LiNR bloom filter (user,
  2026-10-15): the tag `triton-silvertorch-final` holds the code; its
  records stay on the Hub, never cited. LiNR runs the exact filter only; the
  bloom in the study is Meta's, inside SilverTorch.
- **Meta's code as published (pinned `21aa35e`, built `-O3`) was the parity
  reference, not an arm.** The adapted code was gated bit-exact against it;
  it left the workspace at RENAME, and its last parity run stands behind the
  tag `meta-published-parity-final`.
- **The final run is final** (user, 2026-10-13): the FULL-RERUN at tag
  `final-freeze` is the one result set. No code change triggers a rerun of
  its legs. What it did not remeasure (the adapted code's parity gates and
  their timings, the data facts) stands as final and citable with its
  source named. SYNTH-MID (20 / 30 / 50 % synthetic pass rates on LAION
  10 M / 30 M plus a timed `n_probe` sweep) is the one leg added after the
  freeze, at the same tag.
- **The adapted arm's score path is Meta's default, fp16** (user,
  2026-10-13); the int32 path is the parity ablation only.
- **Filtering in the study** (user, 2026-10-12): SilverTorch runs its own
  bloom with false positives allowed and **no recheck in timed code**;
  recall is graded against the exact filtered oracle, so a false positive in
  the top-k counts as a miss. LiNR runs its exact clause filter only.
  SilverTorch's exact clause mode is our adaptation and leaves the
  head-to-head. Reverse (NOT) clauses have no bloom form, so those filters
  are LiNR-exact only. False-positive claims are made only on SilverTorch's
  own bloom, untimed.
- **LiNR's recall is end-task recall, ours is fidelity to exact search**
  (user, 2026-10-12): no LiNR recall claim is called reproduced or not; our
  numbers are stated under our metric (share of the exact filtered top-k
  returned), with the Goodreads correlation to end-task recall beside them.
- **One pod, one venv; pooled seed variance** (user, 2026-10-15): results
  are not split or labelled by machine. SilverTorch's measured seed spread
  is pooled per dataset and applied as the error bar to every SilverTorch
  number on that dataset; a dataset without seeds uses the pool over all
  datasets; LiNR keeps what it has.
- **Scope: model-based retrieval** (user, 2026-10-10): retrieval as torch
  ops inside the model graph, the frame both papers use. Baselines are what
  a practitioner can put in that graph; standalone ANN libraries (Faiss,
  cuVS, HNSW, filtered-graph indexes) are out of the study.
- **The project ships a library plus a benchmark** (user, 2026-10-10): an
  open-source library of the methods and an easy way to benchmark them on
  more datasets (`eval-data ingest`).
- **V3's bit budget is one bit per coordinate** (user, 2026-10-10): OPORP
  gives at most D bits; no `k_bits` > D.
- **The export** (user, 2026-10-15/16): the exported repository carries
  the code, the human docs (`README.md`, `docs/`, `retrieve/docs/`) and an
  AGENTS.md that points at nothing that is gone; no `articles/`, no
  `.chains/`, no `research/`. It provides the means to run evaluations and
  get the numbers (`bench run` / `campaign` / `merge` / `report`) and
  reproduces no paper table or figure.
- **The private Hub stays as it is** (user, 2026-10-16):
  `pinkmeme/eval-results` keeps its superseded subtrees; a clean public
  repository may come at F4.
- **Nothing is citable until its gate passed** (AGENTS.md rule 2);
  [validation](validation.md) says which gates have passed.

## Campaign design

The rules the final run was planned and run under.

- **Datasets**: goodreads (E1c, 0.8 M, d128), arXiv (3 M, d128), PubMed
  (10 M, d768), LAION at 1 / 3 / 10 / 30 M (d256, `filter` only) and
  Yambda-500m (unfiltered).
- **Three kept sweeps per dataset** (one high pass rate, one low, one
  conjunctive or reverse): goodreads `c0_genre`, `c1_lang_reverse`, `all4`;
  arXiv `c3_nversions` (0.444), `c0_maincat` (0.136), `all4` (0.0094);
  PubMed `c0_mesh` (0.0002), `c3_journal_reverse` (0.9993), `all5` (0.018).
- **A synthetic selectivity suite** (`synth`): a nested, per-item uniform
  pass flag (`u_i < p`, seed 20261008) at the ten rates of
  `layout.SYNTH_RATES` (0.001 to 1) on the real embeddings, with seeded
  draws as a seed axis. It is IVF's worst case (the filter is independent
  of the embedding); p = 1.0 is the unfiltered point. The
  cluster-correlated variant (`arxiv-corr-synth`) is IVF's best case.
- **Grid**: batch sizes {1, 16} (bs 64 in the large-scale and timed-tune
  legs), k {100, 1000} (1000 is the closest to LiNR's 2000). Seed 0
  everywhere; seeds 1-2 on the timed bloom legs, where a build, a pool or a
  synthetic draw is random. The deterministic arms (V1, V2, postfilter)
  compute quality once and reuse it across seeds. Postfilter α ∈ {1, 8}.
- **SilverTorch at its tuned point per dataset size** (user, 2026-10-11):
  per-filter probe curves, the matched-recall point at recall 0.90 and 0.95
  (0.99 a stretch); a fixed `n_probe` across 0.8-30 M compares different
  recall levels, so the paper's `n_probe` 24 is one reference row only.
  Every SilverTorch cell scans at most a quarter of the lists (`n_probe` ≤
  `n_lists` / 4); at 10-30 M `n_lists` is swept over {4096, 16384, 65536}.
- **IVF tuned per dataset size** (user, 2026-10-08): `n_lists` and
  `n_probe` depend on N and are tuned, not fixed at the library default;
  tuning is quality-only, seed 0, and its records are never paper numbers.
  Picks in [validation](validation.md#inputs-to-the-final-run-method-not-results).
- **Timing**: CUDA-graph is the headline mode, eager the secondary. Every
  ratio claim is timed interleaved: the arms of one comparison run
  round-robin in one process, so clock drift cancels.
- **Baselines stay torch-importable** (user): the generic-torch
  postfilter, LiNR V1 / V2 on the torch backend (the native floor) and a
  `torch.compile(mode="max-autotune")` arm of LiNR's torch-reference ops.
- **LiNR V4 left the library** (user): it is ours, not LiNR's; tag
  `linr-v4-final`.
- **Claims drive cells**: every planned cell maps to a figure or table row;
  a cell that maps to none is cut.

## Library

- **SilverTorch ships as Meta's official code, adapted** (Goal above): the
  `silvertorch/` uv workspace member (Apache-2.0, Meta's `LICENSE` and file
  headers kept, each changed file marked, every change a row of its
  `CHANGES.md` with its gate in [validation](validation.md#adapted-silvertorch-gates)),
  a dependency of `torchretrieve` reached through `SilverTorch(backend="cuda")`.
  Our phase 1 (k-means, int8 codes, probe selection) stays ours.
- **Our kernel effort goes to LiNR and to the adapted SilverTorch code.**
  The hand-written CUDA C++ and CuTe DSL SilverTorch backends were deleted
  once the official backend's parity gate was green (tag
  `cuda-cute-backends-final`); the Triton SilverTorch backend followed (Goal
  above).
- **Meta's shape**: `retrieve.modules` (the `nn.Module`s and builders) and
  `retrieve.ops` (registered kernels, one namespace per backend), plus
  `retrieve.indexing` and `retrieve.functional`. Meta's code never enters
  `retrieve/`; the adapted version is its own workspace member. See
  [architecture](../docs/system/architecture.md).
- **LiNR V1-V3 are library modules**; the harness keeps only a name to
  class table.
- **Op names, module names, buffer names and op schemas are stable.** A
  state dict written by an earlier release loads into the current modules.
- **k-means++ is opt-in** (`kmeans_init="random"` is the default): every
  recorded number was taken on random init, and k-means++ flipped no verdict
  on LAION 30 M ([validation](validation.md#inputs-to-the-final-run-method-not-results)).
- **Stream compaction is deterministic**: survivors come out in ascending
  item order on both backends, and a rerun is byte-identical on the
  `[:counts]` prefix ([kernels](../docs/system/kernels.md)). Chosen over a
  wider quality tolerance because two golden cells could not reproduce
  themselves.
- **LiNR's exact scorers store items fp16 and return fp32 scores**
  (`PostfilterKNN`, `PrefilterKNN`, every backend): fp16 storage is the
  LiNR paper's and keeps the item table at `N × D × 2` bytes; fp32 scores
  are what the exact-algorithm gate needs (fp16 scores missed it at 10 M).
  Cost: the `[B, N]` score buffer doubles. On CPU the operands are cast to
  fp32 instead ([kernels](../docs/system/kernels.md#score-conventions)).
- **Int8 quantization uses one global scale**, as the SilverTorch paper
  does.

## Harness

- **Timings compare only within one box** (2026-10-10): the pods shared the
  GPU model but not the driver, CPU or power limit, so every ratio or
  crossover is read inside one leg on one box.
- **The baseline is generic torch** (user): a dense matmul over the whole
  item table, `torch.topk(K)`, then drop the ids that fail the filter. No
  library or ANN baselines.
- **Three packages, one dependency direction**: `bench` → `training` →
  `eval_datasets`, enforced by `tests/test_dependency_direction.py`. The
  library retrieves; the harness measures.
- **`triton` is LiNR's campaign backend**: the fastest arm on every LiNR
  algorithm measured (eager median torch / triton 4.16× on V1, 7.27× on V2,
  10.19× on V3). The torch arms return only as baselines.
- **No unfiltered `quality` suite.** The synth suite's p = 1.0 point is the
  unfiltered cell; Yambda runs `filter_kind none`.
- **Modes: `eager` everywhere, `graph` on every capturable arm.** A run
  with a narrowed mode set records `status: partial`, which the report
  treats as not citable.
- **Clocks cannot be locked** in the container. Records carry the SM clock
  sampled under load after every timing window and an `unstable` flag
  (window spread over 5 %). A batch-size-1 comparison narrower than about
  21 % is noise.
- **Batch scaling has one listed exception** (user, 2026-10-16): the
  compiled torch baseline arm's bs-1 fast path (goodreads-synth LiNR V2
  torch `max-autotune`, `p1`, k 1000) fails `median_ms(bs 16) < 16 ×
  median_ms(bs 1)`; the check is not loosened.
- **Query preparation is outside the timed forward, for every arm** (user,
  2026-10-10): each arm's query-side filter encoding runs in its
  `prepare_queries` and is recorded as `query_prep_ms`
  ([evaluation](../docs/system/evaluation.md#query-preparation)).
- **Results storage** (user): no results in git. A run appends JSONL to a
  local, gitignored results tree; a finished leg is aggregated into
  `results.parquet` and published with its JSONL and samples to the private
  Hub repo `pinkmeme/eval-results` (`bench upload`, `bench fetch` back).
  Raw outputs behind a documented finding go to the same repo under
  `artifacts/<plan>/`. Git keeps code, prose, gate reports, the golden
  cells (a test fixture) and the sha256 of each Hub manifest. The `.git`
  history still holds files removed from `HEAD`; rewriting it is a separate
  decision.

## Datasets

- **The study's datasets: semantic search on text plus recsys** (user):
  arXiv, PubMed + MedCPT, Goodreads, Re-LAION at 1-30 M and Yambda-500m.
  Each filtered dataset has real filters and an open or local query
  encoder.
- **Out of the study**: YFCC-10M (user, 2026-10-13: its uint8 CLIP image
  descriptors are not trained for inner product; every YFCC result is
  unusable), OpenAlex and Semantic Scholar SPECTER2 (same N, width and
  domain as PubMed), Amazon Reviews 2023, Cohere Wikipedia, yambda-5b and
  KuaiRand-27K. None of them has code in the harness (user, 2026-10-17)
  or a suite.
- **Yambda-500m in, unfiltered** (user, 2026-10-13): how recall against the
  exact top-k tracks end-task recall (held-out Listen+ events) across dims;
  SilverTorch adapted only, an `n_lists` / `n_probe` sweep at d64 / d128 /
  d256 on the E1c checkpoints, the first 10,000 test users, an exact scan as
  the reference.
- **LAION up to 30 M, `filter` only** (user, 2026-10-10):
  Re-LAION-2B-en-research-safe captions encoded by us (nomic-embed-text-v1.5,
  Matryoshka d256, a native width of that encoder), metadata tags as
  filters; no 100 M point.
- **PubMed scores by dot product** (user, 2026-10-13): MedCPT is a
  dot-product model, so its vectors are stored as shipped
  (`normalize: false`).
- **PubMed runs as a 10 M slice**, not the full ~36 M: at 10 M the items,
  codes and attributes fit in about 40 GB, and 10 M matches the papers'
  pool size.
- **No PCA.** Every dataset runs at its encoder's native width.
- **The router is dropped** (2026-10-11): on PubMed 10 M it was not on the
  recall-latency Pareto front between IVF and exact search.

## Sequential encoder

The trainer's checkpoints replace the published gSASRec ones as the history
encoder behind the sequential benchmarks.

- **Output contract.** Item embeddings are `[N, D]` from an embedding
  table, query embeddings `[B, D]` from `encode.py`, scoring is the dot
  product. D is 64, 128 or 256.
- **The encoder is the gSASRec body** (`SASRecBlock`), with the E1c recipe
  as the trainer's defaults (user). The trainer reads item ids and
  positions only; a `timestamps` column is ignored.
- **Two losses** (user): gBCE with per-position negatives (the published
  baseline; one shared vector per step collapses it,
  [evidence](artifacts/seqrec-encoder/gate-b-yambda-d64/README.md)); and
  sampled softmax on L2-normalized embeddings with logQ (temperature 0.05,
  in-batch positives plus shared uniform negatives), the default
  ([recipe search](validation.md#recipe-search-yambda-500m-d64)).
- **HSTU was tried and dropped**, with its code: it did not beat the
  gSASRec body with logQ.
- **Success is measured on the same test file with the same eval code**:
  a new encoder must beat the published gSASRec checkpoint of the same D,
  re-scored on today's `trainer/test.parquet`, on test NDCG@10 **and**
  R@100, on yambda-500m and goodreads-work-id, with one shared recipe
  ([bars](validation.md#bars)).
- **The harness uses the E1c checkpoints** (user): goodreads and
  yambda-500m `sasrec-ssm-logq-d{dim}`. The golden baseline stays on
  `gsasrec-d128-drop0.5-id`, because it compares two harnesses on fixed
  inputs.
- **Out of scope:** a LLaMA block, row-wise Adagrad, a bf16 table,
  FuXi-style channels and multi-GPU training.

## Environment

- **GPU work runs on RunPod pods** (AGENTS.md rule 1), launched with
  `infra/runpod/pod.sh`; A100-SXM4-80GB is the reference GPU for every
  citable number. One job per GPU at a time, each with its own
  `TORCHINDUCTOR_CACHE_DIR` ([roadmap](roadmap.md#multi-gpu-execution)).
- **Disks**: the repository on `/workspace`, everything large on the
  ephemeral container disk ([pod storage](environment/storage.md)).
- **`ncu` is blocked**; kernel attribution uses `torch.profiler`.
- **The sequential-encoder results were measured on an H100 80GB HBM3
  pod**; every number records the GPU name.
