# Future work & research directions

> **Status:** idea catalog (2026-07-03). Not a commitment queue — a vetted menu. Each entry
> names the repo components it builds on, the external prior art, the expected payoff, and a
> rough cost. Complements the execution plans:
> [kernels-layers-design.md](kernels-layers-design.md) /
> [evaluation-refactor.md](evaluation-refactor.md) (refactor track),
> [torch-export-refactor.md](torch-export-refactor.md) and
> [live-update-api.md](live-update-api.md) (deferred features),
> [00-roadmap.md](00-roadmap.md) Stage 3/4 (kernel optimizations, thesis measurements).
>
> Entries marked **[thesis]** are candidate thesis/paper contributions with a clear ablation
> story; entries marked **[eng]** are engineering value without novelty claims.

---

## Part I — Engineering improvements

### I1. GPU CI **[eng]**

The correctness + parity suite is GPU-only (skipped without CUDA via
[tests/conftest.py](../../retrieve/tests/conftest.py)); nothing runs it automatically —
`.pre-commit-config.yaml` covers lint only. Every refactor phase in the companion plans leans
on "run the suite on a dev GPU". Options, cheapest first:

1. A `just test-gpu` / documented manual gate + a PR checklist line (status quo, formalized).
2. Scheduled nightly on a rented GPU runner (Modal / Lambda / a lab box with a self-hosted
   GitHub runner): `uv sync && pytest retrieve/tests -x -q` + the eval unit tests. ~30 min of
   A10 time per night.
3. Same runner, plus the golden-cell regression from
   [evaluation-refactor.md §Conventions](evaluation-refactor.md) (one `linr_v3` filter cell,
   asserting byte-identical quality columns).

Recommendation: (2) now, (3) after the eval refactor lands. Cost: a day of setup.

### I2. Fused / in-kernel top-K selection **[eng → thesis-adjacent]**

The repo's standing design rule is "top-K is never in-kernel; host `torch.topk` (CUB) beats
anything we can write in pure Triton without a warp-level radix-select primitive"
([kernels.md](../system/kernels.md)). That was measured for *this* codebase, but the external
evidence says the ceiling is higher: RadiK's optimized radix select reports large-k wins over
prior GPU selection ([RadiK, arXiv 2501.14336](https://arxiv.org/abs/2501.14336)), RTop-K
reports up to 4.8× over `torch.topk` on row-wise selection
([RTop-K, ICLR 2025](https://arxiv.org/pdf/2409.00822)), and AIR Top-k's iteration-fused
design minimizes memory traffic ([SC'23 study](https://dl.acm.org/doi/10.1145/3581784.3607062)).

Concrete experiment (bounded, one week): for the two hottest consumers —
`codesigned_probe_score*` (scores `[B, P]`, P = n_probe × max_cluster_size, then host topk)
and `oporp_1bit_match_topk` full-scan (`[B, N]` scores) — benchmark (a) current host
`torch.topk`, (b) cuVS/RAFT's `select_k` via `pylibraft` on the same score buffer, (c) a
two-pass Triton threshold-then-compact sketch. Decision rule: only replace the host topk if
end-to-end cell latency (eval harness bs=1/16 rows) improves ≥ 10% — the score-buffer write is
often the true bottleneck, in which case *fusing selection into the scoring kernel* (avoiding
the `[B, P]` HBM round-trip entirely) is the more interesting follow-up and becomes
thesis-adjacent (it changes the memory-traffic model of Algorithm 1's Phase 4).

### I3. Hardware popcount + allocator hygiene (roadmap Stage 3, quantified) **[eng]**

Stage 3's two deferred items stay valid; adding measurement targets so they can be
accepted/rejected quickly:

- Swap the 5-step SWAR `_popcount_int64` for PTX `popc.b64` via
  `tl.extra.cuda.libdevice` / inline asm in `oporp_1bit_match_topk`. Expected: modest —
  the kernel is bandwidth-bound at large N (score write dominates); measure with the tuner
  regimes; accept if ≥ 5% on the full-scan path at N ≥ 1M. Keep the SWAR twin for the torch
  reference (bit-exact pairing is load-bearing).
- Allocator hygiene in `fused_masked_knn_topk`/`oporp` host tails (kill remaining `cat`-pads
  in favor of preallocate+slice) — after [kernels-layers-design.md](kernels-layers-design.md)
  K2/K4.1 these all flow through two helpers, so this becomes a two-site change.

### I4. Persistent kernels for small-batch serving **[eng → thesis-adjacent]**

All kernels launch one grid per forward; at bs=1–4 (the online-serving regime the thesis
targets) launch + tile-scheduling overhead is a visible fraction of the sub-millisecond
latencies in the results. VecFlow explicitly credits **persistent kernels** for its
small-batch streaming throughput ([VecFlow, SIGMOD'25](https://arxiv.org/abs/2506.00812)).
A persistent variant of `codesigned_probe_score` (grid = #SMs, work-queue over (b, tile)
cells) is a contained experiment; the eval harness's bs=1 rows give the before/after directly.
Interaction to respect: the current design already amortizes launches via cudagraph capture
(`mode="reduce-overhead"`), so measure against *captured* baselines — the win may be small
until capture is off (e.g., dynamic shapes in production serving).

### I5. Quantization upgrades: RaBitQ, int4, PQ **[thesis]**

The library ships three quantizers (per-row int8, global int8, Sign-OPORP 1-bit, plus SimHash).
Two upgrades with strong prior art:

- **RaBitQ** ([SIGMOD'24](https://dl.acm.org/doi/10.1145/3654970),
  [arXiv 2405.12497](https://arxiv.org/abs/2405.12497)): random-rotation binary codes with an
  **unbiased distance estimator and a sharp theoretical error bound** — precisely what
  Sign-OPORP/SimHash lack (the thesis currently justifies them empirically). Implementation
  fits the existing infrastructure almost exactly: codes are D-bit strings (same `[N, W]`
  int64 packing, same popcount kernel), plus per-item norms and a rescale in the estimator —
  i.e., a third subclass of the planned `_PackedBitsKNN` base
  ([kernels-layers-design.md](kernels-layers-design.md) K4.2) with a modified score epilogue.
  The extended multi-bit variant gives a knob comparable to `SimHashKNN.k_bits`. Ablation:
  RaBitQ vs OPORP vs SimHash on the existing yambda/goodreads/arxiv quality configs — one new
  algo file in the eval harness, the deep-sweep YAML pattern already exists.
- **int4 / dp4a-packed codes** for `SilverTorch.item_codes`: halves index memory again vs
  int8; the codesigned kernel's `tl.dot(int8)` path would become unpack+dot or a packed-dp4a
  trick. Worth a memory/recall Pareto point; lower priority than RaBitQ (which attacks the
  same memory axis with theory attached).
- Classic PQ/OPQ is deliberately *not* recommended — it abandons the "index as plain tensors +
  one fused kernel" design that is the thesis's systems contribution; RaBitQ keeps it.

### I6. Serving story: `torch.export` + AOTI end-to-end demo **[eng]**

[torch-export-refactor.md](torch-export-refactor.md) makes layers export-clean but explicitly
ships no `.pt2` artifacts. The missing last mile — one demo repo/script that exports a
`SilverTorch` retrieval graph, `aoti_compile_and_package`s it, and serves it from the C++
runner — would substantiate the "SilverTorch-style in-model serving" claim end-to-end (the
paper's predictor is C++; [silvertorch.md §3](../../articles/silvertorch.md)). Depends on: the
export plan landing. Deliverable: `examples/serve_aoti/` + a latency comparison against the
eager+cudagraph path. This is also where the state-dict round-trip gap
([kernels-layers-design.md](kernels-layers-design.md) Additional-1) must be resolved.

### I7. Multi-GPU sharding (revive from backup) **[eng → thesis "future work" section]**

`ShardedSilverTorch` plans are preserved in `docs/plans-silvertorch-backup/`. The paper's
scale-out design (shard ANN+bloom per GPU, replicate OverArch, gather pre-filtered results —
[silvertorch.md §5.3](../../articles/silvertorch.md)) is straightforward on the refactored
base: per-shard `SilverTorch` modules + an all-gather merge layer. Only worth doing with
access to a multi-GPU box and a concrete catalog that doesn't fit one card (the 15M synth
arxiv catalog fits; a 100M+ synth catalog per I8/R7 would be the driver).

### I8. Benchmark harness upgrades **[eng, feeds thesis Ch.6]**

Beyond roadmap Stage 4a (throughput, p99, build-time, per-query latency vectors):

- **Pareto-frontier reporting at fixed recall** — the standard comparison protocol in the
  filtered-ANN literature (Big-ANN'23 filter track; VecFlow reports speedup *at 95% recall*).
  Today's rows have both recall and latency; add an analysis script
  (`evaluation/analysis/pareto.py`) that interpolates each algo's parameter sweep to
  latency-at-recall∈{0.90, 0.95, 0.99} and emits the thesis table. Pure post-processing on
  existing JSONs + the deep-sweep configs.
- **Energy / cost-efficiency**: SilverTorch's headline includes TCO; the harness can sample
  `nvidia-smi --query-gpu=power.draw` (or NVML) during the perf window and report J/query.
  Cheap (a background sampler thread around `measure_forward_cuda`), and gives the thesis a
  cost-efficiency figure mirroring the paper's Table 2 at academic scale.
- **Provenance columns** (`extra.gpu`, `extra.commit`, `extra.torch`) — 5 lines, see
  [evaluation-refactor.md §Additional](evaluation-refactor.md).
- **Selectivity as a first-class column**: per-sweep mean pass-rate (the filter's mean
  `counts/N` over the query pool) is computed implicitly everywhere but never emitted; adding
  it turns every filter plot's x-axis from "sweep name" into a quantitative selectivity axis —
  needed by R2 and by thesis §6.4's per-clause analysis.

---

## Part II — Research directions

### R1. Filter-aware IVF partitioning **[thesis — strongest candidate]**

The codesigned kernel prunes by *geometry first* (probe top-n_p clusters), *predicate second*
(bloom/exact test inside probed clusters). When the predicate is selective and **correlated
with cluster structure** (e.g., goodreads `c0_genre` — genres cluster in embedding space),
most probed items fail the filter and recall collapses unless n_probe grows. The literature
has two stronger designs: **label-centric IVF** — group items by attribute value, then
spatially within each label; single- and multi-label AND/OR supported on GPU
([VecFlow, SIGMOD'25](https://arxiv.org/abs/2506.00812)) — and **IVF²**, fusing inverted
(attribute) and spatial indexes, winner of the
[NeurIPS'23 Big-ANN filter track](https://big-ann-benchmarks.com/neurips23.html).

Concrete thesis-scale version on this codebase: build per-attribute-value sub-IVFs for the
top-V most frequent values of the most selective clause (label-centric for the head), fall
back to the existing codesigned path for the tail — all still "index as tensors + one probe
kernel", preserving the systems story. Evaluate on the existing goodreads/arxiv sweeps across
the selectivity spectrum (I8's selectivity column gives the x-axis). Hypothesis: at pass-rates
below ~5%, label-centric probing dominates codesigned bloom by an order of magnitude in
probed-item count — mirroring VecFlow's motivation — while above ~20% the existing co-design
wins on memory.

### R2. Selectivity-adaptive query planning **[thesis]**

The repo already implements the full strategy menu: post-filter mask (`PostfilterKNN`),
pre-filter compact + sparse rescore (`PrefilterKNN` — LiNR's finding that V1 vs V2 flips with
pass rate, [linr.md §5.3](../../articles/linr.md)), 1-bit cascade (`OneBitKNN→PrefilterKNN`),
and fused codesigned IVF (`SilverTorch`). The eval results quantify exactly where each wins —
but *per sweep*, offline. The research question: **choose the strategy per query** from a
cheap selectivity estimate (bloom-signature popcount of the query, clause-value frequency
tables — both O(1) lookups on data already resident), with a cost model calibrated from the
harness's own measurements. Prior art frames this as learned/planned hybrid search
([ACORN's predicate-agnostic goal, SIGMOD'24](https://arxiv.org/abs/2403.04871); learned
query planning for filtered ANN, [arXiv 2602.17914](https://arxiv.org/pdf/2602.17914)) but
none of it targets the all-GPU, in-model setting. Deliverable: a `PlannedRetrieval` module
that dispatches among existing layers + a calibration script over eval JSONs; evaluate
end-to-end on mixed-selectivity query streams (the per-user goodreads qa already induces a
wide selectivity distribution within one sweep).

### R3. Theoretically-grounded 1-bit codes (RaBitQ) vs Sign-OPORP/SimHash **[thesis]**

Engineering half in I5; the research half: a controlled comparison of 1-bit code families
*under a fixed kernel/memory budget* (same `[N, W]` packing, same popcount kernel, same
candidate_pool cascade) measuring recall-vs-k_bits and recall-vs-candidate_pool on all three
datasets, plus RaBitQ's error-bound-driven candidate_pool selection (choose pool size per
query from the estimator's confidence rather than a fixed 5000/10000). Fixed-pool vs
bound-driven-pool is a clean, novel-enough ablation for a thesis chapter and directly improves
the linr_v3 cascade. ([RaBitQ, SIGMOD'24](https://dl.acm.org/doi/10.1145/3654970).)

### R4. Learned-similarity OverArch stage **[thesis — ties the two papers together]**

SilverTorch's biggest quality lever is the OverArch re-ranker (+2.4–28.2% recall,
[silvertorch.md §6](../../articles/silvertorch.md)); LiNR's is MoL-with-clusters (+15–23%
HitRate, [linr.md §3.3/§5.1](../../articles/linr.md)); and MoL now has published exact/
approximate top-k retrieval algorithms with error bounds and open code
([RAILS, WWW'25](https://arxiv.org/abs/2407.15462), [github.com/bailuding/rails](https://github.com/bailuding/rails)).
The repo has everything needed to reproduce this at academic scale: the two-stage cascade
pattern (`linr_v3`'s stage1→stage2 wiring incl. the single-cudagraph stitch), SASRec
embeddings, and the eval oracle machinery. Work: train a small MoL head on yambda/goodreads
user-item pairs (the gSASRec trainer already produces the embeddings), add an
`overarch_rerank` algo (ANN top-K₀ → MoL scores → top-k), measure recall-vs-K₀-vs-latency.
This is the one direction that upgrades the thesis from "systems reproduction + kernels" to
"systems + modeling", and it reuses the harness unchanged (one new algo file + params).

### R5. NOT-predicates for approximate filters on GPU **[thesis — small but novel]**

`BloomFilter` is conjunctive-only; reverse clauses are exact-filter-only (bloom mode raises,
[bloom.py:42-46](../../retrieve/src/retrieve/layers/filters/bloom.py#L42-L46)), which cost
three goodreads sweeps until the exact-kernel path landed. Bloom filters fundamentally cannot
answer NOT (false positives become false *negatives* under complement — retrieval-fatal). The
question "what is the right GPU-resident approximate structure for negated set-membership at
recommendation cardinalities" appears unstudied: candidates are a **second bloom filter over
the complement vocabulary** (viable when per-clause vocabularies are small and closed — true
for goodreads language/format), **cuckoo/quotient filters** (support deletion, still FP-only),
or **exact narrow attrs for reverse clauses + bloom for forward ones** (a hybrid the codesigned
kernel could take today — it already receives both buffer kinds; the SilverTorch layer just
forbids the combination). Even a negative result with measurements ("hybrid exact-reverse +
bloom-forward dominates complement-bloom at all realistic vocabulary sizes") is a tidy paper
section and removes a real feature gap.

### R6. Streaming index freshness **[thesis-adjacent, feature-driven]**

[live-update-api.md](live-update-api.md) covers flat-layer upserts; the open design problem is
SilverTorch's IVF (upsert ⇒ cluster assignment; deletes ⇒ shrinking clusters; drift ⇒
re-clustering). The SilverTorch paper sketches a "fresh index" merged at query time and leaves
details to future work ([silvertorch.md §7](../../articles/silvertorch.md)); streaming
GPU-resident IVF is an active topic ([SIVF, arXiv 2601.11808](https://arxiv.org/pdf/2601.11808)).
A contained version for this repo: fresh items go to a small flat side-index
(`PostfilterKNNInt8` over the append buffer — exact, no clustering), query = fused merge of
probed-IVF top-k and side-index top-k (one extra `masked_topk` over concatenated scores),
background re-cluster folds the side-index in when it exceeds a threshold. Measure recall
drift + latency vs side-index size; the LiNR +6% freshness lift
([linr.md §6](../../articles/linr.md)) motivates the whole exercise.

### R7. Scale ceiling: 100M–1B items on one GPU **[thesis — headline-number potential]**

LiNR reports 1B×64d fp16 → 1-bit on a single V100 ([linr.md §5.3.2](../../articles/linr.md));
the repo's measured ceiling so far is the 15M synth-arxiv catalog. The synth generator
([datasets/synth_arxiv.py](../../evaluation/datasets/synth_arxiv.py) — cluster + slerp
sampling, sharded emission) scales by construction; the eval harness already warns and guides
at >100M ([loaders.py:98-104](../../evaluation/retrieval/loaders.py#L98-L104)). Experiment
ladder: 50M → 100M (d=128, 1-bit codes = 800 MB; int8 = 12.8 GB — both fit an 80 GB card
alongside scratch) → 250M+ (1-bit only). Measure the full algo matrix where memory permits;
report the recall/latency/memory scaling curves and the crossover where IVF beats full-scan
1-bit. This produces the thesis's "billion-scale on one GPU" claim with honest constraints,
and stresses exactly the code paths the refactor plans harden (bucketing, grid caps —
`clause_compact`'s 3-D grid already anticipates N>16M).

### R8. Evaluation methodology for filtered retrieval **[thesis §method, low-cost]**

Two measured-but-unformalized choices in the harness are worth writing up as methodology
contributions (they generalize beyond this system):

- **Oracle-relative recall for filtered cells** (recall vs filtered-FullScan top-K rather
  than held-out interactions, with `-1` short-fill semantics and per-row denominator
  `min(pass_count, k)` — [oracle.py](../../evaluation/retrieval/oracle.py),
  [sweep.py:597-604](../../evaluation/retrieval/sweep.py#L597-L604)) — versus the common but
  misleading "held-out recall under filters" (targets often fail the filter). Formalize, and
  quantify how much the two disagree on goodreads.
- **Approximate-filter scoring**: bloom cells are scored against the *exact* oracle, so bloom
  FPs manifest as recall/precision effects rather than being baked into ground truth —
  combine with Yambda's Coverage@k ([yambda.md §4](../../articles/yambda.md)) and a
  measured-FPR column to characterize approximate filtering end-to-end. Cheap: the harness
  already computes everything except FPR (= mean pass-rate delta between bloom and exact
  masks — one extra pass at oracle build time).

### Prioritization

| # | direction | payoff | cost | depends on |
|---|---|---|---|---|
| R1 | filter-aware IVF | high (core thesis claim) | 3–6 wk | I8 selectivity col; refactors help |
| R4 | MoL OverArch | high (quality axis) | 3–5 wk | none (harness as-is) |
| R7 | 100M–1B scaling | high (headline) | 2–3 wk + GPU time | synth generator (exists) |
| R3 / I5 | RaBitQ codes | medium-high | 2 wk | K4.2 `_PackedBitsKNN` |
| R2 | adaptive planning | medium-high | 3–4 wk | I8, eval JSONs (exist) |
| I8 | Pareto/energy/selectivity | medium (enables R1/R2) | 1 wk | E5 helps |
| R5 | NOT-filters | medium (novel niche) | 1–2 wk | none |
| I2 | fused top-k | medium | 1 wk experiment | K2 |
| R6 | streaming freshness | medium | 3–4 wk | live-update-api |
| I1 | GPU CI | medium (safety) | 1–2 d | none |
| I4 | persistent kernels | low-medium | 1–2 wk | none |
| I6 | AOTI serving demo | low-medium | 1–2 wk | torch-export plan |
| I3 | popc/allocator | low | days | K2 |
| I7 | multi-GPU | low (hardware-gated) | 2–4 wk | backup plans |

## Sources

- SilverTorch — [arXiv 2511.14881](https://arxiv.org/abs/2511.14881) (SIGIR'26); local copy
  [articles/silvertorch.md](../../articles/silvertorch.md)
- LiNR — CIKM'24; local copy [articles/linr.md](../../articles/linr.md)
- Yambda-5B — [arXiv 2505.22238](https://arxiv.org/abs/2505.22238) (RecSys'25); local copy
  [articles/yambda.md](../../articles/yambda.md)
- RAILS / MoL — [arXiv 2407.15462](https://arxiv.org/abs/2407.15462) (WWW'25 oral),
  [code](https://github.com/bailuding/rails)
- RaBitQ — [SIGMOD'24](https://dl.acm.org/doi/10.1145/3654970),
  [arXiv 2405.12497](https://arxiv.org/abs/2405.12497)
- VecFlow — [arXiv 2506.00812](https://arxiv.org/abs/2506.00812) (SIGMOD'25)
- ACORN — [arXiv 2403.04871](https://arxiv.org/abs/2403.04871) (SIGMOD'24)
- Big-ANN benchmarks NeurIPS'23 filter track — [big-ann-benchmarks.com](https://big-ann-benchmarks.com/neurips23.html)
- RadiK — [arXiv 2501.14336](https://arxiv.org/abs/2501.14336); RTop-K —
  [arXiv 2409.00822](https://arxiv.org/pdf/2409.00822) (ICLR'25); GPU top-k study —
  [SC'23](https://dl.acm.org/doi/10.1145/3581784.3607062)
- SIVF streaming IVF — [arXiv 2601.11808](https://arxiv.org/pdf/2601.11808); learned filtered-ANN
  planning — [arXiv 2602.17914](https://arxiv.org/pdf/2602.17914)
- PyTorch user-defined Triton kernels —
  [tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html);
  FlagGems — [github.com/FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems)
- cuVS/Faiss GPU — [NVIDIA blog](https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/)
