# Reproduction fidelity — what we changed, why, and what measured it

> **Status:** written 2026-09-15 on `dev/f1-f3-paper` (roadmap **F1**, paper gap
> **G1**). Paper material for §3 "Reproduction methodology" of
> [reproducibility-paper.md](../plans/reproducibility-paper.md) §C.4. It is a
> *document of record*, not a plan: every row names the plan section that
> measured it, and a row whose gate has not passed says so in the row.
>
> **Sources.** The papers as frozen in [articles/](../../articles/)
> (`silvertorch.md`, `linr.md`) — cited, never edited. The implementation as
> documented in [../system/](../system/) (`architecture.md`, `kernels.md`,
> `filtering.md`, `evaluation.md`, `datasets.md`). The measurements in the
> plans' validation records, named per row.
>
> **CLAUDE.md rule 2 applies to every number here.** A measurement quoted below
> is quoted because a validated gate produced it. Where a number would be
> natural but its gate has not passed, the row says **not yet validated** and
> names the roadmap step it waits on. Nothing here is a performance claim about
> the current code: the Triton-vs-official head-to-head is roadmap **B3** and
> the campaign is **D1**. Neither had run when this was written; **B3 has since
> passed** (2026-09-15) and its results are
> [official-vs-reimplementation.md](official-vs-reimplementation.md), which is
> where every speed claim lives. Three rows below were updated on 2026-09-16 by
> what B3 measured: **OF-4**, **OF-7** and **R-3**.

## 1. Naming, once, for the whole paper

The thesis called our SilverTorch layer *QuantizedIVF* and described it as a new
algorithm proposed in that work. It is not: it is **SilverTorch's Algorithm 1**
(`articles/silvertorch.md` §4.4, `alg:partial_bloom`), reimplemented from the
paper's text. The name is retired; roadmap **F1** rewrote
[../thesis/main.tex](../thesis/main.tex) accordingly and added the missing
citation of the SilverTorch paper, which the thesis did not carry at all.

The three vocabularies, from
[library-harness-boundary.md](../plans/library-harness-boundary.md) §6:

| paper | library class | harness algo key | in this paper's prose |
|---|---|---|---|
| SilverTorch Algorithm 1 | `SilverTorch` | `silvertorch` | "SilverTorch Algorithm 1 (reimplemented)" on first use, "SilverTorch" after |
| Meta's released ops | `retrieve.ops.official`, `SilverTorch(backend="official")` | `silvertorch` + `backend: official` | "the official kernels" / "Meta's implementation" |
| LiNR V1 (post-filter mask) | `LiNRV1` over `PostfilterKNN` | `linr_v1_filter_mask` | "LiNR V1" |
| LiNR V2 (exact pre-filter) | `LiNRV2` over `PrefilterKNN` | `linr_v2` | "LiNR V2" |
| LiNR V3 (1-bit Sign-OPORP + rerank) | `LiNRV3` over `OneBitKNN` → `PrefilterKNN` | `linr_v3` | "LiNR V3" |
| *beyond both papers*: int8 dense | `LiNRV4` over `PostfilterKNNInt8` | `linr_v4` | "LiNR V4 (ours)" — stated as ours, every time |
| bloom / exact attribute filter | `BloomFilter` / `ExactAttributeFilter` | `filter_kind: bloom` / `clause` | — |

Two names in the tables below are ours and must never be attributed to a paper:
**LiNR V4** (an int8 dense variant the LiNR paper does not contain) and
SilverTorch's **`filter_mode="exact"`** (an exact fused predicate; the paper's
filter is a bloom index and is approximate by construction).

## 2. How to read the tables

| column | meaning |
|---|---|
| *paper* | what the paper says, with the section |
| *ours* | what the code does today (`docs/system` is the authority; if they disagree the code is right) |
| *why* | the reason for the difference — a constraint, a measurement, or a deliberate extension |
| *measured in* | the validation record that established it; **"—"** means the row is a design fact, not a measurement |

## 3. SilverTorch Algorithm 1 — deviations

| # | paper | ours | why | measured in |
|---|---|---|---|---|
| ST-1 | Bloom signature hashes each `(feature, value)` pair with `K` hash functions (§4.1) | splitmix-style mix with a **per-clause salt** folded in before the position mask, so value `V` in clause 0 and in clause 3 land on different bits; the salt is a `clause_salt` buffer, a pure function of the clause index | without it, single-clause queries over overlapping small-integer value vocabularies (years, version counts, licence codes) leak ~25–30 % of non-matching items as cross-clause false positives | [filtering.md → Bloom hash keys](../system/filtering.md#bloom-hash-keys-clause_idx-value); the buffer form's bit-equality with the pre-B5 inline form is pinned by `test_bloom_hash.py`, GPU-green in [O §14.2](../plans/silvertorch-official-integration.md) |
| ST-2 | bloom width per bundle of 2048 docs, `B = max_terms_in_bundle · k · b_multiplier`; heuristic `bits = max_values × hashes × 3`; FPR 6.98 % at 512 bits → 0.067 % at 1024 (§4.1, §6) | **fixed** `m_bits` (1024 in every recorded run), `k_hash = 5`, one width for the whole index | a fixed width is a buffer, not a per-bundle table; it is also what makes our index directly comparable across datasets. Consequence: matched memory ≠ matched FPR against the official index, so the two must be compared at *both* operating points | design fact; the comparison protocol is [O §4.3](../plans/silvertorch-official-integration.md); FPR on **real** attributes is **not yet validated** — roadmap **D3** |
| ST-3 | filtering is the bloom index only; false positives are accepted because the downstream OverArch removes them (§4.1) | a second, **exact** fused predicate (`filter_mode="exact"`, AND-of-OR over `[N, C, A_max]` narrow attrs, reverse clauses supported) alongside `filter_mode="bloom"` | we have no OverArch stage, so a false positive is a quality loss with nothing downstream to absorb it; the exact mode makes the bloom's FPR cost *measurable* by difference | design fact ([architecture.md → SilverTorch](../system/architecture.md#silvertorch)) |
| ST-4 | runtime DSL: arbitrary nested AND/OR/NOT over `feature = value`, parsed to an operation array and walked on a stack (§4.1, §4.4) | fixed clause schema, **conjunctive only**: `C` clauses declared at index time, `[B, C]` query, AND across clauses, OR within a clause's value list; reverse (NOT) clauses in the *exact* path only, never in the bloom path | LiNR's schema (`articles/linr.md` §3.1) is the one both algorithms share in this library, which is what lets any filter plug into any retriever; a DSL is a host-side parser feeding `combine_masks` and is deliberately out of scope | design fact ([filtering.md → Out of scope](../system/filtering.md#out-of-scope)) |
| ST-5 | int8 quantization at publish: global min/max across all embeddings, scaled to [-128, 127] (§4.2) | items: one **global** per-tensor scale (as the paper); queries: a **per-row** scale, applied in the epilogue as two left-associated fp32 multiplies `(dot.float() · q_scale[b]) · global_scale` | the query scale cannot be global — it is computed per forward. The association is fixed because re-associating it changes the last bits and would break backend parity | [kernels.md → Score conventions](../system/kernels.md#score-conventions); bit-identity of the epilogue across `triton` / `torch` / `official(int32)` is [O §14](../plans/silvertorch-official-integration.md) (B2, `torch.equal`) |
| ST-6 | per-embedding scales are available in the released kernel (`per_embedding_scale`) | **not used**; the global-vs-per-row scale ablation must run on the int32 path with per-row scales applied host-side | the official kernel casts the int32 dot to fp16 *before* dividing, so `per_embedding_scale` returns `inf` in **every** output slot for any `D ≥ 5` at full-range codes — measured, not inferred. This is a finding about the released code, not a choice of ours | [O §13.2](../plans/silvertorch-official-integration.md) (A3): `ones(N, fp16)` → `inf` in all 256 slots at `D = 128` |
| ST-7 | KMeans++-based training on GPUs at publish (§3) | Lloyd's k-means, `n_iter = 10`, **random** init by default; `kmeans_init="kmeans++"` (greedy D²) is implemented and opt-in | every recorded number was taken on random init, so the default cannot move without invalidating them (plan **L** D9). k-means++ at `n_lists = 8192` costs 9.5 s, so cost is not the reason | [L §12.2](../plans/library-api-refactor.md) (L2). **The thesis text said k-means++ and the code did not** — corrected in `main.tex` at F1. Whether init changes quality is **not yet validated** — roadmap **D1** |
| ST-8 | Phase 4 is a global top-k inside the design (§4.4 Algorithm 1) | one Triton launch for phases 2+3, then a **host-side `torch.topk`** over the `[B, P]` score buffer | CUB's top-k beats anything we write in pure Triton; the launch boundary is where the paper's phases 3 and 4 split anyway | design fact ([architecture.md](../system/architecture.md)); the epilogue's `-1` / `-inf` sentinel is [O §14.7](../plans/silvertorch-official-integration.md) |
| ST-9 | transposed, cluster-major bloom index: one 64-bit AND tests 64 items; 1-bit masks, not 8-bit bool (§4.1, §4.4) | our Triton bloom reads **row-wise per-item signatures** `[N, W]` int64; the transposed layout is not implemented in Triton | the transposed index *was* implemented, in a hand-written CUDA backend, and was bit-exact with Triton on scores; it was deleted at roadmap B4 once Meta's own ops passed the parity gate, under CLAUDE.md rule 5. Rewriting it in Triton is roadmap **G-a** (O §8 TF-1) | the CUDA backend's record — kernel-only bloom scoring 147.8 → 58.6 µs after a memory-level-parallelism fix, against Triton's 124.6 µs at `B=16, P=58k, D=128` — is [archive/cuda-silvertorch-handoff.md §13](../plans/archive/cuda-silvertorch-handoff.md). **That backend no longer exists**; no claim about the current code follows from it, and the Triton transposed index is **not yet validated** (G-a) |
| ST-10 | the co-designed index reverses filter and probe order and "the results remain the same"; 1.79×–2.15× latency, scratch 35.6 → 18.2 MB at 20 M items, probe 32 (§4.4, §6) | the same reversal, fused into one kernel; the separate-filter arm exists as `OfficialConfig(bloom_path="full")` | the ablation is exactly reproducible on our side — the two arms are one constructor flag | the arm exists and is parity-tested ([O §14](../plans/silvertorch-official-integration.md), B1/B2). **The ablation itself is not yet validated** — roadmap **D1** (S9 cells) / **G-b** |
| ST-11 | served through a C++ predictor; the model is lowered and scripted at publish (§3) | each algo is wrapped in `torch.compile(dynamic=True, mode="reduce-overhead")` by the harness and measured eager *and* under CUDA-graph replay; the library never compiles internally | we measure a research library, not a serving stack; the compile wrapper is what makes the CUDA-graph number meaningful, and the library stays a plain `nn.Module` | capture is verified: `cudagraph_skips == 0` on **20/20** cells, [H §12.5](../plans/evaluation-harness-v2.md) (C4) |
| ST-12 | end-to-end results at 10 M and 80 M items, 2-GPU shard for 80 M, OverArch + Value Model, 12-way multi-embedding queries, QPS under a 200 ms P99 budget (§6) | single GPU, ≤ 5.4 M items in the thesis, no OverArch, no Value Model, single-embedding queries, latency rather than client-side QPS | out of scope and declared as such; see [reproducibility-paper.md §B.4](../plans/reproducibility-paper.md) for the full list and the one-line reason per item | — |

## 4. LiNR V1–V4 — deviations

| # | paper | ours | why | measured in |
|---|---|---|---|---|
| LN-1 | all reported numbers use **un-fused** individual ops (§6), and the paper predicts a fused implementation would change its V1/V2 conclusion | V2's pre-filter is a **fused** eval + stream compaction (`clause_compact` / `bloom_compact`), and V2's scoring is a fused gather + dot + top-k (`fused_masked_knn_topk`) | the fusion is the point of using Triton; it is also the single most likely reason the thesis measures V2 *faster* than V1 at high pass rate where the paper measures the opposite | the contradiction is recorded in [reproducibility-paper.md §B.2](../plans/reproducibility-paper.md) L1. Locating the crossover is **not yet validated** — roadmap **G-b** (P G11) |
| LN-2 | V1/V2/V3 only; fp16 embeddings | **LiNR V4 added**: int8 dense, `torch._int_mm` + int32 top-k | halves index memory; the int32 dot is a positive monotonic transform of the fp32 dot, so ranking needs no rescale | design fact ([kernels.md](../system/kernels.md)); the memory halving is in the thesis table and is a *pre-v2-harness* number (§6 below) |
| LN-3 | V3 is Sign-OPORP at 512 bits keeping ~1 % of items (§5.3) | Sign-OPORP with a `candidate_pool` query parameter; the thesis swept `P_c` 2 000–32 000 on a 797 k catalogue (0.25–4 %) | our catalogues are 1–2 orders smaller, so "1 %" is a different absolute pool | bit-width and the 1 %-retained point are **not yet validated** — roadmap **G-b** (P G15) |
| LN-4 | V2 is an exact filtered top-K | it is — the candidate sets of our two backends are **identical** on every row (0 / 9 859 differ on counts or ids, `torch.equal`) — but the two backends' **top-k lists differ**: `jaccard@100` 0.998743, `score_max_abs_diff` 9.77e-3 | not a selection bug. `fused_masked_knn_topk` **accumulated in fp16** (`tl.sum` inherits the operand dtype; the compiled PTX has 8 `add.f16` and no f32 adds) while its own docstring and `kernels.md` promised fp32. On goodreads the partial sums reach \|s\| = 31 against a top-100 near 0…−1 — catastrophic cancellation, **0.0276** max abs score error against an fp64 dot. All 626 swapped pairs lie inside that error and the top-k epilogue ranked correctly in 626/626 | [linr-v2-backend-parity.md §6.1](../plans/linr-v2-backend-parity.md) (L4). **The fix is roadmap L5 and has not landed**: fp32 accumulation was measured at 0.9408 ms vs 0.9636 ms at `B=16` (i.e. not slower) and would move parity to 0.999751, but **those numbers are not yet validated** and every V2 number in this paper must come from the post-L5 campaign (D1). The finding — that a parity suite comparing an implementation against a reference *at the same precision* is structurally blind to reduction width — is the citable part |
| LN-5 | metrics: average + p95 latency, recall of the label set @2000 (§5.3) | median / p20 / p80 over a query pool, recall / ndcg @ {100, 500, 1000} against an exact filtered fp32 oracle | our protocol predates the comparison; mean / p95 / p99 / QPS and K = 2000 are additions the paper needs | **not yet validated** — roadmap **D1** (P G3) |
| LN-6 | native TF/PyTorch masking and indexing is ~100× slower than the custom CUDA filter kernel (§6) | we report torch-eager, torch-compiled and Triton on the same cells | a `torch.compile`d mask is not the same baseline as eager PyTorch indexing; whichever way it comes out is a headline finding | **not yet validated** — roadmap **D1** (P G5/L7) |
| LN-7 | live index updates, TorchScript serving stack, MoL / Hadamard similarity, the A/B lift (§4.4, §5, §6) | none of these | out of scope; see [reproducibility-paper.md §B.4](../plans/reproducibility-paper.md) | — |

## 5. Deviations in **Meta's released code** against **Meta's paper**

These are findings about the reference implementation, not choices of ours. They
are the reason "which SilverTorch do you mean?" is a real question.

| # | finding | evidence | measured in |
|---|---|---|---|
| OF-1 | the released bloom index is **not the same hash** as ours and can never be bit-compared: murmur3 over `(feature_id, value, seed)`, width per 2048-doc bundle from the bundle's maximum term count, `hash_k` raw hashes of which a term ANDs the first `k` *distinct* positions | source-level, then confirmed by the parity gate matching **FPR and memory** instead of bits: byte-exact memory at `b_multiplier = m_bits / (max_terms · 5)`, FPR 0.0000 for both blooms on synthetic attributes | [O §4.3](../plans/silvertorch-official-integration.md), [O §14.4](../plans/silvertorch-official-integration.md) (B2). Real-attribute calibration is **not yet validated** — roadmap **D3** |
| OF-2 | the mask bit order is **HIGH-bit-first** on the GPU path (doc `d` at bit `63 - (d % 64)` of word `d // 64`) and **LOW-bit-first** in the CPU reference in the same tree | three independent probes agree (packed output, hand-built mask, round trip) | [O §13.2](../plans/silvertorch-official-integration.md) (A3); pinned by parity test T3 |
| OF-3 | the official path is **eager only**. Five of seven ops fail CUDA-graph capture; `generate_column_info_for_clusters` captures; `bloom_index_search_batch` *captures and then faults on replay* with an illegal memory access, because the host plan decode and its pageable upload are not in the graph | measured per op, each probe in its own subprocess (a failed capture poisons the CUDA context) | [O §13.2](../plans/silvertorch-official-integration.md) (A3). In the harness, `official` records `mode: graph` as null with `reason: not_capturable`, and that is the only reason it carries — 4/4 cells, [H §12.5](../plans/evaluation-harness-v2.md) |
| OF-4 | host overhead per forward is an order of magnitude above ours: `fused_kmean_ann` costs **19 launches and 3 host syncs** (the paper's design implies far fewer; only 2 of the 19 are the scoring work), `…_with_partial_masks` 4 syncs, a bloom forward ≈ 32 launches and ≥ 5 syncs, against our **one launch** per forward | `torch.profiler`, and fd-2 capture of c10's sync warnings (the Python `warnings` instrument reports zero syncs and is wrong — a `TORCH_WARN` from a C++ custom op never becomes a Python warning) | [O §13.2](../plans/silvertorch-official-integration.md) (A3), sync instrument re-validated in [O §14.5](../plans/silvertorch-official-integration.md). **This is a host-side count, not a kernel-speed claim**; the kernel-only comparison is roadmap **B3**, now run — it confirms the count at the timing level (the official payload prep costs 268–896 µs over 73–95 launches per forward against our 34–58), [O §16.2](../plans/silvertorch-official-integration.md) |
| OF-5 | expression parsing costs **58.7 µs per call / 3.67 µs per query** at `B = 16` on the CPU, inside the timed op | `torch.utils.benchmark`, IQR ≤ 0.2 µs | [O §13.2](../plans/silvertorch-official-integration.md). Reported as its own row and excluded from any kernel comparison; every timed official cell runs with `cache_plans: false` so the parse is paid ([H §12.6](../plans/evaluation-harness-v2.md)) |
| OF-6 | the README's own verification command (`pytest silvertorch/`) **fails at collection**, 3 errors, before a test runs: Meta's internal `@oss-disable` comment stripping is line-based and mangled three test files. Two ops (`is_topk`, `take_top_k_and_gather_from_main_and_fresh`) are registered in source but absent from `setup.py`, so they exist in no OSS build | excluding the three files: 99 passed / 3 subtests in 11.6 s, every CUDA test included — green for everything the OSS build ships | [O §13.1](../plans/silvertorch-official-integration.md) (A2) |
| OF-7 | the official and our Triton implementation agree to `jaccard@100` **0.999849 / 0.999805** on goodreads (`n_probe` 24 / 32) but only **0.985033 / 0.984882** on arXiv, where `score_max_abs_diff` is an order of magnitude *smaller* (4.8e-4 vs 5.5e-3) | **not** the filter, and no longer unresolved: the *unfiltered* cells split the same way and the int32 path is bit-exact on both datasets, so the whole deficit is the fp16 score path meeting arXiv's score distribution — 95.3 % of arXiv queries have their rank-100/101 gap inside one fp16 ulp, against 3.1 % of goodreads ones | [H §12.6](../plans/evaluation-harness-v2.md) (C4); **explained by B3**, [O §16.4](../plans/silvertorch-official-integration.md), which falsifies this row's original "looser filter" reading and puts the cost at 3.1e-4 recall@100 on arXiv / 4e-6 on goodreads — [official-vs-reimplementation.md](official-vs-reimplementation.md) §6.1 |
| OF-8 | scores are bit-exactly reproducible between the official kernel and ours on the int32 path (`divisor_for_int8 = -1`), and the fp16 serving path (`divisor = 2^k`) costs only fp16 rounding (rel. 6.2e-5, inside the 2⁻¹¹ bound) | `torch.equal` on the int32 path on every regime cell; bloom ⊇ exact and ⊆ exact with 0 false negatives / positives | [O §14](../plans/silvertorch-official-integration.md) (B2, gate green) |

## 6. Deviations of our **harness** against the papers' protocols — and against itself

| # | what | why it matters | measured in |
|---|---|---|---|
| HB-1 | the papers replay production traffic (SilverTorch: 5 000 real requests; LiNR: 25 k queries with geo and reverse-company clauses); we synthesise query predicates from item attributes over public datasets, with a documented pass-rate per clause condition | the pass-rate distribution is the single variable that decides V1-vs-V2 and the liquidity curve, and ours is not theirs | design fact ([evaluation.md](../system/evaluation.md)); the controlled pass-rate sweep is **not yet validated** — roadmap **G-b** (P G11) |
| HB-2 | ground truth is **our own** exact filtered fp32 oracle, not the papers' label sets | it makes recall oracle-relative under filters and interaction-relative without them, and the two must never be plotted together | oracle correctness is checked where a shipped ground truth exists: our exact filtered oracle reproduced YFCC-10M's shipped `GT.public.ibin` on **100 000 / 100 000** queries, id-exact, `max_abs_distance_error = 0.0` ([roadmap E1](../plans/00-roadmap.md)) |
| HB-3 | the SM clock **cannot be locked** on this box, so timing records a sampled clock instead | every latency number must say which estimator it uses; comparing two estimators of the same clock failed 92 of 99 rows of a gate that in fact passed | [H §12.4](../plans/evaluation-harness-v2.md) (C4) and [../paper/provenance-and-disclosure.md](provenance-and-disclosure.md) §4, which is where the disclosure lives |
| HB-4 | seed 0 only, everywhere, so far | no CIs, no paired tests; every "A is faster than B" sentence in the thesis is unreplicated | **not yet validated** — roadmap **D1** (P G4) |
| HB-5 | the thesis's tables and figures were produced by the **pre-v2 harness** | they are not paper material. The v2 harness closed on what it proves about *itself*; the paper's tables come from `report.py` over the D1 records | [H WP-4's amendment block](../plans/evaluation-harness-v2.md); the disclosure of what was and was not compared is [provenance-and-disclosure.md](provenance-and-disclosure.md) §5 |

## 7. Defects in **our own** implementation, found by reproducing

The reproducibility value of this section is the mechanism, not the fix. Each of
these was invisible to the test suite that was supposed to catch it, and each
became visible only because a *third* implementation disagreed.

| # | defect | how it surfaced | magnitude | state |
|---|---|---|---|---|
| D-1 | `fused_masked_knn_topk` (LiNR V2/V3 `triton`) **accumulated in fp16** where its own docstring and `kernels.md` promised fp32 | a third implementation — the `torch` backend's cuBLAS `bmm`, which accumulates fp32 and rounds only its output — disagreed. The parity suite could not see it: it compares Triton against a reference **at the same precision**, so the whole class of reduction-width bugs is structurally invisible to it | **0.0276** max abs score error against an fp64 dot under catastrophic cancellation (unnormalised goodreads embeddings, \|s\| to 31 against a top-100 near 0); 624 of 9 859 rows swap a rank-100 pair; the `torch` side's own residual is 124 rows of exact fp16 output ties broken by `torch.topk` | **Fix pending: roadmap L5.** The docs were corrected to state what the code does; the kernel is untouched so far. Post-fix numbers exist but are **not yet validated**. [linr-v2-backend-parity.md §6.1](../plans/linr-v2-backend-parity.md) |
| D-2 | `clause_compact` / `bloom_compact` claimed each row's base with an `atomic_add`, so survivors came out in **tile-completion order** — order-nondeterministic run to run | LiNR V2 and V3 on `triton` could not reproduce **themselves** across processes. The downstream tie-breakers turn candidate order into quality noise: `PrefilterKNN`'s top-k, and `OneBitKNN`'s massively-tied integer Hamming ranking, which decides pool *membership* | own-noise **2.0e-6** on `linr_v2` and **6.8e-5** on `linr_v3` — 2× and 68× the harness's own 1e-6 comparison tolerance, measured across three identical runs | **Fixed (L3).** Two-phase kernel (per-tile counts → exclusive scan → write at fixed offsets), ascending item order `torch.equal` to `ops.reference`, identical launch-to-launch in and across processes. Cost **×1.02–1.05** under CUDA graph. The plan's original design was **falsified by measurement** — recomputing the predicate costs +43–58 % end to end, because `clause_compact` is 34–47 % of a V2 forward and is instruction-bound. Confirmed against real data afterwards: the two cells land **3.6e-11** and **1.6e-9** from the golden. [deterministic-compaction.md §7](../plans/deterministic-compaction.md), [H §12.3](../plans/evaluation-harness-v2.md) |
| D-3 | k-means used atomics and was **non-deterministic**, so the centroids — and therefore the IVF lists, the probed cells and the scored items — changed run to run on every backend | it had been *misread as tie order*: the pre-fix harness showed a 1.7e-4 `torch`-vs-`triton` gap on SilverTorch and the record attributed it to ties. After the deterministic fit the gap is **exactly 0.0** at k = 100 / 500 / 1000, on recall and ndcg | the fix moved 5 of 11 golden cells by 5e-6 to 1.6e-4, i.e. 5× to 160× the comparison tolerance, **up as well as down** — so a gate run against the old cells would have failed on all five, unattributably | **Fixed** (C4 library prerequisite (ii), `d5d824b`); two seed-0 fits are `torch.equal` at 1.22× the old wall time. [H §11.2–§11.3](../plans/evaluation-harness-v2.md) |
| D-4 | a `[1]`-view `counts` output tripped inductor's 16-byte `assert_alignment` and killed the **compiled** bloom-V2 forward at `B = 1` only | found while landing D-2; invisible to the compile suite, whose shapes (`N = 512, B = 4`) never hit it | compiled forward failed outright at `B = 1` on goodreads shapes | **Fixed** and pinned by `test_compiled_batch_of_one_bloom_v2`. [deterministic-compaction.md §7.1](../plans/deterministic-compaction.md) |
| D-5 | a Triton epilogue that reduces before it scans makes the *whole kernel* 1.7–1.8× slower at `B = 1` — Triton lays the tile out for the first reduction it meets | found by building three epilogue variants and timing them rather than reasoning about them | 0.121 ms → 0.217 ms on the predicate launch, no change at `B = 16` | **Avoided** in the shipped kernel; the ordering constraint is a comment in `compact_stash`. [deterministic-compaction.md §7.1](../plans/deterministic-compaction.md) |
| D-6 | the fetched datasets were the **pre-`3b1b5b3` 1-indexed `[N+1, …]`** layout; the loader read attribute row `i` for item `i` | loud on goodreads, **silent on arXiv** — it showed up only as `cos(query, target)` falling 0.99 → 0.62 | every filtered arXiv number before the fix was wrong and looked fine | **Fixed** (detected and dropped in the loader). The project's HF mirror still publishes the 1-indexed layout, which is a disclosure item, not a bug: [provenance-and-disclosure.md](provenance-and-disclosure.md) §3. [H §10](../plans/evaluation-harness-v2.md) (A1) |
| D-7 | inductor's on-disk FX cache does **not** invalidate when a `@triton_op` host wrapper's Python source changes | a kernel edit produced numbers from the old code | any kernel-change measurement taken without clearing it | **Operational rule**: a private `TORCHINDUCTOR_CACHE_DIR` per job, or `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`. [roadmap §1 status block](../plans/00-roadmap.md); it also means concurrent workers must not share `/tmp/torchinductor_root` ([H §11.5](../plans/evaluation-harness-v2.md)) |

## 8. Residuals we cannot explain, listed as unexplained

A reproducibility paper is worth more with these in it than without.

| # | residual | what was ruled out, by measurement | state |
|---|---|---|---|
| R-1 | `linr_v4` (both backends) lands **7.3e-5** from the re-derived baseline on `recall@100`, against a 1e-6 comparison tolerance | *not* the library (bit-identical across the trees), *not* the `k_max` slice (a rerun at `--k 100` reproduces the new harness's own number to the last digit), *not* eager-vs-compiled (bit-identical). The int8 path's boundary ties do move with the query-batch chunk shape — but **the chunk-shape attribution is falsified**: at chunk 64 the cell lands **2.9e-4** away, four times *further* than chunk 16's 7.3e-5, and moves k = 500/1000 by 2.3–2.6e-4 where chunk 16 was within 1e-5 | **unexplained.** [H WP-4 amendment](../plans/evaluation-harness-v2.md), correction in [V §11.2](../plans/evaluation-package-layout.md) (C5) |
| R-2 | arXiv `silvertorch` `recall@100` lands **2.0e-6** away — about 2 single-hit changes in 10 000 rows | not the slice and not the batch shape, both tested | **unattributed**, accepted as a bounded residual |
| R-3 | the official backend clears `jaccard@100` 0.9998 against our Triton on goodreads and **0.985** on arXiv (OF-7) | not a harness artifact; `score_max_abs_diff` is *smaller* on arXiv | **Resolved by B3** and therefore no longer a residual: it is fp16 resolution against arXiv's score distribution, not a filter effect, and it costs 3.1e-4 recall@100 ([O §16.4](../plans/silvertorch-official-integration.md), [official-vs-reimplementation.md](official-vs-reimplementation.md) §6.1) |

## 9. What this document does not contain

- **No performance comparison of our Triton kernels against the official ones.**
  That is roadmap **B3** (paper gap G2) and the paper section built from it is
  **F2**, [official-vs-reimplementation.md](official-vs-reimplementation.md).
  Nothing in §3–§5 above may be read as a speed claim.
- **No campaign numbers.** Every quality and latency table in the paper comes
  from `report.py` over the **D1** records (roadmap D4). The thesis's tables were
  produced by the pre-v2 harness and are superseded, not cited.
- **No claim of equivalence between the v2 harness and the pre-v2 harness.** What
  was and was not compared is stated once, in
  [provenance-and-disclosure.md](provenance-and-disclosure.md) §5.
- **No venue or reviewer reasoning.** That research lives in
  [reproducibility-paper.md](../plans/reproducibility-paper.md) Part A and is not
  extended here.
