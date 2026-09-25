# The official kernels against our reimplementation

> **Status:** written 2026-09-16 on `dev/f2-official-section` (roadmap **F2**,
> O §10 WP-9, paper gap **G2**).
> Paper material for §3 "Reproduction methodology" (the official-kernel check)
> and §5.1 "RQ1" of reproducibility-paper.md
> §C.4.
>
> **References.** Labels such as O §16, H §12, V §11 and L §12 name
> sections of the design plans this section was written from (O: official
> integration, H: harness v2, V: harness package layout, L: library
> refactor). Those plans are retired; each link now points to the artifacts
> or wiki page that carries the fact, and a label with no link has no
> public source beyond this text.
>
> **Source.** Every measurement below is
> [O §16](../artifacts/official-silvertorch/b3/tables.md), the validation record of
> roadmap **B3**, executed 2026-09-15 on the A100 box; raw scripts, JSON, harness
> JSONL and the full table dump are in
> [artifacts/official-silvertorch/b3/](../artifacts/official-silvertorch/b3/).
> B3 is closed ([validation](../validation.md#official-against-our-triton-reimplementation-citable-contested)),
> so under CLAUDE.md rule 2 these are paper material — with the limits
> §8 and §9 state, which are not decoration.
>
> **What is not here.** The campaign has not run. Roadmap **D1** is in progress
> (stage D1-a at the time of writing); every line that waits on it says so and
> names the stage, and §10 is the list. No number in this document was invented
> to fill one of those holes.
>
> **Companions.** The design differences between the two implementations are
> [reproduction-deviations.md](reproduction-deviations.md) §3 and §5 — cited here,
> not restated. The box, the estimator and the disclosure are
> [provenance-and-disclosure.md](provenance-and-disclosure.md).

> **Orchestrator note on citability, 2026-09-16.** This document treats B3's
> numbers as paper material. The basis, stated so a reader can disagree with
> it: B3's checkbox is flipped because its gate — "JSON + tables appended to
> O" — was delivered, and the orchestrator independently re-derived its parity
> figures from the committed records (the jaccard and `score_max_abs_diff`
> values in §6) rather than accepting the report. What was **not** re-run is
> the 3.5 h of timing measurement itself.
>
> The records were produced on `dev/b3-head-to-head`, and
> `bench report`'s citability check flags any record from a branch other than
> `development` — **by design, and it will flag these**. The judgement here is
> that the flag does not indicate a defect in this case: B3 changed no library
> or harness code, so the tree it measured is `development`'s tree at
> `e23309c`, and every record carries that `code_version` and commit. A
> measurement-only branch is not the hazard the rule was written for.
>
> If the user prefers the strict reading of CLAUDE.md rule 2 — that no number
> is paper material until it has been reproduced on `development` — then the
> fix is one re-run of B3's two legs, not a change to any text below, and this
> blockquote is what should be revisited.

## 1. The question, and the short answer

The plan that built this arm asked whether *"a ~600-line Triton reimplementation
is competitive with the vendor's ~9.4k-line CUDA C++"*
([O](../system/kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend), header). Meta's ops tree is
**9,427 lines** of `.cpp/.cu/.cuh/.h`, ≈ 3.5k of it the `fused_kmean_ann` scorer
family; our phases 2+3 are **310 lines** of Triton (294 more for the exact
variant, 110 shared, 165 for the bloom hash) ([O §1](../system/kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend)).

The answer is two-sided, and the two sides do not cancel:

- **End to end, our Triton arm is the fastest in every cell of the matrix** — every
  dataset, every filter mode, every probe width, eager and under CUDA graph —
  by **1.2–1.9×** against the official arm on the unfiltered and bloom cells and
  **2.9–10.1×** on the exact cells, and by **7.3–32.7×** against the `torch`
  eager floor.
- **Kernel-only, Meta's scoring kernel is faster than ours in every cell** — by
  **1.15–3.2×** on arXiv and by **10.9–17.8×** on Goodreads.

Both are true because they measure different things, and the reason is nameable:
their op spends **268–896 µs** per forward building its payload across **73–95
kernel launches** against our **34–58**, which is more than their scorer saves;
and our scorer's 10× deficit on Goodreads is **our index layout, not our kernel**
(§4). A reader who wants one sentence should take this one: *the vendor's kernel
is better than ours, our system is faster than theirs, and neither fact is an
artifact of the other.*

## 2. What was compared, and what makes the arms comparable

Both arms are the same `SilverTorch` module with `backend="triton"` or
`backend="official"`; `backend="torch"` is the readable eager floor. Phase 1
(centroid matmul + probe top-k) and the `torch.topk` epilogue are **identical
code in both arms**; only Algorithm 1's phases 2+3 differ.

| control | how it was enforced | where |
|---|---|---|
| same index in both arms | asserted per mode and dataset, not assumed: `centroids_equal` **True**, the official arm's `item_codes` `torch.equal` to the Triton arm's under the cluster sort permutation **True**, `global_scale` equal **True** — 6/6 checks | [O §16.1](../artifacts/official-silvertorch/b3/tables.md) |
| the official expression parse is paid every forward | `OfficialConfig(cache_plans=False)` on **all 156** official perf entries, in both tiers, so the CPU parse is inside the timed call | O §16.1; the parse cost itself is [OF-5](reproduction-deviations.md) |
| identical top-k | the same host `masked_topk` in both arms (324–339 µs Goodreads, 119–125 µs arXiv) | O §16.2 |
| one estimator, stated | 50 warm-ups, 3 windows, per-call CUDA events, median of window medians, `spread` recorded, `unstable` above 5 %; SM clock sampled **under load** after the last window | O §16.1, [provenance §4](provenance-and-disclosure.md) |
| host-side rows are labelled | the two parser rows of §5 are `perf_counter` walls, because CUDA events would time an empty timeline | O §16.3 |

Datasets: Goodreads work-id d128 (797,084 items, `c0_genre`) and arXiv papers
d128 (2,988,996 items, `c0_maincat`), `users_limit 10000`, **seed 0**,
`n_lists 1024`, `n_probe ∈ {24, 32}`, our bloom at `m_bits 1024, k_hash 5`, the
official bloom at `b_multiplier 10.0, hash_k 7, bloom_path="partial"`
([provenance §7](provenance-and-disclosure.md) for the data).

## 3. End to end — the system comparison

`k = 100`, seed 0, eager, `bs = 16`, ms per forward; 30 harness records, all at
one `code_version`, `dirty: false`, `sm_mhz_load` **1410 MHz on all 30**
([O §16.5](../artifacts/official-silvertorch/b3/tables.md)).

| dataset | filter | n_probe | `triton` | `official` | `torch` | official / triton | torch / triton |
|---|---|---|---|---|---|---|---|
| arXiv | none | — | **0.668** | 0.846 | 4.852 | 1.27× | 7.26× |
| arXiv | bloom | 24 | **1.094** | 1.292 | 8.170 | 1.18× | 7.47× |
| arXiv | bloom | 32 | **1.084** | 1.451 | 10.806 | 1.34× | 9.97× |
| arXiv | exact | 24 | **0.701** | 7.110 | 7.863 | 10.14× | 11.22× |
| arXiv | exact | 32 | **0.709** | 7.131 | 10.410 | 10.06× | 14.68× |
| Goodreads | none | — | **0.857** | 1.325 | 16.918 | 1.55× | 19.74× |
| Goodreads | bloom | 24 | **1.112** | 2.063 | 28.516 | 1.86× | 25.64× |
| Goodreads | bloom | 32 | **1.350** | 2.260 | 37.870 | 1.67× | 28.05× |
| Goodreads | exact | 24 | **0.921** | 3.044 | 27.575 | 3.31× | 29.94× |
| Goodreads | exact | 32 | **1.119** | 3.291 | 36.638 | 2.94× | 32.74× |

Ratios are computed from the medians in O §16.5; the roadmap's one-line summary
of B3 rounds them to 1.3–1.6× unfiltered, 1.2–1.9× bloom, 2.9–10.1× exact, and
the per-cell figures above are the ones to quote. `exact` is the filter mode the
SilverTorch paper does not have — it is ours
([ST-3](reproduction-deviations.md)) — and on the official arm it runs through
our adapter's full-`N` mask, which §8 says how to read.

Four things the table says:

1. **The co-design is visible in Meta's own numbers, and it orders the two arms'
   filter modes oppositely.** For the official arm, bloom (partial masks over
   probed clusters) is *cheaper* than exact (a full-`N` `filtering_bit_mask`):
   1.29 vs 7.11 ms on arXiv, 2.06 vs 3.04 ms on Goodreads. For Triton the order
   reverses (1.09 vs 0.70, 1.11 vs 0.92) because our exact predicate is fused
   into the scorer while our bloom pays a query-side hash (§5). This is the
   paper's §4.4 co-design claim (S9) reproduced *from the vendor's side* — but it
   is **not** the controlled ablation, which needs `bloom_path="full"` cells and
   is **not yet validated — D1-e / G-b** ([ST-10](reproduction-deviations.md)).
2. **Graph mode is Triton-only.** All **78** official `graph` entries are null
   with `reason: not_capturable` — the official path is eager only
   ([OF-3](reproduction-deviations.md), O D7). `torch.compile(mode="reduce-overhead")`
   puts Triton at 0.35–1.21 ms at `bs = 16`: 1.6–2.6× over its own eager cell on
   arXiv, 1.09–1.15× on the Goodreads filter cells, and **0.96× — slightly
   slower — on Goodreads `none`**, the one cell where capture does not pay.
   Triton is the fastest arm in every graph cell too. The **eager column is the
   comparable number**; the graph column is a deployed-best-case that exists for
   one arm only.
3. **`torch` is a floor, not a straw man, and it is bit-exact**: `jaccard@100`
   1.0 and `score_max_abs_diff` 0.0 against Triton on all six filter cells, at
   7.3–32.7× the latency and 1.7–9.6 GiB of peak forward memory against Triton's
   32–150 MiB.
4. **At `bs = 1` the margin is 1.1–2.0× and is not a result** — see §9.

## 4. Kernel-only — where the vendor wins, and why we still do not lose

Algorithm 1 phases 2+3 only, `bs = 16`, `n_probe 24`, phase 1 precomputed and
excluded. `device µs` is one `torch.profiler` call classified by kernel name
([O §16.2](../artifacts/official-silvertorch/b3/tables.md)); the classifier is a
name heuristic and the raw per-kernel lists are in the artifacts.

| dataset | mode | arm | wall ms | device µs | scorer µs | prep µs | topk µs | launches | peak MiB |
|---|---|---|---|---|---|---|---|---|---|
| Goodreads | none | `triton` | **0.7517** | 721 | 337.0 | 41 | 335 | 38 | 37.7 |
| Goodreads | none | `official-fp16` | 1.3074 | 1146 | **30.9** | 710 | 338 | 75 | 214.6 |
| Goodreads | none | `official-int32` | 1.3750 | 1203 | 33.1 | 771 | 335 | 73 | 233.3 |
| Goodreads | bloom | `triton` | **1.0069** | 960 | 520.8 | 88 | 339 | 58 | 37.7 |
| Goodreads | bloom | `official-fp16` | 2.1563 | 1181 | **29.2** | 730 | 331 | 93 | 214.7 |
| Goodreads | exact | `triton` | **0.8177** | 778 | 397.6 | 37 | 339 | 35 | 37.7 |
| Goodreads | exact | `official-fp16` | 3.0496 | 1343 | 26.8 | 834 | 324 | 79 | 228.3 |
| arXiv | none | `triton` | **0.4977** | 302 | 129.8 | 40 | 124 | 38 | 10.8 |
| arXiv | none | `official-fp16` | 0.9362 | 529 | **113.3** | 268 | 120 | 75 | 60.9 |
| arXiv | bloom | `triton` | **0.8953** | 452 | 228.6 | 86 | 124 | 58 | 10.8 |
| arXiv | bloom | `official-fp16` | 1.6064 | 543 | **76.8** | 294 | 120 | 93 | 61.0 |
| arXiv | exact | `triton` | **0.4770** | 423 | 260.8 | 33 | 125 | 34 | 10.8 |
| arXiv | exact | `official-fp16` | 7.2117 | 815 | 81.7 | 271 | 119 | 77 | **826.7** |

**Meta's scoring kernel is faster than ours in every cell**: 1.15× arXiv
unfiltered (113 vs 130 µs), 3.0× arXiv bloom (77 vs 229 µs), and **10.9×**
(31 vs 337 µs) to **17.8×** (29 vs 521 µs) on Goodreads.

**The 10× is our probe layout, not our scorer.** Our padded IVF allocates
`max_tensor_size_per_row = n_probe × max_cluster_size` slots per query: on
Goodreads that is **611,520 slots** where the 24 probed clusters hold **≈ 18.7 k
real items**, so **97 % of what our kernel walks is `-1` padding**. On arXiv's
less skewed IVF it is 171,648 slots and **59 %** padding. The official op reads a
CSR and visits only real items. "At equal `n_probe`" is therefore equal *recall*
but **not equal work**, and that — not bandwidth, not instruction mix, and not
the 310-vs-3,500 line difference — is what the Goodreads factor measures. Both
arms' scorers lower to the same `dp4a` instruction path
([O §8 TF-3](../roadmap.md#phase-g-after-the-paper)), so the comparison is
purely about what each design *reads*.

**The official arm gives all of it back in payload prep.** Its scans,
`repeat_interleave`s, fills and gathers cost **268–834 µs across 73–93 launches**
in the rows above (O §16.2's prose quotes 268–896 µs over 73–95 launches across
the full record) against Triton's 34–58 launches, so the official arm's *total
device time* is **1.20–1.93×** ours in these rows even where its scorer is 10×
faster. The launch-count finding of [OF-4](reproduction-deviations.md) — a
host-side count when it was made — is now confirmed at the timing level.

The shared `topk` epilogue is 324–339 µs (Goodreads) / 119–125 µs (arXiv) in
every row and dominates the unfiltered Triton cell, which is why the wall ratios
are much smaller than the scorer ratios. **A reader who wants the scorer
comparison must read the scorer column, and a reader who wants the system
comparison must read §3 — neither substitutes for the other.**

## 5. Phase 2 alone — the transposed index, replicated against Meta's code

Full-`N` mask, `bs = 16` ([O §16.3](../artifacts/official-silvertorch/b3/tables.md)):

| dataset | op | arm | median ms | p99 ms | timer |
|---|---|---|---|---|---|
| Goodreads | `bloom_match` (row-wise, full `N`) | ours | 0.4517 | 0.5610 | CUDA events |
| Goodreads | `bloom_index_search_batch` (transposed, packed) | official | **0.2279** | 0.2588 | CUDA events |
| Goodreads | `…_return_partial_response` (probed clusters only) | official | 0.4488 | 0.4971 | CUDA events |
| Goodreads | `build_query_signatures` (our query-side bloom bits) | ours | 0.3687 | 0.4191 | CUDA events |
| arXiv | `bloom_match` (row-wise, full `N`) | ours | 1.4809 | 1.4898 | CUDA events |
| arXiv | `bloom_index_search_batch` (transposed, packed) | official | **0.2443** | 0.2814 | CUDA events |
| arXiv | `…_return_partial_response` (probed clusters only) | official | 0.4513 | 0.5039 | CUDA events |
| arXiv | `build_query_signatures` (our query-side bloom bits) | ours | 0.3780 | 0.4580 | CUDA events |
| both | `queries_to_expressions` (host, incl. its D2H) | official | 0.0272/0.0273 | — | `perf_counter` |
| both | `parse_expression_query_batch` (host) | official | 0.0468 | — | `perf_counter` |

- **SilverTorch's transposed-index claim (S13) replicates, and it scales the way
  the argument predicts.** The official transposed search is **2.0× faster at
  0.8 M items and 6.1× at 3.0 M**; our row-wise `bloom_match` grows **3.3×** from
  0.8 M to 3.0 M items while theirs grows **1.07×**. This is measured against the
  vendor's code, not inferred from the hand-written CUDA backend that was deleted
  at roadmap B4 ([ST-9](reproduction-deviations.md)).
- **Their partial-response path is *slower* than their own full-`N` search** at
  these sizes (0.449 vs 0.228 ms; 0.451 vs 0.244 ms) — 13 launches, 2 syncs and a
  `repeat_interleave` against one launch. The co-design still wins end to end
  (§3) because it shrinks what the *scorer* then reads, not because phase 2 gets
  cheaper. That is a useful correction to a natural misreading of the paper's
  §4.4.
- **Our query-side bloom hashing costs 0.37 ms at `bs = 16`** — more than Meta's
  entire mask search — and it sits inside our fused bloom forward, accounting for
  most of the gap between our `none` (0.75 ms) and `bloom` (1.01 ms) kernel-only
  cells. Under CUDA graph it largely disappears.
- **Host parse**: 46.8 µs/call at `bs = 16` (2.9 µs/query) plus 27.2 µs for
  `queries_to_expressions` including its `.tolist()` sync — ≈ 74 µs of CPU work
  per official bloom forward, paid inside every timed call here
  ([OF-5](reproduction-deviations.md)).

**Bloom selectivity and memory at the shipped settings** (O §16.3):

| dataset | exact pass rate | our bloom FP rate | official FP rate | our bloom MiB | official index MiB |
|---|---|---|---|---|---|
| Goodreads `c0_genre` | 0.3323 | **0.000000** | **0.000000** | 97.3 (`m_bits = 1024`) | **33.3** (`b_multiplier = 10`) |
| arXiv `c0_maincat` | 0.1357 | **0.000000** | **0.000000** | 364.9 | **71.3** |

Both blooms have **zero false positives** on these single-clause sweeps, so the
matched-FPR protocol of [O §4.3](../system/kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend) is
**undefined here** — there is no FPR to match. What is comparable is memory at
equal (zero) FPR, where the official index is **2.9× / 5.1× smaller**: our fixed
`m_bits = 1024` ([ST-2](reproduction-deviations.md)) is simply over-provisioned
at these vocabularies. The FPR-versus-width curve (S8) is **not yet validated —
D3**.

## 6. Parity — bit-exact where it can be, and where fp16 costs something

Kernel-only, 512 queries per cell, against the Triton arm on the same batches
([O §16.4](../artifacts/official-silvertorch/b3/tables.md)):

| dataset | mode | score path | `jaccard@100` | `score_max_abs_diff` |
|---|---|---|---|---|
| Goodreads | none | int32 | **1.000000** | **0.0** |
| Goodreads | none | fp16 | 1.000000 | 2.885e-03 |
| Goodreads | bloom / exact | int32 | 0.999961 | **0.0** |
| Goodreads | bloom / exact | fp16 | 0.999961 | 2.916e-03 |
| arXiv | none | int32 | **1.000000** | **0.0** |
| arXiv | none | fp16 | 0.982928 | 4.814e-04 |
| arXiv | bloom / exact | int32 | **1.000000** | **0.0** |
| arXiv | bloom / exact | fp16 | 0.985788 | 4.814e-04 |

**The official int32 path is bit-exact against our Triton kernel on both
datasets in all three filter modes** — confirmed at scale on real data, not only
in B2's unit regimes ([OF-8](reproduction-deviations.md)). Goodreads' 0.999961
comes with `score_max_abs_diff = 0.0`: identical scores, ids differing only where
scores tie. This is the reproduction result that matters most for RQ1: an
independent implementation written from the paper's text produces the vendor
kernel's exact arithmetic.

### 6.1 The Goodreads/arXiv jaccard split is fp16 resolution, not filtering

End to end the official arm reaches `jaccard_vs_first@100` **0.9998** on
Goodreads and **0.985** on arXiv. The natural reading — recorded as unresolved in
[OF-7](reproduction-deviations.md) and [R-3](reproduction-deviations.md), and as
a hypothesis in O's steer — was "more near-ties at the rank-100 boundary under a
looser filter". **B3 falsifies it.** It is near-ties, but it has nothing to do
with the filter and everything to do with the embeddings:

- the **unfiltered** cells split the same way (arXiv 0.9838 end to end, 0.9829
  kernel-only, against Goodreads 0.9999 / 1.0000), so no filter is involved;
- on the **int32** path both datasets are bit-exact in all three modes, so it is
  the score path, not the candidate selection.

| dataset | mode | rank-100/101 gap p10 | gap median | score@100 median | one fp16 ulp there | rows with gap < 1 ulp |
|---|---|---|---|---|---|---|
| Goodreads | none | 9.46e-04 | 6.97e-03 | 0.5623 | 2.75e-04 | **3.3 %** |
| Goodreads | bloom / exact | 1.10e-03 | 6.85e-03 | 0.6320 | 3.09e-04 | **3.1 %** |
| arXiv | none | 1.38e-05 | 7.92e-05 | 0.8363 | 4.08e-04 | **95.3 %** |
| arXiv | bloom / exact | 1.41e-05 | 8.57e-05 | 0.8303 | 4.05e-04 | **94.5 %** |

arXiv's Nomic text embeddings put the 100th and 101st candidate **8.6e-5 apart on
a score of 0.83** — a fifth of an fp16 ulp — while Goodreads' gSASRec scores are
22× further apart than their ulp. Under a score path that rounds to fp16, **95 %
of arXiv queries can swap their boundary ranks and ≈ 3 % of Goodreads queries
can.** The mechanism is a property of the *corpus's score distribution meeting
fp16 resolution*, and it is the number to quote rather than the jaccard alone.

What it costs in the metric anyone cares about — recall@100 against the exact
oracle, end to end, 10,000 queries:

| dataset | `triton` | `official` | Δ |
|---|---|---|---|
| arXiv `c0_maincat`, n_probe 24 | 0.884044 | 0.883735 | **3.1e-4** |
| Goodreads `c0_genre`, n_probe 24 | 0.912800 | 0.912796 | **4e-6** |
| Goodreads `c0_genre`, n_probe 32 | 0.936908 | 0.936910 | **−2e-6** (official higher) |

A 0.985 jaccard that costs 3.1e-4 of recall is a ranking-order difference at a
boundary, not a quality difference; saying only "0.985" would overstate it, and
saying only "3.1e-4" would hide the mechanism. **This supersedes the "looser
filter" reading recorded in [OF-7](reproduction-deviations.md) /
[R-3](reproduction-deviations.md)**, which predate B3.

## 7. Memory

- **Index.** The official index is smaller wherever our padded layout bites:
  **2.70×** on Goodreads `none` (110 vs 297 MiB), 2.75× Goodreads bloom, 1.90×
  Goodreads exact, 1.63× arXiv bloom — but only **1.01–1.02×** on the arXiv
  `none` and exact cells, where the CSR's two permutation vectors nearly cancel
  the padded table they replace. The skew that costs us kernel time in §4 is the
  same skew that costs us ≈ 200 MiB on Goodreads.
- **Forward peak.** The official arm's peak is ≈ 1.9× ours on the bloom and
  `none` cells and **26× ours on the arXiv exact cell** (827 vs 32 MiB) — the
  latter is our adapter, not Meta's op (§8).
- **The `torch` floor** peaks at 1.7–9.6 GiB against Triton's 32–150 MiB at
  `bs = 16`.

## 8. Two caveats that change how the numbers must be read

**(a) Every official *exact* number is an upper bound, and the cost is ours.**
The arXiv exact cell's 7.21 ms wall has only **0.50–0.82 ms of device time**. The
rest is `ops/official/adapter.pack_mask` — **our** adapter building a full-`N`
`filtering_bit_mask` over 2.99 M items, whose `[B, N/64, 64]` intermediate is the
**826 MiB** peak and whose reduction the kernel-name classifier files under
"quantize". `filter_mode="exact"` is not in the SilverTorch paper at all
([ST-3](reproduction-deviations.md)) and Meta's ops provide no exact predicate
([O §1.1](../system/kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend)), so this path is a
bridge we built. **The 10.1× and 3.3× exact ratios of §3 therefore bound what
that path costs today, not what Meta's kernels cost**; a transposed or chunked
packer is Phase G work and B3 deliberately measured rather than fixed it.

**(b) No claim here rests on a `bs = 1` margin.** 31 of 468 end-to-end perf
entries are `unstable`, **26 of them at `bs ∈ {1, 8}`**, with spreads to 22.5 %.
Of the 156 `bs = 16` entries, 5 exceed the 5 % threshold; the only one inside a
headline ratio is arXiv bloom `official` at `n_probe 24`, at **8.1 %** — read
that 1.18× as 1.1–1.3×. 14 of 30 cells carry `unstable: true`, every one either a
`bs ∈ {1, 8}` eager entry or a `clocks_drift` flag raised by a host-bound arm
dropping the card from 1410 to 1155–1395 MHz. Those clock drops are not noise:
they are the official arm at `bs = 1`, whose GPU is idle *inside* the measured
call while the host does a `.tolist()` sync and a CPU expression parse. Read
those rows as host-bound by construction, and see
[provenance §4](provenance-and-disclosure.md) for why batch 1 is not a 5 % target
on this box.

## 9. What this comparison cannot say

Stated here rather than left for a reviewer to find:

1. **No graph-mode comparison exists.** The official path is eager only — five of
   seven ops fail CUDA-graph capture and `bloom_index_search_batch` captures and
   then faults on replay ([OF-3](reproduction-deviations.md), O D7). All 78
   official `graph` entries are null. Comparing our graph column against their
   eager column would be a category error; §3's eager column is the comparison.
2. **No occupancy, sector or bandwidth analysis.** `ncu` is blocked in this
   container ([provenance §1](provenance-and-disclosure.md)), so every kernel
   number here is `torch.profiler` **device time**, and the split into
   scorer / mask / prep / topk is a **kernel-name heuristic** (O §16.2). The
   arXiv official-exact row's "quantize" entry is in fact our packer's reduction.
3. **One seed, one `k`, one dimension, one box, one GPU.** Seed 0 throughout;
   the parity analysis is at `k = 100` (the `k ∈ {500, 1000}` jaccards exist in
   the JSONL and are unanalysed); d128 only; a single shared A100 whose SM clock
   **cannot be locked**. **No confidence intervals and no paired tests stand
   behind any ratio in this document — D1-b.**
4. **No matched-FPR bloom comparison.** Undefined at these settings (§5); matched
   *memory* is reported instead.
5. **Neither paper's production configuration is reproduced**: single GPU, ≤ 3.0 M
   items here, no OverArch, no Value Model, single-embedding queries
   ([ST-12](reproduction-deviations.md)). No number here is compared against a
   latency printed in either paper.
6. **The synthetic layout ladder of O §9a
   was not run**; the two real datasets' own `P` (611,520 and 171,648 at
   `n_probe 24`) replaced it, which is where the pad-tax finding came from.
7. **`build_s` is not compared.** Index-build time for the two bloom indexes is a
   D1 column.

## 10. What is still missing, and which stage supplies it

Every row is a hole this section leaves open on purpose. Until the named step's
gate is green the item is not paper material (CLAUDE.md rule 2).

| what the section still needs | waits on |
| --- | --- |
| multi-seed cells, bootstrap CIs and paired tests behind **every ratio above** | **D1-b** (P G4) |
| the paper's actual latency / QPS / recall tables — these 30 records are a head-to-head run, not the campaign; every table is `report.py` over the campaign records | **D1** → **D4** |
| mean / p95 / p99, per-query latency vectors, open-loop QPS under a P99 budget | **D1** (P G3) |
| the `k ∈ {500, 1000}` official-vs-Triton ratios and jaccards | **D1** |
| the full unfiltered (`quality`-suite) sweep; B3 narrowed it to `k = 100`, `bs ∈ {1, 8, 16}` | **D1-c** |
| the controlled S9 co-design ablation (`bloom_path="full"` vs `"partial"`, latency and scratch vs probe count) | **D1-e** / **G-b** |
| bloom FPR and memory against filter width on real attributes, both blooms (S8) | **D3** |
| external baselines at matched recall (Faiss GPU/CPU, HNSW, cuVS) | **D2** (P G5, G13, G14) |
| a Triton transposed bloom index, and any claim that the §5 phase-2 gap has been closed | **G-a** (O §8 TF-1) |
| any post-fix number for the padded probe layout (TF-9) or for `pack_mask` (§8a) | **Phase G**, after D1 — a library change invalidates every recorded cell through the tree-hash resume key |
| scale beyond 3.0 M items | **G-b** (P G10); the large datasets are **E2**–**E4** |

## 11. What the reproduction got wrong before it measured

A reproduction paper's value is that it *checked*. Ours wrote its expectations
down before the run (O §9), and
two of the four were wrong — one of them in both directions.

| expectation, as written | verdict |
|---|---|
| (i) kernel-only, unfiltered, large `P`: "parity within ±20 %", the headline being "a 310-line Triton kernel within X % of the vendor's" | **Wrong in both directions.** Meta's scorer is 1.15–17.8× faster than ours, and our *forward* is still 1.7–2.1× faster than theirs because their op spends 268–896 µs in prep. The Goodreads factor is our layout's pad tax, not bandwidth. |
| (ii) eager wall at `bs = 1` / small `P`: official 2–3× slower from launches and syncs | **Confirmed, milder**: 1.4–2.5× kernel-only, 1.1–2.0× end to end — and the card visibly drops clock in exactly those cells. |
| (iii) bloom kernel-only at `bs = 16`: the official partial mask + masked scorer beats our row-wise fused kernel by ≈ 2× | **Wrong as stated.** Their *phase 2* beats ours by 2.0–6.1×, but their bloom *forward* is **1.8× (arXiv) / 2.1× (Goodreads) slower** than our fused one, and their partial-response op is slower than their own full-`N` search. |
| (iv) quality: identical candidate sets up to fp16 ties | **Confirmed and quantified**: int32 bit-exact; fp16 costs 3.1e-4 recall@100 on arXiv and 4e-6 on Goodreads (§6). |

The corrections are not cosmetic. (i) redirected our kernel work: the scorer
retunes we had queued are second-order next to the **padded probe layout itself**,
which is what costs 10.9–17.8× on a skewed IVF. (iii) strengthened the case for a
Triton transposed bloom index while removing the reason we thought we needed it.
Both are Phase G, and both are stated here as *motivation*, not as results.
