---
title: method-differences
created: 2026-10-12
updated: 2026-10-12
type: comparison
tags: [paper]
sources: [articles/linr.md, articles/silvertorch.md, official-fork/CHANGES.md, retrieve/src/retrieve/modules/filters.py, retrieve/src/retrieve/modules/linr.py, retrieve/src/retrieve/modules/silvertorch.py, retrieve/src/retrieve/ops/official/, evaluation/bench/, research/decisions.md, research/paper/reproduction-deviations.md, research/paper/meta-published-vs-adapted.md, research/paper/results.md]
---

# Method differences: what we report against what the papers describe

What each reported number measures, method by method, against what LiNR's
paper ([articles/linr.md](../../articles/linr.md)), SilverTorch's paper
([articles/silvertorch.md](../../articles/silvertorch.md)) and Meta's
published code (`backend="official"`, pinned `21aa35e`) do. Filtering
comes first, since that is where the methods differ most and where our
configuration is being changed now.

This page summarises; the per-item records stay where they are:
[reproduction-deviations](reproduction-deviations.md) (paper against us,
with the evidence),
[meta-published-vs-adapted](meta-published-vs-adapted.md) and
[`official-fork/CHANGES.md`](../../official-fork/CHANGES.md) (what our
adaptation of Meta's code changed), [results](results.md) (the numbers).
Paper sections are the arXiv numbering ([claims](claims.md)). Anything not
confirmed from code or text is marked **unverified**.

## 1. What the reported numbers mean

| quantity | what it is |
|---|---|
| **recall@k** | per query, the share of the exact filtered top-k that the method returns (the k items with the highest fp32 inner product over the whole catalogue that pass the **exact** filter), averaged over queries; it is not the papers' end-task recall ([§4.1](#41-metric)) |
| **held-out (end-task) recall** | per query, whether the held-out item (the next interaction, the paired document) is in the top-k; recorded beside recall@k and used on Goodreads to show the two move together |
| **latency** | median milliseconds per batch, one process, one A100-SXM4-80GB, CUDA-graph replay unless the table says eager, query preparation and index build not included |
| **ratio** | SilverTorch's latency over the faster of LiNR V1 and V2 in the same cell (same dataset, filter, filter kind, batch size, k); below 1 means SilverTorch is faster |
| **tuned point** | the fastest timed SilverTorch setting (lists, probes) whose recall@k reaches the target (0.90 or 0.95); "not reached" means no timed setting did, and "top" is the best recall seen |
| **pass rate** | the share of the catalogue that passes the **exact** filter, averaged over queries; on synthetic filters it is the rate the filter was built for |
| **bloom false-positive rate** | among items that fail the exact filter, the share our standalone bloom filter lets through, averaged over queries; on SilverTorch cells it is still **our** bloom's rate, not that of Meta's bloom index ([§2.5](#25-pass-rate-and-false-positive-rate-definitions)) |
| **QPS** | batch size over per-call wall time, one closed-loop client in process: not the papers' client-side QPS under a P99 budget |
| **index MiB / build s** | the bytes of every buffer of the built module (index, codes, filter) / wall time of construction and index registration |
| **unstable** | the three timing windows' medians spread more than 5 %, or the SM clock drifted more than 5 %; clocks cannot be locked on the pods |

## 2. Filters

### 2.1 The two filter kinds

| | exact clause filter | bloom filter |
|---|---|---|
| our name | `ExactAttributeFilter`, `filter_kind: clause`; SilverTorch `filter_mode="exact"` | `BloomFilter`, `filter_kind: bloom`; SilverTorch `filter_mode="bloom"` |
| LiNR's name | attribute-based matching (ABM): clauses, one of them a "reverse matching clause" (§3.1) | none: LiNR has no bloom filter. **LiNR with a bloom filter is ours** (user: evaluate LiNR with both kinds to see the cost of a probabilistic filter) |
| SilverTorch's name | the "forward index", the GPU baseline the paper compares its bloom index against (§4.1, §6.2.2) | the "bloom index" (§4.1) |
| what a query can say | `C` clauses fixed at index time; per clause one query value; an item passes a clause when any of its values equals it (negated for a reverse clause); AND across clauses ([filtering](../../docs/system/filtering.md#linr-paper-31--fixed-clause-schema-no-dsl)) | the same clauses, forward only: no reverse clause, because NOT over a bloom would drop true matches (`filters.py:50`, `adapter.py:102-110`) |
| answer | exact: no false positives or negatives | a superset of the exact answer: false positives (hash collisions), no false negatives |
| what the papers allow | LiNR: the same schema (§3.1). SilverTorch: nested AND / OR / NOT over `feature = value` (§4.3); our schema covers the conjunctive part only ([ST-4](reproduction-deviations.md#3-silvertorch-as-we-run-it--deviations-from-the-paper)) | SilverTorch: the bloom with NOT through its expression stack (§4.3), how a NOT stays exact over a bloom is not said (**unverified**) |
| reverse sweeps | run | none: goodreads `c1_lang_reverse`, `all4` and PubMed `c3_journal_reverse`, `all5` have no bloom cell (`evaluation/config/goodreads.yaml:14`, `pubmed.yaml`); YFCC runs clause only ([decisions](../decisions.md#datasets)) |

### 2.2 Where each method's filter runs, and what happens to false positives

| method | exact clause filter | bloom filter | false positives |
|---|---|---|---|
| **LiNR V1** | over all N, fused into the score write: failing items get score −∞, then top-k (`linr.py`, `LiNRV1.forward`). Paper: multiply the scores by the 0-1 masks, so failing items score 0 (§3.1.1) | our bloom over all N, the same fused −∞ write | kept: a false positive keeps its true score and can enter the top-k |
| **LiNR V2** | a compact list of passing ids (`clause_compact`), then gather + dot + top-k over it. Paper: slice the matrix to the passing items, then matmul and top-k (§3.1.2) | the same with `bloom_compact` | kept, as candidates |
| **LiNR V3** | pre-filter as V2, then 1-bit Hamming top-`candidate_pool` among the survivors, then exact rescoring (`LiNRV3.forward`) | the same with the bloom | kept, as candidates |
| **SilverTorch, paper** | not in the method | after probe selection, on the probed clusters' items only, fused with scoring (Algorithm 1, §4.3) | accepted: "eliminated by subsequent ranking stages" (§4.1) |
| **SilverTorch, Meta's code** | no exact filter: the scorer takes any bit mask (`filtering_bit_mask`), which our adapter filled with our exact mask ([filtering](../../docs/system/filtering.md#implementation-status)) | Meta's bloom over the probed clusters (the partial path, default) or over all N (`bloom_path="full"`) | returned ([OF-12](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper)) |
| **SilverTorch, adapted (the study)** | our exact test inside Meta's search, on the probed docs only (CHANGES C9). **Ours, not Meta's** (user: "exact on silvertorch is our adaptation") | Meta's bloom, evaluated inside the payload kernels on the probed docs only (C8); the full-N path is refused (K fix 17) | with the recheck (C13) every bloom survivor is tested exactly, so none is returned; without it they are returned as in Meta's code ([§2.4](#24-the-recheck)) |
| **postfilter baseline** | top-`alpha·k` by score over all N, then the filter on those candidates, first k survivors kept ([evaluation](../../docs/system/evaluation.md#the-postfilter-baseline)) | the same with the bloom | kept |

In both SilverTorch arms the filter never sees items in unprobed clusters,
so a passing item there is a miss whatever the filter does; that loss is
the IVF partition's, not the filter's.

### 2.3 The bloom implementations and their parameters

Three different blooms appear in the study. They share nothing but the
idea; equal widths do not give equal false-positive rates.

| | SilverTorch paper (§4.1, §6.2.2) | Meta's code as published | adapted (the study's SilverTorch) | LiNR's bloom (ours) |
|---|---|---|---|---|
| layout | an M-bit signature per item, stored transposed so one 64-bit AND tests 64 items | one bloom per 2048-doc bundle ([OF-1](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper)) | Meta's bundle layout | an M-bit signature per item, row-wise `[N, M/64]` int64 ([filtering](../../docs/system/filtering.md#implementation-status)) |
| width | M per item; 512-2048 bits measured; heuristic max values × hashes × 3 | `max terms in the bundle · k · b_multiplier` bits per doc; `m_bits` ignored ([OF-10](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper)); `b_multiplier` 10 in our config (`ops/official/__init__.py:123`) | `m_bits` per doc in every bundle (C12, `bloom_width="configured"`); **1024** in the study | `m_bits`, **1024** in the study |
| hashes | K = 5 | murmur3 over `(feature, value, seed)`; the search ANDs the first `k` distinct of 7 stored hashes | as Meta, `k` = **5** | splitmix-style, a per-clause salt, `k_hash` = **5** ([ST-1](reproduction-deviations.md#3-silvertorch-as-we-run-it--deviations-from-the-paper)) |
| sweep | FPR at 512 / 768 / 1024 / 2048 bits | one width | `m_bits` 64-2048, `k_hash` 3 / 5 (the bloom-width suite, quality only) | follows the suite width (`evaluation/config/suites.yaml:259`) |
| reported FPR | 6.98 at 512 bits, 0.067 at 1024 (units not stated; read as %); elsewhere 0.00173 % at 1024 (§6.2.4), inconsistent | — | **not measured directly**: see [§2.5](#25-pass-rate-and-false-positive-rate-definitions) | the record's `bloom_fp_rate` |

Study defaults: `m_bits` 1024, `k_hash` 5 for both arms
(`evaluation/bench/algos.py:52`, `research/artifacts/numbers/suites.yaml:1-3`).
How a query is encoded: LiNR's bloom ORs the bit positions of every active
`(clause, value)` into one query signature and tests `(q & sig) == q`
([filtering](../../docs/system/filtering.md#bloom-hash-keys-clause_idx-value));
for Meta's bloom the active clauses are rendered as `"0:v AND 1:w"`,
parsed by Meta's parser into plans once per batch outside the timed call,
and searched by Meta's ops (`adapter.py:102-131`,
[evaluation](../../docs/system/evaluation.md#query-preparation)).

### 2.4 The recheck

The adapted code can test every bloom survivor exactly before scoring
(CHANGES C13, `OfficialConfig.recheck`, `retrieve/src/retrieve/ops/official/__init__.py:111-115`).
Bloom ∧ exact is the exact filter, so with the recheck the bloom changes
speed, not results. It is not in Meta's code or either paper.

- **Library default:** on for `backend="official-fork"`
  (`OfficialConfig.recheck_for`, `__init__.py:164-170`); Meta's code has none.
- **The study's setting:** **off** (user, 2026-10-12: "NO RECHECKS in timed
  code, only as part of grading the recall"). SilverTorch is evaluated with
  its probabilistic bloom, false positives allowed; the cost shows up as
  lost recall against the exact oracle. LiNR's bloom never had a recheck.
- **What the current numbers used:** the harness cannot yet set the flag
  (`algos.official_config` passes only `bloom_path` / `score_path`,
  `evaluation/bench/algos.py:87`), so **every SilverTorch bloom number in
  [results](results.md) ran with the recheck on** and its recall equals the
  exact filter's. The recheck-off numbers are not yet measured.
  [decisions](../decisions.md#goal-and-scheduling) still records "recheck on
  by default"; the user does not stand behind it.

### 2.5 Pass rate and false-positive rate definitions

| name | definition | source |
|---|---|---|
| pass rate (ours) | mean over kept queries of (items passing the **exact** filter) / N | `evaluation/bench/oracle.py:106-109` |
| synthetic rate `p` | each item draws `u ~ U(0,1)` once; it passes the rate-`p` clause iff `u < p`, so the sets nest and are unrelated to the embeddings; `arxiv-corr-synth` instead passes the items near the query's centroid | [datasets](../../docs/system/datasets.md#synthetic-selectivity-attrs) |
| local pass rate | the share of a query's **unfiltered** top-100 that passes; what SilverTorch's recall follows (C6) | [results](results.md#claim-verification) |
| bloom false-positive rate | mean over queries with at least one failing item of (bloom passes − exact passes) / (N − exact passes), computed with **our standalone bloom** (LiNR's) at the cell's `m_bits` / `k_hash` | `oracle.py:112-118`, `run.py:89-102` |
| LiNR's "pass rate" | items passing per query: high-pass set ≈ 1.7 M of 15.5 M on average (two clauses: geo, reverse company); low-pass set adds a title clause, at most 1.2 M, most queries thousands | linr.md §5.3 |
| SilverTorch's | no pass-rate axis; FPR against bloom bits on 5,000 production filter queries (6 features, ~10 values per item) | silvertorch.md §6.2.2 |

**The false-positive rate on SilverTorch cells is not SilverTorch's.**
On `official` and `official-fork` cells the harness builds the standalone
filter on the Triton backend (`algos.filter_backend`) and computes
`bloom_fp_rate` from it ([evaluation](../../docs/system/evaluation.md#bloom-widths-as-build-params)).
The C4 rates in [results](results.md#claim-verification) (PubMed 2.4 % at 64
bits to 9e-6 at 1024, and "Meta's code as published, 9e-6") come from that
column (`research/artifacts/results-draft/build_results.py:521-528`), so
they are **our** bloom's rates at the same width and hash count, not those
of Meta's bloom with or without the recheck. The adapted bloom's own rate
before the recheck is **unverified**; its effect on results was measured
only as "0 false positives returned" with the recheck on
([gate R](meta-published-vs-adapted.md#3-the-parity-gate)) and as false
positives in Meta's results at 30 M
([OF-12](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper)).

### 2.6 How a false positive reaches recall

Recall is graded against the exact filtered fp32 oracle, so the "recheck"
is in the grading, never in the timed call
(`evaluation/bench/metrics.py:36-52`):

- a false positive that outranks a true passing item pushes it out of the
  top-k: one miss;
- a false positive in a slot that would otherwise be empty (fewer than k
  items pass) costs nothing: the denominator is the oracle's own list,
  `min(k, items passing)`; it lowers precision@k only;
- queries with no passing item are left out of the average
  (`run.py`, `oracle_rows`);
- so the cost is largest at low pass rates with many passing items near
  the query (a false positive competes with true items) and smallest when
  hardly anything passes. In SilverTorch the competition is only among the
  probed clusters' items, where a very selective filter can leave only
  false positives ([OF-12](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper):
  a one-item filter returned nothing but false positives at 30 M).

## 3. Method by method

Columns: what the paper says / what Meta's code does (LiNR has no public
code) / what we do / why / effect on the reported numbers / source.

### 3.1 LiNR V1 (similarity masking)

| aspect | paper | public code | we do | why | effect on numbers | source |
|---|---|---|---|---|---|---|
| filter | exact clause masks, multiplied into the scores (failing items → 0), then top-k | none | exact clause filter fused into the score write (failing → −∞), or our bloom | 0 would let failing items outrank passing items with negative scores; −∞ is the exact semantics | exact V1 equals the oracle up to fp16 (§4.4); bloom V1 can lose recall to false positives | linr.md §3.1.1; `linr.py` `LiNRV1`; `functional.masked_topk` |
| scoring | fp16 embeddings, custom CUDA filter op, separate un-fused ops in TF / PyTorch | none | fp16 items, fp32 accumulate (cuBLAS), Triton filter kernel fused with the score write | the fusion is the point of a GPU library | our V1 is faster than an un-fused one; the V1 / V2 crossover moves ([LN-1](reproduction-deviations.md#4-linr-v1v3--deviations)) | linr.md §3.1.1, §6; [LN-1](reproduction-deviations.md#4-linr-v1v3--deviations) |
| quality | "no recall drop" (exhaustive) | — | exact search over fp16 items, graded against the fp32 oracle | — | ≥ 0.99 recall@k everywhere except YFCC synth at k 1000 (0.983-0.995, fp16 storage, flagged) | [HB-6](reproduction-deviations.md#6-deviations-of-our-harness-against-the-papers-protocols--and-against-itself) |

### 3.2 LiNR V2 (explicit pre-filtering)

| aspect | paper | public code | we do | why | effect on numbers | source |
|---|---|---|---|---|---|---|
| filter | exact clauses first, slice the passing items | none | `clause_compact` (or `bloom_compact`): one fused filter + stream-compaction kernel, ascending item order | slicing in torch copies the matrix; the fused kernel is what makes V2 fast | V2 faster than the paper's at high pass rate | linr.md §3.1.2; [filtering](../../docs/system/filtering.md#caller-patterns) |
| scoring | matmul + top-k on the slice, each query of a batch run separately | none | one fused gather + dot (fp32 accumulate) + top-k kernel over the batch's candidate lists | batching | V2 at bs 64 / 256 slower than V1 on every real filter; PubMed `c3_journal_reverse` / `all5` and YFCC at bs 256 time out | linr.md §5.3.2; [results](results.md#claim-verification) C1 |
| quality | exact | — | exact (clause) / bloom-limited (bloom) | — | as V1 | — |

### 3.3 LiNR V3 (1-bit Sign-OPORP, then rerank)

| aspect | paper | public code | we do | why | effect on numbers | source |
|---|---|---|---|---|---|---|
| filter | pre-filter as V2 | none | pre-filter as V2 (clause or bloom), then Hamming top-`candidate_pool` among survivors, then exact rescore | as the paper | a bloom false positive can take a pool slot | linr.md §3.2, Fig. 3; `linr.py` `LiNRV3` |
| bits | 512 bits at d128 in the benchmark; "1-bit embedding of the same dimension" in §3.2 | none | at most D bits (`k_bits` = D: 128 at d128, 192 YFCC, 768 / 256 PubMed) | OPORP as defined gives at most D bins | ¼ of LiNR's bit budget at d128: stage-1 recall can only be lower | [LN-8](reproduction-deviations.md#4-linr-v1v3--deviations); [decisions](../decisions.md#goal-and-scheduling) |
| pool | keep ~1 % of items (0.5-2.5 % swept) | — | pool as a fraction of the mean pass count (0.5-20 %), at least 2,000 | our catalogues are 1-2 orders smaller | 1 % reaches 0.89 median recall; parity needs 20 % | [evaluation](../../docs/system/evaluation.md#pool-fractions); [results](results.md#claim-verification) C2 |

### 3.4 SilverTorch

| aspect | paper | Meta's code as published | we do (adapted) | why | effect on numbers | source |
|---|---|---|---|---|---|---|
| filter | bloom index, probe then filter (Algorithm 1), false positives accepted | bundle bloom on the probed clusters or all N, false positives returned; no exact filter | study: Meta's bloom on the probed docs, **recheck off**; exact clause mode (C9) exists but is ours | user directive 2026-10-12 | the headline tables in [results](results.md) still use the exact clause mode and the bloom rows ran with the recheck on, so neither yet shows the bloom's false-positive cost | §2 above; CHANGES C8, C9, C13 |
| bloom width | M per item, 512-2048 | from the bundle's max term count, `m_bits` ignored | `m_bits` 1024 per doc (C12) | `m_bits` must mean something to sweep it | widths are comparable across datasets; not Meta's default index | [OF-10](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper); CHANGES C12 |
| quantization | int8 items, global min/max scaled to [−128, 127]; dp4a | int8 kernels; fp16 score path (`divisor`) by default | items: one global symmetric scale (abs max / 127); queries: a per-row int8 scale; **int32 score path** | the int32 path is bit-exact with Meta's; per-row query scale because the query is quantized per call | int8 sets a recall ceiling (C6: PubMed 0.92 vs 0.996 at fp16 on the same lists); fp16 is a reported variant only | silvertorch.md §4.2; [ST-5](reproduction-deviations.md#3-silvertorch-as-we-run-it--deviations-from-the-paper); [decisions](../decisions.md#goal-and-scheduling); `indexing/quantize.py:34` |
| clustering | k-means++ on GPU | none (phase 1 is ours) | Lloyd's k-means, 10 iterations, random init | every record used random init | whether init changes quality is **unverified** | [ST-7](reproduction-deviations.md#3-silvertorch-as-we-run-it--deviations-from-the-paper) |
| probes / lists | 24 probes in production, 32 / 64 / 256 in the analyses; 9,000 clusters at 81 M | — | lists 1,024-65,536 tuned per size; probes doubled from 8 to a quarter of the lists; 24 kept as a reference row | the tuned point is the fair comparison; 24 gives 0.05-0.89 recall here | the head-to-head is at equal recall, not at 24 probes | [results](results.md#at-the-papers-production-setting-24-probes); [evaluation](../../docs/system/evaluation.md#ivf-tuning) |
| execution | in a C++ predictor | eager only, host syncs every call ([OF-3, OF-4](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper)) | one search op, no host sync, CUDA-graph replay | without capture the timing is host overhead | the adapted code is 0.12-0.82 of Meta's eager time (median per dataset and filter) with identical ids and scores | [meta-published-vs-adapted](meta-published-vs-adapted.md); results C7 |
| co-design ablation | probe-then-filter 1.79-2.15× faster than filter-then-probe | both paths | full path refused (K fix 17) | — | C5 is measured on Meta's code as published only | results C5 |
| what is around it | user tower, OverArch, Value Model, 12-way queries, 2-GPU shard at 80 M | — | none | out of scope | no end-to-end QPS or TCO claim is tested | [ST-12](reproduction-deviations.md#3-silvertorch-as-we-run-it--deviations-from-the-paper) |

## 4. Everything else that changes a number

### 4.1 Metric

| | LiNR paper | SilverTorch paper | ours |
|---|---|---|---|
| recall | "recall label@2000": product labels, protocol not published in enough detail to rebuild (§5.3) | ANN recall at top-k 2048 / 4096 against Faiss (definition not stated, **unverified**); recall@20-1000 against user interactions with OverArch (§6.2.4) | recall@k against the exact filtered fp32 top-k ([§1](#1-what-the-reported-numbers-mean)) |
| bridge | — | — | on Goodreads our recall and held-out recall move together: Pearson 0.88-0.94 per filter; once ours reaches 0.90, held-out recall is 0.93-1.0 of the exact scan's ([validation](../validation.md#numbers-adapted-silvertorch-over-the-study-grid)) |
| consequence | no LiNR recall claim is called reproduced or not ([decisions](../decisions.md#goal-and-scheduling)) | — | — |

### 4.2 k, batch size, protocol

| | LiNR paper | SilverTorch paper | ours | effect |
|---|---|---|---|---|
| k | 2000 | 1024 end to end; 2048 / 4096 ANN | 100 and 1000; 2048 on arXiv, Goodreads, PubMed | k 1000 reaches the target in fewer cells than k 100 |
| batch | 1, 16 | 16 (ANN); QPS end to end | 1, 16, 64; LiNR also 256 | V1 / V2 order flips with batch size (C1) |
| latency statistic | average and p95 | average per batch; QPS at a 200 ms P99 budget | median per batch (mean, p95, p99 also recorded) | — |
| what is timed | the serving call (**unverified**) | client to server end to end; ANN with 50 warm-up + 100 batches | the module's forward only: query preparation (bloom query signatures, Meta's expression parse) and index build outside, reported separately; SilverTorch's forward includes probe selection | our latencies have no network, user tower or parse |
| warm-up / repeats | — | 5 runs averaged (end to end) | 50 warm-up calls, then 3 windows of 1,000-5,000 calls; the median window's median | — |
| mode | TorchScript serving | C++ predictor | CUDA-graph replay (headline), eager beside it; Meta's published code eager only | — |
| clocks | — | — | not lockable; sampled SM clock and an `unstable` flag per entry | sub-millisecond bs-1 differences under ~21 % are noise ([decisions](../decisions.md#harness)) |
| seeds | — | — | seed 0 in every reported number; seeds with the final rerun | no confidence intervals yet |

Sources: [evaluation](../../docs/system/evaluation.md#measurement-protocol),
[provenance §4](provenance-and-disclosure.md#4-timing-which-clock-estimator-and-why-the-question-is-not-pedantic),
[results](results.md#uncertainty).

### 4.3 Hardware, data, encoders

| | LiNR paper | SilverTorch paper | ours |
|---|---|---|---|
| GPU | A100 for the stretch test; the GPU of Tables 3-4 is not stated | A100 40 GB | one A100-SXM4-80GB, timings compared only within one box ([decisions](../decisions.md#harness)) |
| catalogue | 15.5 M LinkedIn jobs, 25 k queries (internal) | 10 M / 20 M / 40 M / 80 M production items, 5,000 replayed requests (internal) | Goodreads 0.8 M, arXiv 3 M, PubMed 10 M, YFCC 10 M, LAION 1 / 3 / 10 / 30 M |
| dimension | 128, fp16 | 128 | 128 (Goodreads, arXiv), 192 (YFCC), 256 (LAION), 768 (PubMed); native widths, no PCA |
| encoders | two-tower member / item model | User Tower (HSTU-style) | Goodreads: our sequential encoder (gSASRec body); arXiv, LAION: nomic-embed-text-v1.5; PubMed: MedCPT; YFCC: the Big-ANN CLIP descriptors ([provenance §7](provenance-and-disclosure.md#7-data-provenance)) |
| filters | geo, reverse company, job title; one attribute per item and clause | 6 features, ~7 conditions per query, AND / OR / NOT | real attributes (genre, language, category, MeSH, journal, tags, domain …), up to 4 values per item and clause; query values taken from the held-out target's attributes (`evaluation/eval_datasets/common.py:16`); plus synthetic rates 0.001-1 |
| baselines | TF vs PyTorch implementations | Faiss CPU / GPU, HNSW, inverted and forward index | each other, plus generic torch (dense matmul, top-k, then filter) and torch eager / compiled; no Faiss ([decisions](../decisions.md#harness)) |

### 4.4 The oracle and stored precision

| method | items stored | scored in | against an fp32 oracle |
|---|---|---|---|
| LiNR V1 / V2 | fp16 (as the paper) | fp32 accumulate | equal up to fp16 rounding; visible only on YFCC's narrow cone ([HB-6](reproduction-deviations.md#6-deviations-of-our-harness-against-the-papers-protocols--and-against-itself)) |
| LiNR V3 | 1-bit codes, then fp16 rescore | Hamming, then fp32 | pool size decides |
| SilverTorch | int8 codes, global scale | int32 dot, fp32 epilogue | int8 and the probed lists decide |
| oracle | fp32 | fp32, exact filter, ties to the lowest id | — (checked against YFCC's shipped ground truth: 100,000 / 100,000 queries id-exact, [HB-2](reproduction-deviations.md#6-deviations-of-our-harness-against-the-papers-protocols--and-against-itself)) |
