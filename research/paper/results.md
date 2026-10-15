---
title: results
created: 2026-10-11
updated: 2026-10-15
type: summary
tags: [paper]
sources: [research/artifacts/results-draft/build_results.py, research/artifacts/results-draft/page/pagedata.py, research/artifacts/hub-index.md, research/validation.md, research/paper/claims.md]
---

# Results (draft): SilverTorch and LiNR under filtering

> **Preliminary, not citable.** Every number comes from the final rerun, whose gate is still open
> ([validation](../validation.md#full-rerun)). Where each number comes from is in the last section. The same text,
> with interactive figures, is the page "SilverTorch vs LiNR"; both are built from one data file.

Two families of filtered GPU retrieval on one A100. **SilverTorch** (Meta's official code, adapted by us) probes a few
clusters of an int8 index and filters them with a bloom filter, which lets some false positives through. **LiNR** (our
implementation from the paper) filters exactly: V1 scores every item, then masks; V2 gathers the passing items, then
scores them; V3 ranks everything with 1-bit codes, then rescores a candidate pool. Five datasets, from 0.8 M to 30 M
items, and synthetic filters with a controlled pass rate.

**Recall** here measures fidelity to exact search: the share of the exact filtered top-k (the highest inner products
over the whole catalogue, under the same filter) that a method returns. A bloom false positive in the top-k counts as a
miss. This is not the papers' end-task recall, but on Goodreads the two move together: Pearson 0.99 at k 100 and 0.97 at k 1000 over SilverTorch's bloom settings, 0.87–0.93 per filter over every method. Times are medians per
batch, with SilverTorch and LiNR timed in the same process.

## 1. Quality, speed and memory under filtering

Table 1 shows each method on one real filter per dataset, at k = 100 and a batch of 16. SilverTorch appears at its
default of 24 probes and at the fastest setting that reaches recall 0.90 and 0.95. LiNR V3 appears at the smallest pool
(as a share of the passing items) that reaches the same targets. Speed-up is against exact LiNR V1. Three filters are
shown here; the page has every filter and batch size.

**Table 1a.** PubMed 10M, filter `c0_mesh` (pass 0.02 %), d = 768, k = 100, batch 16.

| Algorithm | Recall@100 | t_med, ms | Speed-up vs V1 | Setting | Index, MiB |
|---|--:|--:|--:|--:|--:|
| LiNR V1 | 1.000 | 11.909 | 1.00× | exact | 16,174 |
| LiNR V2 | 1.000 | 6.338 | 1.88× | exact | 16,174 |
| LiNR V3, pool for 0.90 | 0.999 | 6.360 | 1.87× | pool 0.5 % | 17,090 |
| LiNR V3, pool for 0.95 | 0.999 | 6.360 | 1.87× | pool 0.5 % | 17,090 |
| SilverTorch, 24 probes | 0.580 | **0.241** | **49.4×** | 4096 lists / 24 | 8,710 |
| SilverTorch, tuned for 0.90 | 0.909 | 0.795 | 15.0× | 4096 / 512 | 8,710 |
| SilverTorch, tuned for 0.95 | 0.952 | 2.240 | 5.32× | 65536 / 16384 | 8,891 |

**Table 1b.** LAION 30M, filter `tags4` (pass 0.27 %), d = 256, k = 100, batch 16.

| Algorithm | Recall@100 | t_med, ms | Speed-up vs V1 | Setting | Index, MiB |
|---|--:|--:|--:|--:|--:|
| LiNR V1 | 0.999 | 16.809 | 1.00× | exact | 16,022 |
| LiNR V2 | 0.999 | 19.915 | 0.84× | exact | 16,022 |
| LiNR V3, pool for 0.90 | 0.948 | 19.558 | 0.86× | pool 10 % | 16,937 |
| LiNR V3, pool for 0.95 | 0.983 | 19.673 | 0.85× | pool 20 % | 16,937 |
| SilverTorch, 24 probes | 0.492 | **0.289** | **58.2×** | 4096 lists / 24 | 11,448 |
| SilverTorch, tuned for 0.90 | 0.917 | 2.396 | 7.02× | 16384 / 2048 | 11,461 |
| SilverTorch, tuned for 0.95 | 0.955 | 4.414 | 3.81× | 16384 / 4096 | 11,461 |

**Table 1c.** LAION 1M, filter `c0_domain` (pass 0.95 %), d = 256, k = 100, batch 16.

| Algorithm | Recall@100 | t_med, ms | Speed-up vs V1 | Setting | Index, MiB |
|---|--:|--:|--:|--:|--:|
| LiNR V1 | 1.000 | 0.655 | 1.00× | exact | 534 |
| LiNR V2 | 1.000 | 0.854 | 0.77× | exact | 534 |
| LiNR V3, pool for 0.90 | 0.949 | 0.811 | 0.81× | pool 0.5 % | 565 |
| LiNR V3, pool for 0.95 | 0.952 | 0.820 | 0.80× | pool 20 % | 565 |
| SilverTorch, 24 probes | 0.567 | **0.208** | **3.15×** | 512 lists / 24 | 382 |
| SilverTorch, tuned for 0.90 | 0.923 | 0.598 | 1.10× | 8192 / 4096 ◐ | 390 |
| SilverTorch, tuned for 0.95 | 0.962 | 0.837 | 0.78× | 8192 / 6144 ◐ | 390 |

The results show three main patterns. **First**, at 24 probes SilverTorch is 3–58× faster than V1 at a batch of 16, but
it returns a different answer: recall@100 is 0.33–0.89. Its time barely moves with catalogue size (0.19–0.29 ms per batch from 0.8 M to 30 M items), while V1's grows with it (0.57 ms on Goodreads, 16.8 ms on LAION 30M), consistent with SilverTorch scanning only 24 of 512–4,096 clusters and V1 scoring every item. **Second**, held to recall 0.90, SilverTorch still wins by a wide margin wherever a
modest share of clusters is enough: 15× V1 on PubMed's MeSH filter, 7.0× on LAION 30M's tag filter, 6.7× on arXiv's main
category. Where it must probe half the clusters (LAION's domain filter at 1–10 M items, its tag filter at 1 M), it lands
between 0.7× and 1.4× of V1's speed. **Third**, among LiNR's variants exact V1 is the fastest at a batch of 16 on every
filter but one. The exception is PubMed's MeSH filter, where only 0.02 % of items pass and V2 halves the time, consistent with it gathering just those. V3 at a pool reaching 0.90 beats the faster exact variant only on LAION's domain filter, by 2–7 %.

## 2. Speed at equal recall

Table 2 holds SilverTorch to a recall target and compares it with the fastest LiNR that reaches the same target: exact V1
or V2, or V3 at its smallest sufficient pool. Each cell shows how many times faster SilverTorch is (bold: SilverTorch
faster). "Share" is the fraction of clusters SilverTorch probes. ◐: half or more of the clusters; "top": not reached
within three quarters, with the best recall reached. Real filters, k = 100; the page has the full grid (k = 100, 1000,
2048, synthetic filters).

**Table 2.**

| Dataset · filter | Pass | 0.90: share | bs 1 | bs 16 | bs 64 | 0.95: share | bs 1 | bs 16 | bs 64 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Goodreads 0.8M · `c0_genre` | 33 % | 0.80 % | **1.29×** | **2.98×** | **6.18×** | 1.6 % | **1.27×** | **2.98×** | **5.63×** |
| LAION 1M · `c0_domain` | 0.95 % | 50 % ◐ | 0.92× | **1.09×** | 0.75× | 75 % ◐ | 0.72× | 0.78× | 0.52× |
| LAION 1M · `tags4` | 0.27 % | 50 % ◐ | **1.04×** | 0.71× | 0.44× | 75 % ◐ | 0.80× | 0.49× | 0.29× |
| arXiv 3M · `all4` | 0.94 % | 25 % | **3.22×** | **4.96×** | **4.77×** | 25 % | **3.22×** | **3.43×** | **2.82×** |
| arXiv 3M · `c0_maincat` | 14 % | 3.1 % | **2.03×** | **6.72×** | **5.97×** | 25 % | **2.02×** | **2.41×** | **1.69×** |
| arXiv 3M · `c3_nversions` | 44 % | 3.1 % | **2.82×** | **6.50×** | **5.72×** | 25 % | **2.64×** | **1.78×** | **1.17×** |
| LAION 3M · `c0_domain` | 0.95 % | 50 % ◐ | 0.76× | **1.35×** | 0.87× | 75 % ◐ | 0.64× | **1.01×** | 0.60× |
| LAION 3M · `tags4` | 0.27 % | 25 % | **1.94×** | **2.19×** | **1.41×** | top 0.91 | – | – | – |
| LAION 10M · `c0_domain` | 0.95 % | 50 % ◐ | 0.99× | **1.33×** | 0.78× | 75 % ◐ | 0.79× | 0.95× | 0.54× |
| LAION 10M · `tags4` | 0.27 % | 25 % | **4.22×** | **3.78×** | **2.26×** | 25 % | **2.17×** | **1.50×** | 0.86× |
| PubMed 10M · `c0_mesh` | 0.02 % | 12 % | **3.20×** | **7.97×** | **7.87×** | 25 % | **1.35×** | **2.83×** | **2.47×** |
| LAION 30M · `c0_domain` | 0.95 % | 25 % | **3.03×** | **4.38×** | **2.78×** | top 0.91 | – | – | – |
| LAION 30M · `tags4` | 0.27 % | 12 % | **7.39×** | **7.02×** | **4.38×** | 25 % | **5.15×** | **3.81×** | **2.29×** |

Three patterns stand out. **First**, the share of clusters SilverTorch must probe decides the outcome. Over all 276 cells,
183 reach the target within a quarter of the clusters, and there SilverTorch beats the best LiNR in 181, with a median of
0.29 of its time (about 3.4× faster). Another 51 cells reach the target only by probing half to three quarters of the
clusters, and SilverTorch wins 13 of those. The other 42 do not reach it even at three quarters. **Second**, the hard
cases are filters whose passing items are thinly spread across clusters: LAION's domain filter, and its tag filter on
the small LAION subsets. This is consistent with the classic shortage of candidates: a query's nearest clusters hold few items that pass, so recall has to be bought with more probes, while an exact scan pays the same price whatever the filter.
**Third**, SilverTorch's lead narrows at a batch of 64 (on LAION 30M's tag filter: 7.4× at one query, 4.4× at 64). This is consistent with V1's full scan being one matrix product shared by the whole batch, while each SilverTorch query probes its own clusters and so amortises less.

## 3. Effect of batch size on speed

**Table 3.** Time per query (time per batch ÷ batch size). arXiv 3M, filter `c0_maincat`, k = 100.

| Algorithm | B = 1 | B = 16 | B = 64 |
|---|--:|--:|--:|
| LiNR V1 | 0.825 ms | 0.111 ms | 61.8 µs |
| LiNR V2 | 0.392 ms | 0.224 ms | 0.212 ms |
| LiNR V3, pool for 0.90 | 0.499 ms | 0.209 ms | 0.192 ms |
| LiNR V3, pool for 0.95 | 0.521 ms | 0.223 ms | 0.207 ms |
| SilverTorch, 24 probes | **0.193 ms** | **12.0 µs** | **5.7 µs** |
| SilverTorch, tuned for 0.90 | **0.193 ms** | 16.6 µs | 10.4 µs |
| SilverTorch, tuned for 0.95 | 0.194 ms | 46.2 µs | 36.6 µs |

Batching helps the three methods very differently. LiNR V1's cost per query falls 13× on arXiv from one query to 64 and 34× on PubMed, consistent with it reading the whole index once per batch. V2's falls only about 2×, consistent with it gathering the passing items separately for each query. For one query, V2 is faster than V1 below a pass rate of 0.85–0.97; at a batch of 16
the crossover drops to 0.014–0.065, and at 64 V1 wins on every filter. For one query SilverTorch takes about 0.19 ms at most settings on every dataset, consistent with a fixed cost per call rather than the work done; batching is then nearly free until the probed work dominates: at 24 probes on arXiv its cost per query falls 34×, to 6 µs.

## 4. Scaling with catalogue size

LAION runs at 1, 3, 10 and 30 M items with the same two filters, which isolates the effect of size. Table 4 shows
SilverTorch at recall 0.90 against the best LiNR, and at 24 probes against the best exact LiNR. Table 5 shows Meta's
co-design, timed on Meta's code as published because the adapted code has no whole-index path: filtering only inside
the probed clusters, against building the filter over the whole index first. The paper reports 1.79–2.15×.

**Table 4.**

| Filter · size | 0.90: share | bs 1 | bs 16 | bs 64 | 24 probes: recall | 24 probes: bs 16 |
|---|--:|--:|--:|--:|--:|--:|
| `c0_domain` · 1M | 50 % ◐ | 0.92× | **1.09×** | 0.75× | 0.567 | 3.10× |
| `c0_domain` · 3M | 50 % ◐ | 0.76× | **1.35×** | 0.87× | 0.493 | 8.40× |
| `c0_domain` · 10M | 50 % ◐ | 0.99× | **1.33×** | 0.78× | 0.496 | 26.0× |
| `c0_domain` · 30M | 25 % | **3.03×** | **4.38×** | **2.78×** | 0.551 | 55.0× |
| `tags4` · 1M | 50 % ◐ | **1.04×** | 0.71× | 0.44× | 0.398 | 3.30× |
| `tags4` · 3M | 25 % | **1.94×** | **2.19×** | **1.41×** | 0.334 | 8.90× |
| `tags4` · 10M | 25 % | **4.22×** | **3.78×** | **2.26×** | 0.381 | 29.1× |
| `tags4` · 30M | 12 % | **7.39×** | **7.02×** | **4.38×** | 0.492 | 58.2× |

**Table 5.** Whole-index filter time ÷ probe-then-filter time (median; range over filters and probes).

| Catalogue | Batch 16 | Batch 64 |
|---|--:|--:|
| 1 M items | 0.82× (0.81–0.83) | 0.89× (0.83–0.93) |
| 3 M items | 0.82× (0.82–0.83) | **1.09×** (0.86–1.28) |
| 10 M items | **1.01×** (0.81–1.19) | **1.57×** (1.12–2.40) |
| 30 M items | **1.64×** (1.10–2.21) | **3.18×** (1.70–5.65) |

The results show three patterns. **First**, at a fixed setting SilverTorch's lead grows in proportion to the catalogue:
at 24 probes it is 3× V1 at 1 M and 55–58× at 30 M, while its own time stays near 0.2 ms. **Second**, at equal recall
the lead also grows, from 0.4–1.0× at 1 M to 4.4–7.4× at 30 M on the tag filter. The share of clusters needed for 0.90
falls from a half to an eighth. With the cluster counts used, a cluster at 30 M holds about 1,800 items against about
120 at 1 M. Under the tag filter (0.27 % pass) that is, on average, about five passing items per probed cluster instead of 0.3, consistent with far fewer probes being needed to find 100 passing neighbours. **Third**, the co-design pays only at 10–30 M items: 1.6× at a batch of 16
and 3.2× at 64 at 30 M, against the paper's 1.79–2.15×, and about even or slower at 1–3 M. That is consistent with the whole-index filter being cheap when the index is small, so skipping it saves little.

## 5. Robustness to filter selectivity

The synthetic filters pass 0.1 %, 1 %, 10 % or 100 % of the items at random, so selectivity is the only thing that
changes.

**Table 6.** LAION-synth 30M, batch 16, k = 100.

| Pass rate | V1, ms | V2, ms | SilverTorch 24: recall | SilverTorch 24: ms | SilverTorch at 0.90: ms |
|---|--:|--:|--:|--:|--:|
| 0.1 % | 16.37 | 14.25 | 0.356 | 0.21 | 1.86 |
| 1.0 % | 16.40 | 15.07 | 0.589 | 0.25 | 1.17 |
| 10 % | 16.39 | 22.57 | 0.714 | 0.35 | 1.16 |
| 100 % | 16.47 | 99.01 | 0.765 | 0.69 | 1.68 |

The methods fall into three groups. **Exact V1** takes the same time at every pass rate, consistent with it scoring every item and masking afterwards. **Exact V2**'s cost grows with the pass rate, consistent with it gathering the passing items: on LAION-synth 30M it takes 14
ms at 0.1 % and 99 ms with no filter; on PubMed-synth, 5 ms and 146 ms. V2 is the right choice only for very selective
filters. **SilverTorch** behaves as the shortage of candidates predicts. At 24 probes its recall falls from 0.77 with no filter to 0.36 at 0.1 %, consistent with fewer of the items in its nearest clusters passing. Restoring 0.90 costs more probes at the rarest filter
(1.9 ms against 1.2 ms at 1–10 %), but that is still 9–14× under V1.

## 6. Index memory and build time

**Table 7.** GPU index size (MiB) and SilverTorch's build time, from its smallest to its largest number of clusters.

| Dataset | d | LiNR V1 / V2 | LiNR V3 | SilverTorch | SilverTorch / V1 | Build, s (clusters) |
|---|--:|--:|--:|--:|--:|--:|
| Goodreads 0.8M | 128 | 292 | 304 | **209** | 0.72 | 2.0 (4,096) |
| LAION 1M | 256 | 534 | 565 | **382**–390 | 0.72 | 0.6–7 (512–8,192) |
| arXiv 3M | 128 | 1,095 | 1,140 | **777** | 0.71 | 3.8 (2,048) |
| LAION 3M | 256 | 1,602 | 1,694 | **1,146**–1,161 | 0.72 | 5.4–40 (2,048–16,384) |
| PubMed 10M | 768 | 16,174 | 17,090 | **8,710**–8,891 | 0.54 | 85.6–1,426 (4,096–65,536) |
| LAION 10M | 256 | 5,341 | 5,646 | **3,819**–3,880 | 0.72 | 34.7–578 (4,096–65,536) |
| LAION 30M | 256 | 16,022 | 16,937 | **11,448**–11,509 | 0.71 | 105.0–1,738 (4,096–65,536) |

SilverTorch's index is 0.54–0.72 of LiNR's, consistent with storing each vector in int8 (one byte per dimension against fp16's two) plus a 1,024-bit (128-byte) bloom filter per item. That overhead weighs most at d = 128 and least at d = 768. V3 adds
its 1-bit codes to V1's index (+4–6 %). More clusters cost build time, not memory: 65,536 clusters make the index only
0.5 % larger than 4,096, but take 17× longer to build (105 s against 1,738 s at 30 M).

## 7. Exact or probabilistic filter

To isolate the price of SilverTorch's bloom filter, the same search ran with an exact filter fused in its place, at the
same clusters and probes. Table 8 gives the bloom's time as a share of the exact filter's (median over batch sizes and
probe settings): below 1, the bloom is faster. The two filters come from separate runs on the same GPU, and recall is
identical for both.

**Table 8.**

| Dataset · filter | Clusters | Items per cluster | Probes | Bloom ÷ exact (range) |
|---|--:|--:|--:|--:|
| Goodreads 0.8M · `c0_genre` | 4,096 | 195 | 24–64 | 1.06 (0.97–1.06) |
| LAION 1M · `tags4` | 8,192 | 122 | 4096–6144 | 1.50 (1.08–1.72) |
| arXiv 3M · `all4` | 2,048 | 1,459 | 24–512 | 1.06 (0.89–1.08) |
| arXiv 3M · `c0_maincat` | 2,048 | 1,459 | 24–512 | **0.95** (0.90–1.06) |
| arXiv 3M · `c3_nversions` | 2,048 | 1,459 | 24–512 | **0.94** (0.90–1.06) |
| LAION 3M · `c0_domain` | 2,048 | 1,465 | 24 | 1.11 (0.92–1.13) |
| LAION 3M · `c0_domain` | 16,384 | 183 | 8192–12288 | **0.82** (0.73–0.94) |
| LAION 3M · `tags4` | 2,048 | 1,465 | 24 | 1.08 (0.99–1.10) |
| LAION 3M · `tags4` | 8,192 | 366 | 2048 | 1.13 (1.12–1.17) |
| LAION 3M · `tags4` | 16,384 | 183 | 4096 | 1.38 (1.10–1.53) |
| LAION 10M · `c0_domain` | 65,536 | 153 | 32768–49152 | **0.77** (0.74–0.91) |
| LAION 10M · `tags4` | 4,096 | 2,441 | 1024 | **0.82** (0.73–0.97) |
| LAION 10M · `tags4` | 16,384 | 610 | 4096 | 1.01 (0.99–1.02) |
| LAION 10M · `tags4` | 65,536 | 153 | 16384 | 1.64 (1.18–1.68) |
| PubMed 10M · `c0_mesh` | 4,096 | 2,441 | 24–512 | **0.87** (0.50–1.05) |
| PubMed 10M · `c0_mesh` | 16,384 | 610 | 2048 | **0.71** (0.62–0.94) |
| PubMed 10M · `c0_mesh` | 65,536 | 153 | 8192–16384 | **0.85** (0.78–0.98) |
| LAION 30M · `c0_domain` | 4,096 | 7,324 | 24 | **0.91** (0.81–1.08) |
| LAION 30M · `c0_domain` | 16,384 | 1,831 | 4096 | **0.88** (0.87–0.90) |
| LAION 30M · `c0_domain` | 65,536 | 458 | 16384 | **0.80** (0.74–0.94) |
| LAION 30M · `tags4` | 4,096 | 7,324 | 24–1024 | **0.87** (0.66–1.09) |
| LAION 30M · `tags4` | 16,384 | 1,831 | 2048–4096 | **0.88** (0.81–0.96) |
| LAION 30M · `tags4` | 65,536 | 458 | 8192–16384 | 1.06 (1.00–1.08) |

The picture is mixed. **First**, the bloom is faster where the filter is expensive to check exactly: PubMed's MeSH filter
(0.71–0.87 of the exact time) and LAION's domain filter at deep probes (0.77–0.88). That is consistent with a few bitwise ANDs per item costing less than walking several exact clauses. **Second**, the exact filter is as fast or faster on LAION's four-tag filter when
clusters are small: 1.1–1.6× with 120–370 items per cluster, about even at 450–600. With 1,800 items per cluster (LAION
30M) the bloom is faster again (0.88). That is consistent with the bloom paying a fixed cost per probed cluster, which dominates when clusters are small; the exact kernel reads only 5–24 bytes per item. At 24 probes on LAION 3M the exact filter is also about
10 % faster, consistent with there being little probed work for the bloom to save. On arXiv and Goodreads the two are within 6 %.
**Third**, the bloom's approximation costs nothing measurable at Meta's default width: at the timed settings at most 0.049 % of the items the bloom passes actually fail the filter (none at 34 of 51 graded settings), there are no false negatives, and recall moves by at most 0.000002. Narrower filters let more
through: at 64 bits, 2.3 % of PubMed's failing items pass, and recall at a fixed setting falls from 0.69 to 0.50.

## 8. Sensitivity to parameters

**Table 9.** SilverTorch's number of clusters on LAION 30M (d = 256): index, build, recall at a share of clusters probed,
and the fastest timed setting reaching 0.90 at a batch of 16.

| Filter · clusters | Index, MiB | Build, s | Recall at 1/8 | Recall at 1/4 | Probes for 0.90 | t_med, ms |
|---|--:|--:|--:|--:|--:|--:|
| `tags4` · 4,096 | 11,448 | 105 | 0.899 | 0.947 | 1,024 | 3.88 |
| `tags4` · 16,384 | 11,461 | 399 | 0.917 | 0.955 | 2,048 | 2.40 |
| `tags4` · 65,536 | 11,509 | 1738 | 0.930 | 0.961 | 8,192 | 3.30 |
| `c0_domain` · 4,096 | 11,448 | 105 | 0.847 | 0.897 | – | – |
| `c0_domain` · 16,384 | 11,461 | 399 | 0.857 | 0.903 | 4,096 | 3.38 |
| `c0_domain` · 65,536 | 11,509 | 1738 | 0.866 | 0.908 | 16,384 | 4.09 |

**Table 10.** LiNR V3's recall@100 against its candidate pool, as a share of the passing items (floor: 2,000 candidates).

| Dataset · filter | Pass | 0.5 % | 1 % | 2 % | 5 % | 10 % | 20 % |
|---|--:|--:|--:|--:|--:|--:|--:|
| arXiv 3M · `c0_maincat` | 14 % | 0.521 | 0.638 | 0.750 | 0.872 | **0.937** | **0.975** |
| Goodreads 0.8M · `c0_genre` | 33 % | 0.841 | 0.867 | **0.920** | **0.965** | **0.984** | **0.994** |
| LAION 30M · `tags4` | 0.27 % | 0.781 | 0.781 | 0.781 | 0.881 | **0.948** | **0.983** |
| PubMed 10M · `c0_mesh` | 0.02 % | **0.999** | **0.999** | **0.999** | **0.999** | **0.999** | **0.999** |

For SilverTorch, recall follows the *share* of clusters probed, not their count. Finer clustering raises recall slightly
at the same share (+0.03 at an eighth on the tag filter) but needs four times the probes for that share, consistent with each probe carrying a fixed cost. The middle setting, 16,384 clusters (about three times √N), is the fastest to reach 0.90: 2.4 ms,
against 3.9 ms at 4,096 and 3.3 ms at 65,536. The best number of clusters sits near the usual √N rule of thumb. For LiNR
V3, recall rises with the pool until the pool holds nearly every passing item. The pool never drops below 2,000
candidates, which is why recall is flat at small shares. On PubMed's MeSH filter (0.02 % pass), even that floor holds all
passing items, so recall is 0.999 from the smallest pool. On arXiv's main category (13.6 % pass), 0.90 needs 10 % of the
passing items.

## 9. Seed to seed

The timed comparison was rerun at seeds 1 and 2: a new clustering, a new synthetic draw and a new timing run. Seed 2 ran
on a different machine from seeds 0–1, with the same GPU model. Table 11 gives the spread over the seeds at the same
setting, with each tuned cell held at the setting picked at seed 0: SilverTorch's time (max ÷ min − 1), its recall (max
− min), its time over the best exact LiNR at each setting, and over the best LiNR at each tuned cell. Table 12 checks
the machines against each other on LiNR V1, whose time does not depend on the clustering.

**Table 11.**

| Dataset | Seeds | Settings | Time: median | Time: max | Recall: max | vs exact LiNR | Tuned cells | vs best LiNR | Winner changes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| arXiv 3M | 3 | 54 | 1.4 % | 10.2 % | 0.002 | 1.3 % | 39 | 1.2 % | 0 |
| arXiv-synth 3M | 2 | 33 | 1.3 % | 9.8 % | 0.007 | 1.4 % | 21 | 0.5 % | 0 |
| Goodreads 0.8M | 3 | 30 | 5.1 % | 8.7 % | 0.008 | 4.9 % | 15 | 4.4 % | 0 |
| Goodreads-synth 0.8M | 2 | 27 | 5.0 % | 7.4 % | 0.013 | 5.4 % | 18 | 5.6 % | 0 |
| LAION 10M | 3 | 15 | 0.6 % | 11.3 % | 0.002 | 0.7 % | 6 | 0.3 % | 0 |
| LAION-synth 10M | 3 | 75 | 0.5 % | 10.5 % | 0.004 | 0.8 % | 21 | 0.8 % | 0 |
| LAION 1M | 3 | 6 | 2.7 % | 12.3 % | 0.009 | 2.8 % | 0 | – | 0 |
| LAION 30M | 3 | 33 | 0.5 % | 8.3 % | 0.002 | 0.5 % | 12 | 0.5 % | 0 |
| LAION-synth 30M | 2 | 78 | 0.2 % | 1.4 % | 0.002 | 0.4 % | 24 | 0.2 % | 0 |
| LAION 3M | 3 | 12 | 0.8 % | 14.8 % | 0.003 | 0.8 % | 3 | 0.4 % | 0 |
| PubMed 10M | 3 | 39 | 0.6 % | 53.6 % | 0.007 | 0.7 % | 9 | 0.8 % | 0 |
| PubMed-synth 10M | 3 | 51 | 1.0 % | 11.4 % | 0.005 | 1.1 % | 15 | 0.5 % | 0 |

**Table 12.** LiNR V1's time at seeds 1–2 over seed 0, same cell.

| Cells | Seed | Machine | Settings | Median | Range |
|---|--:|--:|--:|--:|--:|
| LAION 10M, LAION 1M, LAION 3M, PubMed 10M | 1 | same | 27 | 1.000 | 0.995–1.018 |
| LAION 10M, LAION 1M, LAION 3M, PubMed 10M | 2 | another | 27 | 1.000 | 0.998–1.003 |
| Goodreads 0.8M, arXiv 3M | 2 | another | 30 | 1.000 | 0.988–1.004 |
| Goodreads 0.8M, LAION 30M, arXiv 3M | 1 | same | 39 | 1.000 | 0.996–1.002 |
| LAION 30M | 2 | another | 9 | 1.000 | 0.999–1.001 |

The comparison barely moves. At the same setting, SilverTorch's time varies by a median 0.2–5.1 % per dataset, and recall
by at most 0.013. Held at its seed-0 setting, SilverTorch's time over the best LiNR's spreads by a median 0.2–5.6 %, and
none of the 183 tuned cells change winner. At that setting 3 cells (LAION-synth 10M with no filter) dip below the target at a later seed, by at most 0.0001. LiNR V1 stays within 1.8 % of its seed-0 time on
every machine. SilverTorch's shortest times are another matter: the settings under 0.25 ms, close to its fixed cost per
call, run 5–9 % slower at seed 2 on Goodreads, arXiv, their synthetic sets and LAION 30M, whose seed 2 ran on a
different machine from seeds 0–1. The cells moved between the other two machines show no such shift. This is consistent
with the machines differing in fixed cost per call, not in GPU throughput, and it is the main source of spread on those
datasets; it changes no winner. The largest spreads in SilverTorch's time come from PubMed's MeSH filter at deep probes
(5.9–9.1 ms at a batch of 64), consistent with a new clustering changing how many passing items the probed clusters
hold. Re-picking the setting at each seed instead changes one winner (LAION-synth 10M with no filter, bs 64, k 100, 0.95: 0.73–0.74 at the seed-0 pick, 0.73–1.26 re-picked). That is a pick sitting exactly on the
target, not a change in speed. Seeds 0–2 cover every real dataset and the synthetic PubMed and LAION 10M sets; synthetic Goodreads and arXiv have seeds 0 and 2, and synthetic LAION 30M seeds 0 and 1. LAION 1M's deep settings ran at seed 0 only, by design.

## 10. Conclusions

1. **SilverTorch** is the faster method wherever it reaches the required recall within about a quarter of its clusters.
   It is several times faster than the best LiNR at recall 0.90, and its lead grows with catalogue size, up to 7× at
   30 M items. Its index is also 28–46 % smaller. Its weakness is a shortage of candidates: filters whose passing items
   are thinly spread across clusters force it to probe half the index or more, and there exact LiNR is as fast or
   faster.
2. **LiNR V1** is the right exact method for batches: its time per query falls fastest with batch size, and from a batch of 16 it beats V2 on all but the rarest filters. **LiNR V2** wins for one query at a time and for very selective
   filters. **LiNR V3** rarely beats the faster exact variant at equal recall.
3. The **bloom filter** is a sound engineering choice at Meta's default width. It is faster than a well-built exact
   filter where the filter is expensive to check, at no measurable cost in recall. It is not faster everywhere, and the
   co-design around it pays only at 10 M items and above.

## 11. The papers' claims

LinkedIn's claims are C1–C3, Meta's C4–C7 ([claims](claims.md)).

|  | Verdict | The paper | We measured |
|---|--:|--:|--:|
| **C1** | shape holds, location does not | LiNR V2 (gather, then score) beats V1 (score, then mask) below about 11 % passing. | For one query V2 wins below 0.85–0.97 passing; at a batch of 16 only below 0.014–0.065; at 64 and 256 never. |
| **C2** | not comparable | LiNR V3's 1-bit prefilter is about 10 % faster at near-equal recall. | At the smallest pool reaching 0.90, V3 takes 0.84–1.37 of V2's time for one query and 0.71–1.07 at batches of 16 and 64, where V1 beats both. |
| **C3** | does not hold as stated | Native PyTorch masking is about 100× slower than a custom kernel. | 1.4–1.9× at one query and 5.8–6.7× at a batch of 16 against plain PyTorch; about even with torch.compile at one query. |
| **C4** | holds on Meta's bloom as adapted | SilverTorch's 1024-bit bloom filter lets through about 0.067 % false positives. | At most 0.000004 % of failing items pass; recall moves by at most 0.000002 against the exact filter. |
| **C5** | wins at 10–30 M only | Filtering inside the probed clusters is 1.79–2.15× faster than filtering the whole index. | 2.14× at 30 M, 1.18× at 10 M, 0.83× at 1 M (median, Meta's code as published). |
| **C6** | partially | int8 SilverTorch cannot reach 0.95 recall; selective filters need more probes. | 0.95 is reached within a quarter of the clusters on 9 of 17 real filters; the clusters probed, not int8, set the limit. |
| **C7** | holds (earlier gate) | Our adaptation returns what Meta's published code returns. | Identical ids and scores in all 1,536 gate cells, at the adapted code's review head; not rerun in the final rerun. |

## 12. Provenance and what may be cited

**Nothing on this page is citable yet**: the final rerun's gate is open ([validation](../validation.md#full-rerun)).

- **Code and runs.** All records come from the final rerun at tag `final-freeze` (`b3bddc8`), on A100-SXM4-80GB GPUs,
  collected into one total `results.parquet`. The exact-filter row in section 7 is the same tree with the exact clause
  filter built in (tag `final-freeze-clause`, `500284e`); its bloom path is byte-identical.
- **Records.** These are the Hub subtrees `fr/<leg>-<pod>-s<seed>[-rerun]` on `pinkmeme/eval-results`, each with its
  manifest in [hub-index](../artifacts/hub-index.md): 95 subtrees and 10,411 records; 138 of 1266 timed points stay unstable in every run. The tables and the page data are published as
  `artifacts/results-draft/<date>-fr` (manifest in the same index).
- **Scripts.** [`build_results.py`](../artifacts/results-draft/build_results.py) builds the tables,
  [`product_recall.py`](../artifacts/results-draft/product_recall.py) the recall bridge,
  [`gls_adapted.py`](../artifacts/results-draft/gls_adapted.py) the local-pass-rate check, and
  [`page/pagedata.py`](../artifacts/results-draft/page/pagedata.py) the data for the page and this file
  ([`results_md.py`](../artifacts/results-draft/page/results_md.py)).
- **Timing.** CUDA-graph median per batch, with Meta's default fp16 score path. SM clocks cannot be locked in the
  container, so every record carries its sampled SM clock and an unstable flag. A leg's cells that are unstable get one
  rerun, which replaces them; a cell unstable in every run keeps its record and its flag. The bloom-against-exact pairs
  drop any pair unstable on either side; both runs sampled 1,410 MHz at the median.
- **Rules behind the numbers.** SilverTorch's bloom recheck is off when timed, and its false positives are counted in a
  separate, untimed pass on Meta's bloom only. LiNR uses the exact filter only. Recall is exact filtered top-k recall,
  so no claim is called "not reproduced" on recall alone. The headline comparison sets SilverTorch against the fastest
  LiNR at the same recall. Seed 2 of every cell ran on a different machine from its seeds 0–1, with the same GPU model.
  Every number is seed 0 except section 9.
- **Citability.** Sections 1–9 and the claims table will become citable together when the final rerun's gate passes. C7
  quotes an earlier gate, at the adapted code's review head, which already stands. It was not rerun in the final rerun
  (user's call).
