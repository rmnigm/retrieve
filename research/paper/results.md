---
title: results
created: 2026-10-11
updated: 2026-10-12
type: summary
tags: [paper]
sources: [research/artifacts/results-draft/build_results.py, research/artifacts/numbers/README.md, research/artifacts/hub-index.md, research/validation.md, research/paper/claims.md]
---

# Results (draft): SilverTorch, adapted from Meta's code, against LiNR

> **Numbers superseded by the FULL-RERUN, not for the paper.** The figures on this page come from the NUMBERS legs and the final pass at earlier tags; the paper's tables are regenerated from the FULL-RERUN's `results.parquet` ([roadmap](../roadmap.md) step 7, [validation](../validation.md#full-rerun)). The structure and the rulings stand.

**Status: citable**, with the exclusions below. The SilverTorch numbers are
measured on the adapted code (the NUMBERS legs, whose gate is green:
[validation](../validation.md#numbers-adapted-silvertorch-over-the-study-grid));
the LiNR numbers from the final pass are covered by D1-G, also green
([decisions](../decisions.md#goal-and-scheduling)). Quoted only with their
flag: the cells unstable in every run. LiNR V2 at bs 256 on PubMed `all5` is a **TIMEOUT**: no result with the 80 GB GPU full; on `c3_journal_reverse` it finished at 2.55 s a batch (55× V1). LiNR's timings in the
head-to-head come from the NUMBERS legs, where LiNR ran in the same process
as SilverTorch.

The study compares two families of filtered retrieval on one A100
(A100-SXM4-80GB): SilverTorch, which is Meta's official published code
adapted by us (the changes are listed in `official-fork/CHANGES.md`), and
LiNR V1 to V3, which we implemented from the paper.

## Where the numbers come from

- **Head-to-head:** the NUMBERS timed legs ([plan](../artifacts/numbers/README.md)).
  For every filter, SilverTorch is timed at the probe counts where its recall
  first reaches 0.90 and 0.95 (at k 100 and k 1000), at the highest probe
  count, and at the paper's 24 probes. Each point runs in one process with
  LiNR V1 and V2, interleaved, at bs 1 / 16 / 64. The probe counts come from
  the adapted code's quality-only probe curves (`n_probe` doubling from 8 to
  a quarter of the lists), with 4096, 16384 and 65536 lists from 10 M items.
- **Unstable records:** a cell measured twice keeps its stable record. The
  cells still unstable are kept and flagged (`flagged` in `h2h.csv` and
  `tuned.csv`): five sub-millisecond SilverTorch cells on LAION 1 M,
  unstable in both runs, two PubMed SilverTorch records and one arXiv k 2048
  LiNR V2 record. Six tuned points rest on a flagged cell.
- **Claims C1-C6:** LiNR from the final pass plus the LiNR-only NUMBERS legs
  (bs 64 / 256, correlated filters, k 2048, the V3 candidate pool); the
  SilverTorch probe curves, ceiling split and local pass rate (C6) and the bloom
  widths (C4, NUMBERS A4) from the adapted code. C4's "as published" column and
  C5 come from Meta's code as published in the final pass (C5 only there: the
  adapted code refuses the full-mask path), and C7 from the adaptation's gate
  grid.
- **The script:** [`build_results.py`](../artifacts/results-draft/build_results.py)
  writes every table and figure to Hub `artifacts/results-draft/20261013`
  ([hub-index](../artifacts/hub-index.md)).

All latencies are per batch, CUDA-graph mode, median. "Recall" is **Our recall@k** is fidelity to exact search: per query, the share of the exact filtered top-k (the k items with the highest inner product over the whole catalogue, computed in fp32 under the same filter) that the method returns, averaged over queries.
It is not the papers' end-task recall (held-out interactions or labels
found); on Goodreads the two move together (Pearson 0.88-0.94 per filter
over SilverTorch, LiNR V1 / V2 and V3 configurations; once our recall
reaches 0.90, end-task recall is 0.93-1.0 of the exact scan's (median 0.99
over the approximate configurations),
[validation](../validation.md#numbers-adapted-silvertorch-over-the-study-grid)). "Ratio" is SilverTorch's latency over the
faster of LiNR V1 and V2 in the same cell: below 1 means SilverTorch wins.

## The tuned point per dataset size

Tables: `by_size.csv`, `tuned.csv`. Figure: `fig_tuned_vs_n`. A cell's
tuned point is the fastest timed SilverTorch setting (lists, probes) that
reaches the target. This sets **SilverTorch at recall 0.90 / 0.95 against
exact LiNR** (recall ≈ 1.0): it is not a comparison at matched recall, and
part of SilverTorch's speed is bought with the recall it gives up. The
matched-recall row, LiNR V3 (approximate) at the same 0.90 / 0.95, is
pending. "Not reached" means not reached within a 25 % scan
(`n_probe` ≤ `n_lists` / 4, the tuning cap), on our k-means index. Real
filters, exact clause filtering, k 100:

| dataset (N, d) | filter | lists / probes at 0.90 | ratio at 0.90, bs 1 / 16 / 64 | ratio at 0.95, bs 1 / 16 / 64 |
|---|---|---|---|---|
| Goodreads (0.8 M, 128) | `c0_genre` | 4096 / 32-128 | 0.58 / 0.25 / 0.16 | 0.58 / 0.26 / 0.19 |
| | `c1_lang_reverse` | 4096 / 128-256 | 0.51 / 0.31 / 0.25 | 0.51 / 0.41 / 0.40 |
| | `all4` | 4096 / 512-1024 | 0.56 / 0.46 / 0.52 | 0.56 / 0.73 / 0.82 |
| LAION 1 M (256) | `c0_domain`, `tags4` | not reached (top 0.85 / 0.83) | — | — |
| arXiv (3 M, 128) | `c0_maincat` | 2048 / 64-128 | 0.46 / 0.18 / 0.22 | 0.57 / 0.63 / 0.98 |
| | `c3_nversions` | 2048 / 64-128 | 0.27 / 0.19 / 0.23 | 0.31 / 0.47 / 0.65 |
| | `all4` | 2048 / 256 | 0.29 / 0.32 / 0.39 | 0.33 / 0.53 / 0.64 |
| LAION 3 M (256) | `tags4` | 8192 / 2048 | 0.50 / 0.43 / 0.66 | not reached (top 0.91) |
| | `c0_domain` | not reached (top 0.87) | — | — |
| PubMed (10 M, 768) | `c0_mesh` | 4096 / 512 | 0.41 / 0.27 / 0.34 | not reached (top 0.9499) |
| | `c3_journal_reverse` | 65536 / 1024 | 0.045 / 0.20 / 0.47 | not reached (top 0.92) |
| | `all5` | not reached (top 0.90) | — | — |
| LAION 10 M (256) | `tags4` | 4096 / 1024 or 16384 / 4096 | 0.26 / 0.39 / 0.66 | 0.41 / 0.45 / 0.78 |
| | `c0_domain` | not reached (top 0.90) | — | — |
| LAION 30 M (256) | `tags4` | 16384 / 2048 or 65536 / 8192 | 0.15 / 0.20 / 0.33 | 0.22 / 0.38 / 0.63 |
| | `c0_domain` | 16384 / 4096 | 0.42 / 0.40 / 0.71 | not reached (top 0.91) |

"Top" is the highest recall at any timed point; the probe curves stop at a
quarter of the lists.

- **Where SilverTorch reaches recall 0.90 on a real filter, it is faster
  than the best exact LiNR at every size**, from 0.8 M to 30 M items: 0.045-0.71 of
  LiNR's latency at 0.90, at every batch size. At 0.95 it stays faster, but
  its lead shrinks at bs 64 (0.98 on arXiv `c0_maincat`).
- **Where it does not reach the target within a 25 % scan, there is no
  SilverTorch point to compare:** LAION `c0_domain` up to 10 M, LAION 1 M
  `tags4`, PubMed `all5`, and 0.95 on PubMed. More lists do not lift it
  much: on LAION 1 M / 3 M, 4096-16384 lists end at 0.82-0.91, and fp16
  scoring would add at most 0.009, so the limit is the IVF partition rather
  than int8 (C6).
- **LiNR's faster variant on real filters** is V2 at bs 1 almost everywhere,
  and V1 at bs 16 and 64. PubMed `c3_journal_reverse` (pass rate ≈ 1) is the
  exception: V1 is faster there at every batch size.
- **0.99 is not reached** by any timed point.

### At the paper's production setting (24 probes)

Table: `at24.csv`. At 24 probes SilverTorch with the exact clause filter is
faster in all 273 cells (k 100 and 1000): 0.016-0.99 of LiNR at bs 1 and
0.02-0.395 at bs 16. But its recall@100 there is far from exact: 0.42-0.89
on Goodreads, 0.68-0.85 on arXiv, 0.39-0.81 on PubMed, 0.33-0.57 on LAION,
and 0.05 on the most selective synthetic filter; at k 1000 it is lower
still (Goodreads 0.15-0.67, synthetic down to 0.006).
24 probes is a reference row, not a comparison at equal quality.

### Synthetic filters: the pass rate decides who reaches the target

Table: `tuned.csv`. Figure: `fig_h2h_synth`. At 0.90, k 100:

- **Pass rate 0.001:** SilverTorch does not reach 0.90 on goodreads-synth,
  or LAION 1 M / 3 M synth within a 25 % scan. It does reach it
  on arxiv-synth (0.57 / 0.35 / 0.59 of LiNR at bs 1 / 16 / 64) and at 10 M
  and 30 M with more lists (0.25-0.72).
- **Pass rate 0.01 to 1:** SilverTorch wins wherever it reaches 0.90 (0.03-0.94
  of LiNR), except LAION 1 M synth at bs 64 and pass rate ≥ 0.1 (1.11-1.29).
- **LiNR V1 wins at broad filters with large batches when SilverTorch needs
  many probes.** All 17 cells where SilverTorch reaches the target and still
  loses are at pass rate ≥ 0.1 and bs ≥ 16: synthetic filters at ≤ 3 M (most
  at 0.95), arXiv `c0_maincat` / `c3_nversions` at k 1000, and LAION 10 M /
  30 M synth at pass rate ≥ 0.1, k 1000. The worst is LAION 3 M synth at pass rate
  1, bs 64, 0.95: 2.7× LiNR.

### k 1000, k 2048 and bloom filtering

- **k 1000:** fewer cells reach the target (24 of 44 clause cells per batch size at 0.90, 17 at 0.95,
  against 35 and 26 of 44 at k 100). SilverTorch wins every reached cell at
  bs 1 (median ratio 0.42 at 0.90), but LiNR V1 wins 7 of 17 at 0.95, bs 64.
- **k 2048** (the papers' own k; arXiv, Goodreads, PubMed): SilverTorch wins
  at 0.90 and 0.95 on arXiv `c0_maincat` (0.23-0.72) and Goodreads `c0_genre`
  (0.30-0.81). On PubMed `c0_mesh` it stays below 0.90 at the timed points.
- **Bloom filtering:** the head-to-head with SilverTorch's own bloom (no
  recheck, false positives allowed) against LiNR's exact filter comes from
  the bloom legs, still running. The bloom cells timed so far ran with a
  recheck, so they are not reported.

### Index memory and build time

Table: `memory.csv`. The adapted code's int8 index is about 70 % of LiNR's
fp16 index at ≤ 3 M items (arXiv 776 vs 1,095 MiB) and 56 % on PubMed
(8.8 vs 15.8 GiB). It needs a k-means build: 2-4 s at ≤ 3 M, 29-85 s at
10 M with 4096 lists, and 10-29 min with 65536 lists at 10-30 M. LiNR builds
in about a second.

## Claim verification

The reproduction (each paper's claims against our measurement) is kept
apart from the head-to-head above. Claims C1-C7 are listed in
[claims](claims.md).

### LiNR (our implementation)

| claim | verdict | evidence |
|---|---|---|
| C1: V1 beats V2 at high pass rate, V2 wins at low | **reproduced in shape, not in location.** A crossover exists at bs 1, but at a pass rate of 0.85-0.97, not LiNR's ≈ 0.11: at LiNR's own operating point V2 wins (0.29-0.68 of V1), with fused kernels the paper itself says would change the picture (LN-1, L-12). Weak at bs 16, gone at bs 64 / 256. At bs 1, V2 wins below a pass rate of 0.85-0.97 at every N and d, and by 1.6-5.1× at 0.001. At bs 16, V2 wins only below about 0.02, by at most 25 %. At bs 64 and 256, V1 is faster on every real filter (V2 1.3-35× slower at bs 64, LAION 2.0-2.2×; at bs 256 V2 is 2.2-55× slower where it finished and times out on PubMed `all5`). Correlated filters (arxiv-corr-synth) change nothing: V2 at 0.35-0.41 of V1 at bs 1, 0.92-1.26 at bs 16. At k 2048, V2 wins at bs 1 (0.08-0.77) and at bs 16 only on PubMed | `c1_crossover.csv`, `c1_curves.csv`, `c1_real.csv`, `linr_numbers.csv` |
| C2: V3 keeping 1 % gives about 10 % at near-parity recall | **not comparable as stated.** LiNR's recall is end-task recall (recall@2000 against its product's relevance labels), and the evaluation protocol is not published in enough detail to rebuild. Under our metric (share of the exact filtered top-k returned), on the real filters a 1 % candidate pool reaches a median recall of 0.89, at 0.83 of V2's latency at bs 16 and 1.08 at bs 1. Parity with the exact scan (0.99) needs a 20 % pool, where V3 is no faster than V2 (1.01 at bs 16, 1.14 at bs 1). Median recall rises with the pool from 0.84 (0.5 %) to 0.99 (20 %). On LAION 1-30 M a 1 % pool reaches 0.89 and V3 runs at 1.11× V2 at bs 1 and 0.97× at bs 16; 20 % reaches 0.98 (V2 there from the final pass, not interleaved) | `c2_v3.csv` |
| C3: the custom filter kernel is about 100× faster than native masking | **direction reproduced, size not.** Eager PyTorch is 1.4-2.0× slower at bs 1 and 5.8-6.8× at bs 16. `torch.compile` max-autotune runs at 0.5-1.05× of Triton at bs 1 (median 0.85-0.96, so mostly faster) and 1.5-2.7× slower at bs 16 | `c3_torch.csv` |

### SilverTorch: the adapted code and Meta's code as published

| claim | verdict | evidence |
|---|---|---|
| C4: bloom false-positive rate about 0.067 % (6.7e-4 as a fraction) at 1024 bits | **ours is far lower, on SilverTorch's own bloom** (our rates are fractions). Measured on the adapted code's bloom index with no recheck, over the whole index (survivors minus exact passes, per query): at the study's width (1024 bits per doc, k_hash 5) 0 on arXiv and Goodreads and 4e-8 on PubMed `c0_mesh`, against the paper's 6.7e-4. It falls steeply with width: PubMed `c0_mesh` (k_hash 3) 2.3 % at 64 bits, 7.7e-6 at 1024, 8.5e-7 at 2048. No false negatives anywhere. The bloom index costs 1.2 GiB on PubMed at 1024 bits (2.4 at 2048). What the false positives cost SilverTorch's recall is measured per cell on the bloom legs (recall with the recheck off against the exact filter's, the same index and probes) | `c4_bloom.csv` |
| C5: co-design (probe then filter) 1.79-2.15× faster than filter then probe | **Measured on Meta's code as published only:** holds at 30 M (1.07-2.20 at bs 16, 1.69-5.56 at bs 64) and at 10 M bs 64 (1.13-2.35); reversed at 0.8-3 M at bs 1-16 (0.78-0.89). The adapted code refuses the full-mask path by design (K fix 17), so it has no C5 column | `c5_meta.csv` |
| C6: int8 IVF cannot reach 0.95 recall; selective filters need more probes | **partially, on the adapted code.** At 24 probes recall falls with selectivity (0.93 to 0.05 on goodreads-synth from pass rate 1 to 0.001), and per query it follows the *local* pass rate (the share of the query's unfiltered top-100 that passes) more than the global one: Spearman 0.24-0.73 against 0.01-0.60 at 32 probes on Goodreads and arXiv, higher in all 45 (filter, probe count) cells. With enough probes, 0.95 is reached on Goodreads, arXiv, LAION `tags4` at 10 M and 30 M, and most synthetic filters. It is not reached on LAION `c0_domain` at any size (top 0.85-0.91), on PubMed (`c0_mesh` 0.9499; `c3_journal_reverse` 0.92, the int8 ceiling, fp16 0.996). The split on the adapted index, over the same probed lists (int8 as shipped / per-row int8 / fp16, 16384 lists / 4096 probes): PubMed `c3_journal_reverse` 0.92 / 0.94 / 0.996 and `c0_mesh` 0.946 / 0.953 / 0.976, so on PubMed int8 is the limit; LAION 1 M / 3 M at most 0.009 from fp16, so there the IVF partition is the limit | `c6_recall.csv`, `c6_ceiling.csv`, `c6_local_pass.csv` |
| C7 (restated): the adaptation keeps Meta's results and removes its overheads | **ids and scores identical to Meta's code in 1,488 / 1,488 gate cells at ≤ 10 M (30 M in [validation](../validation.md#meta-fork-backendofficial-fork)); eager time 0.15-0.82 of Meta's (median per dataset and filter), at most 1.03.** On the NUMBERS legs, eager and graph return the same ids on all 2,115 adapted cells | `adaptation.csv`, `summary.json` |

The four paper claims tested only by arithmetic or not testable here
(production pipelines, TCO, A/B tests, scale above our data) are listed in
[claims](claims.md) and are not repeated here.

## Uncertainty

Every number on this page is seed 0, so it carries no confidence interval
and no paired test. Seeds and error bars come with the final full rerun
(user, 2026-10-11); the interleaving (SilverTorch and LiNR in one process per
cell group) is what keeps the ratios comparable meanwhile.

## Still open

- Reconciling [claims](claims.md) with the verdicts above.
- Seeds and confidence intervals come with the final full rerun; every
  number here is seed 0.
