---
title: abstract
created: 2026-10-11
updated: 2026-10-11
type: summary
tags: [paper]
sources: [docs/paper/results.md, docs/paper/claims.md, docs/paper/ecir-criteria.md, docs/artifacts/abstract-draft/repro-abstracts.md, arXiv:2407.13218v3, arXiv:2511.14881v5]
---

# Abstract (draft)

**Draft, NOT CITABLE.** The findings below are stated in words, with no
numbers, as agreed with the user: the SilverTorch-vs-LiNR numbers wait for
the adapted official code's rerun. Each finding is a reading of the
provisional [results draft](results.md), and the
[status table](#what-each-sentence-rests-on) gives its source and its
state. Choose one of the three versions; the other two can supply
phrases. The research behind the choices is in
[repro-abstracts](../artifacts/abstract-draft/repro-abstracts.md).

## Main version (balanced: reproduction plus head to head; 200 words)

**Title:** *Reproducing Model-Based Filtered Retrieval on GPUs: SilverTorch
and LiNR Head to Head*

> Model-based retrieval runs search and attribute filtering inside the recommendation model as GPU operators, not as a separate ANN service. Two deployed systems take opposite routes: LinkedIn's LiNR scans every item exactly and filters before or after scoring, while Meta's SilverTorch probes an int8 IVF index and evaluates a Bloom filter inside the probe. Both evaluate only on internal data, and LiNR has no public code. We reproduce the testable claims of both papers on one A100, using five public datasets with real attribute filters (0.8-30M items, 128-768 dimensions) and a controlled filter-selectivity axis. We implement LiNR's three variants from the paper. For SilverTorch we use Meta's released code, adapted to remove host overheads while returning the same results. Several claims hold only in part: LiNR's custom filter kernel beats native PyTorch masking by far less than reported, and SilverTorch's co-design speedup appears only at large indexes and batches. Head to head at matched recall, SilverTorch wins for broad filters up to a few million items; exact LiNR wins for highly selective filters and at the largest indexes. We map how selectivity, batch size, k, recall target, dimension and index size move this crossover, and release all code and results.

## Alternate A (reproduction first, claim by claim; 200 words)

**Title:** *Reproducing LiNR and SilverTorch: What Holds for GPU Filtered
Retrieval Outside LinkedIn and Meta*

> LinkedIn's LiNR and Meta's SilverTorch argue that filtered retrieval belongs inside the recommendation model as GPU operators, and both report large speedups on internal data. Neither result had been checked on public data, and LiNR ships no code. We list every quantitative claim of both papers, mark which can be tested outside production, and test those on one A100, using five public datasets with real attribute filters from 0.8 to 30 million items. We implement LiNR's variants from the paper and run SilverTorch from Meta's released code. The claims hold in direction more often than in size. LiNR's crossover between post- and pre-filtering appears for single queries but fades at larger batches. Its custom filter kernel beats native masking by a small factor, not two orders of magnitude, and its 1-bit variant helps only in part. SilverTorch's Bloom filter is more accurate than reported. Its co-design of search and filtering pays off only at large indexes and batches, and its int8 index reaches high recall only with far more probes than its production setting. We release a version of Meta's code adapted to remove the host-side overheads that reproducing it exposed, with the same results, and our harness and records.

## Alternate B (the missing comparison, practitioner view; 196 words)

**Title:** *Exact Scan or Int8 IVF? Where LiNR and SilverTorch Each Win in
GPU Filtered Retrieval*

> Inside a recommendation model on a GPU, filtered top-k retrieval can either scan every item exactly, as LinkedIn's LiNR does, or probe a quantised IVF index with the filter fused into the probe, as Meta's SilverTorch does. Each paper compares itself with CPU services or standalone ANN libraries, never with the other family, and neither tests on public data. We supply this missing comparison. Our testbed has five public datasets with real attribute filters (0.8 to 30 million items, 128 to 768 dimensions) and a synthetic filter-selectivity axis on real embeddings. On it we run LiNR's three variants, implemented from the paper, against SilverTorch, run from Meta's released code adapted to remove its host overheads. We compare latency at matched recall on one A100. At its production probe count SilverTorch is always faster, but often far from exact. At matched recall it keeps the lead for broad filters on indexes of up to a few million items. Exact scanning wins once filters are highly selective or the index reaches ten million items. Larger batches favour the IVF and larger k favours the scan. We also test each paper's own claims and release all code and measurements.

## What each sentence rests on

Every finding is **provisional**. The LiNR findings wait on D1-G being
declared green. The SilverTorch findings wait on the adapted code's
acceptance and rerun; until then its latency is bridged from the Triton
campaign ([results](results.md#where-the-numbers-come-from)).

| phrase (version) | source | state | could change after the rerun? |
|---|---|---|---|
| five public datasets, 0.8-30M items, 128-768 dimensions; one A100 (all) | Goodreads 0.8 M, arXiv 3 M, PubMed 10 M d768, YFCC-10M d192, Re-LAION 1-30 M d256 ([datasets](../system/datasets.md)); the A100 reference GPU (AGENTS.md rule 1) | setup fact | the adapted SilverTorch has no LAION cells yet. If they do not land, the head-to-head range ends at 10 M, and the 30 M point is LiNR and Meta's code only |
| adapted to remove host overheads, "returning the same results" (all) | `official-fork/CHANGES.md` on the META-FORK branch; the gate grid: ids equal to Meta's on every gate cell ([results](results.md#where-the-numbers-come-from)) | **not yet validated**: the META-FORK rows are "not yet validated" and the review has not accepted | if the review or rerun finds a mismatch, the phrase becomes "returning Meta's results up to ties" or is dropped |
| LiNR's custom filter kernel beats native masking "by far less than reported" / "a small factor, not two orders of magnitude" (main, A) | C3 in [results](results.md#claim-verification) | LiNR, NOT CITABLE until D1-G | unlikely (LiNR records are final-pass) |
| V1/V2 crossover "for single queries but fades at larger batches" (A) | C1 | as above | unlikely |
| 1-bit variant "helps only in part" (A) | C2 | as above | unlikely |
| Bloom filter "more accurate than reported" (A) | C4 (our false-positive rate is lower than the paper's) | measured with Meta's code; the adapted code's bloom recall gate is still running | no: the adapted default adds an exact recheck, so its false-positive rate is zero by construction; the sentence is about Meta's design |
| co-design pays off "only at large indexes and batches" (main, A) | C5: holds for Meta's code at 10-30 M and bs 64, reversed at 0.8-3 M | Meta's code as published: measured; adapted: provisional (neutral at bs 1, a gain at bs 64) | the sentence is about Meta's code. If the paper's C5 verdict moves to the adapted code, write "pays off at large batches" |
| int8 reaches high recall "only with far more probes" than production (A) | C6, `c6_ceiling.csv` | SilverTorch recall comes from the Triton records (ids shown equal); provisional | the rerun adds the adapted code's deep probe curves |
| at the production probe count SilverTorch is "always faster, but often far from exact" (B) | [results](results.md#at-silvertorchs-production-setting-24-probes): faster in every gated cell, recall from 0.05 to about 0.95 | bridged latency; provisional | the direction is wide (1.3-50×) and unlikely to flip; check after the rerun |
| SilverTorch wins broad filters "up to a few million items" (all) | matched recall 0.95, k 100: Goodreads and arXiv, real and synthetic filters with pass rate ≥ 0.01 | bridged; provisional | the margins are 0.23-0.93 of LiNR's time; recheck the cells near 0.9 |
| exact LiNR wins "highly selective filters" (all) | pass rate 0.001 (synthetic), PubMed `c0_mesh` | provisional | holds while SilverTorch's recall at the gated probes stays below target |
| exact LiNR wins "at the largest indexes" / "once the index reaches ten million items" (main, B) | PubMed and YFCC 10 M: SilverTorch does not reach 0.95 within the gated probes | **the weakest sentence.** With 16384 lists and 4096 probes the Triton records reach 0.89-0.96 at 10-30 M, so the outcome depends on the probe budget | **likely to change.** The rerun's deep probe curves decide it. If SilverTorch reaches the target at 10 M, rewrite as "and at ten million items only when the probe budget is capped" or drop the clause |
| "larger batches favour the IVF and larger k favours the scan" (B) | [results](results.md#how-the-parameters-move-the-crossover) (batch size, k) | provisional | unlikely to flip |
| "both evaluate only on internal data" / "neither tests on public data" (main, B) | [claims](claims.md): Meta's 10 M / 80 M pools and LinkedIn's 15.5 M jobs are internal | **to check** against SilverTorch §6.2: the dataset of the 20 M × 128 kernel benchmark is not named in our claims table | if that set is public, write "both report their system results on internal data" |
| "LiNR has no public code" (main, A) | [ECIR criteria](ecir-criteria.md#impact) | as of 2026-10-10 | recheck before submission |
| "never with the other family" (B) | LiNR predates SilverTorch; SilverTorch's baselines are CPU services, Faiss and HNSW ([claims](claims.md) S-1, S-6) | **to check** what SilverTorch's "GPU baselines" in S-1 are (if one is a brute-force GPU scan, soften to "never with each other") | |

## Why these choices

Read against the 19 abstracts in
[repro-abstracts](../artifacts/abstract-draft/repro-abstracts.md):

1. **No numbers is the norm.** 17 of 19 reproducibility abstracts give no
   measured result. They give counts and ranges instead. So the drafts
   lose nothing by waiting for the rerun. Only setup ranges appear (items,
   dimensions, one A100). One headline ratio can be added after the rerun
   if it is robust (BERT4Rec and SEATER each carry one).
2. **The five usual moves, in order.** Motivation → what is reproduced
   and why it was hard (internal data, no LiNR code) → setup in counts →
   findings in words → artifacts last. All three versions follow it.
3. **Conditions, not verdicts.** The well-received abstracts state where
   a claim holds (PLAID "in low-latency settings", BERT4Rec "only with
   sufficient training", FPSR "in long-tail scenarios"). Each of our
   findings names its condition: batch size, index size, selectivity,
   probe count. This is also the paper's thesis: the claims depend on
   the regime.
4. **The missing comparison as the headline.** PLAID, NCF-vs-MF and the
   BERT4Rec loss paper each add the comparison the original left out and
   lead with it. That is Alternate B, and the head-to-head sentence in the
   main version. It answers ECIR's novelty question ("new baselines or
   experiments") directly.
5. **The adaptation as a finding.** SEATER's ECIR 2026 abstract turns a
   bottleneck found while reproducing into its second contribution; the
   graph-bisection study (ECIR 2019) reports its implementation work.
   Our adapted code is presented the same way: reproduction exposed host
   overheads, and we release a version without them that returns the same
   results. It is not called a "fork".
6. **Model-based retrieval, not ANN benchmarking.** The first sentence
   defines the frame (GPU operators in the model, not a separate ANN
   service), which pre-empts the reviewer question of why Faiss, cuVS
   and HNSW are absent ([ECIR criteria](ecir-criteria.md#reliability)).
   The filtered-ANN experimental studies (SIGMOD 2026, arXiv 2025) share
   our selectivity axis, but they are ANN-library studies; the drafts do
   not compete with them.
7. **Hedged blame, direct verdicts.** A claim that fails is said plainly
   ("by far less than reported"). Nothing is said about the original
   authors' intent; DPR's "it appears that" is the register to use if a
   defect in Meta's code reaches the abstract.
8. **Double-blind.** No author, lab or repository names. "We release"
   points to the anonymous mirror at submission (release decision 4).

**Which to pick.**

- The **main version** fits ECIR's reliability and novelty questions
  best, and it is safe whichever way the 10 M rows go once that clause is
  edited.
- **Alternate A** is the safest before the rerun: its SilverTorch
  sentences rest on Meta's code and on recall, not on the bridged
  latency.
- **Alternate B** is the strongest pitch, but it rests most on the
  provisional head-to-head.

## Open before this page is final

- The adapted code's rerun: replace every "provisional" row above and
  settle the 10 M sentence.
- D1-G green: the LiNR findings.
- META-FORK accepted: "returning the same results".
- The two checks marked **to check** above (SilverTorch's 20 M data and
  its GPU baselines).
- The user: which version; the title; whether one headline ratio goes in
  after the rerun. Also the ECIR 2027 dates, which this research could not
  confirm (no official 2027 call was found on 2026-10-11).
