---
title: abstract
created: 2026-10-11
updated: 2026-10-12
type: summary
tags: [paper]
sources: [research/paper/results.md, research/paper/claims.md, research/paper/ecir-criteria.md, research/artifacts/abstract-draft/repro-abstracts.md, arXiv:2407.13218v3, arXiv:2511.14881v5]
---

# Abstract

## Submitted (2026-10-11)

The ECIR 2027 reproducibility-track abstract as agreed with the user and
registered on 2026-10-11 (paper deadline 19 October 2026). The earlier
drafts are superseded and removed; the research behind the wording is in
[repro-abstracts](../artifacts/abstract-draft/repro-abstracts.md).

**Title (recommended, not confirmed in the session; check against the
EasyChair record):** *Model-Based Filtered Retrieval on GPUs: Reproducing
LiNR and SilverTorch*

**Abstract (204 words), verbatim as last agreed:**

> Model-based retrieval implements nearest-neighbour search and attribute filtering as PyTorch operators in the model code, executed inside PyTorch inference rather than by a separate ANN index. This removes a dedicated serving stack and lets search, filtering and scoring be co-designed within one model graph. Two such approaches target semantic search and recommendation: LiNR, an exhaustive GPU scan with pre- or post-filtering, and SilverTorch, an int8 IVF index with a Bloom filter fused into the probe. Each was evaluated only on proprietary data, and they were not compared empirically. We reproduce and test their claims on five public datasets with real and synthetic attribute filters at scales up to 30M items, implementing LiNR from its paper and adapting Meta's released SilverTorch code. For LiNR, the pre- versus post-filtering trade-off holds for single queries but largely disappears in batches, and its 1-bit variant is faster only for batched queries and recovers recall with enough bits per item. For SilverTorch, filtering only the probed clusters beats filtering the full index first only at 30M items, and recall at the default probe count falls with filter selectivity. At matched recall, IVF search is faster for broad filters and exhaustive scanning for highly selective ones. We release all code and measurements.

**Wording note.** The abstract says "pre- or post-filtering" for LiNR; the
paper text uses LiNR's own terms (V1 and V2 both pre-filter: V1 masks, V2
gathers; LiNR argues against post-filtering, §5.3, §6). The abstract stays as
submitted.

The last suggested edit, "recovers recall **only** with enough bits per
item", was not confirmed; if the registered text differs from the quote
above, the EasyChair record is the source of truth.

**Keywords (suggested):** Model-based Retrieval; Filtered Vector Search;
Approximate Nearest Neighbor Search; GPU Retrieval; Recommender Systems;
Reproducibility.

**Topics (suggested):** System aspects; Search and ranking; Recommender
systems; Evaluation research.

## What each sentence rests on

No number of the SilverTorch-vs-LiNR comparison is in the abstract. The
paper (deadline 19 October) must stand behind each sentence:

| sentence | source | state |
|---|---|---|
| each evaluated only on proprietary data; not compared empirically | both papers' text: LiNR's job-search sets (~15.5 M jobs) and internal data; SilverTorch's 10 M / 80 M pools "sampled from production", its 20 M and 40 M benchmarks on internal queries; SilverTorch cites LiNR in related work only | verified (arXiv 2407.13218v3, 2511.14881v5) |
| five public datasets, real and synthetic filters, up to 30M items | [datasets](../../docs/system/datasets.md) | setup fact |
| adapting Meta's released SilverTorch code | `official-fork/` ([Meta as published vs adapted](meta-published-vs-adapted.md)) | review ACCEPT (`1f6fed8`, tag `campaign-fork`); the abstract claims no property of it |
| LiNR pre- vs post-filtering: holds for single queries, largely disappears in batches | C1 ([claims](claims.md), [results](results.md#claim-verification)) | LiNR final pass, D1-G green ([validation](../validation.md)); the NUMBERS bs 64 / 256 legs strengthen it (V1 faster on every real filter there) |
| LiNR 1-bit variant: faster only for batched queries, recall recovered with enough bits | C2 | LiNR final pass; "recall" here must be ours (share of the exact filtered top-k returned), not LiNR's end-task recall, which is not comparable; our code length is capped at the embedding dimension against LiNR's 512 bits ([LN-8](reproduction-deviations.md#4-linr-v1v3--deviations)): say so in the paper |
| SilverTorch co-design beats filtering the full index first only at 30M items | C5, Meta's code as published | measured on the published code; the adapted code is not yet timed on it |
| recall at the default probe count falls with filter selectivity | C6 | holds on the adapted code: at 24 probes 0.93 at pass rate 1 down to 0.05 at 0.001 (goodreads-synth); NUMBERS legs, gate green |
| at matched recall, IVF faster for broad filters, exhaustive scan for highly selective ones | the head-to-head in [results](results.md#the-tuned-point-per-dataset-size) | **holds, with one exception the paper should state.** Measured on the NUMBERS legs (gate green): wherever SilverTorch reaches 0.90 / 0.95 it is faster at bs 1; on highly selective filters (pass rate 0.001 at ≤ 3 M, LAION `c0_domain`, YFCC) it cannot reach the target and the exhaustive scan wins. The exception: on broad filters (pass rate ≥ 0.1) at bs ≥ 16 and 0.95, LiNR V1 wins on ≤ 3 M synthetic sets (up to 2.7×) |
| we release all code and measurements | [release and licenses](release-and-licenses.md) | the anonymous mirror must exist by the paper deadline |

## Open before the paper

- The anonymous code mirror for review.
- A GenAI statement may be requested at camera-ready (drafted in the
  session, not recorded here).
