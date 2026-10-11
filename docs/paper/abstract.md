---
title: abstract
created: 2026-10-11
updated: 2026-10-11
type: summary
tags: [paper]
sources: [docs/paper/results.md, docs/paper/claims.md, docs/paper/ecir-criteria.md, docs/artifacts/abstract-draft/repro-abstracts.md, arXiv:2407.13218v3, arXiv:2511.14881v5]
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
| five public datasets, real and synthetic filters, up to 30M items | [datasets](../system/datasets.md) | setup fact |
| adapting Meta's released SilverTorch code | `official-fork/` ([Meta as published vs adapted](meta-published-vs-adapted.md)) | the adaptation's review is pending; the abstract claims no property of it |
| LiNR pre- vs post-filtering: holds for single queries, largely disappears in batches | C1 ([claims](claims.md), [results](results.md#claim-verification)) | LiNR final pass; D1-G green ([validation](../validation.md)) |
| LiNR 1-bit variant: faster only for batched queries, recall recovered with enough bits | C2 | LiNR final pass; our code length is capped at the embedding dimension against LiNR's 512 bits ([LN-8](reproduction-deviations.md#4-linr-v1v4--deviations)): say so in the paper |
| SilverTorch co-design beats filtering the full index first only at 30M items | C5, Meta's code as published | measured on the published code; the adapted code is not yet timed on it |
| recall at the default probe count falls with filter selectivity | C6 | to be re-read on the adapted code's NUMBERS legs |
| at matched recall, IVF faster for broad filters, exhaustive scan for highly selective ones | the head-to-head in [results](results.md#who-wins-where-head-to-head) | **provisional** until the adapted code's NUMBERS legs |
| we release all code and measurements | [release and licenses](release-and-licenses.md) | the anonymous mirror must exist by the paper deadline |

## Open before the paper

- The adapted code's NUMBERS legs: confirms or changes the matched-recall sentence.
- `results.md` still carries "NOT CITABLE until D1-G"; D1-G is green.
- The anonymous code mirror for review.
- A GenAI statement may be requested at camera-ready (drafted in the
  session, not recorded here).
