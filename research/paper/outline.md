---
title: outline
created: 2026-10-12
updated: 2026-10-12
type: summary
tags: [paper]
sources: [https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers, research/paper/ecir-criteria.md, research/paper/claims.md, research/paper/abstract.md, research/paper/results.md, research/validation.md, research/artifacts/hub-index.md]
---

# Paper outline: sections, page budget, claims and exhibits

The plan the paper is written against (roadmap PAPER). It fixes the section
list, the page budget, what each section must claim and which table or
figure carries it, and the exact source of every exhibit. The material
pages under `research/paper/material/` collect the facts each section
needs; the [ECIR criteria](ecir-criteria.md) page maps the review
questions to this outline's sections.

**The study**, in the paper's words: Meta's official SilverTorch code,
adapted (improvements listed in CHANGES), against LiNR V1-V3, implemented
from the paper. Never "a fork" in prose; our Triton SilverTorch
reimplementation is not in the paper. Meta's code as published appears
only as the parity reference and the "as published" column of T3.

**Format.** ECIR 2027 reproducibility track, paper due 19 October 2026
([abstract](abstract.md)). The 2027 call is not online yet; the 2024-2026
calls agree on **12 pages LNCS plus references, appendices inside the
12**, double-blind, code and data available to reviewers through an
anonymous repository. The call asks every paper to say why the originals
were chosen, how the reproduction was attempted and whether it worked,
what communication with the authors took place, the difficulties, and
the outcome.

**How the survey shaped it** (eleven repro papers read in full, plus the
abstract survey in [repro-abstracts](../artifacts/abstract-draft/repro-abstracts.md)):
results organised claim by claim with the verdict stated up front (as the
SIGIR 2026 *Hypencoder Revisited* study does); the original's experiment
kept apart from what we add; efficiency compared as ratios with both
machines named; interpretations of the text cited to the original's
section; a success criterion stated before the results; a Pareto figure
early.

## Numbers and citability

Every number in the paper comes from one of three sources, and none is
citable before its gate in [validation](../validation.md) is green
(AGENTS.md rule 2). The NUMBERS and META-FORK numbers are citable from
the merge of `dev/results-draft` into `staging` (`4365f8a` and later;
controller ruling, 2026-10-12):

| source | what it gives | where |
|---|---|---|
| LiNR final pass | every LiNR V1-V3 number, the postfilter and `torch.compile` arms (C1-C3) | Hub `campaign-final/merged-4016/results.parquet`, library `e16512f5`; [validation](../validation.md#campaign-v2-phase-v-not-yet-validated), EXHIBITS row |
| NUMBERS legs | every SilverTorch number on the adapted code (C4-C6, the head-to-head), LiNR beside it where a leg ran it (V3 pools, bs 64, k 2048, correlated filters) | Hub `numbers/*` ([hub-index](../artifacts/hub-index.md)) |
| META-FORK gates | the adapted code against Meta's code as published (C7, T3) | Hub `artifacts/meta-fork/*`; [validation](../validation.md#meta-fork-backendofficial-fork-not-yet-validated) |

The tables and figures are built by the results-draft script
([`build_results.py`](../artifacts/results-draft/build_results.py), Hub
`artifacts/results-draft/<date>`); the file names below (`c1_crossover.csv`
…) are its outputs. Its refresh on the NUMBERS legs is in progress
([results](results.md)); the final tables are regenerated from the last
full rerun (roadmap FULL-RERUN). Recall is recall@k against the exact
filtered top-k throughout; the originals report end-task recall
(label@2000, E-Task / C-Task), so no claim is called "not reproduced" on
recall alone.

## Page budget

| § | section | pages | exhibits |
|---|---|---|---|
| — | title, abstract, keywords | 0.5 | — |
| 1 | Introduction | 1.0 | — |
| 2 | The originals, their claims, related work | 1.25 | T1 (claims and verdicts) |
| 3 | Reproduction setup | 1.75 | T-data (datasets) |
| 4 | The claims, one by one | 3.25 | F1, F2, F4 |
| 5 | SilverTorch against LiNR | 2.0 | F3, T2 |
| 6 | Meta's code as published and as adapted | 0.75 | T3 |
| 7 | Deviations, difficulties, communication | 0.75 | — |
| 8 | Limitations, availability, conclusion | 0.75 | — |
| | **total** | **12.0** | 4 tables, 4 figures |

The survey's LNCS papers never let results drop below 4 pages; §4 + §5
hold 5.25. If the paper runs long, cut in this order: related work in §2
to three sentences; F4 to text (C4 and C5 are each one sentence and one
range); T-data merged into the §3 prose; §7's residuals to one line.
Nothing goes to an appendix (appendices count toward the 12).

## §1 Introduction (1.0 p)

**Claims the section makes.**
- Model-based retrieval: nearest-neighbour search and attribute filtering as
  PyTorch operators inside the model graph, not a separate ANN service.
  Two industrial systems claim it: LiNR (LinkedIn, CIKM '24) and
  SilverTorch (Meta, SIGIR '26).
- **Why these two** (the call asks): both from large deployments, both
  evaluated only on proprietary data, never compared with each other;
  SilverTorch cites LiNR in related work only. One ships official code,
  one does not, so the study covers both reproduction modes: adapting
  released code and implementing from text.
- What we do: test every testable quantitative claim on five public
  datasets (0.8-30 M items, d 128-768) with real and synthetic filters on
  one A100, and compare the two systems at matched recall.
- Contributions, as a numbered list (survey pattern): (1) claim verdicts
  for both papers; (2) the first head-to-head, as a regime map over pass
  rate, N and batch size; (3) Meta's code adapted, held bit-exact to the
  published code, faster end to end; defects found in the released code;
  (4) a controlled-selectivity benchmark on public data and the released
  harness.
- The outcome in one sentence per system (the abstract's findings).

**Sources.** [abstract](abstract.md) (sentence-by-sentence table);
[claims](claims.md) (venues, arXiv versions);
[ECIR criteria](ecir-criteria.md), Impact rows. Material: **C1**.

## §2 The originals, their claims, related work (1.25 p)

**Claims.**
- LiNR in one paragraph: exhaustive GPU scan, V1 (score all then mask),
  V2 (gather passing items then score), V3 (1-bit Sign-OPORP pre-score,
  keep a pool, rescore), custom filter kernels.
- SilverTorch in one paragraph: int8 IVF with the filter fused into the
  probe; clause filters and a Bloom index; co-design (probe then filter).
- **T1** lists every quantitative claim of both papers (L-1…L-12,
  S-1…S-16) compressed to rows: claim, testable here (yes / partly / no
  and why), our claim id C1-C7, verdict. The untestable rows stay
  (production pipelines, A/B tests, TCO, 240 M-1 B scale, live updates):
  the call asks whether the right portions were replicated, and no
  surveyed paper lists its untestable claims, so this is cheap
  distinction.
- Related work (≤ 0.3 p): filtered ANN benchmarks and studies (Big-ANN
  NeurIPS '23 filtered track, whose YFCC-10M set we use; Iff et al.,
  SIGIR '26, filtered ANN on transformer embeddings; the SIGMOD '26
  attribute-filtering study); GPU filtered search systems (candidates
  for C1 to check: VecFlow, SIGMOD '26); why standalone ANN libraries (Faiss, cuVS,
  HNSW) are out of scope: they are services outside the model graph, not
  ops in it.

**Exhibit T1.** Source: [claims](claims.md) (the L-* / S-* tables and the
C1-C7 matrix) and the verdicts from [results](results.md#claim-verification)
after its refresh; the final-pass T1 at Hub
`artifacts/exhibits/20261011-final/t1.md` holds the LiNR verdict wording.
Material: **C1** (paper descriptions, related work), **C3** (verdict
column).

## §3 Reproduction setup (1.75 p)

**Claims.**
- **What counts as reproduced** (stated before any result; the BERT4Rec
  study's ±5 % rule is the model): a *shape* claim (a crossover exists, an
  effect has a direction, recall falls with selectivity) is reproduced when
  the shape appears on our data; a *magnitude* claim (~100×, 1.79-2.15×,
  "keep 1 % for ~10 %", FPR at 1024 bits) is reproduced when our CI
  overlaps the stated range on comparable settings, and "holds in
  direction, not in size" otherwise; a claim on a setup we cannot build is
  "not testable" with the reason. The recall-definition caveat sits here.
- **Arms.** LiNR V1-V3 implemented from the paper in Triton, with the
  points where the text is silent and what we chose (LN-* in
  [deviations](reproduction-deviations.md#4-linr-v1v3--deviations));
  SilverTorch as Meta's official code, adapted (CHANGES), with Meta's code
  as published as the bit-exact reference (one sentence here, §6 for
  detail); two native baselines for C3: eager torch masking and
  `torch.compile(max-autotune)`; the generic postfilter (matmul → top-k →
  drop).
- **Data (T-data).** Goodreads 0.8 M d128, arXiv 3 M d128 / d256, YFCC-10M
  d192 (Big-ANN filtered track), PubMed 10 M d768, Re-LAION 1 / 3 / 10 /
  30 M d256: encoder, real filter attributes and their pass rates, the
  synthetic uniform and cluster-correlated filters on real embeddings,
  query counts, ground truth (exact filtered oracle; checked against
  YFCC's shipped GT on all 100,000 queries).
- **Protocol.** One A100-SXM4-80GB (LiNR used an A100, SilverTorch
  A100-40G: same GPU class, so ratios and shapes compare); torch 2.10,
  CUDA 12.8; per-batch latency in CUDA-graph mode, median; arms timed
  interleaved in one process; percentile-bootstrap CIs on ratios; SM
  clocks cannot be locked in the container, so the sampled clock and an
  `unstable` flag are recorded per measurement; warm-up and what the timed
  region contains (kernels only; adapter work outside).
- **Tuning.** IVF `n_lists` per dataset size and the probe curve (`n_probe`
  doubling to `n_lists` / 4); results at the papers' operating point
  (24 probes) and at matched recall 0.90 / 0.95; k 100 and 1000, k 2048 on
  a subset (the papers' own k); bs 1 / 16 / 64; seeds where a CI rests on
  them.
- **Compute.** Total GPU-hours of the final pass and the NUMBERS legs (one
  line; 2 of 11 surveyed papers report cost).

**Exhibit T-data.** Source: [datasets](../../docs/system/datasets.md)
(per-dataset sections, Synthetic selectivity attrs);
[provenance §7](provenance-and-disclosure.md#7-data-provenance); pass
rates from the final-pass records (`pass_rate` per sweep). Protocol:
[evaluation](../../docs/system/evaluation.md#measurement-protocol),
[statistics](../../docs/system/evaluation.md#statistics),
[provenance §1-§4](provenance-and-disclosure.md#4-timing-which-clock-estimator-and-why-the-question-is-not-pedantic).
Material: **C1** (data), **C2** (arms, protocol, tuning, compute).

## §4 The claims, one by one (3.25 p)

One subsection per claim, each in the same order: the original's claim
with its number → our measurement on the comparable setting → verdict in
the §3 vocabulary → the mechanism, where we found one. LiNR first, then
SilverTorch.

| sub | claim | what the section must state | exhibit · source |
|---|---|---|---|
| 4.1 | **C1** V1 beats V2 at high pass rate, V2 wins at low (L-3…L-5) | the crossover in pass rate at bs 1 and bs 16 across N and d; at bs 16 V2's edge is small and confined to very low pass rates; real filters | **F1** (V2 / V1 latency vs pass rate, bs 1 and bs 16, one line per N): `c1_curves.csv`, `c1_crossover.csv`, `c1_real.csv`; seeds 0-2 on the crossover cells |
| 4.2 | **C2** V3 keeping ~1 % gives ~10 % at near-parity recall (L-6) | V3 against V2 at matched recall, by batch size; recall vs bits per item (our code length is capped at d; LiNR used 512 bits); the pool curve | text + one row group in T2 or a line in F3: `c2_v3.csv`, NUMBERS `numbers/L2-v3pool`, `numbers/L2b-v3pool-laion` |
| 4.3 | **C3** custom filter kernel ~100× over native masking (L-2) | direction holds, size far below 100×; eager vs compiled torch; at bs 1 compiled torch can win, and why (reduction order, fp16 top-k keys) | text with ranges: `c3_torch.csv`; mechanism: [validation](../validation.md) V1-FUSE row |
| 4.4 | **C4** Bloom FPR ≈ 0.067 at 1024 bits, memory (S-8…S-10, S-14) | the FPR and memory against width on real attributes; the paper's own two FPR figures disagree (S-9 vs S-10); the transposed layout is not in Meta's released code (OF-1), so that half is a paper-vs-code finding | **F4a** (FPR and memory vs bits): `c4_bloom.csv`, NUMBERS `bloomwidth` |
| 4.5 | **C5** co-design 1.79-2.15× over full mask then probe (S-3, S-11) | evaluated on Meta's code as published only (`backend="official"`): holds only at 30 M, reversed below by host work in the partial path. The adapted code has no full path to compare: it refuses `bloom_path="full"` by design (K fix 17 in `official-fork/CHANGES.md`), so its column reads "refused by design (K17)" | **F4b** (full / partial ratio vs N, Meta's code as published): `c5_meta.csv`, NUMBERS `codesign` (published-code cells). `c5_adapted.csv` (`82fafb6`, before K17) is not cited |
| 4.6 | **C6** int8 IVF: recall ceiling < 0.95, selective filters need more probes (S-6, S-7, S-16) | recall at 24 probes falls with N and selectivity; 0.95 is reachable on ≤ 3 M with more probes / lists, not within the gated probes at 10-30 M; uniform vs cluster-correlated filters; the local pass rate explains per-query recall | **F2** (recall vs pass rate at fixed probes, uniform and correlated, one line per N): `c6_recall.csv`, `c6_ceiling.csv`; NUMBERS `A1-*` quality curves; local pass rate from Hub `artifacts/exhibits/20261009-1057-gls` |

The verdicts are restated in T1 (§2), so §4 does not repeat them in a
closing table. C7 is §6.

Material: **C3** (every number and verdict), **C2** (the mechanism notes
and what each arm runs).

## §5 SilverTorch against LiNR (2.0 p)

Neither paper ran this comparison; it is the study's own result and the
novelty the call asks for. Kept apart from §4 (survey pattern 2).

**Claims.**
- At SilverTorch's production setting (24 probes) it is faster than the
  best exact LiNR variant on every gated cell, but its recall there is
  far from exact on selective filters and at 10 M+.
- At matched recall (0.90 / 0.95, k 100), IVF wins for broad filters and
  small N; LiNR's exhaustive scan wins for highly selective filters
  (pass rate ≲ 0.001 synthetic, PubMed `c0_mesh`) and where IVF cannot
  reach the target within the probe cap (10 M rows); k 1000 narrows
  SilverTorch's window; batch size favours SilverTorch.
- The regime map in words: which arm to pick by pass rate × N × batch.
- Index memory and build time: int8 IVF about half of LiNR's fp16 index;
  k-means build cost.

**Exhibits.**
- **F3** recall-latency Pareto per dataset (SilverTorch probe curve against
  LiNR V1 / V2 points and V3's pool curve), the paper's lead figure; one
  panel per dataset size, bs 16. Source: `h2h.csv`, NUMBERS `A2-*` timed
  legs, LiNR from `campaign-final`.
- **T2** matched-recall winner table (dataset × filter × bs → winner and
  ratio with CI). Source: `matched.csv`; paired CIs from the interleaved
  SilverTorch / LiNR groups (NUMBERS item 5); memory from `memory.csv`.

Material: **C3**.

## §6 Meta's code as published and as adapted (0.75 p)

**Claims.**
- What the adaptation changed, by kind (host syncs removed, plan decode
  moved out of the call, CUDA-graph capture, compile support, the bloom's
  configured width and exact recheck), each change gated.
- The parity gate: ids and scores `torch.equal` to the published code on
  every gate cell with Meta's bloom semantics; the bloom changes are the
  only intended difference, and they remove false positives.
- Speed: eager time a fraction of the published code's on the gate grid,
  never slower beyond the A/A floor; the published path is host-bound.
  No new GPU run feeds this: the numbers are the META-FORK gate grid's.
- What this cannot say (the published code was not built for our harness;
  its `-O3` build is ours).

**Exhibit T3.** Rows: gate (X, R, S, capture, compile) and speed ranges
per dataset; the co-design row reads "refused by design (K17)" in the
adapted column (§4.5). Speed rows come from the META-FORK gate grid
only: the adapted code against Meta's `-O3` build, eager, the 1,536 gate
cells plus the 30 M cells (review ACCEPT at `1f6fed8`, tag
`campaign-fork`; the 30 M review item B1 is Hub
`artifacts/meta-fork/laion30m-b1-32c0d61`). Source:
[Meta as published vs adapted](meta-published-vs-adapted.md) §2-§5;
[validation](../validation.md#meta-fork-backendofficial-fork-not-yet-validated);
Hub `artifacts/meta-fork/pod-b-32c0d61`, `head-82fafb6`,
`laion30m-82fafb6`, `laion30m-b1-32c0d61`; `adaptation.csv`. Material:
**C2**.

## §7 Deviations, difficulties, communication (0.75 p)

**Claims.**
- The deviations that change a verdict, each one line with its id:
  LiNR's text vs our implementation (LN-*, notably the code-length cap,
  LN-8); Meta's released code vs Meta's paper (OF-*: the bloom layout, the
  ignored `m_bits`, bundle false positives); our harness vs the papers'
  protocols (HB-*). The full list stays in the released repository.
- Difficulties: the LiNR filter sets and production data are not public;
  the published SilverTorch path was host-bound and not capturable;
  clocks cannot be locked on rented GPUs.
- **Communication** (the call asks): what contact with the LinkedIn and
  Meta teams took place and what it settled. Open with the user
  (roadmap "Needs the user"); the paper states the facts, including "no
  reply" if so.
- Unexplained residuals, listed as such.

**Sources.** [deviations](reproduction-deviations.md) §3-§8. Material:
**C2** (LN, OF, HB lists), **C3** (which deviations move a verdict).

## §8 Limitations, availability, conclusion (0.75 p)

**Claims.**
- Limits: one GPU model, ≤ 30 M items, no production pipeline or model
  quality; exact top-k recall, not end-task recall; unlocked clocks
  (recorded); the goodreads encoder and data cannot be redistributed.
- Availability: anonymous code mirror for review, the results archive,
  one command per stage, the pinned environment; a permanent archive
  (Zenodo) at camera-ready.
- Conclusion: one sentence per system and one for the head-to-head.

**Sources.** [release and licenses](release-and-licenses.md);
[provenance §8](provenance-and-disclosure.md#8-reproducing-the-environment);
[ECIR criteria](ecir-criteria.md), Availability rows. Material: **C1**
(data licences), **C2** (reproduction commands, environment).

## Open before the paper can be written

- The results-draft refresh on the NUMBERS legs ([results](results.md));
  collector C3 starts after it is merged.
- The goodreads end-task-recall check (product recall against our recall),
  the bridge for the recall caveat in §3 and §4.
- Contact with the original authors, and the anonymous mirror (the user).
- The claims list restated after the refresh ([claims](claims.md)).
