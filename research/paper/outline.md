---
title: outline
created: 2026-10-12
updated: 2026-10-12
type: summary
tags: [paper]
sources: [https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers, research/paper/ecir-criteria.md, research/paper/claims.md, research/paper/abstract.md, research/paper/results.md, research/validation.md, research/artifacts/hub-index.md, research/paper/material/coverage.md]
---

# Paper outline: sections, page budget, claims and exhibits

The plan the paper is written against (roadmap PAPER). It fixes the section
list, the page budget, what each section must claim and which table or
figure carries it, and the exact source of every exhibit. The material
pages under `research/paper/material/` hold the facts each section needs
([data, introduction, literature](material/data-intro-literature.md),
[method](material/method.md), [results](material/results.md)); the
[coverage](material/coverage.md) page maps every citable Hub artifact,
validation row, claim and deviation to its section. The
[ECIR criteria](ecir-criteria.md) page maps the review questions to this
outline's sections.

**The study**, in the paper's words: Meta's official SilverTorch code,
adapted (improvements listed in CHANGES), against LiNR V1-V3, implemented
from the paper. Never "a fork" in prose; our Triton SilverTorch
reimplementation is not in the paper and no Triton SilverTorch record is a
source. Meta's code as published appears as the parity reference, the "as
published" column of T3, and the arm of C5 (the adapted code has no
full-mask path).

## Open for the user

Everything else on this page and the material pages has a source. These
need a decision or an action only the user (or the controller) can take:

1. **Author contact and upstream issues** (roadmap "Needs the user"). §7's
   communication line states what happened, "no reply" included. Open
   questions for the authors: LiNR's filter sets and V1 / V2 setup, the
   SilverTorch FPR inconsistency (S-9 0.067 at 1024 bits vs S-10
   0.00173 %); upstream issues OF-3, OF-4, OF-6, OF-9, OF-10.
2. **Anonymous mirror and the release decisions**
   ([release §5](release-and-licenses.md#5-decisions-for-the-user)):
   goodreads checkpoints, Re-LAION vectors, PubMed slice, the public result
   repos, `eval-results` shape, GitHub visibility. §8's availability
   paragraph depends on them.
3. **C1 verdict wording** — *ruled* (user, 2026-10-13): "reproduced in
   shape, not in location": a V1 / V2 crossover exists at bs 1 at every N and d, but at
   pass rate 0.85-0.97, where LiNR's high-pass point (≈ 0.11) has V1 3.0×
   faster and ours has V2 faster (0.29-0.68 of V1). The mechanism is our
   fused V2 (LN-1), which LiNR §6 itself predicts (L-12).
4. **C4's false-positive rates** — *resolved* (controller and user,
   2026-10-13): the harness's `bloom_fp_rate` is its standalone filter
   module's, not SilverTorch's, and is not used. C4 reports SilverTorch's
   own bloom, measured on the adapted index with no recheck
   (`fp_adapted.py`, Hub `artifacts/results-draft/20261013-bloom-fp`); what
   its false positives cost in recall comes from the bloom legs.
5. **Mention repository stars (34 / 13 forks) and citation counts?**
   OpenAlex gives LiNR 1, SilverTorch 0 (a floor). The material page argues
   impact from deployment scale and released code instead.

## Numbers and citability

Every number in the paper comes from one of four sources, all citable
(AGENTS.md rule 2): their gates are green in [validation](../validation.md).

| source | what it gives | where |
|---|---|---|
| LiNR final pass | every LiNR V1-V3 number, the postfilter and `torch.compile` arms (C1-C3); Meta's code as published for C5 | Hub `campaign-final/merged-4016/results.parquet` (library `e16512f5`); [validation](../validation.md#campaign-v2-phase-v), rows EXHIBITS and D1-G (green, user 2026-10-11) |
| NUMBERS legs | every SilverTorch number on the adapted code (C4 recall and memory, C6, the head-to-head), LiNR beside it where a leg ran it (V3 pools, bs 64 / 256, k 2048, correlated filters) | Hub `numbers/*` ([hub-index](../artifacts/hub-index.md)); [validation](../validation.md#numbers-adapted-silvertorch-over-the-study-grid) (green, controller ACCEPT 2026-10-12, three exclusions) |
| META-FORK gates | the adapted code against Meta's code as published (C7, T3) | Hub `artifacts/meta-fork/*`; [validation](../validation.md#meta-fork-backendofficial-fork) (review ACCEPT at `1f6fed8`) |
| bloom legs (**pending**, due 2026-10-13 16:00Z) | SilverTorch's bloom without recheck (curves, timed), LiNR V1 / V2 with the exact filter interleaved (the bloom side of §5) | Hub `numbers/A1-bloom`, `numbers/A2-bloom`; validation NUMBERS once their gate is green |
| results-draft checks | the end-task bridge, the C6 ceiling split and local pass rate on adapted indexes | Hub `artifacts/results-draft/20261012-checks`, `-adapted`; validation NUMBERS rows |

The tables and figures are built by
[`build_results.py`](../artifacts/results-draft/build_results.py) into Hub
`artifacts/results-draft/20261012` (manifest `73d3c2ed…`); the file names
below (`c1_crossover.csv` …) are its outputs, and the
[results draft](results.md) states them in prose. The final tables are
regenerated from the last full rerun (roadmap FULL-RERUN), which also
brings seeds and confidence intervals: every number in this pass is seed 0
except the C1 seed cells, and no table carries a CI (controller ruling,
2026-10-12). The exclusions, each listed and never quoted as measured:
LiNR V2 at bs 256 on PubMed `c3_journal_reverse` / `all5`
(TIMEOUT); cells unstable in every run (flagged; 30 of 1,641 `h2h` rows).

Recall is recall@k against the exact filtered top-k throughout; the
originals report end-task recall (label@2000, E-Task / C-Task), so no claim
is called "not reproduced" on recall alone. On Goodreads the two move
together (the end-task bridge, §3).

## Page budget

| § | section | pages | exhibits |
|---|---|---|---|
| — | title, abstract, keywords | 0.5 | — |
| 1 | Introduction | 1.0 | — |
| 2 | The originals, their claims, related work | 1.0 | T1 (claims and verdicts) |
| 3 | Reproduction setup | 1.75 | T-data (datasets) |
| 4 | The claims, one by one | 3.25 | F1, F2, F4 |
| 5 | SilverTorch against LiNR | 2.25 | F3, T2 |
| 6 | Meta's code as published and as adapted | 0.75 | T3 |
| 7 | Deviations, difficulties, communication | 0.75 | — |
| 8 | Limitations, availability, conclusion | 0.75 | — |
| | **total** | **12.0** | 4 tables, 4 figures |

The survey's LNCS papers never let results drop below 4 pages; §4 + §5
hold 5.5. If the paper runs long, cut in this order: related work in §2
to three sentences; F4 to text (C4 and C5 are each one sentence and one
range); T-data merged into the §3 prose; §7's residuals to one line.
Nothing goes to an appendix (appendices count toward the 12).

## §1 Introduction (1.0 p)

**Claims the section makes.**
- Model-based retrieval: nearest-neighbour search and attribute filtering as
  PyTorch operators inside the model graph, not a separate ANN service.
  Two industrial systems claim it: LiNR (LinkedIn, CIKM '24) and
  SilverTorch (Meta, SIGIR '26). Their own words are in
  [material C1 §2.4](material/data-intro-literature.md#24-the-papers-own-model-based-retrieval-wording-for-1-and-the-faiss--cuvs-exclusion).
- **Why these two** (the call asks): both from large deployments, both
  evaluated only on proprietary data, never compared with each other
  (SilverTorch cites LiNR in §2.1 and §7 only; verified). One ships
  official code, one does not, so the study covers both reproduction
  modes: adapting released code and implementing from text.
- What we do: test every testable quantitative claim on five public
  datasets (0.8-30 M items, d 128-768) with real and synthetic filters on
  one A100, and compare SilverTorch at recall 0.90 / 0.95 with exact LiNR,
  and with LiNR V3 at the same recall (the matched-recall row).
- Contributions, as a numbered list (survey pattern): (1) claim verdicts
  for both papers; (2) the first head-to-head, as a regime map over pass
  rate, N and batch size; (3) Meta's code adapted, held bit-exact to the
  published code, 0.15-0.82 of its eager time; defects found in the
  released code; (4) a controlled-selectivity benchmark on public data and
  the released harness.
- The outcome in one sentence per system (the abstract's findings).

**Sources.** [abstract](abstract.md) (sentence-by-sentence table and the
LiNR wording note under it: V1 and V2 both pre-filter, masking vs
gathering); [claims](claims.md) (venues, arXiv versions);
[ECIR criteria](ecir-criteria.md), Impact rows. Material: **C1** Part 2.

## §2 The originals, their claims, related work (1.0 p)

**Claims.**
- LiNR in one paragraph: exhaustive GPU scan, V1 (score all, mask), V2
  (gather passing items, score), V3 (1-bit Sign-OPORP pre-score, keep a
  pool, rescore), custom filter kernels. Both V1 and V2 are pre-filtering
  in LiNR's terms (masking vs gathering); "post-filtering" is kept for our
  baseline.
- SilverTorch in one paragraph: int8 IVF with the filter fused into the
  probe; clause filters and a Bloom index; co-design (probe then filter).
- **T1** lists every quantitative claim of both papers (L-1…L-12,
  S-1…S-16) compressed to rows: claim, testable here (yes / partly / no
  and why), our claim id C1-C7, verdict. The untestable rows stay
  (production pipelines, A/B tests, TCO, 240 M-1 B scale, live updates):
  the call asks whether the right portions were replicated, and no
  surveyed paper lists its untestable claims.
- Related work (≤ 0.3 p): filtered ANN benchmarks are CPU-side library
  studies (Big-ANN NeurIPS '23 filtered track; Iff
  et al., SIGIR '26; Li et al., SIGMOD '26); GPU filtered search exists as
  standalone systems (VecFlow, SIGMOD '26; CAGRA / cuVS); none evaluates
  filtering as ops inside a model graph, which is why Faiss, cuVS and HNSW
  are out of scope.

**Exhibit T1.** Source: [claims](claims.md) (the L-* / S-* tables and the
C1-C7 matrix); verdicts from [results](results.md#claim-verification) and
[material C3 T1](material/results.md#t1-verdict-column-2). Material: **C1**
(paper descriptions, related work with BibTeX-ready citations), **C3**
(verdict column).

## §3 Reproduction setup (1.75 p)

**Claims.**
- **What counts as reproduced** (stated before any result; the BERT4Rec
  study's ±5 % rule is the model): a *shape* claim (a crossover exists, an
  effect has a direction, recall falls with selectivity) is reproduced when
  the shape appears on our data; a *magnitude* claim (~100×, 1.79-2.15×,
  "keep 1 % for ~10 %", FPR at 1024 bits) is reproduced when our ratio
  falls in the stated range on comparable settings, and "holds in
  direction, not in size" otherwise; a claim on a setup we cannot build is
  "not testable" with the reason. Every ratio is one seed, interleaved in
  one process; seeds and CIs come with the full rerun. Where the criterion
  is ambiguous per claim:
  [material C2 Q18](material/method.md#q18-success-criterion-per-claim-type-draft-for-3).
- **Recall.** Ours: per query, the share of the exact filtered top-k (the
  k highest inner products over the whole catalogue in fp32, same filter)
  that the method returns; a hit is an exact id match, so an id tied at
  the k-th score but not the oracle's counts as a miss. The papers' recall
  is end-task. **Bridge** (Goodreads, 17 configurations per filter):
  Pearson 0.88-0.91 at k 100, 0.91-0.94 at k 1000; where our recall ≥ 0.90,
  end-task recall is 0.93-1.0 of the exact scan's (median 0.99).
- **Arms.** LiNR V1-V3 implemented from the paper in Triton, with the
  points where the text is silent and what we chose (LN-* in
  [deviations](reproduction-deviations.md#4-linr-v1v3--deviations));
  SilverTorch as Meta's official code, adapted (CHANGES), with Meta's code
  as published as the bit-exact reference (one sentence here, §6 for
  detail); two native baselines for C3: eager torch masking and
  `torch.compile(max-autotune)`; the generic postfilter (matmul → top-k →
  drop).
- **Data (T-data).** Goodreads 0.8 M d128, arXiv 3 M d128 / d256,
  PubMed 10 M d768, Re-LAION 1 / 3 / 10 /
  30 M d256: encoder, real filter attributes and their pass rates (mean
  1.9e-4 to 0.999; per query, one item to all), the synthetic uniform and
  cluster-correlated filters on real embeddings, query counts, ground
  truth (exact filtered oracle; checked by the exact LiNR arms, ≥ 0.99
  against it in every cell).
- **Protocol.** One A100-SXM4-80GB per job. SilverTorch's paper used
  A100-40G; LiNR's latency tables name no GPU (its stretch and live-update
  tests say "single A100"), so ratios and shapes are compared, not
  milliseconds. torch 2.10, CUDA 12.8; per-batch latency in CUDA-graph
  mode (eager for Meta's code as published, which cannot be captured),
  median of three windows; arms timed interleaved in one process; SM
  clocks cannot be locked in the container, so the sampled clock and an
  `unstable` flag are recorded per measurement; what the timed region
  contains (kernels only; query preparation outside, reported apart).
- **Tuning.** IVF `n_lists` per dataset size (4096, 16384, 65536 from
  10 M) and the probe curve (`n_probe` doubling from 8 to `n_lists` / 4);
  SilverTorch at recall targets 0.90 / 0.95 (against exact LiNR, and against
  LiNR V3 at the same recall), and at the papers' operating
  point (24 probes) as a reference row; k 100 and 1000, k 2048 on a subset
  (the papers' own k); bs 1 / 16 / 64 (256 for LiNR where it fits); seed 0,
  seeds 0-2 on the C1 crossover cells.
- **Compute** (one line, labelled approximate; 2 of 11 surveyed papers
  report cost): final pass 73.5 GPU-h (sized from its records), NUMBERS
  about 56 GPU-h (pod 1's driver log about 26, pods b and d about 30 by the
  plan), the C4 / C6 adapted checks about 1.2 GPU-h
  ([provenance §1](provenance-and-disclosure.md#1-hardware)).

**Exhibit T-data.** Source: [material C1 §1.1-§1.4](material/data-intro-literature.md#part-1-data-t-data-3)
(per dataset and per sweep, from the final-pass parquet and its per-query
sidecars); [datasets](../../docs/system/datasets.md);
[provenance §7](provenance-and-disclosure.md#7-data-provenance). Protocol:
[evaluation](../../docs/system/evaluation.md#measurement-protocol),
[provenance §1-§4](provenance-and-disclosure.md#4-timing-which-clock-estimator-and-why-the-question-is-not-pedantic).
Material: **C1** (data), **C2** (arms, protocol, tuning, compute).

## §4 The claims, one by one (3.25 p)

One subsection per claim, each in the same order: the original's claim
with its number → our measurement on the comparable setting → verdict in
the §3 vocabulary → the mechanism, where we found one. LiNR first, then
SilverTorch.

| sub | claim | what the section must state | exhibit · source |
|---|---|---|---|
| 4.1 | **C1** V1 beats V2 at high pass rate, V2 wins at low (L-3…L-5) | a crossover exists at bs 1 at every N and d, at p 0.85-0.97 (LiNR: below 0.11); V2 1.6-5.1× faster at p 0.001. At bs 16 V2 wins only below p ≈ 0.02-0.065, by at most 25 %, never on laion1m-synth. At bs 64 V2 is slower on every real filter (1.3-35×); at bs 256 it times out on two. Correlated filters change nothing. Seeds 0-2 at the crossover cells move V2 / V1 by ≤ 0.006. Mechanism: fused V2 (LN-1, L-12); batching favours V1's one shared matrix | **F1** (V2 / V1 latency vs pass rate, bs 1 and bs 16, one line per N): `c1_curves.csv`, `c1_crossover.csv`, `c1_real.csv`, `linr_numbers.csv`, figure `fig_c1_v2_over_v1` |
| 4.2 | **C2** V3 keeping ~1 % gives ~10 % at near-parity recall (L-6) | not comparable as stated (end-task recall, unpublished protocol). Under our metric a 1 % pool reaches median 0.89 at 0.83 of V2 at bs 16 and 1.08 at bs 1; parity (0.99) needs 20 %, where V3 is no faster. LAION 1-30 M: 1 % reaches 0.89, 1.11× V2 at bs 1, 0.97× at bs 16. Our code length is capped at d (LN-8; LiNR 512 bits); near parity at 1 % needs 768 bits (PubMed). On Goodreads' end task a 1 % pool keeps 0.89-0.95 of the exact scan's recall at k 100 | text + one line in F3: `c2_v3.csv` (NUMBERS L2, L2b), `product_recall.csv` |
| 4.3 | **C3** custom filter kernel ~100× over native masking (L-2) | direction holds, size far below 100×: eager torch 1.4-2.0× slower at bs 1, 5.8-6.8× at bs 16; `torch.compile` max-autotune 0.5-1.05× of ours at bs 1 (mostly faster: another reduction order and fp16 top-k keys), 1.5-2.7× slower at bs 16 | text with ranges: `c3_torch.csv`; mechanism: [validation](../validation.md#library-gates) V1-FUSE row |
| 4.4 | **C4** Bloom FPR ≈ 0.067 at 1024 bits, memory (S-8…S-10, S-14) | SilverTorch's own bloom, measured on the adapted index with no recheck: its false-positive rate falls with width (PubMed `c0_mesh`, k_hash 3: 2.3 % at 64 bits, 7.7e-6 at 1024) and at the study's 1024 bits / k_hash 5 is 0 on arXiv and Goodreads and 4.3e-8 on PubMed, far below the paper's 0.067 (%) under either reading; no false negatives. Width costs memory: PubMed index 9.1 → 11.4 GiB from 64 to 2048 bits, 1,221 MiB per 1024 bits at 10 M items, which is N · m_bits / 8 bytes, the arithmetic behind the paper's own memory figures (5 GB at 1024 bits for 40 M). Meta's code as published admits false positives that cost recall: PubMed `c0_mesh` recall@100 0.649 against the adapted code's 0.664 (k_hash 3), and at 30 M (OF-12). The paper's two FPR figures disagree (S-9 vs S-10); the transposed layout is not in Meta's released code (OF-1). **Pending A1-bloom:** what the false positives cost in recall per real bloom sweep (recall without the recheck against the exact filter's, same index and probes). **F4a** (memory and FPR vs bits): `c4_bloom.csv` (`fp_adapted.py`, NUMBERS A4) |
| 4.5 | **C5** co-design 1.79-2.15× over full mask then probe (S-3, S-11) | evaluated on Meta's code as published only (`backend="official"`, eager): holds at 30 M (1.07-2.20 at bs 16, 1.69-5.56 at bs 64) and at 10 M bs 64 (1.13-2.35); mixed at 3 M bs 64 (0.87-1.25) and 10 M bs 16 (0.83-1.20); reversed at 0.8-3 M bs 1-16 (0.78-0.89). Mechanism: host work in the published partial path grows with the sub-query count (OF-9). The adapted code refuses `bloom_path="full"` by design (K fix 17), so its column reads "refused by design (K17)" | **F4b** (full / partial vs N, Meta's code as published): `c5_meta.csv` (final pass `codesign`, `nsweep-codesign`, `codesign-laion30m`). `c5_adapted.csv` is not cited |
| 4.6 | **C6** int8 IVF: recall ceiling < 0.95, selective filters need more probes (S-6, S-7, S-16) | at 24 probes recall falls with selectivity (goodreads-synth 0.93 → 0.05 from p 1 to 0.001) and is far from exact on real filters (0.33-0.89). With more probes 0.95 is reached on Goodreads, arXiv, LAION `tags4` at 10 / 30 M and 18 of 27 synthetic sweeps, not on LAION `c0_domain` or PubMed. The ceiling split on adapted indexes (int8 shipped / per-row / fp16, same probed lists): PubMed `c3_journal_reverse` 0.92 / 0.94 / 0.996 and `c0_mesh` 0.946 / 0.953 / 0.976, so on PubMed int8 is the limit; LAION 1 / 3 M fp16 adds ≤ 0.009, so there the partition is. No mechanism is on record for PubMed's int8 loss (its geometry is ordinary: random-pair cosine 0.67). Per query, recall follows the local pass rate more than the global one (Spearman higher in 45 / 45 cells; 0.24-0.73 vs 0.01-0.60 at 32 probes) | **F2** (recall vs pass rate at fixed probes, uniform and correlated, one line per N): `c6_recall.csv` (NUMBERS A1 / A1b), `at24.csv`, `c6_ceiling.csv` and `c6_local_pass.csv` (Hub `artifacts/results-draft/20261012-adapted`), `score_gaps_*.json` |

The verdicts are restated in T1 (§2), so §4 does not repeat them in a
closing table. C7 is §6.

Material: **C3** (every number and verdict), **C2** (the mechanism notes
and what each arm runs).

## §5 SilverTorch against LiNR (2.25 p)

Neither paper ran this comparison; it is the study's own result and the
novelty the call asks for. Kept apart from §4 (survey pattern 2).

**Claims.**
- Where SilverTorch reaches recall 0.90 on a real filter, it is faster than
  the best exact LiNR at every size from 0.8 M to 30 M (not matched recall:
  LiNR's recall is ≈ 1.0): 0.045-0.71 of
  LiNR's latency at 0.90, every batch size; at 0.95 its lead shrinks at
  bs 64 (up to 0.98).
- Where it does not reach the target within a 25 % scan (`n_probe` ≤
  `n_lists` / 4) there is no SilverTorch point (LAION `c0_domain` up to
  10 M, LAION 1 M, PubMed `all5`, 0.95 on PubMed). 0.99 is reached
  nowhere.
- **Matched recall (pending, after the bloom legs):** LiNR V3 (approximate)
  at 0.90 / 0.95, timed and interleaved with SilverTorch's bloom at the same
  points (user, 2026-10-13).
- At its production setting (24 probes) SilverTorch is faster in 344 of
  345 cells (0.016-1.06 of LiNR at bs 1; the exception is LAION 1 M bloom
  `c0_domain` at bs 1, against exact LiNR) but its recall there is far from
  exact: a reference row, not a comparison at equal quality.
- Synthetic filters: at p ≥ 0.01 SilverTorch wins wherever it reaches the
  target (0.03-0.94); LiNR V1 wins the 17 reached cells at p ≥ 0.1 with
  bs ≥ 16 where many probes are needed (worst 2.7×). k 1000 shrinks the
  reached cells (24 of 44 clause cells per batch size at 0.90 against 35 of 44 at k 100). Bloom
  filtering: SilverTorch wins every bloom cell it reaches (0.09-0.85).
- Index memory and build time: the adapted int8 clause index is 0.56-0.72
  of LiNR's fp16 index (PubMed 8.8 vs 15.8 GiB), but its bloom index is
  larger than LiNR's at ≤ 3 M; k-means build 2-4 s at ≤ 3 M, 10-29 min
  with 65536 lists, against about a second for LiNR.
- The regime map in words: which arm to pick by pass rate × N × batch ×
  target recall.
- **Pending the A2-bloom leg** (user directive, 2026-10-12): SilverTorch is
  evaluated with its probabilistic bloom filter as Meta ships it, false
  positives allowed and no recheck in the timed code (the recheck becomes
  grading only: recall against the exact oracle counts a false positive as
  a miss); SilverTorch's exact clause mode is our adaptation and leaves the
  head-to-head (at most a reference row). The bloom-side head-to-head
  (SilverTorch bloom against LiNR exact, interleaved per cell) comes from `numbers/A1-bloom` (curves) and `numbers/A2-bloom`
  (timed), due 2026-10-13 16:00Z (controller bloom plan, chain note
  `numbers/2026-10-12-211500000-controller-bloom-plan.md`). The clause-filter claims above stand as
  measured; which of them stay in §5 is set when the leg lands.

**Exhibits.**
- **F3** recall-latency Pareto per dataset (SilverTorch probe curve against
  LiNR V1 / V2 points and V3's pool curve), the paper's lead figure; one
  panel per dataset size, bs 16. Source: `h2h.csv` (graph, interleaved
  with LiNR V1 / V2, 1,641 rows, 30 flagged), `c2_v3.csv` for V3; figures
  `fig_h2h_synth`, `fig_tuned_vs_n` as drafts.
- **T2** winner table, SilverTorch at 0.90 / 0.95 against exact LiNR, with
  the LiNR V3 matched-recall row (dataset × filter × bs → winner and
  ratio), point ratios without CIs (seed 0; CIs come with FULL-RERUN).
  Source: `tuned.csv`, `by_size.csv`; memory from `memory.csv`.

Material: **C3**.

## §6 Meta's code as published and as adapted (0.75 p)

**Claims.**
- What the adaptation changed, by kind (host syncs removed, plan decode
  moved out of the call, one-call forward, CUDA-graph capture, compile
  support, the bloom's configured width and exact recheck), each change
  gated.
- The parity gate: ids and scores `torch.equal` to the published code on
  every gate cell with Meta's bloom semantics (1,488 / 1,488 at ≤ 10 M,
  276 / 276 at the head, 64 / 64 bloom at 30 M); the bloom changes are the
  only intended difference, and they remove false positives.
- Speed: eager time 0.15-0.82 of the published code's (median per dataset
  and filter), never slower beyond the A/A floor (max 1.03); 30 M median
  0.279; 9-10 kernels and no host sync a forward against 50 kernels and
  3-6 syncs. The numbers are the META-FORK gate grid's (controller ruling:
  no NUMBERS leg ran `backend="official"`).
- What this cannot say: the published code was not built for our harness;
  its `-O3` build is ours (OF-11); phase 1 (k-means, query prep) is ours in
  both columns.

**Exhibit T3.** Rows: gate (X, R, S, capture, compile) and speed ranges
per dataset × filter kind; the co-design row reads "refused by design
(K17)" in the adapted column (§4.5). Source:
[Meta as published vs adapted](meta-published-vs-adapted.md) §2-§5;
[validation](../validation.md#meta-fork-backendofficial-fork);
`adaptation.csv` (from `artifacts/meta-fork/head-82fafb6`), Hub
`artifacts/meta-fork/pod-b-32c0d61` (head), `laion30m-b1-32c0d61` (30 M).
Material: **C2** Q5-Q9, **C3** §6.

## §7 Deviations, difficulties, communication (0.75 p)

**Claims.**
- The deviations that change a verdict, each one line with its id: LN-1
  (fused V2, C1), LN-5 / HB-2 (recall definition, C2), LN-8 (V3 bits, C2),
  LN-6 (which "native" baseline, C3), OF-9 (host-bound partial path, C5),
  OF-10 / OF-1 / OF-12 (Meta's bloom: width ignored, no transposed layout,
  false positives in results; C4). The full list stays in the released
  repository.
- Defects reproducing found in our own code (D-1 fp16 accumulation in V2,
  D-2 non-deterministic compaction, D-3 non-deterministic k-means, D-6 the
  1-indexed mirror, D-9 tie order), all fixed: one line, as findings.
- Difficulties: the LiNR filter sets and production data are not public;
  the published SilverTorch path was host-bound and not capturable;
  clocks cannot be locked on rented GPUs; the LiNR V2 bs 256 timeouts.
- **Communication** (the call asks): open item 1.
- Unexplained residuals: none on the study's arms (R-1 and R-2 concern
  arms outside it).

**Sources.** [deviations](reproduction-deviations.md) §3-§8. Material:
**C2** Q8, Q17 (LN, OF, HB lists), **C3** §7 (which deviations move a
verdict).

## §8 Limitations, availability, conclusion (0.75 p)

**Claims.**
- Limits: one GPU model, ≤ 30 M items, no production pipeline or model
  quality; exact top-k recall, not end-task recall (bridged on one
  dataset); unlocked clocks (recorded); seed 0 (seeds and CIs with the full
  rerun); the goodreads data cannot be redistributed.
- Availability: anonymous code mirror for review, the results archive,
  one command per stage, the pinned environment; the released code is
  about 8,000 lines of library Python, 6,200 of harness, and Meta's
  adapted extension; a permanent archive (Zenodo) at camera-ready.
- Conclusion: one sentence per system and one for the head-to-head.

**Sources.** [release and licenses](release-and-licenses.md);
[provenance §8](provenance-and-disclosure.md#8-reproducing-the-environment);
[ECIR criteria](ecir-criteria.md), Availability rows. Material: **C1**
§1.5 (data licences), **C2** Q16 (reproduction commands, environment).

## Candidate findings

The strongest results the material supports, ranked by what they add to
the paper, each with its source. All seed 0, citable. Items marked *new*
are not yet carried by the sections above or by the results draft.

1. **At recall 0.90, SilverTorch is faster than exact LiNR wherever it
   reaches the target within a 25 % scan.** Real filters, k 100: 36 of 51 (dataset, filter, bs)
   cells reach 0.90, and SilverTorch is faster in all 36 (0.045-0.71 of the best
   exact LiNR); 0.99 is reached nowhere. Its edge grows with N where it
   reaches the target: tuned bs 1 ratio 0.51-0.58 on Goodreads 0.8 M,
   0.27-0.46 on arXiv 3 M, 0.15-0.42 on LAION 30 M. (`tuned.csv`,
   `by_size.csv`, `fig_tuned_vs_n`; §5)
2. **What caps SilverTorch's recall depends on the dataset.** On the
   same probed lists, fp16 scoring lifts PubMed `c3_journal_reverse` from
   0.92 (int8 as shipped) to 0.996 and `c0_mesh` from 0.946 to 0.976, past
   0.95, so on PubMed int8 is the limit; on LAION 1 / 3 M fp16 adds at most
   0.009, so there the IVF partition is. No mechanism is on record for
   PubMed's int8 loss. (`c6_ceiling.csv`; §4.6)
3. **LiNR's V1 / V2 crossover exists, but at the other end of the pass-rate
   axis.** At bs 1, V2 wins below p 0.85-0.97 at every N and d; at
   LiNR's own high-pass point (p ≈ 0.11) V2 is 0.29-0.68 of V1, where LiNR
   has V1 3.0× faster. Batching reverses it: V2 is slower on every real
   filter at bs 64 (1.3-35×) and times out at bs 256 on two. Seeds 0-2
   move the ratio by ≤ 0.006. (`c1_crossover.csv`, `c1_curves.csv`,
   `linr_numbers.csv`; §4.1)
4. **Meta's code, adapted, is bit-exact with the published code and
   0.15-0.82 of its eager time**, from 50 kernels and 3-6 host syncs a
   forward to 9-10 kernels and none, which makes it capturable and
   compilable. (validation META-FORK; `adaptation.csv`; §6)
5. **The co-design gain is real only at the original's scale.** On Meta's
   published code, partial (probe then filter) beats full at 30 M (up to
   2.20× at bs 16, 5.56× at bs 64) and is slower at 0.8-3 M (full / partial
   0.78-0.89), where its host work costs more than the scorer saves. (`c5_meta.csv`, OF-9; §4.5)
6. **The 100× native-masking penalty is 1.4-6.8× in eager PyTorch and
   disappears under `torch.compile` at bs 1** (0.5-1.05× of the custom
   kernel). (`c3_torch.csv`; §4.3)
7. **Our recall tracks end-task recall** on Goodreads (Pearson 0.88-0.94
   per filter), so the exact-top-k results speak to the papers' end-task
   claims; above 0.90 of ours, end-task recall is a median 0.99 of the
   exact scan's. (`product_recall.csv`; §3)
8. *New:* **Meta's published bloom costs recall at 10 M, not only at
   30 M.** On PubMed `c0_mesh` (rarest-MeSH filter, about 1,900 passing
   items of 10 M) the published bloom's recall@100 is 0.649 against the
   adapted code's 0.664 at k_hash 3 (0.6640 vs 0.6640 at k_hash 5);
   arXiv and Goodreads differ by < 4e-5. Meta's own false-positive count
   is not recorded (open item 4). OF-12 so far records the effect at 30 M
   only.
   (`c4_bloom.csv`; §4.4 and OF-12)
9. *New:* **The paper's bloom memory figures are exact arithmetic,
   N · m_bits / 8 bytes**, and the adapted index grows by exactly that
   (1,221 MiB per 1024 bits at 10 M); the published bundle bloom is
   smaller (PubMed 7.6 GiB index against the adapted 10.0 GiB at 1024
   bits). (`c4_bloom.csv`; §4.4)
10. *New:* **LiNR V1 scales linearly in N and runs near memory bandwidth.**
    At bs 1, d256: 0.51 / 1.23 / 3.57 / 10.8 ms at 1 / 3 / 10 / 30 M, about
    1.42 TB/s of item reads at 10-30 M (70 % of the A100-80GB's 2.04 TB/s;
    derived). LiNR's 4.8 ms for 15.5 M × 128 fp16 is about 0.83 TB/s.
    This is why a fused scan is hard to beat for exact search and why the
    crossover moves. (`linr_vs_n.csv`; §4.1, §5)
11. *New, an interpretation to check:* **at pass rate 0.001 SilverTorch
    reaches 0.90 at 10 M and 30 M but not at 0.8-3 M** (except arXiv);
    larger N leaves more passing items per query (10,000-30,000 against
    800-3,000). The 1 M LAION curve is short (512 lists), which confounds
    the 1 M point. (`tuned.csv`; §5)
12. **Cluster-correlated filters are no harder for IVF than uniform ones,
    and easier at p 0.01** (0.95 at 256 probes against 512 for uniform),
    while LiNR's V2 / V1 does not change (0.35-0.41 at bs 1). (`c6_recall.csv`,
    `linr_numbers.csv`; §4.6, §4.1)
