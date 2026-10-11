---
title: repro-abstracts
created: 2026-10-11
updated: 2026-10-11
type: summary
tags: [paper]
sources: [https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers, arXiv abstract pages and the venue pages linked in the table]
---

# How reproducibility papers write their abstracts

The research behind [the abstract draft](../../paper/abstract.md). It covers
19 abstracts from the ECIR, SIGIR and RecSys reproducibility tracks and
adjacent venues (2019-2026), read on 2026-10-11. Nine are on GPU or ANN
efficiency, or compare method families. Where only a summary was
available (the ECIR 2026 session listings), the row says so. Word counts
are approximate. "Numbers" means measured results, not counts of datasets
or methods.

## The abstracts

| # | venue · year | title | link | structure | numbers | how partial or negative outcomes are phrased | artifacts |
|---|---|---|---|---|---|---|---|
| 1 | SIGIR 2024 repro | A Reproducibility Study of PLAID | [arXiv 2404.14989](https://arxiv.org/abs/2404.14989) | ~235 w, 9 sentences: method → "we reproduce and fill gaps" → parameter study → **missing baseline** (BM25 + ColBERTv2 rerank) → where each wins → mechanism (clusters ≈ tokens) → lesson for efficiency evaluation | none | regime-conditional: rerank wins "in low-latency settings", "cannot reach peak effectiveness at higher latency"; a fix extends the frontier | not in abstract |
| 2 | ECIR 2026 repro | Down with the Hierarchy: The 'H' in HNSW Stands for "Hubs" | [arXiv 2412.01940](https://arxiv.org/abs/2412.01940) · [ECIR](https://ecir2026.dryfta.com/programme/abstract-archive/abstract/public/121/down-with-the-hierarchy-the-h-in-hnsw-stands-for-hubs) | ~250 w, 10 sentences: workload motivation → HNSW dominates → question ("is the hierarchy necessary?") → study at larger scale than prior work → finding → explanatory hypothesis → implication | none | a design claim of the original is shown to "provide no benefit" in high dimensions; the explanation is labelled a hypothesis | not in abstract |
| 3 | RecSys 2022 repro | A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation | [arXiv 2207.07483](https://arxiv.org/abs/2207.07483) | review of inconsistent prior comparisons → original code fails at defaults → succeeds with ~30× training → own implementation replicates 3 of 4 datasets, 95 % less training → verdict with condition | few, ratio-style (30×, 95 %, 3/4) | conditional verdict: state of the art "only with sufficient training" | own implementation (named) |
| 4 | RecSys 2023 | The Effect of Third Party Implementations on Reproducibility | [arXiv 2307.14956](https://arxiv.org/abs/2307.14956) | ~107 w, 5 sentences: field context → new angle (unofficial implementations) → 6 implementations vs official on 5 datasets → "alarming" → call to action | none | adjective + call to action, no magnitudes | no |
| 5 | RecSys 2019 | Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches | [arXiv 1907.06902](https://arxiv.org/abs/1907.06902) | scope (18 algorithms) → reproducible 7 → 6 lose to simple baselines → 1 loses to tuned linear → methodological lesson | counts only | blunt counts ("only 7 … could be reproduced") | not in abstract |
| 6 | RecSys 2020 | Neural Collaborative Filtering vs. Matrix Factorization Revisited | [arXiv 2005.09683](https://arxiv.org/abs/2005.09683) | background → claim under test → revisit original experiments → dot product wins with tuning → mechanism (MLP struggles to learn a dot product) → **serving cost** argument → recommendation | none | direct ("substantially outperforms"), justified by mechanism and cost | no |
| 7 | RecSys 2022 repro | Revisiting the Performance of iALS on Item Recommendation Benchmarks | [arXiv 2110.14037](https://arxiv.org/abs/2110.14037) | baseline's reputation → tuning recipe → re-evaluation on 4 benchmarks → competitive | one count ("at least half of the comparisons") | reverses a prior negative result with tuning | tuning recipe |
| 8 | ECIR 2022 repro (preprint title differs) | Another Look at DPR: Reproduction of Training and Replication of Retrieval | [arXiv 2104.05740](https://arxiv.org/abs/2104.05740) | ~170 w, 5 sentences: dense vs sparse → original → independent implementation with the authors' checkpoints → "largely verify" + one under-reported baseline → improved end to end | none | hedged attribution: "it appears that the original authors under-report" | toolkits named (Pyserini, PyGaggle) |
| 9 | RecSys 2023 | Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec? | [arXiv 2309.07602](https://arxiv.org/abs/2309.07602) | ~148 w, 7 sentences: field → two models → common belief → "But" confound (loss) → controlled finding reverses belief → secondary finding | none | pivot word "But", then a direct claim under a stated condition ("if both models are trained with the same loss") | no |
| 10 | ECIR 2019 repro | Compressing Inverted Indexes with Recursive Graph Bisection: A Reproducibility Study | [ECIR doi](https://doi.org/10.1007/978-3-030-15712-8_22) · [NSF PAR](https://par.nsf.gov/biblio/10171634) | ~160 w, 8 sentences: why reordering matters → few public implementations → the algorithm → clean-room reimplementation + challenges → core results reproduced and **generalise** → implementation released | none | confident: "successfully reproduce the core results" | implementation released |
| 11 | ECIR 2019 repro | An Experimental Study of Index Compression and DAAT Query Processing Methods | [ECIR doi](https://doi.org/10.1007/978-3-030-15712-8_23) · [NSF PAR](https://par.nsf.gov/biblio/10171629) | ~143 w, 4 sentences: progress → methods compared to few baselines, "no clear overall picture" → what is evaluated → research questions on how the best choice **depends on** compression, reordering, k | none | findings left as questions | no |
| 12 | ECIR 2025 repro | On the Reproducibility of Learned Sparse Retrieval Adaptations for Long Documents | [arXiv 2503.23824](https://arxiv.org/abs/2503.23824) | ~167 w, 9 sentences: task → cost problem → LSR → prior adaptations → aim → key finding → re-evaluation across lengths → extra analyses → code | segment counts only | "confirmed" plus extension; partial scope implied | code link |
| 13 | ECIR 2026 repro (session summary) | Efficient Optimization of Hierarchical Identifiers for Generative Recommendation | [ECIR session](https://ecir2026.dryfta.com/programme/abstract-archive/program/detail/11/reproducibility-i-recommender-systems) | reproduce SEATER → extend to a new dataset (Yambda) → **bottleneck found** (tree construction) → greedy and hybrid fixes → quality kept at ~5-8 % of build time → code | one efficiency ratio | the reproduction found an efficiency defect and fixed it, as a contribution | anonymous code link |
| 14 | ECIR 2026 repro (session summary) | RecRankerEval: A Reproducible Framework for Deploying and Evaluating LLM-based Top-k Recommenders | same session | reproduce RecRanker → find inflated results (ground truth in prompts) → framework over five dimensions → reproduced on three datasets, not on a fourth → what improves results | none | names the dataset where it did not reproduce | framework |
| 15 | ECIR 2026 repro (session summary) | A Reproducible and Fair Evaluation of Partition-aware Collaborative Filtering | same session | transparent benchmark of FPSR / FPSR+ → not consistently best → competitive, wins in long-tail scenarios | none | "not consistently top performers" + where they do win | code link |
| 16 | ECIR 2026 repro (session summary) | A Systematic Reproducibility Study of BSARec for Sequential Recommendation | same session | reproduce → component analysis → small gain from the claimed component, larger from an overlooked detail (padding) | none | component's claimed benefit shrinks to "slight" | not stated |
| 17 | SIGMOD 2026 (experimental study, filtered ANN) | Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study | [arXiv 2508.16263](https://arxiv.org/abs/2508.16263) | ~235 w, 9 sentences: need → no unified comparison → interface + taxonomy + component analysis → 10 algorithms, 12 methods, 4 datasets, **selectivity 0.1 %-100 %** → strengths, limitations, selection guidelines → code | scale and selectivity ranges only | trade-offs without a single winner; guidance | GitHub link |
| 18 | arXiv 2025 (benchmark, filtered ANN) | Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors | [arXiv 2507.21989](https://arxiv.org/abs/2507.21989) | ~190 w, 6 sentences: embeddings → FANNS definition → gap (no public attribute-rich embedding sets) → new dataset (2.7 M arXiv) → 11 methods across filter types, k, scale, selectivity → "eight key observations" | dataset size only | findings deferred to the paper | dataset |
| 19 | ECIR 2026 resource (contrast) | Evaluating the Efficiency and Effectiveness of Learned Sparse Retrieval with the lsr_benchmark | [ECIR](https://ecir2026.dryfta.com/programme/abstract-archive/abstract/public/130/evaluating-the-efficiency-and-effectiveness-of-learned-sparse-retrieval-with-the-lsr-benchmark) | ~170 w, 6 sentences: trade-off → no efficiency methodology → benchmark → design → hosted API → directional finding ("more effective but less efficient than BM25") | counts only | directional trade-off, no magnitudes | benchmark, API |

We did not obtain the abstract of the SIGIR 2023 *ColBERT to Col\** study
(the ACM page refused the fetch and the author PDF did not parse). The
SIGIR 2026 JPQ replicability study has no public full text yet. Neither is
counted above.

## Patterns

1. **Length.** 107-250 words, median about 170. ECIR's LNCS template has
   no word limit; the track's 2026 call asked for the abstract a week
   before the paper.
2. **Order.** Almost every abstract follows the same five moves:
   motivation (1-2 sentences) → what is reproduced and why it is hard (no
   code, internal data, inconsistent prior results) → setup in counts
   (datasets, methods, scale ranges) → findings in words → artifacts. The
   artifacts sentence appears in about half of them (#10, #12, #13, #15,
   #17), always last.
3. **Measured numbers are rare.** 17 of 19 give no measured result. The
   numbers that do appear are counts (#5: 7 of 18; #3: 3 of 4 datasets),
   scale ranges (#17: selectivity 0.1-100 %) or one headline ratio (#3,
   #13). So an abstract with no numbers is the norm, not a gap.
4. **Partial and negative outcomes are stated, with their condition.**
   The strongest abstracts give the condition under which a claim holds
   or fails: #1 "in low-latency settings", #3 "only with sufficient
   training", #9 "if both models are trained with the same loss", #15
   "not consistently … but in long-tail scenarios". A claim that fails is
   phrased directly (#2, #6, #9). Blame on the original is hedged (#8 "it
   appears that").
5. **The missing comparison is a contribution.** #1 (rerank vs PLAID), #6
   (dot product vs MLP) and #9 (controlled loss) each add the comparison
   the original left out, and that comparison becomes the headline. #6
   adds a serving-cost argument.
6. **Regime language for efficiency work.** #1, #11, #15 and #17 say which
   method wins *where* (latency budget, compression choice, long tail,
   selectivity) instead of naming one winner.
7. **Efficiency defects found while reproducing are findings.** #13 turns
   a bottleneck into its second contribution. #10 reports implementation
   challenges as part of the study.
8. **Generalisation is claimed explicitly** when the study goes beyond the
   original data (#10 "generalises to other collections", #13 new
   dataset, #2 more large-scale datasets than prior work).

## What the ECIR reviewers are asked

The 2026 call ([call](https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers);
mapped to our work in [ECIR criteria](../../paper/ecir-criteria.md)) asks
about four things:

- **Reliability:** the methodology fits the original challenge, the
  baselines are representative and tuned, parameters are described, the
  right parts are replicated.
- **Impact:** importance, how clear the conclusions are, how central the
  topic and the original paper are.
- **Novelty:** new insights, new baselines or experiments, new evaluation
  criteria.
- **Availability:** code and data at review time, a permanent
  repository, enough documentation, agreement with the shared material.

The call does not require a successful reproduction, but it asks for a
"precise and rigid evaluation of the process". It does not accept
same-team repeatability. It says nothing on industry systems or abstract
content. An abstract can signal each criterion in one phrase:

- reliability: "the testable claims", "public data", "tuned to matched
  recall";
- impact: two deployed systems that had not been reproduced, one with no
  public code;
- novelty: the head-to-head and its crossover map, the controlled
  selectivity axis, the adapted code;
- availability: what is released.
