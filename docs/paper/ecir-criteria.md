---
title: ecir-criteria
created: 2026-10-10
updated: 2026-10-10
type: summary
tags: [paper]
sources: [https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers, docs/decisions.md, docs/paper/claims.md, docs/validation.md, docs/roadmap.md]
---

# The ECIR criteria, question by question

Roadmap F-CRIT. Each review question of the ECIR reproducibility track,
mapped to what answers it in our work, its state, and the step that closes
the gap. F5 writes the paper against this page
([decisions](../decisions.md#goal-and-scheduling)).

**Where the questions come from.** The ECIR 2026 call
([call](https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers)),
quoted in its own words. The 2027 call was not online on 2026-10-10; when it
appears, compare the lists and update this page.

**How to read the tables.**

- *State* is one of three values. **answered**: the work and its evidence
  exist. **partly**: exists, but incomplete or not yet citable. **gap**:
  nothing answers it yet.
- Every number of ours cited here is **NOT CITABLE** until D1-G
  (CLAUDE.md rule 2). Under the user's claims-first rule (2026-10-10), the
  runs now only decide C1-C7 and the new takes. The citable numbers come from
  F-REPRO's final pass. So "answered" means *the question has its answer
  and its evidence*, not that the numbers are final.
- *Paper section* names the section of the planned 12-page paper. The
  section list is a proposal for F5:
  - §1 introduction;
  - §2 the two originals and their claims;
  - §3 setup: scope, datasets, arms, harness, tuning;
  - §4 results claim by claim;
  - §5 official kernels vs ours;
  - §6 deviations and defects;
  - §7 new takes;
  - §8 availability and limits.

Claims C1-C7 and the exhibits (T1-T3, F1-F4) are defined in
[claims](claims.md). The current verdicts come from the EXHIBITS row of
[validation](../validation.md#campaign-v2-phase-v-not-yet-validated),
updated from the exhibits control notes of 2026-10-10:

| claim | verdict so far |
|---|---|
| C1 | holds in shape at 3 M, bs 16 |
| C2 | does not hold so far |
| C3 | does not hold (custom vs native 1.4-12×, not ~100×) |
| C4 | holds in part |
| C5 | does not hold, reversed in Meta's code (full / partial 0.82-0.88, host work in the partial path); in our backend the direction depends on `n_probe` (C5-OURS, a new take) |
| C6 | holds |
| C7 | holds in part |
| new: local pass rate l_q governs IVF recall | holds on goodreads and arXiv |
| new: l_q router | dropped: not on the recall-latency Pareto front on PubMed 10 M (decisions); one line in the paper as a tried idea at most |

## Reliability

| question (ECIR) | what answers it | paper section · exhibit | evidence | state | gap → step |
|---|---|---|---|---|---|
| Is the evaluation methodology aligned with the research challenges of the reproduced experiment? | Both papers do **model-based retrieval**: filtered retrieval as GPU ops inside the model graph, not a separate ANN service. That is our scope: the claims are tested as torch-importable ops at the papers' operating points ([decisions](../decisions.md#goal-and-scheduling), scope). We keep the production claims we cannot test, each with the reason (T1 "untestable" rows). For each claim the axis is the one it turns on: the pass rate for C1 / C3 (`synth` suite, a controlled p axis on real embeddings), N and `n_probe` for C6, bloom width for C4, co-design on/off for C5. Ground truth is our exact filtered oracle, checked against YFCC's shipped GT on 100,000 / 100,000 queries ([provenance §7](provenance-and-disclosure.md#7-data-provenance)). Ratios are timed interleaved; clocks are sampled and flagged ([decisions](../decisions.md#harness)) | §3; T1 | [claims](claims.md#claim-to-evidence-matrix); [evaluation](../system/evaluation.md#measurement-protocol); [validation](../validation.md#harness-gates) | answered | The paper has to state the frame and its limits once: no production pipeline, ≤ 30 M items, one A100 → F5 |
| Are the baselines representative of the relevant algorithm types and techniques? | Per algorithm type, the arm a practitioner would put in the graph: <ul><li>generic torch postfilter (dense matmul → top-k → drop), α {1, 8};</li><li>LiNR V1 / V2 on the torch backend (the "native masking" floor of C3);</li><li>V1 / V2 / V3 in Triton;</li><li>SilverTorch torch reference (unfused IVF), plus a `torch.compile(max-autotune)` arm;</li><li>SilverTorch in our Triton and Meta's official kernels.</li></ul> Exact (V1, V2), quantized (V3, int8 IVF) and filter-structure (clause, bloom) families are all present. Faiss / cuVS / HNSW are out by decision: they are standalone ANN services, not ops in a model graph, so they sit outside model-based retrieval ([decisions](../decisions.md#harness), [backlog](../backlog.md#baselines-outside-the-study)) | §3, §8 threats | [decisions](../decisions.md#campaign-v2-user-2026-10-08) (baselines); [evaluation](../system/evaluation.md#the-postfilter-baseline) | partly | A reviewer will ask about ANN libraries. SilverTorch's own S-6 compares with Faiss. The argument is the model-based-retrieval framing (user, 2026-10-10); the paper states it in the introduction, the setup and threats → F5. The C3 torch arms run on goodreads-synth only (SYNTH-TRIM), plus one bs-1 Triton-vs-compiled pair at 0.8 M and 3 M (pod 1) |
| Are the parameter and hyperparameter settings clearly described? | <ul><li>Every record carries its `params`, `config_sha`, `code_version` and env.</li><li>The grid is in `suites.yaml` and decisions: bs {1, 16}, k {100, 1000}, `n_probe` {24, n95}, pools, bloom m_bits × k_hash, synth rates, seeds.</li><li>IVF `n_lists` / `n_probe` per dataset, with the tuning rule and the 25 % cap ([decisions](../decisions.md#campaign-v2-user-2026-10-08)).</li><li>Encoders and their prefixes ([datasets](../system/datasets.md)); the E1c training recipe ([checkpoints](../system/checkpoints.md#hyperparameter-notes)).</li><li>Kernel autotune configs (`tune-kernels`).</li></ul> | §3 + an appendix table | [evaluation](../system/evaluation.md#config-one-yaml-per-dataset--suitesyaml); `evaluation/config/` | partly | <ul><li>12 pages cannot hold all this, so the paper needs one parameter table plus a pointer to the configs at the tag → F5.</li><li>The goodreads encoder's recipe is described, but its weights may not be shareable ([release §5](release-and-licenses.md#5-decisions-for-the-user)) → REL-LIC decision 1.</li></ul> |
| Are the algorithms and baselines adequately tuned? | <ul><li>IVF is tuned per dataset size (IVF-TUNE: goodreads 4096 / 64, arXiv 2048 / 256, YFCC and PubMed 4096 / 1024 at the cap).</li><li>SilverTorch is reported at the paper's `n_probe` 24 and at matched recall (n95), plus full `n_probe` curves (`deep`, synth sweep).</li><li>The postfilter runs α {1, 8}; the torch arm gets `torch.compile(max-autotune)`.</li><li>Triton kernels are autotuned per shape.</li><li>V3 is swept over pool fraction and `k_bits` (V-V3BITS, V3-BITS-PUBMED).</li><li>Matched-recall latency comes from `stats.at_recall`.</li></ul> | §3; F3 (Pareto), T2 | [decisions](../decisions.md#campaign-v2-user-2026-10-08) (IVF tuned); [evaluation](../system/evaluation.md#ivf-tuning) | partly | <ul><li>The official backend runs Meta's defaults (`score_path` fp16, plan cache off), which the paper must say.</li><li>YFCC / PubMed n95 is not reached at the cap (0.72 / 0.873), which must be reported as such.</li><li>V3 at LiNR's 512 bits is below D only at d768 → V3-BITS-PUBMED.</li><li>The goodreads `deep` points → V-GR-DEEP.</li></ul> |
| Does the study replicate the appropriate portions of the original paper? | Every quantitative claim of both papers is listed: LiNR L-1…L-12 and SilverTorch S-1…S-16, each marked testable / partly / no with the reason. The testable ones map to C1-C7 ([claims](claims.md)). Not testable: production pipelines, A/B tests, TCO, live updates, the 240 M / 1 B stress points | §2; T1 | [claims](claims.md) | answered | T1 must show the untestable rows too, so the paper covers the originals whole → EXHIBITS (T1), F5 |

## Impact

| question (ECIR) | what answers it | paper section · exhibit | evidence | state | gap → step |
|---|---|---|---|---|---|
| How important is reproducing these experiments to the community? | <ul><li>Two industrial systems (LinkedIn LiNR, CIKM '24; Meta SilverTorch, SIGIR '26) claim model-based retrieval: filtered retrieval as ops inside the model graph on the GPU, not a separate ANN service.</li><li>Neither result was reproduced on public data.</li><li>One has official code: Meta's kernels, which we run as a reference.</li><li>The other has none: we reimplemented LiNR V1-V3 from the text.</li><li>Filtered ANN is an active benchmark topic (Big-ANN NeurIPS'23 filtered track, whose YFCC-10M set we use).</li></ul> | §1 | [claims](claims.md); [official vs ours](official-vs-reimplementation.md#1-the-question-and-the-short-answer) | partly | The motivation is not written yet → F5 |
| How clear are the conclusions? (the brief's wording: "how obvious") | The conclusions are one verdict per claim (T1). They are not obvious in advance, and several go against the papers: <ul><li>C3: the custom-kernel ~100× over native masking does not hold; we see 1.4-12×, and at bs 1 a compiled torch V1 beat Triton V1 before V1-FUSE (being rechecked).</li><li>C5: the co-design speedup is reversed by host work in Meta's partial path; in our own backend the full mask is slower at small `n_probe` and faster at 128.</li><li>C2: V3's "keep 1 %" does not hold so far.</li><li>C1: the V1/V2 crossover exists at 3 M bs 16 near p 0.28, but at bs 1 V2 wins everywhere.</li></ul> | §4 (one subsection per claim); T1 | [validation](../validation.md#campaign-v2-phase-v-not-yet-validated) (EXHIBITS row) | partly | Verdicts are NOT CITABLE until F-REPRO → D1-G. C2's 512-bit question → V3-BITS-PUBMED. C3's bs-1 wording → the pod-1 Triton-vs-compiled pair. C5 inside our backend → C5-OURS cells (arXiv done, goodreads one sweep, PubMed 4 cells). Each verdict ships with its mechanism: V-PROF3 for C5, the reduction order for C3 |
| If validated, do the reproduced works advance a central IR topic? | Filtered model-based retrieval at scale (0.8-30 M items, d128-d768): <ul><li>when an exact fused scan beats IVF, depending on pass rate and batch (C1, F1);</li><li>how IVF recall collapses at low pass rate (C6, F2; uniform vs cluster-correlated filters);</li><li>bloom filters vs clause filters (C4).</li></ul> These are the decisions a practitioner faces when building a retrieval stage | §1, §4, §7 | F1, F2, F3, F4a | partly | The F2 real-vs-synth overlay and the correlated variant → V-AX-CORR, EXHIBITS |
| Is the original paper central or marginal to the community? | LiNR is a CIKM '24 applied paper, SilverTorch a SIGIR '26 paper with official code. Both are recsys / IR systems papers from large deployments. Citation counts are not gathered | §1 | [claims](claims.md) (arXiv versions) | partly | One sentence with venue and adoption evidence, if any → F5 |

## Novelty

| question (ECIR) | what answers it | paper section · exhibit | evidence | state | gap → step |
|---|---|---|---|---|---|
| Does the reproduction yield new insights not reported in the original paper? | <ul><li>**Local pass rate.** IVF recall follows each query's *local* pass rate l_q (the share of its unfiltered top-100 that pass) far better than the global p: Spearman 0.82 / 0.48 vs 0.38 / 0.30 (goodreads / arXiv). A pass-rate router is mixed: it helps on arXiv (0.65× all-exact latency at recall 0.958), not on goodreads, and its threshold does not transfer (EXHIBITS ideas #1-#2).</li><li>**Co-design in our backend.** With our kernels the full mask vs the partial bloom path flips with `n_probe`: full / partial 1.28 eager, 0.99-1.21 graph at `n_probe` ≤ 32, 0.85-0.93 at 128 bs 16 (C5-OURS, arXiv)</li><li>**Where the time goes.** Meta's official path is host-bound: device time is a small part of end to end, and 44 of 48 eager arms are host-bound (C5, C7, V-PROF3).</li><li>**Reduction order.** A compiled torch V1 wins at bs 1 through a different reduction order and fp16 top-k keys (C3).</li><li>**Bits vs dimension.** Our OPORP gives at most D bits; `k_bits` 64 at D 128 loses up to 26 recall points (C2).</li><li>**Defects.** Defects in Meta's released code against Meta's paper (OF-*), and defects that reproducing found in our own code ([deviations §5, §7](reproduction-deviations.md)).</li></ul> | §5, §6, §7 | [validation](../validation.md#campaign-v2-phase-v-not-yet-validated); [exhibits artifact](../artifacts/campaign-v2/exhibits/README.md) | partly | <ul><li>The router as a measured arm → V-ROUTER, then a library method → ROUTER-LIB.</li><li>The defect ledger → EXHIBITS idea #6.</li><li>Upstream issues and author contact → "Needs the user" on the [roadmap](../roadmap.md#needs-the-user).</li><li>The deviations drafts still cite retired plan labels → F5.</li></ul> |
| Does it introduce new baselines or experiments? | <ul><li>New baselines: the generic-torch postfilter, a `torch.compile(max-autotune)` arm, Meta's official kernels against an independent Triton implementation.</li><li>New experiments: <ul><li>the controlled-selectivity `synth` suite (uniform, worst case for IVF);</li><li>its cluster-correlated variant (best case);</li><li>real bloom FPR / memory vs width on real attributes (C4);</li><li>a 30 M d256 scale point (Re-LAION);</li><li>four public datasets with real filters (goodreads, arXiv, YFCC-10M, PubMed 10 M) beyond both papers' internal data.</li></ul></li></ul> | §3, §4, §7 | [decisions](../decisions.md#campaign-v2-user-2026-10-08); [datasets](../system/datasets.md) | partly | Legs still open: V-AX-SYNTH, V-YFCC, V-GR-DEEP, V-PUBMED, V-AX-CORR, D3, V-LAION30 |
| Does it propose new evaluation criteria, such as measures or statistical tests? | <ul><li>Percentile-bootstrap CIs (B = 10,000, fixed seed) on latency, recall and ratios.</li><li>Paired ratios over interleaved rounds (ABAB in one process, so clock drift cancels); "no difference" when the CI holds 1.0.</li><li>Matched-recall latency (`at_recall`, never extrapolated).</li><li>QPS at recall 0.95 per selectivity band (Big-ANN style).</li><li>Per-query local pass rate as an explanatory measure.</li><li>Kernel-only vs end-to-end and host-share as separate measures.</li><li>Sampled SM clock and an `unstable` flag, instead of claiming locked clocks.</li></ul> | §3 (statistics), §4, §5 | [evaluation](../system/evaluation.md#statistics); [provenance §4](provenance-and-disclosure.md#4-timing-which-clock-estimator-and-why-the-question-is-not-pedantic) | partly | <ul><li>Multi-seed CIs on the headline cells → V-SEEDS, deferred to F-REPRO (controller, 2026-10-10).</li><li>QPS bands in a report exhibit → EXHIBITS idea #6.</li></ul> |

## Availability

| question (ECIR) | what answers it | paper section · exhibit | evidence | state | gap → step |
|---|---|---|---|---|---|
| Are the code and datasets available to reviewers at review time? | The code is on public GitHub, but under the author's name. Review needs the anonymous mirror ([release §4](release-and-licenses.md#4-double-blind-review)). Data per dataset ([release §1](release-and-licenses.md#1-datasets)): <ul><li>arXiv and YFCC derived files may be shared;</li><li>PubMed: vectors and attrs, or the recipe;</li><li>Re-LAION: vectors, gated or not;</li><li>goodreads: recipe only.</li></ul> Results are on the private Hub `pinkmeme/eval-results` | §8 | [release-and-licenses](release-and-licenses.md); [hub-index](../artifacts/hub-index.md) | gap | Anonymous mirror, results archive for reviewers, anonymous data access → F4 (needs the user's accounts and REL-LIC decisions 1-6) |
| Is the shared material in a permanent repository? | Planned: a Zenodo DOI with the tagged code, the cited results subset and Meta's pinned sdist (Apache-2.0 allows it) | §8 | [roadmap](../roadmap.md) F4 | gap | F4 after D1-G |
| Are the experiments documented well enough for other researchers to reproduce them? | <ul><li>One command per stage: `uv sync`, `eval-data <dataset> all`, `bench campaign --suite … --resume`, `bench report --manifest`.</li><li>The environment is pinned (`uv.lock`, torch 2.10.0+cu128, triton 3.6.0, the official pin `21aa35e`), with a pod image.</li><li>Every record carries code_version, config, env and clocks.</li><li>System pages: [evaluation](../system/evaluation.md), [datasets](../system/datasets.md), [checkpoints](../system/checkpoints.md); library guide in `retrieve/docs/`.</li></ul> | §8 + artifact README | [provenance §8](provenance-and-disclosure.md#8-reproducing-the-environment); [README](../../README.md) | partly | <ul><li>A one-command reproduction at the final tag → F-REPRO, F4.</li><li>`campaign.yaml` for one code_version → F-REPRO.</li><li>Adding a dataset without an ETL → H-ADDDATA.</li><li>The `torchretrieve` sdist has no LICENSE file → F4 (the package is not on PyPI yet; README and AGENTS.md now say so).</li></ul> |
| Are there discrepancies between the paper and the shared material? | `bench report` generates every table and figure from the records the manifest names, so the paper's numbers cannot drift from the data. Deviations from the originals are listed with IDs ([deviations](reproduction-deviations.md)) | §6, §8 | [evaluation](../system/evaluation.md#campaign-manifest) | partly | <ul><li>Known discrepancies in shared material today: the arXiv Hub card is stale; the Hub mirror uses the legacy `[N+1, …]` layout ([provenance §7](provenance-and-disclosure.md#7-data-provenance)); [provenance §7](provenance-and-disclosure.md#7-data-provenance) still describes goodreads on gSASRec.</li><li>The paper drafts cite retired plan labels.</li></ul> → F4 (cards), F5 (drafts), D1-G (the report over the final manifest) |
| Is the shared material complete enough to replicate the experiments exactly? | Exact replication is in reach where the inputs are shareable: <ul><li>arXiv, YFCC and PubMed (deterministic rebuild, shown bit-equal for PubMed);</li><li>the synth attrs (seeded);</li><li>the oracles (content-fingerprinted).</li></ul> Timing replicates only statistically (clocks unlocked; CIs given) | §8 | [datasets](../system/datasets.md#pubmed); [validation](../validation.md#datasets) | partly | <ul><li>**Goodreads**: exact replication needs our checkpoint, and the data's terms forbid redistribution → REL-LIC decision 1.</li><li>Re-LAION needs the gated source → REL-LIC decision 2.</li><li>The final tag and rerun → F-REPRO.</li></ul> |

## Gaps by step

Every "partly" or "gap" above, by the step that closes it:

| step | closes |
|---|---|
| **F4** | code and data at review time, permanent repository, sdist license, Hub cards, anonymous mirror |
| **F-REPRO → D1-G** | citable verdicts, one code_version, one-command reproduction, multi-seed CIs (V-SEEDS folded in) |
| **F5** | the model-based-retrieval framing (scope) and the ANN-library exclusion it implies; parameter table; motivation and venue impact; deviations rewritten against C1-C7 |
| **V-AX-SYNTH, V-YFCC, V-GR-DEEP, V-PUBMED, V-AX-CORR, D3, V-LAION30, V3-BITS-PUBMED, C5-OURS cells** | the remaining experiments behind C1-C6 and the new experiments |
| **EXHIBITS #1/#2/#6** | the new insights (local pass rate, QPS bands, defect ledger); the l_q router was tried and dropped |
| **H-ADDDATA** | "documented well enough", benchmark side |
| **Needs the user** | REL-LIC decisions 1-6 ([release §5](release-and-licenses.md#5-decisions-for-the-user)); author contact and upstream issues |
