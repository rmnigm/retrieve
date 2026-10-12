---
title: coverage
created: 2026-10-12
updated: 2026-10-12
type: summary
tags: [paper]
sources: [research/artifacts/hub-index.md, research/validation.md, research/paper/claims.md, research/paper/reproduction-deviations.md, research/paper/outline.md]
---

# Coverage: every citable source and claim, and where the paper uses it

Each row maps a citable source to the [outline](../outline.md) section that
uses it, or says why it is not used. "§" is the outline's section; T / F
are its exhibits. Built from the [hub-index](../../artifacts/hub-index.md)
rows under `numbers/`, `campaign-final`, `artifacts/meta-fork/` and
`artifacts/results-draft/`, the [validation](../../validation.md) rows
marked citable, the claims C1-C7 and every deviation id in
[deviations](../reproduction-deviations.md). Proposals for citable material
the outline did not yet use are marked **add**.

## 1. Hub: NUMBERS legs (`numbers/*`, citable, gate green)

| subtree | what | used in |
|---|---|---|
| `numbers/A0-k1000-check` | k 1000 on the adapted backend at every width, 7 / 7 | §3 tuning (one clause: k 1000 runs at every width); not an exhibit |
| `numbers/A1-small` | adapted probe curves, Goodreads / arXiv / synth | §4.6 F2, `c6_recall`; per-query sidecars feed the local pass rate (`c6_local_pass`) |
| `numbers/A1-pubmed` | adapted probe curves, PubMed | §4.6 F2, the 0.95 ceiling rows |
| `numbers/A1-yfcc` | adapted probe curves, YFCC + synth | not used: YFCC is out of the study (user, 2026-10-13) |
| `numbers/A1-laion10m` | adapted probe curves, LAION 10 M + synth | §4.6, §5 (N axis) |
| `numbers/A1-laion30m` | adapted probe curves, LAION 30 M real | §4.6, §5 |
| `numbers/A1-laion30m-synth` | adapted probe curves, laion30m-synth | §4.6 (p 0.001 needs 65536 / 16384) |
| `numbers/A1b-laion-small-lists` | LAION 1 / 3 M at 4096-16384 lists | §4.6 (more lists do not lift 1 / 3 M), §5 not-reached rows |
| `numbers/A2-small` | timed, interleaved with LiNR, small datasets | §5 F3 / T2 (`h2h`, `tuned`) |
| `numbers/A2-small-rerun` | the 69 unstable A2-small cells re-timed | §5 (supersedes those cells); §3 one line on unstable cells |
| `numbers/A2b-laion-small-lists` | timed, LAION 1 / 3 M | §5 |
| `numbers/A2b-laion-small-lists-rerun` | 15 unstable cells re-timed (7 stay flagged) | §5 flags |
| `numbers/A2-pubmed` | timed, PubMed | §5 T2 (`c3_journal_reverse` 0.045) |
| `numbers/A2-yfcc` | timed, YFCC + synth | not used: YFCC is out of the study |
| `numbers/A2-laion10m` | timed, LAION 10 M + synth | §5 |
| `numbers/A2-laion30m-real` | timed, LAION 30 M real | §5 |
| `numbers/A2-laion30m-synth` | timed, laion30m-synth | §5 |
| `numbers/A3-st-k2048` | timed SilverTorch vs LiNR at k 2048 | §5 (k 2048 rows) |
| `numbers/A4-bloomwidth` | adapted bloom width grid, quality only | §4.4 F4a: recall per width, index memory per width; its `bloom_fp_rate` column is the harness's standalone filter's, not used (SilverTorch's own rate: `artifacts/results-draft/20261013-bloom-fp`) |
| `numbers/L1-linr-bs64` | LiNR bs 64 / 256 | §4.1 (V2 slower at bs 64, TIMEOUTs at 256) |
| `numbers/L1b-linr-bs64-laion` | LiNR bs 64, LAION | §4.1 (LAION 2.0-2.2) |
| `numbers/L2-v3pool` | LiNR V3 pool 0.5-20 % | §4.2, F3 V3 curve |
| `numbers/L2b-v3pool-laion` | LiNR V3 pool, LAION | §4.2 (LAION rows, now in `c2_v3`) |
| `numbers/L3-linr-corr` | LiNR on correlated filters | §4.1 (no change), §4.6 (correlated vs uniform) |
| `numbers/L4-linr-k2048` | LiNR at k 2048 | §4.1, §5 |
| `numbers/A1-bloom`, `numbers/A2-bloom` | **pending** (due 2026-10-13 16:00Z): SilverTorch bloom without recheck, curves and timed against LiNR V1 / V2 exact, interleaved | §4.4 (what the index's false positives cost), §5 bloom side |

## 2. Hub: the final pass (`campaign-final*`, citable, D1-G green)

| subtree | used in |
|---|---|
| `campaign-final/merged-4016` | the final pass's one table: every LiNR number (C1-C3), Meta as published (C4 "as published" rows, C5), the data facts (T-data), the seed cells; §3, §4.1-§4.5 |
| `campaign-final` (root) | superseded by `merged-4016`; not cited. The validation EXHIBITS row's C5 range (1.08-2.16 / 1.69-5.53) comes from it; cite `c5_meta.csv` (1.07-2.20 / 1.69-5.56) |
| `campaign-final/{laion1m,laion3m,laion10m}-nsweep`, `laion30m-nsweep-p1` | LiNR V1 / V2 against N on LAION (`linr_vs_n`, `linr_mem_n`): §4.1 V1 scaling, §5 memory; **add** the V1-bandwidth line (outline candidate 10) |
| `campaign-final/{laion1m,…,laion30m,arxiv}-synth-nsweep-synth` | C1 crossover against N (`c1_crossover`, F1) |
| `campaign-final/{laion1m,…,laion30m}-nsweep-codesign` | C5 against N on Meta's code as published (`c5_meta`, F4b) |
| `campaign-final/{arxiv,pubmed}-filter-d1-st`, `goodreads-synth-synth-d1-st` | restored SilverTorch records of three d-run legs; their SilverTorch is Triton (out); the official bloom records in them feed `c4_bloom`'s "as published" rows only where `merged-4016` reads them |
| `campaign-final/ceiling-int8-{goodreads,arxiv,yfcc10m,pubmed,laion30m}` | Triton-index ceiling checks: **not cited** (controller c3 ruling 2); replaced by `artifacts/results-draft/20261012-adapted` |
| `campaign-final/{goodreads,arxiv}-synth-c3-d256` | C3 at d256 (`c3_torch`): §4.3 |
| `artifacts/exhibits/20261011-final` | T1 fragment from the final pass (LiNR verdict wording): §2 T1 |
| `artifacts/d1g-green/batch-vs-sum-20261011` | the 25 `bs16 < 16 × bs1` entries explained: §3 or §8, one line |

## 3. Hub: META-FORK (`artifacts/meta-fork/*`)

| subtree | used in |
|---|---|
| `artifacts/meta-fork/head-82fafb6` | the 1,488-cell X grid (without YFCC) and its speed: §6 T3 (`adaptation.csv`) |
| `artifacts/meta-fork/pod-b-32c0d61` | the head's 300 X cells and 36 R cells, speed at the head: §6 T3 |
| `artifacts/meta-fork/laion30m-b1-32c0d61` | X 64 / 64 bloom at 30 M, speed median 0.279: §6 T3 |
| `artifacts/meta-fork/head-r-82fafb6` | gate R (0 false positives, configured width + recheck): §4.4 (0 false positives), §6 |
| `artifacts/meta-fork/laion30m-82fafb6`, `laion30m-8ff11e0` | earlier 30 M grids (168 / 168, 72 / 72), marked NOT CITABLE in hub-index; the head's records supersede them; T3 names them only as history if at all |
| `artifacts/meta-fork/c0`, `c3-c4`, `k18-32c0d61-partial` | intermediate gates (NOT CITABLE per hub-index); not used |

## 4. Hub: results draft (`artifacts/results-draft/*`)

| subtree | used in |
|---|---|
| `artifacts/results-draft/20261012` | every table and draft figure (manifest `73d3c2ed…`): T1 verdicts, F1-F4, T2, T3, memory |
| `artifacts/results-draft/20261012-checks` | the end-task bridge (§3 recall paragraph, §4.2); the score gaps (PubMed's geometry, §4.6) |
| `artifacts/results-draft/20261012-adapted` | C6 ceiling split and local pass rate on adapted indexes (§4.6) |
| `artifacts/results-draft/20261011` | the first draft on pre-NUMBERS records; superseded, not cited |

## 5. Validation rows marked citable

| row | used in |
|---|---|
| Library suite (1017 passed), harness suite (970 passed, 161 s) | §8 availability: the gates a reviewer can rerun (material C2 Q16) |
| V1-FUSE | §4.3 mechanism (compiled torch's bs 1 win: reduction order, fp16 top-k keys) |
| META-FORK: build, Meta's suite (122), library tests, X, speed, S, capture, compile, R, C10, C11 | §6 T3 (X, R, S, capture, compile, speed); C10 / C11 are design facts for CHANGES, not paper rows |
| NUMBERS: gate, eager / graph ids 2,115 / 2,115, unstable cells, probe curves, timed legs, LiNR legs | §3 (protocol, unstable cells), §4, §5 |
| NUMBERS: end-task bridge | §3 recall paragraph |
| NUMBERS: YFCC precision losses | not used: YFCC is out of the study |
| NUMBERS: bloom width (A4) | §4.4, with the FPR attribution caveat |
| NUMBERS: C6 ceiling split, local pass rate | §4.6 |
| NUMBERS: provenance (two A100 builds in one table) | §3 protocol, one clause |
| Campaign v2: EXHIBITS, D1-G | §3 (the final pass as one table, eager / graph ids 5,942 equal / 0 differ), §2 T1 |
| Datasets: YFCC GT check 100,000 / 100,000 | not used: YFCC is out of the study (user, 2026-10-13) |
| Official against our Triton reimplementation (citable, contested; not in the study) | not used: Triton SilverTorch is out of the paper |

## 6. Claims C1-C7

| claim | section | exhibit |
|---|---|---|
| C1 | §4.1 | F1, T1 |
| C2 | §4.2 | F3 V3 curve, T1 |
| C3 | §4.3 | T1, text |
| C4 | §4.4 | F4a, T1 |
| C5 | §4.5 | F4b, T1 |
| C6 | §4.6 | F2, T1 |
| C7 | §6 | T3 |
| the head-to-head (no claim) | §5 | F3, T2, T4, F5 |
| untestable L-/S- rows (L-7…L-11, S-1, S-2, S-4, S-5, S-15) | §2 | T1 rows "not testable" with the reason |

## 7. Deviations

| ids | section | note |
|---|---|---|
| LN-1, LN-3, LN-5, LN-6, LN-8 | §7 (one line each), §4.1-§4.3 | they move C1, C2, C3 |
| LN-2 (our V4), LN-4 (V2 exact), LN-7 (not implemented) | §3 arms, one clause; LN-7 in T1's untestable rows | no verdict moves |
| OF-1, OF-9, OF-10, OF-12 | §7, §4.4, §4.5 | they move C4 and C5; OF-12 also at 10 M on PubMed (`c4_bloom`, outline candidate 8) |
| OF-3, OF-4, OF-5, OF-6, OF-11 | §6 (what the adaptation removed), §7 (upstream issues) | timing magnitudes are pre-final: mechanism only |
| OF-2, OF-7, OF-8 | §6, one clause (bit order handled in the adapter; fp16 path 3.1e-4 recall; int32 bit-exact) | OF-8 is the parity gate's basis |
| ST-1…ST-12 | §2 SilverTorch paragraph (ST-1, ST-2, ST-5, ST-9, ST-10), §3 arms (ST-3 exact mode is ours, ST-4 filter schema, ST-7 k-means, ST-6 / ST-8 kernel facts), §2 T1 untestable (ST-11, ST-12) | ST-3 changes with the user's bloom directive: the study runs Meta's bloom without our recheck |
| HB-1…HB-5 | §3 (HB-1 synthesised predicates, HB-2 oracle, HB-3 clocks, HB-4 seeds), §8 (HB-5 thesis tables not used) | — |
| D-1, D-2, D-3, D-6, D-9 | §7 "defects reproducing found in our own code" | all fixed |
| D-4, D-5, D-7, D-8 | not in the paper: engineering defects of our harness / compile path with no bearing on a number (D-8's fix is the timed-region rule in §3) | — |
| R-1, R-2, R-3, R-4 | §7: "no unexplained residual on the study's arms"; R-1 (V4) and R-2 (Triton SilverTorch) concern arms outside it, R-3 is OF-7, R-4 a golden-cell residual on LiNR V2 / V3 at 4.5e-4 / 1.7e-5 | R-4 is on a study arm; **add** one clause to §7 that it is a golden-baseline residual below any reported precision |

## 8. Citable but not yet used: proposals

- **LiNR V1 at memory bandwidth** (`linr_vs_n.csv`): one sentence in §4.1
  explaining the crossover location (outline candidate 10).
- **Meta's bloom costs recall at 10 M** (`c4_bloom.csv`, PubMed `c0_mesh`
  0.649 vs 0.664): one sentence in §4.4 and an OF-12 update (candidate 8).
- **Bloom memory as N · m_bits / 8** (`c4_bloom.csv`): the memory half of
  C4 reproduced by arithmetic (candidate 9).
- **V3's 2,000-item pool floor** (`evaluation.md` pool fractions): one
  clause in §4.2 so the flat LAION curve is not read as a result.
- **Eager torch in graph mode** (`c3_torch.csv`, 1.4-1.9×): one clause in
  §4.3: most of eager's bs 16 penalty is launch overhead.
