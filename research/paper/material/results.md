---
title: results (material C3)
created: 2026-10-12
updated: 2026-10-12
type: summary
tags: [paper]
sources: [research/paper/outline.md, research/paper/results.md, research/paper/claims.md, research/paper/reproduction-deviations.md, research/validation.md, research/decisions.md, research/artifacts/hub-index.md, research/artifacts/results-draft/build_results.py, research/artifacts/campaign-final/README.md, docs/system/evaluation.md]
---

# Material C3: results, verdicts, deviations that move them

Collected facts for the [outline](../outline.md)'s §2 (T1 verdict column),
§4 (C1-C6), §5 (F3, T2), §6 (T3 numbers) and §7 (deviations that move a
verdict). Facts with sources, not prose.

**How each number was checked.** Table numbers are read from Hub
`artifacts/results-draft/20261012` (manifest sha256 `73d3c2ed…`; fetched
with `bench fetch` on 2026-10-12, its 29 files byte-identical to the build
output on pod 1), `artifacts/results-draft/20261012-checks` and
`artifacts/results-draft/20261012-adapted` (`b8999356…`); the seed numbers
from `campaign-final/merged-4016`. "C3-derived" marks a number computed
from those files that no `build_results.py` table carries. The
[results draft](../results.md) and [validation](../../validation.md) agree
with every number here (the six wording fixes of the c3 rulings are in
both, commit `fc70420`), except the C4 attribution in
[§4 C4](#c4-bloom-fpr--0067-at-1024-bits-memory-s-8s-10-s-14).

**Citability.** The NUMBERS legs, the final pass (D1-G) and the META-FORK
gates are green ([validation](../../validation.md#numbers-adapted-silvertorch-over-the-study-grid)),
so their numbers are **citable**, with these exclusions, each listed and
never quoted as measured:
- the yfcc10m-synth LiNR cells at k 1000 (HB-6), included and flagged;
- LiNR V2 at bs 256 on PubMed `c3_journal_reverse` / `all5` and on YFCC: a
  TIMEOUT;
- the cells unstable in every run, kept and flagged.

No Triton SilverTorch record is a source (controller c3 rulings). All
numbers are seed 0 unless stated; no table carries a confidence interval:
seeds and error bars come with roadmap FULL-RERUN (user, 2026-10-11;
results.md "Uncertainty").

## T1 verdict column (§2)

These verdicts use the outline's §3 vocabulary. Each holds only under the
condition given with it.

| claim | verdict | condition | alternative wording (survey pattern 6) |
|---|---|---|---|
| C1 (L-3…L-5) | **reproduced in shape at bs 1, not in location; weak at bs 16; reversed at bs ≥ 64** (final wording: outline open item 3) | a V1 / V2 crossover in pass rate exists at bs 1 at every N and d, at p 0.79-0.97. LiNR's own high-pass setting (p ≈ 0.11) has V1 faster by 3.0×; ours has V2 faster there (fused V2, LN-1 / L-12) | "partially confirmed: the crossover exists, at a far higher pass rate than in the original" |
| C2 (L-6) | **not comparable as stated** (end-task recall, unpublished protocol); under our metric **holds in direction, not in size, at bs 16 only** | 1 % pool: V3 at 0.83 of V2 at bs 16, recall 0.89 (median); slower at bs 1 (1.08). Goodreads end-task bridge below | "cannot be fully assessed due to the recall definition; on our metric, the gain appears only at bs 16" |
| C3 (L-2) | **holds in direction, not in size** | eager torch 1.4-2.0× (bs 1), 5.8-6.8× (bs 16), never near 100×; `torch.compile` is mostly faster at bs 1 | "the gap is significantly narrower than in the original" |
| C4 (S-8…S-10, S-14) | **FPR: far below the paper's at 1024 bits, on SilverTorch's own bloom** (no recheck: 0 on arXiv / Goodreads, 4.3e-8 on PubMed at 1024 bits / k_hash 5); **memory: reproduced as arithmetic** (N · m_bits / 8); no false negatives; the transposed layout is **not in the released code** (OF-1) | `fp_adapted.py` on the adapted index; NUMBERS A4; final-pass `bloomwidth` on Meta's code as published | "confirmed within the original evaluation scope (FPR, memory); the layout claim cannot be assessed on the released code" |
| C5 (S-3, S-11) | **Meta as published: reproduced at 30 M (and 10 M bs 64), mixed at 3 M bs 64 and 10 M bs 16, reversed at 0.8-3 M.** The adapted column reads "refused by design (K17)" | N and batch size decide it; the reversal is host work in the published partial path (OF-9) | "holds at the original's scale, reverses at smaller N" |
| C6 (S-6, S-7, S-16) | **reproduced in shape** (recall falls with selectivity and N at fixed probes); the **0.95 ceiling is dataset-dependent**: reached on Goodreads, arXiv, LAION `tags4` at 10 / 30 M and 18 of 31 synthetic sweeps, not on LAION `c0_domain`, PubMed or YFCC; **int8 is the limit on YFCC and PubMed `c3_journal_reverse`, the IVF partition on LAION 1 / 3 M** | adapted code, probe curves up to `n_lists` / 4; ceiling split on adapted indexes | "partially confirmed: holds on PubMed, YFCC and LAION `c0_domain`, not on Goodreads and arXiv" |
| C7 (ours) | **holds**: bit-exact to Meta's code as published, faster end to end | gate grid ≤ 10 M, the head's 300 cells, the 30 M bloom cells | — |

## §4 claim by claim

### C1: V1 beats V2 at high pass rate, V2 wins at low (L-3, L-4, L-5)

**Original** (arXiv v3 §5.3.1-5.3.2, Tables 3-4, [claims](../claims.md)
L-3 / L-5):
- High-pass set, about 11 % pass, d128 fp16. At B = 1, PyTorch V1 averages
  4.8 ms and V2 14.6 ms, so V2 / V1 = 3.0. At B = 16, V1 takes 22.8 ms.
- Low-pass set: V2 takes 1.9 ms at B = 1 and 21.4 ms at B = 16.
- Recall@2000 (end-task) is 0.688 for all variants.
- LiNR's latencies are un-fused ops (L-12).

**Ours** (`c1_crossover.csv`, `c1_curves.csv`, `c1_real.csv`,
`linr_numbers.csv`; synthetic uniform filters, graph mode, k 100):

| fact | value | source |
|---|---|---|
| bs 1 crossover p | 0.79 (yfcc10m-synth d192) to 0.97 (arxiv-synth d256) at every N and d | `c1_crossover` bs 1 `crossover_p` |
| bs 1, V2 / V1 at p ≈ 0.001 | 0.195 (laion30m) to 0.643 (goodreads): V2 1.6-5.1× faster | `c1_crossover` `v2_over_v1_lowest_p` |
| bs 1, V2 / V1 at p ≈ 0.1 (LiNR's high-pass point) | 0.29-0.68 (V2 faster); LiNR: 3.0 (V1 faster) | `c1_curves` p 0.08-0.12; C3-derived |
| bs 16 crossover p | 0.017 (arxiv d128) to 0.024 (laion30m); 0.065 on arxiv-synth d256; **none** on laion1m-synth or yfcc10m-synth | `c1_crossover` bs 16 |
| bs 16, V2's largest edge | V2 / V1 0.75 (arxiv d256; 25 %); 0.87-0.96 on the other sets with a crossover | `c1_crossover` bs 16 `v2_over_v1_lowest_p` |
| bs 16 at p 1 | V2 / V1 2.6-6.8 | `c1_crossover` `v2_over_v1_p1` |
| bs 16, V2 / V1 at p ≈ 0.1 | 1.12-1.58 | `c1_curves`; C3-derived |
| real filters, bs 1 | V2 faster on every sweep except PubMed `c3_journal_reverse` (p 0.999, 1.07) | `c1_real` |
| real filters, bs 16 | V1 faster everywhere except PubMed `c0_mesh` (p 0.00019): V2 / V1 0.53 | `c1_real` |
| bs 64 | V2 1.3-35× slower on every real filter; LAION 2.0-2.2 | `c1_real`, `linr_numbers` (L1, L1b) |
| bs 256 | V2 3.9-55× slower where it finished; TIMEOUT on PubMed `c3_journal_reverse` / `all5` and YFCC `tags_and` | `linr_numbers` `V2_status` |
| correlated (arxiv-corr-synth, p 0.010-0.107) | bs 1: 0.35-0.41 (k 100), 0.37-0.43 (k 1000); bs 16: 0.92-1.26 | `linr_numbers` (L3) |
| k 2048 | bs 1: V2 / V1 0.08-0.77; bs 16: V2 wins only on PubMed `c0_mesh` (0.52); arXiv 1.34, Goodreads 1.61 | `linr_numbers` (L4) |
| seeds 0-2 | at the bs 16 cells beside the crossover (arxiv-synth p 0.02 / 0.05, laion30m-synth p 0.01 / 0.02, yfcc10m-synth p 0.01 / 0.02; k 100 and 1000), V2 / V1 differs across seeds by ≤ 0.011; 0 / 48 records unstable | `campaign-final/merged-4016`, suites `seeds-c1`, `seeds-c1-laion30m`; C3-derived |
| V1 scaling in N (bs 1, d256, LAION) | 0.51 / 1.23 / 3.57 / 10.8 ms at 1 / 3 / 10 / 30 M; V2 at p ≈ 0.0095 (`c0_domain`) 0.20 / 0.30 / 0.70 / 1.92 ms | `linr_vs_n.csv` |
| V1 effective bandwidth | 30 M × 256 × 2 B / 10.8 ms ≈ 1.42 TB/s, the same at 10 M (1.43); 70 % of the A100-SXM4-80GB's 2.04 TB/s. LiNR's 4.8 ms for 15.5 M × 128 fp16 ≈ 0.83 TB/s | arithmetic on `linr_vs_n.csv`; C3-derived (not a measured bandwidth) |

**Mechanism.** Our V2 is a fused compaction plus scoring kernel (LN-1).
LiNR §6 says its numbers are un-fused and predicts that fusing changes the
V1 / V2 picture (L-12). This explains why the bs 1 crossover moves from
below 11 % to 0.79-0.97. Batching favours V1: V2 gathers per query, V1
scores one shared matrix at memory bandwidth.

### C2: V3 keeping ~1 % gives ~10 % at near-parity recall (L-6)

**Original.** §5.3.1, Figure 7: Sign-OPORP at 512 bits, keeping 1 % for
rescoring, gives about 10 % further latency gain at "nearly parity"
recall@2000. That recall is end-task; Figure 7 sweeps filter size 0.5-2.5 %.

**Ours** (`c2_v3.csv`, suite `n-v3pool` = NUMBERS `L2-v3pool`; real
filters on arXiv, Goodreads, PubMed, YFCC; 10 filter rows per pool):

| pool | median recall@100 | V3 / V2 bs 16 (median) | V3 / V2 bs 1 (median) |
|---|---|---|---|
| 0.5 % | 0.843 | 0.828 | 1.035 |
| 1 % | 0.886 | 0.832 | 1.084 |
| 2 % | 0.927 | 0.857 | 1.083 |
| 5 % | 0.950 | 0.900 | 1.084 |
| 10 % | 0.978 | 0.937 | 1.097 |
| 20 % | 0.992 | 1.012 | 1.141 |

- **The pool's size.** `candidate_pool = max(2000, round(frac × mean pass
  count))` ([evaluation → pool fractions](../../../docs/system/evaluation.md#pool-fractions)):
  a fraction of the items that pass, not of the index (LN-3; LiNR's "1 %"
  is of 15.5 M items).
- **LAION 1-30 M** (`c2_v3.csv`, suite `n-v3pool-laion` = NUMBERS `L2b`, 48
  records; V2 from the final pass's `nsweep`, a different process, so the
  ratio is **not interleaved**): median recall 0.894 at 0.5-1 %, 0.917 at
  2 %, 0.943 at 5 %, 0.949 at 10 %, 0.978 at 20 %; V3 / V2 at bs 1 1.11
  (1 %) to 1.19 (20 %), at bs 16 0.97-0.98 (range 0.81-1.08).
- **Why the LAION curve is flat below 5 %.** The 2,000-item minimum: LAION
  30 M `tags4` passes about 80,000 items a query, so 0.5-2 % rounds up to
  2,000, and every selective LAION filter at ≤ 3 M sits at the floor too.
  C3-derived from the pool rule and the sweeps' pass rates (material C1
  §1.2).
- **Recall vs bits** (`c2_v3` suite `v3bits`, bs 16). Our code length is
  capped at d (LN-8); LiNR used 512.
  - Goodreads d128: 1 % pool 0.64-0.78 at 64 bits, 0.87-0.94 at 128 bits.
  - goodreads-synth: 128 bits 0.87-1.00 against 0.64-1.00 at 64.
  - PubMed: 0.93-0.99 at 256 bits, 0.990-0.998 at 768.
  - So near parity (≥ 0.99) at a 1 % pool needs 768 bits here (PubMed). At
    d128 it is not reached.
- **End-task bridge** (Goodreads d128, `product_recall.csv`, LiNR V3 rows;
  held-out recall over the exact scan's). At a 1 % pool: 0.89
  (`c0_genre`), 0.93 (`c1_lang_reverse`), 0.95 (`all4`) at k 100; 0.73-0.87
  at k 1000. A 5 % pool gives 0.95-0.98 at k 100 and 0.87-0.95 at k 1000.
  The closest we get to LiNR's "nearly parity" in its own terms, on one
  dataset, at k ≤ 1000 (LiNR: 2000). C3-derived from the CSV.

### C3: custom filter kernel ~100× over native masking (L-2)

**Original.** §6: native TF / PyTorch masking and indexing costs about 100×
in latency.

**Ours** (`c3_torch.csv`; suite `c3`: synthetic p 0.01 / 0.1 / 1 on
arxiv-synth d128 / d256, goodreads-synth d128 / d256, laion30m-synth,
yfcc10m-synth; suite `c3-real`: 10 real filters):

| arm against Triton | bs 1 | bs 16 |
|---|---|---|
| eager torch (eager mode) | 1.37-2.01, median 1.76 | 5.78-6.76, median 6.26 |
| eager torch (graph mode) | 1.39-1.58 | 1.68-1.86 |
| `torch.compile` max-autotune, synthetic | 0.50-0.96, median 0.85 | 1.49-2.33 |
| `torch.compile` max-autotune, real | 0.56-1.06, median 0.96 | 1.96-4.82, median 2.21 |

**Mechanism** ([validation](../../validation.md#library-gates) V1-FUSE row,
citable through D1-G): `torch.compile(max-autotune)` V1's bs 1 win comes
partly from another reduction order and fp16 top-k keys; its ids differ
from Triton V1 in 36 / 36 V-PILOT cells at bs 1 and are equal in 36 / 36
at bs 16. The eager-in-graph row shows that most of eager's bs 16 penalty
is launch overhead, not masking.

### C4: bloom FPR ≈ 0.067 at 1024 bits, memory (S-8…S-10, S-14)

**Original.** §6.2.2, Figure 8b, 40 M items:
- FPR 6.98 at 512 bits and 0.067 at 1024 (unit not stated; read as %).
- Memory 2.56 / 3.84 / 5 / 10 GB at 512 / 768 / 1024 / 2048 bits.

§6.2.4 says 0.00173 % → 3.89 % from 1024 to 768 bits (S-10), which is
inconsistent with S-9: the paper's own inconsistency.

**Ours.** SilverTorch's own bloom, measured on the adapted code's bloom
index with no recheck (`fp_adapted.py`, Hub
`artifacts/results-draft/20261013-bloom-fp`): per query (bloom survivors −
exact passes) / (N − exact passes) over the whole index, 10,000 queries,
m_bits 64-2048 × k_hash 3 / 5 on Goodreads `c0_genre`, arXiv `all4` /
`c0_maincat` / `c3_nversions`, PubMed `c0_mesh`; recall and `index_mib` from
NUMBERS `A4-bloomwidth` (`c4_bloom.csv`). The records' `bloom_fp_rate` column
is the harness's standalone filter module's, not SilverTorch's, and is not
used.

| what | value | source / status |
|---|---|---|
| SilverTorch bloom FPR vs width (k_hash 3) | PubMed `c0_mesh` 2.3e-2 / 3.5e-3 / 5.1e-4 / 5.8e-5 / 7.7e-6 / 8.5e-7 at 64 / 128 / 256 / 512 / 1024 / 2048 bits; arXiv `c0_maincat` 2.3e-3 at 64, 0 from 512; Goodreads 4.0e-3 at 64, 0 from 512. At the study's 1024 bits / k_hash 5: 0 (arXiv, Goodreads), 4.3e-8 (PubMed). No false negatives | `c4_bloom` (`fp_rate_mean`); measured |
| adapted false positives in the results | 0 at every width (C13 exact recheck); recall@100 identical at every width per sweep (PubMed `c0_mesh` 0.6640, arXiv `all4` 0.7627, Goodreads `c0_genre` 0.9456) | `c4_bloom`; validation META-FORK gate R; citable |
| adapted index memory vs width | PubMed 9,082 → 10,226 → 11,447 MiB at 64 / 1024 / 2048 bits: +1,221 MiB per 1024 bits = 10 M × 128 B, i.e. N · m_bits / 8; arXiv 799 → 1,141 → 1,506; Goodreads 213 → 305 → 402 | `c4_bloom` `index_mib`; the per-bit slope is C3-derived |
| the paper's memory as arithmetic | 40 M × 1024 / 8 = 5.12 GB; × 512 → 2.56; × 768 → 3.84; × 2048 → 10.24: the S-9 figures exactly | C3-derived |
| Meta as published, index memory | PubMed 7,766 MiB (k_hash 3) / 7,956 (k_hash 5); arXiv 454 / 482; Goodreads 130 / 143 | final pass `bloomwidth` records (Meta's bundle bloom, width from the bundle's term count, OF-1 / OF-10) |
| Meta as published, recall@100 | PubMed `c0_mesh` **0.6493** (k_hash 3) and 0.66397 (k_hash 5) against the adapted 0.66399; arXiv and Goodreads within 4e-5 of the adapted | final pass `bloomwidth` records; C3-derived comparison. Meta's bloom lets non-matching items into the top-100 at 10 M on the rarest filter (about 1,900 passing items), not only at 30 M (OF-12) |

- **S-14 / OF-1:** the transposed layout is not in Meta's released code (a
  per-2048-doc bundle bloom instead). A paper-vs-code finding.
- **Units:** our FPR is a fraction. At 1024 bits SilverTorch's bloom gives
  ≤ 7.7e-6 (0.00077 %, k_hash 3) against the paper's 0.067 (%) or
  0.00173 %: lower under either reading, at different N and attribute sets.

### C5: co-design 1.79-2.15× over full mask then probe (S-3, S-11)

**Original.** §6.2.3, Figure 9, at 20 M × 128: latency 1.55 → 0.72 ms,
1.79-2.15× on average. §6.1.1 gives about 17-25 % QPS end to end.

**Meta as published, eager** (`c5_meta.csv`; final pass `codesign`,
`nsweep-codesign`, `codesign-laion30m`; full / partial, above 1 means
co-design wins):

| N | bs 1 | bs 16 | bs 64 |
|---|---|---|---|
| 0.8 M | 0.78-0.80 | 0.81-0.89 | — |
| 1 M (LAION) | — | 0.80-0.81 | 0.83-0.94 |
| 3 M | 0.79-0.82 | 0.78-0.85 | 0.87-1.25 (mixed) |
| 10 M | — | 0.83-1.20 (median 0.99, mixed) | 1.13-2.35 |
| 30 M | — | 1.07-2.20 | 1.69-5.56 |

- The EXHIBITS row in validation gives 1.08-2.16 (bs 16) / 1.69-5.53
  (bs 64) at 30 M: it was computed on the first, 4,001-record aggregate;
  `c5_meta.csv` reads `merged-4016`. Cite `c5_meta.csv`.
- **Adapted:** refuses `bloom_path="full"` by design (K fix 17); the
  column reads "refused by design (K17)" (controller c2 ruling 2).
  `c5_adapted.csv` (`82fafb6`, before K17) is not cited.
- **Mechanism:** in Meta's published partial path, host work grows with
  the sub-query count. That is OF-9: partial is 14-22 % slower at 0.8-3 M
  and 1.70-3.69× faster at 30 M
  ([deviations](../reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper);
  the magnitudes are v2.1 measurements, mechanism only). The adapted code
  builds the partial mask in its payload kernels, so it is not host-bound.

### C6: int8 IVF recall ceiling < 0.95; selective filters need more probes (S-6, S-7, S-16)

**Original.** §6.2.1, §4.2, §7: "cannot reach 0.95 recall"; "no
measurable recall loss at 64 probes and top-2048"; selective filters may
need more probes. Figure 7 covers 20 M × 128, B = 16, recall 0.35-0.92.

**Ours, adapted code** (`c6_recall.csv` from NUMBERS A1 / A1b; `at24.csv`
from the timed legs).

At 24 probes (`at24`, bs 16, k 100, recall@100):
- goodreads-synth, 4096 lists: 0.930 / 0.775 / 0.365 / 0.047 at p 1 / 0.1 /
  0.01 / 0.001.
- Real filters: Goodreads 0.42-0.89, arXiv 0.68-0.85, PubMed 0.39-0.81, YFCC
  0.44, LAION 0.33-0.57.

Highest recall@100 on the probe curve, with the smallest
(`n_lists` / `n_probe`) reaching 0.90 / 0.95 (k 100):

| dataset | filter (p) | max | 0.90 at | 0.95 at |
|---|---|---|---|---|
| Goodreads 0.8 M | `c0_genre` (0.33) | 0.986 | 4096/32 | 4096/64 |
| | `c1_lang_reverse` (0.62) | 0.984 | 4096/128 | 4096/256 |
| | `all4` (0.041) | 0.971 | 4096/512 | 4096/1024 |
| arXiv 3 M | `c0_maincat` (0.14) | 0.966 | 2048/64 | 2048/128 |
| | `c3_nversions` (0.44) | 0.966 | 2048/64 | 2048/128 |
| | `all4` (0.0094) | 0.963 | 2048/256 | 2048/512 |
| LAION 1 M | `c0_domain` (0.0095) / `tags4` (0.0027) | 0.846 / 0.833 | — | — |
| LAION 3 M | `c0_domain` / `tags4` | 0.871 / 0.909 | — / 8192/2048 | — |
| PubMed 10 M | `c0_mesh` (0.00019) | 0.9495 | 4096/512 | — |
| | `c3_journal_reverse` (0.999) | 0.922 | 4096/256 | — |
| | `all5` (0.018) | 0.898 | — | — |
| YFCC 10 M | `tags_and` (0.018) | 0.718 | — | — |
| LAION 10 M | `c0_domain` / `tags4` | 0.897 / 0.951 | — / 4096/1024 | — / 65536/16384 |
| LAION 30 M | `c0_domain` / `tags4` | 0.908 / 0.961 | 16384/4096 / 4096/1024 | — / 16384/4096 |

- **Synthetic filters.** 0.95 is reached on 18 of 31 synthetic sweeps
  (7 uniform sets × 4 pass rates + 3 correlated), including all three
  arxiv-corr-synth sweeps. Not reached: p 0.001 at 0.8-10 M; laion3m-synth
  p 0.01; any laion1m-synth sweep (only 512 lists, probes ≤ 128: a short
  curve); any yfcc10m-synth sweep (max 0.55-0.81). At 30 M p 0.001, 0.95
  needs 65536 / 16384.
- **Uniform vs correlated** (arXiv, 2048 lists). Correlated p 0.01-0.1
  reaches 0.90 at 64 probes and 0.95 at 256. Uniform p 0.01 needs 256 /
  512, uniform p 0.1 64 / 256. Correlated filters are no harder than
  uniform ones at the same pass rate, and easier at p 0.01.
- **k 1000 is harder.** Max recall@1000 at p 0.001: 0.23 (goodreads-synth),
  0.61 (arxiv-synth). Goodreads `all4`: 0.90.
- **Ceiling split on adapted indexes** (`c6_ceiling.csv`, Hub
  `artifacts/results-draft/20261012-adapted`: one `official-fork` index per
  (dataset, n_lists), its own probes, recall@100 over the same probed lists
  as shipped int8 / per-row int8 / fp16; 26 cells, 16384 / 4096 unless
  stated):

| cell | shipped | per-row | fp16 | limit |
|---|---|---|---|---|
| yfcc10m-synth p 1 (4096 / 256) | 0.526 | 0.753 | 0.972 | int8 |
| yfcc10m-synth p 0.01 (4096 / 256) | 0.750 | 0.874 | 0.976 | int8 |
| PubMed `c3_journal_reverse` | 0.922 | 0.938 | 0.996 | int8 |
| PubMed `c0_mesh` / `all5` | 0.946 / 0.887 | 0.953 / 0.891 | 0.976 / 0.903 | mixed |
| arXiv real and synth | 0.952-0.963 | 0.967-0.973 | 0.989-0.998 | int8, small (+0.03-0.04) |
| Goodreads real and synth | 0.975-0.989 | 0.981-0.993 | 0.987-1.000 | small |
| LAION 1 M / 3 M (8192 / 16384 lists) | 0.833-0.909 | 0.833-0.911 | 0.834-0.914 | partition (fp16 ≤ +0.009) |
| LAION 30 M `c0_domain` / `tags4` | 0.903 / 0.955 | 0.909 / 0.961 | 0.918 / 0.970 | partition, mostly |
| laion30m-synth p 0.01-1 | 0.962-0.963 | 0.970-0.973 | 0.983-0.991 | small |

  A fifth column, "global" (plain torch, the same arithmetic as shipped),
  is within 0.0005 of shipped except on YFCC (0.011-0.018). Per-row and
  fp16 match the earlier Triton-index check within 0.0005.
- **Mechanism, YFCC** (`score_gaps_yfcc_pubmed.json`,
  `score_gaps_laion_arxiv.json`, 500 queries per dataset): the items sit
  in a narrow cone (a random pair's cosine is 0.97; PubMed 0.67, arXiv
  0.64, LAION 10 M 0.52); the top-100 spans 0.0022 in score (others
  0.076-0.155) and the gap from rank 100 to 101 is 3.9e-6 (others 7.8e-5
  to 1.6e-4); fp16 storage moves a score by 1.5e-5, per-row int8 by
  1.9e-4, 9 % of the top-100 span (others 0.4-0.7 %).
- **Local pass rate** (`c6_local_pass.csv`, Hub `20261012-adapted`,
  `gls_adapted.py`): l_q = |unfiltered top-100 ∩ filtered top-100| / 100
  from the oracle blobs, recall@100 from A1-small's per-query sidecars
  (`official-fork`), every curve point at the smallest n_lists, Goodreads
  and arXiv real filters. Spearman(recall, l_q) > Spearman(recall, p_q) in
  45 / 45 cells. At 32 probes:

| sweep | recall | ρ(l_q) | ρ(p_q) |
|---|---|---|---|
| Goodreads `c0_genre` | 0.918 | 0.49 | 0.20 |
| Goodreads `c1_lang_reverse` | 0.760 | 0.63 | 0.39 |
| Goodreads `all4` | 0.485 | 0.73 | 0.60 |
| arXiv `c0_maincat` | 0.875 | 0.38 | 0.01 |
| arXiv `c3_nversions` | 0.865 | 0.24 | 0.19 |
| arXiv `all4` | 0.724 | 0.64 | 0.40 |

## §5 SilverTorch against LiNR

### F3 points (Pareto per dataset)

- **SilverTorch probe curve:** `h2h.csv` (graph) or `h2h_eager.csv`.
  Columns `dataset, filter_kind, sweep, perf_bs, perf_k, n_lists, n_probe,
  recall, st_ms`.
- **LiNR V1 / V2:** the same rows, columns `V1`, `V2` (ms, interleaved in
  the same process), with `linr_best` / `linr_ms` / `st_over_linr` /
  `winner` and the `flagged` and `leg` columns. 1,677 rows; 30 flagged
  (`summary.json`).
- **V3 pool curve:** `c2_v3.csv` (recall and V3 / V2; V3 ms = V3 / V2 ×
  the V2 point), LAION from `n-v3pool-laion` there too.
- Use bs 16. bs 1 changes the picture where LiNR V2 is the best LiNR (real
  filters at bs 1) and on `c3_journal_reverse` (V1).
- Draft figures in the same artifact: `fig_h2h_synth`, `fig_tuned_vs_n`,
  `fig_c1_v2_over_v1` (PDF and PNG).

### T2: matched-recall winners

Source: `tuned.csv` (1,089 rows), `by_size.csv`. A cell = dataset × filter
× bs × k × target. Winner and ratio = SilverTorch over the faster of LiNR
V1 / V2. Clause filters, k 100:

| dataset | filter | 0.90, bs 1 / 16 / 64 | 0.95, bs 1 / 16 / 64 |
|---|---|---|---|
| Goodreads | `c0_genre` | 0.58 / 0.25 / 0.16 | 0.58 / 0.26 / 0.19 |
| | `c1_lang_reverse` | 0.51 / 0.31 / 0.25 | 0.51 / 0.41 / 0.40 |
| | `all4` | 0.56 / 0.46 / 0.52 | 0.56 / 0.73 / 0.82 |
| arXiv | `c0_maincat` | 0.46 / 0.18 / 0.22 | 0.57 / 0.63 / 0.98 |
| | `c3_nversions` | 0.27 / 0.19 / 0.23 | 0.31 / 0.47 / 0.65 |
| | `all4` | 0.29 / 0.32 / 0.39 | 0.33 / 0.53 / 0.64 |
| LAION 3 M | `tags4` | 0.50 / 0.43 / 0.66 | not reached |
| PubMed | `c0_mesh` | 0.41 / 0.27 / 0.34 | not reached |
| | `c3_journal_reverse` | 0.045 / 0.20 / 0.47 | not reached |
| LAION 10 M | `tags4` | 0.26 / 0.39 / 0.66 | 0.41 / 0.45 / 0.78 |
| LAION 30 M | `tags4` | 0.15 / 0.20 / 0.33 | 0.22 / 0.38 / 0.63 |
| | `c0_domain` | 0.42 / 0.40 / 0.71 | not reached |

Not reached at 0.90, so LiNR wins by default: LAION 1 M (both filters),
LAION 3 M / 10 M `c0_domain`, PubMed `all5`, YFCC `tags_and`.

**Counts** (clause; reached / cells, then SilverTorch wins):

| k | target | bs 1 | bs 16 | bs 64 |
|---|---|---|---|---|
| 100 | 0.90 | 35/49, 35 | 35/49, 35 | 35/49, 33 |
| 100 | 0.95 | 26/49, 26 | 26/49, 25 | 26/49, 23 |
| 1000 | 0.90 | 24/45, 24 | 24/45, 24 | 24/45, 21 |
| 1000 | 0.95 | 17/45, 17 | 17/45, 16 | 17/45, 10 |

- Median ratio of reached cells, k 1000, bs 1, 0.90: 0.42.
- **Bloom filters** (reached cells): k 100 8 / 12 at 0.90 and 5 / 12 at
  0.95; k 1000 4 / 12 and 2 / 12. SilverTorch wins all of them, ratio
  0.09-0.85.
- **k 2048** (A3 + L4): arXiv `c0_maincat` 0.23-0.72 and Goodreads
  `c0_genre` 0.30-0.81 at 0.90 / 0.95. PubMed `c0_mesh` tops out at 0.856,
  so LiNR wins.
- **No CIs** (seed 0; controller c3 ruling 7). The interleaved groups
  (`interleave_group` in the A2 / A3 parquets) allow paired CIs once
  FULL-RERUN brings seeds.

**Regime boundaries.**
- **Pass rate.**
  - At p 0.001, SilverTorch reaches 0.90 (bs 16, k 100) on arxiv-synth
    (0.35 of LiNR), laion10m-synth (0.45, 4096 / 1024) and laion30m-synth
    (0.25, 65536 / 8192), not on goodreads-synth (max 0.835), laion1m-synth
    (0.681, a 512-list curve), laion3m-synth (0.829) or yfcc10m-synth
    (0.811). C3-derived from `tuned.csv`.
  - At p ≥ 0.01, it wins wherever it reaches the target (0.03-0.94), except
    LAION 1 M synth at bs 64 and p ≥ 0.1 (1.11-1.29).
  - Real filters at p < 0.01 that SilverTorch does not reach at 0.90:
    LAION 1 M `c0_domain` / `tags4` and LAION 3 M / 10 M `c0_domain`.
    PubMed `c0_mesh` at p 0.00019 does reach 0.90.
  - The binding variable is the recall the curve can reach, not the pass
    rate alone.
- **N.** 0.95 is not reached within the probe cap on PubMed, YFCC or LAION
  `c0_domain` at 10-30 M. LAION `tags4` reaches it at 10 M and 30 M only
  with 16384-65536 lists. Where reached, the tuned bs 1 ratio falls with N:
  0.51-0.58 (Goodreads), 0.27-0.46 (arXiv), 0.26 (LAION 10 M `tags4`),
  0.15-0.42 (LAION 30 M) (`by_size.csv`).
- **k and batch.** All 17 cells that SilverTorch reaches and still loses
  are at p ≥ 0.1 and bs ≥ 16: 15 at bs 64 and 2 at bs 16 (laion3m-synth
  p 1 at 0.95, k 100 and k 1000); the worst: laion3m-synth p 1, bs 64,
  0.95: 2.74 (k 100) and 2.70 (k 1000).
- **At 24 probes** (`at24.csv`): SilverTorch is faster in every one of
  363 cells. bs 1: 0.016-0.986. bs 16: 0.017-0.395.

### Index memory and build time

Source: `memory.csv` (adapted index and LiNR's), `linr_mem_n.csv`.

| dataset | SilverTorch adapted, clause | LiNR V1 / V2 | ratio | build (adapted) |
|---|---|---|---|---|
| Goodreads | 209 MiB | 292 MiB | 0.72 | 1.8 s |
| arXiv | 776 MiB | 1,095 MiB | 0.71 | 3.7 s |
| YFCC | 6,870 MiB | 8,545 MiB | 0.80 | 28.9 s |
| PubMed | 9,015 MiB (8.8 GiB) | 16,174 MiB (15.8 GiB) | 0.56 | 85 s (4096 lists) |
| LAION 10 M | 3,056 MiB (4096 lists) | 5,341 MiB | 0.57 | 34.5 s |

- **Bloom indexes are larger than clause indexes.** Adapted bloom: arXiv
  1,141, Goodreads 306, PubMed 10,236 MiB; above LiNR's on arXiv and
  Goodreads. Meta's published bloom index is smaller (arXiv 483, Goodreads
  145 MiB).
- LiNR V3 adds 2-6 % over V1 / V2. LiNR builds in ≤ 1.1 s.
- **Build times** (adapted): 2-4 s at ≤ 3 M with the default lists (LAION
  3 M 20-40 s at 8192 / 16384); 29-85 s at 10 M with 4096 lists; with
  65536 lists PubMed 1,426 s (23.8 min), LAION 10 M 578 s (9.6 min), LAION
  30 M 1,734-1,742 s (29 min): 10-29 min.

### Goodreads end-task bridge (recall caveat)

`product_recall.json` / `.csv`. 17 configurations per filter: the
SilverTorch curve and tuned points, LiNR V1 / V2, and V3 at pools
0.5-20 %.

- **Pearson:** at k 100, 0.875 (`c0_genre`), 0.881 (`c1_lang_reverse`),
  0.905 (`all4`). At k 1000, 0.906-0.935. Pooled (held-out recall divided
  by the exact scan's): 0.896 / 0.913.
- **Spearman:** 0.70-0.78 at k 100 and 0.90-0.96 at k 1000.
- **Where our recall ≥ 0.90:** end-task recall is 0.93-1.0 of the exact
  scan's, median 0.99 over the approximate configurations (0.998 with the
  exact LiNR rows counted).
- **Where our recall < 0.70:** 0.35-0.97 of the exact scan's (median
  0.75; 27 rows). Four SilverTorch configurations at our recall 0.61-0.70
  keep 0.93-0.97 of end-task recall.

## §6 T3 numbers (Meta as published vs adapted)

Source: `adaptation.csv` (the `head-82fafb6` gate grid), `summary.json`, and
[validation](../../validation.md#meta-fork-backendofficial-fork) (the
head's 300 cells and 30 M). Controller c2 ruling 3: T3's speed rows come
from the META-FORK gate grid.

| row | value |
|---|---|
| X gate, ≤ 10 M | 1,536 / 1,536 cells `torch.equal` (ids and scores), 18 groups by dataset × d × filter kind |
| X gate, head `32c0d61` | 300 / 300 (Meta's bloom semantics, none and exact) |
| X gate, 30 M LAION | bloom partial + full 64 / 64 at `32c0d61`; 168 / 168 at `82fafb6`; none / exact last at `8ff11e0` (72 / 72) |
| eager time, adapted / Meta | per-group medians 0.120 (YFCC clause) to 0.817 (PubMed none); min 0.009; max 1.030 (arXiv d256 none) |
| head speed | 300 X cells median 0.230 (max 0.764); 36 R cells median 0.287 (max 0.788) |
| A/A floor | 1.001-1.035 |
| 30 M speed | median 0.279, min 0.059, max 0.467 (64 records) |
| eager vs graph ids, NUMBERS adapted cells | 2,151 / 2,151 equal |
| S / capture / compile gates | 0 syncs, 9-10 kernels a forward (published: 3-6 syncs, 50 kernels); 1 graph launch a replay; 35 compile tests pass |

Per-dataset ranges for T3 (median of adapted / Meta by filter kind):

| dataset | none | clause | bloom |
|---|---|---|---|
| arXiv d128 | 0.50 | 0.17 | 0.35 |
| arXiv d256 | 0.67 | 0.17 | 0.39 |
| Goodreads | 0.26 | 0.15 | 0.22 |
| PubMed | 0.82 | 0.22 | 0.32 |
| YFCC | 0.63 | 0.12 | — |
| arxiv-synth / goodreads-synth | — | 0.28 / 0.19 | 0.24 / 0.19 |

## §4 / §7: failures, negatives, cost

**Timeouts, gate misses, partial and failed cells the paper's numbers
touch** (sources: [validation](../../validation.md#numbers-adapted-silvertorch-over-the-study-grid),
D1-G row, the leg parquets):

| case | what | rerun? | paper must say |
|---|---|---|---|
| LiNR V2 bs 256 on PubMed `c3_journal_reverse` / `all5`, YFCC `tags_and` | TIMEOUT after 3.6-9.8 h with 80 GB full (`c3_journal_reverse` finished at 2.55 s a batch after 11 h); YFCC V1 bs 256 did not run (same process) | no | listed as TIMEOUT, not measured |
| yfcc10m-synth LiNR k 1000 | V1 recall 0.983-0.995 < 0.99 exact gate (fp16 storage, HB-6); p01 / p1 k 1000 did not run | no; included and flagged (user) | flagged; SilverTorch on YFCC synth at k 100 only |
| A2-yfcc | 72 of 168 rows `partial` (k 100 entries only; 1 failed = p01 V1 gate miss) | no | YFCC synth at k 100 |
| L1-linr-bs64 | 12 of 148 rows `partial` (the bs 256 gaps above); 6 failed records (PubMed OOM) superseded by `ok` | yes | — |
| A2-laion30m-real / L1b | 3 / 4 failed, superseded by `ok` reruns | yes | — |
| unstable | A2-small 69 → 0 after rerun; A2b 15 → 7 (5 SilverTorch at ~0.2 ms on LAION 1 M, 2 LiNR V2); A2-pubmed 2, A3 1 not rerun; 6 tuned points rest on a flagged cell | partly | flagged cells named |
| pilot pass rate (V-PILOT, goodreads-synth) | 0.000990 at target 0.001: 1.01 % low (789 of 797,084), inside binomial noise at N·p 797 | no | achieved pass rates are reported, not targets |
| `bs16 < 16 × bs1` (D1-G) | 25 entries: 24 YFCC `tags_and` V2 / V3 (≈ 18×), explained by the per-query spread (against the same queries, bs 16 is 8-18 % cheaper than 16 bs 1 calls); 1 `torch.compile` V2 at k 1000 (1.58, an autotune pick) | no | one line in §3 or §8 |
| probe cap | curves stop at `n_lists` / 4; laion1m-synth only at 512 lists (≤ 128 probes) | — | "not reached within the probe cap" |
| 0.99 | not reached by any timed SilverTorch point | — | stated |

**Negative results.**
- **Router** (IVF ↔ exact): dropped 2026-10-11. On PubMed 10 M it is
  slower than exact V2 at bs 1 and collapses onto IVF at p ≈ 1; its bs 16
  passes are within 2 % of the front
  ([decisions](../../decisions.md#campaign-v2-user-2026-10-08), Hub
  `campaign-v2.5/pubmed-router`). Its records are pre-final (v2.5) and
  partly Triton SilverTorch: a sentence, no numbers.
- **Our int8 LiNR V4:** removed from the library (LN-2). Not in the study.

**GPU-hours** ([provenance §1](../provenance-and-disclosure.md#1-hardware),
labelled approximate, controller c3 ruling 8):
- Final pass 73.5 GPU-h, sized from its records
  ([campaign-final](../../artifacts/campaign-final/README.md#sizing)).
- NUMBERS about 26 GPU-h on pod 1 (its driver log, 48 runs) plus about 30
  on pods b and d (the plan's estimate). `elapsed_s` cannot be summed: it
  repeats on every perf row of a cell (a naive sum gives 830 h).
- C4 / C6 adapted checks about 1.2 GPU-h (A4 15 min, ceiling 57 min).
- Material C2's record-derived estimate (final ≈ 55, NUMBERS ≈ 82) counts
  differently; the paper uses provenance's row.

## §7: deviations that move a verdict

| id | effect on a verdict |
|---|---|
| LN-1 / L-12 (fused V2) | moves C1's bs 1 crossover from below 11 % (LiNR) to 0.79-0.97: "reproduced in shape", not in location |
| LN-5, HB-2 (recall@k against the exact oracle; LiNR used end-task recall@2000) | makes C2 "not comparable as stated"; any recall-based verdict (C2, C6) is on our metric; the Goodreads bridge links the two |
| LN-3 (pool as a fraction of passing items, min 2,000) | C2's "1 %" is a different absolute pool; the LAION curve is flat below 5 % |
| LN-8 (V3 bits capped at d; LiNR 512) | C2's recall at a 1 % pool is lower at d ≤ 256; near parity only at 768 bits (PubMed) |
| LN-6 (eager vs compiled torch as the "native" baseline) | C3's size depends on the baseline: eager 1.4-6.8×, compiled 0.5-4.8× |
| OF-9 (host-bound partial path in Meta's code) | C5 as published reverses below 30 M |
| OF-10 (Meta's bloom ignores `m_bits`), OF-1 (no transposed layout) | the width curve exists only on the adapted code; S-14 not testable |
| OF-12 (Meta's bloom admits non-matching items) | recorded at 30 M; also costs 0.015 recall@100 on PubMed `c0_mesh` at 10 M (`c4_bloom`); the adapted bloom has 0 false positives (C13) |
| HB-6 (fp16 item storage vs fp32 oracle) | YFCC LiNR k 1000 flagged; does not move a verdict |
| HB-4 (seed 0) | every ratio is unreplicated except the C1 seed cells (≤ 0.011 spread); no CI on T2 |
